# Copyright 2026 Individual Contributor: zhiyu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest

from examples.delta_critic.batch_sample import sampling_settings
from examples.delta_critic.mc_labeling import MCConfig, MCContinuation, label_mc_states, validate_delta_top_logprobs
from examples.delta_critic.policy_mc import export_mc_records
from examples.delta_critic.policy_online import select_uncertainty_indices
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.training_data import continuation_rows


def _continuation_record(token_ids, *, state_id="r:t0", delta_top_logprobs=None):
    row = {
        "state_id": state_id,
        "rollout_id": "r",
        "continuation_id": f"{state_id}:k0",
        "continuation_index": 0,
        "continuation_token_ids": list(token_ids),
        "prompt_token_ids": [1],
        "reward": 1.0,
    }
    if delta_top_logprobs is not None:
        row["delta_top_logprobs"] = delta_top_logprobs
    return row


def _label(state_id="r:t0", *, v_prefix=0.25):
    return {
        "state_id": state_id,
        "rollout_id": "r",
        "token_index": 0,
        "prompt_token_ids": [1],
        "prefix_response_token_ids": [],
        "v_prefix": v_prefix,
    }


def test_uncertainty_continuation_selection_matches_policy_score_with_anchor_budget():
    token_ids = list(range(20))
    top_probs = [
        [{"prob": probability}, {"prob": 1.0 - probability}]
        for probability in [0.01, 0.02, 0.03, 0.9, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5] * 2
    ]
    config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        continuation_selection="uncertainty",
        continuation_state_count=5,
        continuation_min_gap=3,
        continuation_max_candidates=2,
    )
    continuation = _continuation_record(token_ids, delta_top_logprobs=top_probs)

    rows, _ = continuation_rows([continuation], [_label(v_prefix=0.25)], config)

    policy_mask = [float(index >= 3 and index != 0) for index in range(len(token_ids))]
    expected_uncertain = select_uncertainty_indices(
        top_probs,
        policy_mask,
        states_per_response=4,
        min_token_gap=3,
        max_candidates=2,
    )
    selected = rows[0]["terminal_suffix_response_indices"]
    assert selected == sorted({0, *expected_uncertain})
    assert len(selected) <= config.continuation_state_count
    assert selected[0] == 0
    assert all(index >= config.continuation_min_gap for index in selected[1:])
    assert all(right - left >= config.continuation_min_gap for left, right in zip(selected, selected[1:], strict=False))
    assert rows[0]["terminal_comp_target"] == 0.75  # reward - V(anchor)


def test_uncertainty_ties_use_policy_later_index_tie_break():
    token_ids = list(range(12))
    top_probs = [[{"prob": 0.4}, {"prob": 0.3}] for _ in token_ids]
    config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        continuation_selection="uncertainty",
        continuation_state_count=4,
        continuation_min_gap=2,
    )

    rows, _ = continuation_rows([_continuation_record(token_ids, delta_top_logprobs=top_probs)], [_label()], config)

    assert rows[0]["terminal_suffix_response_indices"] == [0, 7, 9, 11]


def test_uncertainty_state_budget_uses_configured_count_including_anchor():
    token_ids = list(range(12))
    top_probs = [[{"prob": 0.4}] for _ in token_ids]
    config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        continuation_selection="uncertainty",
        continuation_state_count=8,
        continuation_min_gap=0,
    )

    rows, _ = continuation_rows([_continuation_record(token_ids, delta_top_logprobs=top_probs)], [_label()], config)

    selected = rows[0]["terminal_suffix_response_indices"]
    assert len(selected) == config.continuation_state_count
    assert selected[0] == 0


def test_missing_uncertainty_metadata_fails_and_legacy_uniform_stays_unchanged():
    continuation = _continuation_record(range(10))
    legacy_config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        continuation_state_count=3,
        continuation_min_gap=2,
    )

    legacy_rows, _ = continuation_rows([continuation], [_label()], legacy_config)
    assert legacy_rows[0]["terminal_suffix_response_indices"] == [0, 4, 8]

    uncertainty_config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        continuation_selection="uncertainty",
    )
    with pytest.raises(ValueError, match="requires stored token-aligned actor delta_top_logprobs"):
        continuation_rows([continuation], [_label()], uncertainty_config)


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        [[]],
        [[{}]],
        [[{"prob": "not-a-number"}]],
        [[{"prob": float("nan")}]],
        [[{"prob": -0.1}]],
        [[{"prob": 1.1}]],
        [["malformed candidate"]],
    ],
)
def test_uncertainty_selection_rejects_malformed_probability_rows(metadata):
    config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        continuation_selection="uncertainty",
    )
    row = _continuation_record([7], delta_top_logprobs=metadata)
    with pytest.raises(ValueError, match="delta_top_logprobs"):
        continuation_rows([row], [_label()], config)


def test_mc_metadata_alignment_and_online_export_preserve_actor_probabilities(tmp_path):
    rollout = {
        "id": "r",
        "prompt_id": "p",
        "split": "train",
        "response_index": 0,
        "prompt_token_ids": [1],
        "response_token_ids": [2],
        "terminal_reward": 0.5,
        "finish_reason": "stop",
        "actor_version": "actor-1",
    }
    states = [
        {
            "state_id": "r:t0",
            "rollout_id": "r",
            "prompt_id": "p",
            "split": "train",
            "token_index": 0,
            "prompt_token_ids": [1],
            "prefix_response_token_ids": [],
        }
    ]
    candidates = [[{"token_id": 7, "prob": 0.7, "logprob": -0.3}]]
    config = MCConfig(
        mode="prefix_only",
        continuations_per_state=1,
        sampling_config={"max_tokens": 4, "delta_top_logprobs": 5},
        actor_version="actor-1",
        continuation_skip_special_tokens=True,
        save_continuations=True,
    )
    labels = label_mc_states(
        [rollout],
        states,
        config,
        sample=lambda request, sampling: [MCContinuation((7,), "x", candidates)],
        decode_prefix=lambda ids: "",
        score=lambda text, row: 1.0,
    )
    record = {"rollout": rollout, "states": states, "labels": labels}

    export_mc_records([record], tmp_path)

    continuation = json.loads((tmp_path / "continuations_train.jsonl").read_text().splitlines()[0])
    assert continuation["delta_top_logprobs"] == candidates
    assert labels[0]["mc_continuations"][0]["delta_top_logprobs"] == candidates

    malformed_config = MCConfig(
        mode="prefix_only",
        continuations_per_state=1,
        sampling_config={"max_tokens": 4},
        actor_version="actor-1",
        continuation_skip_special_tokens=True,
        save_continuations=True,
    )
    with pytest.raises(ValueError, match="must align with token_ids"):
        label_mc_states(
            [rollout],
            states,
            malformed_config,
            sample=lambda request, sampling: [MCContinuation((7,), "x", [])],
            decode_prefix=lambda ids: "",
            score=lambda text, row: 1.0,
        )


def test_offline_sampling_requests_top_probs_only_when_saved():
    config = {
        "smoke": {"states_per_response": 1},
        "rollout": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 20, "top_k_logprobs": 7},
        "state_selection": {"max_candidates": 5},
        "mc": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_continuation_tokens": 8,
            "continuations_per_state": 2,
            "save_continuations": True,
        },
        "model": {"actor_model": "actor-1"},
    }

    _, saved = sampling_settings(config, "prefix_only")
    config["mc"]["save_continuations"] = False
    _, unsaved = sampling_settings(config, "prefix_only")

    assert saved.sampling_config["delta_top_logprobs"] == 7
    assert "delta_top_logprobs" not in unsaved.sampling_config


def test_top_probability_validator_requires_token_alignment():
    with pytest.raises(ValueError, match="must align"):
        validate_delta_top_logprobs([1, 2], [[{"prob": 0.8}]])
