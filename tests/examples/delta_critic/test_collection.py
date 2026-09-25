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
"""Exercise the V1 export -> selection -> MC -> step 1 validation path."""

import ast
import asyncio
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.delta_critic.collection import export_v1_rollouts, prompt_split
from examples.delta_critic.legacy_adapter import adapt_legacy_rows
from examples.delta_critic.mc_labeling import (
    MCConfig, MCContinuation, build_mc_requests, label_mc_states, label_mc_states_async,
)
from examples.delta_critic.state_selection import select_spaced_states, select_states
from examples.delta_critic.v1_sampler import sample_v1_continuations


def _rollout(finish_reason="stop"):
    rows = [{
        "uid": "u1", "prompt_id": "math-1", "session_id": 0, "global_steps": 7,
        "prompts": [10, 11], "responses": [20, 21, 22],
        "response_mask": [1, 1, 1], "rollout_log_probs": [-0.1, -0.2, -0.3],
        "reward_score": 1.0, "finish_reason": finish_reason, "gold_answer": "42",
    }]
    return export_v1_rollouts(
        rows, actor_version="actor-step-7", sampling_config={"temperature": 0.7, "top_p": 0.9},
        eval_fraction=0.2, split_seed="experiment-1",
    )


def _config(mode="paired_next_state", **kwargs):
    return MCConfig(
        mode=mode, continuations_per_state=2, sampling_config={"temperature": 0.8, "max_tokens": 4},
        actor_version="actor-step-7", continuation_skip_special_tokens=True, **kwargs,
    )


def test_v1_export_and_prompt_split():
    rows = _rollout()
    assert rows[0]["response_token_ids"] == [20, 21, 22]
    assert rows[0]["behavior_logprobs"] == [-0.1, -0.2, -0.3]
    assert rows[0]["actor_version"] == "actor-step-7"
    assert rows[0]["split"] == prompt_split("math-1", eval_fraction=0.2, seed="experiment-1")
    assert prompt_split("math-1", eval_fraction=0.2, seed="experiment-1") == rows[0]["split"]
    with pytest.raises(ValueError, match="align"):
        export_v1_rollouts(
            [dict(uid="u", prompt_id="p", global_steps=1, prompts=[1], responses=[2], response_mask=[1],
                  reward_score=1, rollout_log_probs=[0, 0])], actor_version="a",
            sampling_config={"temperature": 1}, eval_fraction=0.2, split_seed="x",
        )
    rows = export_v1_rollouts(
        [dict(uid="u", prompt_id="p", global_steps=1, prompts=[1], responses=[2],
              response_mask=[1], reward_score=1, stop_reason="completed",
              extra_fields={"finish_reason": "length"})],
        actor_version="a", sampling_config={"temperature": 1}, eval_fraction=0, split_seed="x",
    )
    assert rows[0]["finish_reason"] == "length"
    assert rows[0]["stop_reason"] == "completed"


def test_selection_source_tie_break_and_prefix():
    rollout = _rollout()[0]
    assert select_spaced_states([(1, 0), (1, 1), (1, 2)], 2, 2) == [2, 0]
    states = select_states(
        [rollout], strategy="indices", states_per_response=2, min_token_gap=1,
        indices_by_rollout={rollout["id"]: [0, 2]},
    )
    assert [row["token_index"] for row in states] == [2, 0]
    assert states[0]["prefix_response_token_ids"] == [20, 21]
    assert states[1]["prefix_response_token_ids"] == []
    assert states[0]["state_id"] == f"{rollout['id']}:t2"
    with pytest.raises(ValueError, match="top-k"):
        select_states([rollout], strategy="uncertainty", states_per_response=1)


def test_uncertainty_selection_uses_source_score():
    rollout = _rollout()[0]
    rollout["tokens"] = [
        {"token_index": i, "entropy": value, "top1_prob": 0.5,
         "top_candidates": [{"token_id": 100 + i, "prob": 0.7}]}
        for i, value in enumerate([0.1, 0.8, 0.3])
    ]
    states = select_states(
        [rollout], strategy="uncertainty", states_per_response=1,
        entropy_weight=1, low_top1_weight=1, final_window_tokens=0, final_window_weight=0,
    )
    assert states[0]["token_index"] == 1
    assert states[0]["selection_score"] == pytest.approx(1.3)
    assert states[0]["candidate_mass"] == pytest.approx(0.7)


def test_mc_end_to_end_terminal_and_request_dedup():
    rollouts = _rollout()
    rid = rollouts[0]["id"]
    states = select_states(
        rollouts, strategy="indices", states_per_response=2, indices_by_rollout={rid: [1, 2]},
    )
    config = _config(save_individual_rewards=True)
    requests, links = build_mc_requests(rollouts, states, config)
    assert len(requests) == 2  # before t2 == after t1
    assert links[f"{rid}:t2"][1] is None  # terminal shortcut
    calls = []

    def sample(request, sampling):
        calls.append(request.input_ids)
        return [MCContinuation((30,), "1"), MCContinuation((31,), "0")]

    labels = label_mc_states(
        rollouts, states, config, sample=sample,
        decode_prefix=lambda ids: "".join(map(str, ids)),
        score=lambda response, rollout: float(response.endswith("1")),
    )
    by_index = {row["token_index"]: row for row in labels}
    assert len(calls) == 2
    assert by_index[1]["v_prefix"] == 0.5
    assert by_index[1]["mc_num_samples"] == 2
    assert by_index[1]["mc_next_num_samples"] == 2
    assert by_index[2]["v_next"] == 1.0
    assert by_index[2]["mc_next_num_samples"] == 0
    assert by_index[2]["delta"] == 0.5
    assert by_index[2]["mc_actor_version"] == "actor-step-7"
    examples, diagnostics = adapt_legacy_rows(rollouts, labels)
    assert len(examples[0].states) == 2 and not diagnostics


def test_mc_nonterminal_last_token_and_validation():
    rollouts = _rollout(finish_reason="length")
    rid = rollouts[0]["id"]
    states = select_states(rollouts, strategy="indices", states_per_response=1, indices_by_rollout={rid: [2]})
    requests, links = build_mc_requests(
        rollouts, states, _config(treat_length_truncation_as_terminal=False)
    )
    assert len(requests) == 2 and links[states[0]["state_id"]][1] is not None
    missing_finish = deepcopy(rollouts)
    missing_finish[0]["finish_reason"] = None
    with pytest.raises(ValueError, match="Missing finish_reason"):
        build_mc_requests(missing_finish, states, _config())
    bad_prefix = deepcopy(states)
    bad_prefix[0]["prefix_response_token_ids"] = [999]
    with pytest.raises(ValueError, match="prefix_response"):
        build_mc_requests(rollouts, bad_prefix, _config())
    tool_rollout = deepcopy(rollouts)
    tool_rollout[0]["policy_token_mask"][1] = 0
    with pytest.raises(ValueError, match="single-turn"):
        build_mc_requests(tool_rollout, states, _config())


def test_v1_server_client_sampler_round_trip():
    rollouts = _rollout()
    rid = rollouts[0]["id"]
    states = select_states(rollouts, strategy="indices", states_per_response=1, indices_by_rollout={rid: [2]})
    calls = []

    class Client:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(token_ids=[30])

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return "1"

    async def sample(request, config):
        return await sample_v1_continuations(
            request, config, server_client=Client(), tokenizer=Tokenizer()
        )

    labels = asyncio.run(label_mc_states_async(
        rollouts, states, _config(), sample=sample, decode_prefix=lambda ids: "",
        score=lambda response, rollout: float(response == "1"),
    ))
    assert len(calls) == 2
    assert all(call["prompt_ids"] == [10, 11, 20, 21] for call in calls)
    assert all(call["sampling_params"] == {"temperature": 0.8, "max_tokens": 4} for call in calls)
    assert labels[0]["v_prefix"] == 1.0
    assert labels[0]["v_next"] == 1.0


def test_source_selection_and_mc_request_differential():
    root = os.environ.get("VALUE_MODEL_ROOT")
    if not root:
        pytest.skip("Set VALUE_MODEL_ROOT for source differential tests")
    source_dir = Path(root) / "delta_value_llm_exp"

    def source_functions(file, names):
        module = ast.parse((source_dir / file).read_text())
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"Any": Any}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_dir / file), "exec"), namespace)
        return namespace

    selection = source_functions("02_select_states.py", {"select_spaced_states"})
    scored = [(0.7, 0), (0.9, 1), (0.9, 2)]
    source_chosen = selection["select_spaced_states"](
        [(score, index, [], 0.0) for score, index in scored], 2, 2
    )
    assert select_spaced_states(scored, 2, 2) == [item[1] for item in source_chosen]

    source_mc = source_functions(
        "03_estimate_values_mc_vllm.py",
        {"is_terminal_after_last_response_token", "_prefix_key", "build_unique_prefix_requests"},
    )
    rollouts = _rollout()
    rid = rollouts[0]["id"]
    states = select_states(
        rollouts, strategy="indices", states_per_response=2, indices_by_rollout={rid: [1, 2]},
    )
    source_requests, source_links = source_mc["build_unique_prefix_requests"](
        states, {rid: rollouts[0]}, {"mc": {"state_value_mode": "paired_next_state"}}
    )
    requests, links = build_mc_requests(rollouts, states, _config())
    assert {(row["rollout_id"], tuple(row["prefix_response_token_ids"])) for row in source_requests} == {
        (request.rollout_id, request.prefix_response_token_ids) for request in requests
    }
    for state in states:
        before, after = source_links[(rid, state["token_index"])]
        ours_before, ours_after = links[state["state_id"]]
        assert before == (ours_before.rollout_id, ours_before.prefix_response_token_ids)
        assert after == (None if ours_after is None else (ours_after.rollout_id, ours_after.prefix_response_token_ids))
