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
"""Golden semantics and optional differential checks against the source checkout."""

import ast
import importlib.util
import math
import os
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path

import pytest

from examples.delta_critic.batch_adapter import pad_response_rows, read_positions, selected_inference_rows
from examples.delta_critic.contracts import DeltaExample, TokenVector
from examples.delta_critic.legacy_adapter import adapt_legacy_rows, terminal_after_last_token
from examples.delta_critic.target_ops import (
    critic_targets,
    denormalize_predictions,
    fit_normalization,
    normalize,
    selected_labels,
    state_advantages,
    value_model_final_grpo,
)


@pytest.fixture
def legacy():
    rollout = {
        "id": "r1",
        "prompt_id": "p1",
        "prompt_token_ids": [10, 11],
        "response_token_ids": [20, 21, 22, 23, 24, 25],
        "terminal_reward": 1.0,
        "policy_token_mask": [1, 1, 0, 1, 1, 1],
        "finish_reason": "length",
        "split": "train",
    }
    rows = [
        {
            "rollout_id": "r1",
            "state_id": "s1",
            "token_index": 1,
            "v_prefix": 0.2,
            "v_next": 0.3,
            "prefix_response_token_ids": [20],
            "mc_num_samples": 32,
        },
        {"rollout_id": "r1", "state_id": "s4", "token_index": 4, "v_prefix": 0.7, "v_next": 0.4},
    ]
    return rollout, rows


@pytest.fixture
def example(legacy):
    rollout, rows = legacy
    examples, diagnostics = adapt_legacy_rows([rollout], rows)
    assert not diagnostics
    return examples[0]


@pytest.fixture
def config():
    return value_model_final_grpo(
        label_mode="paired_next_state", loss_mask="selected_state", target_normalization="standardize"
    )


def test_golden_local_and_segment(example, config):
    targets, signal = critic_targets(example, config)
    assert targets.values == pytest.approx([0, 0.1, 0, 0, -0.3, 0])
    assert targets.mask == signal == (0, 1, 0, 0, 1, 0)
    segment_targets, _ = critic_targets(example, replace(config, label_mode="selected_segment"))
    assert segment_targets.values == pytest.approx([0, 0.5, 0, 0, 0.3, 0])
    learned = state_advantages(example, config, raw_predictions={1: 0.1, 4: -0.3})
    assert learned.values == pytest.approx([0, 0.1, 0.1, 0.1, -0.3, -0.3])
    assert learned.mask == (0, 1, 1, 1, 1, 1)
    oracle_config = replace(config, advantage_source="mc_label", policy_advantage_scope="selected_token")
    oracle = state_advantages(example, oracle_config)
    assert oracle.values == pytest.approx([0, 0.1, 0, 0, -0.3, 0])
    assert oracle.mask == targets.mask


def test_zero_delta_remains_supervised(example, config):
    states = (replace(example.states[0], delta=0.0), example.states[1])
    targets, signal = critic_targets(replace(example, states=states), config)
    assert targets.values[1] == 0
    assert targets.mask[1] == signal[1] == 1
    targets, _ = critic_targets(example, replace(config, loss_mask="response"))
    assert targets.mask == (1,) * 6


def test_normalizations_and_policy_intersection(example, config):
    targets, _ = critic_targets(example, config)
    target_stats = fit_normalization([targets], population="critic_targets")
    assert (target_stats.count, target_stats.mean, target_stats.std) == pytest.approx((2, -0.1, 0.2))
    normalized_target = normalize(targets, mode="standardize", population="critic_targets", stats=target_stats)
    # Source normalizes masked background too, without enabling its mask.
    assert normalized_target.values[0] == pytest.approx(0.5)
    restored = denormalize_predictions(normalized_target.values, mode="standardize", stats=target_stats)
    assert restored == pytest.approx(targets.values)
    advantage = state_advantages(example, config, raw_predictions={1: 0.1, 4: -0.3})
    stats = fit_normalization([advantage], population="state_advantages")
    assert (stats.count, stats.mean, stats.std) == pytest.approx((5, -0.06, 0.19595917942265426))
    normalized = normalize(advantage, mode=config.advantage_normalization, population="state_advantages", stats=stats)
    assert normalized.values[0] == 0
    padded = pad_response_rows([example], config, [normalized], width=8)[0]
    assert padded["response_valid_mask"] == (1, 1, 1, 1, 1, 1, 0, 0)
    assert padded["state_advantage_mask"] == (0, 1, 1, 1, 1, 1, 0, 0)
    assert padded["policy_loss_mask"] == (0, 1, 0, 1, 1, 1, 0, 0)
    assert stats.count == 5  # includes policy-masked token 2 before final loss masking
    assert padded["delta_scalar_positions"] == (2, 3, 4, 5, 6, 7, -1, -1)
    assert padded["policy_logprob_positions"] == (1, 2, 3, 4, 5, 6, -1, -1)
    with pytest.raises(ValueError, match="matching population"):
        normalize(advantage, mode="standardize", population="state_advantages", stats=target_stats)


def test_read_positions_and_prefix(example):
    assert read_positions(2, 6, left_padding=3)["delta_scalar"] == (5, 6, 7, 8, 9, 10)
    states = (replace(example.states[0], token_index=0), replace(example.states[1], token_index=5))
    rows = selected_inference_rows(DeltaExample(example.rollout, states))
    assert rows[0]["input_ids"] == (10, 11, 20)
    assert rows[0]["value_position"] == 2
    assert rows[1]["input_ids"][-1] == 25
    assert rows[1]["value_position"] == 7
    with pytest.raises(ValueError, match="prompt token"):
        read_positions(0, 1)


def test_legacy_precedence_and_metadata(legacy, config):
    rollout, rows = legacy
    rows[0]["delta"] = 0.9
    examples, diagnostics = adapt_legacy_rows([rollout], rows[::-1])
    assert len(diagnostics) == 1
    assert selected_labels(examples[0], "paired_next_state")[0] == 0.9
    assert examples[0].states[0].v_next == 0.3
    assert examples[0].rollout.metadata["split"] == "train"
    assert examples[0].states[0].mc_num_samples == 32
    assert examples[0].states[1].mc_num_samples is None
    state_advantages(examples[0], config, raw_predictions={1: 0.2, 4: 0.3})
    assert rows[0]["delta"] == examples[0].states[0].delta == 0.9
    rollout["response_token_ids"][0] = 999
    assert examples[0].rollout.metadata["response_token_ids"][0] == 20


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r, s: s.append(dict(s[0], state_id="other")), "Duplicate selected"),
        (lambda r, s: s[0].update(token_index=6, prefix_response_token_ids=r["response_token_ids"]), "out of range"),
        (lambda r, s: s[0].update(prefix_response_token_ids=[99]), "prefix_response"),
        (lambda r, s: s[0].update(prompt_token_ids=[99]), "prompt_token"),
        (lambda r, s: s[0].update(rollout_id="absent"), "missing rollout"),
        (lambda r, s: s[0].pop("v_prefix"), "Missing required"),
        (lambda r, s: r.update(policy_token_mask=[1]), "binary entries"),
        (lambda r, s: s[1].update(state_id="s1"), "Duplicate state_id"),
        (lambda r, s: s[0].update(v_prefix=float("nan")), "finite"),
    ],
)
def test_invalid_legacy(legacy, mutation, match):
    rollout, rows = legacy
    mutation(rollout, rows)
    with pytest.raises(ValueError, match=match):
        adapt_legacy_rows([rollout], rows)


def test_empty_zero_variance_and_missing_prediction(example, config):
    for vector in [TokenVector((), ()), TokenVector((0, 0), (1, 1))]:
        with pytest.raises(ValueError):
            fit_normalization([vector], population="critic_targets")
        with pytest.raises(ValueError):
            fit_normalization([vector], population="state_advantages")
    no_states = replace(example, states=())
    assert critic_targets(no_states, config)[0].mask == (0,) * 6
    with pytest.raises(ValueError, match="all selected"):
        state_advantages(example, config, raw_predictions={1: 0.1})
    with pytest.raises(ValueError, match="requires delta or v_next"):
        selected_labels(replace(example, states=(replace(example.states[0], v_next=None),)), "paired_next_state")
    advantage = state_advantages(example, config, raw_predictions={1: 0.1, 4: -0.3})
    with pytest.raises(ValueError, match="cannot truncate"):
        pad_response_rows([example], config, [advantage], width=5)
    assert normalize(advantage, mode="none", population="state_advantages") == advantage
    assert denormalize_predictions([0, 1], mode="none") == (0, 1)


@pytest.mark.parametrize(
    "finish,length,assume,expected",
    [
        ("eos", False, False, True),
        ("stop", False, False, True),
        ("length", True, False, True),
        ("length", False, False, False),
        ("abort", True, True, False),
        (None, True, True, True),
    ],
)
def test_terminal(finish, length, assume, expected):
    assert (
        terminal_after_last_token(
            finish, treat_length_truncation_as_terminal=length, assume_legacy_last_token_terminal=assume
        )
        is expected
    )


def test_terminal_unknown_requires_choice():
    with pytest.raises(ValueError, match="Missing finish_reason"):
        terminal_after_last_token(
            None, treat_length_truncation_as_terminal=True, assume_legacy_last_token_terminal=False
        )


def test_external_value_model_differential(legacy, example, config):
    """Optional: compare actual source functions, without importing its GPU stack."""
    source = os.environ.get("VALUE_MODEL_ROOT")
    if source is None:
        pytest.skip("Set VALUE_MODEL_ROOT to compare against the original checkout")
    root = Path(source) / "delta_value_llm_exp"
    spec = importlib.util.spec_from_file_location("source_offline_grpo", root / "offline_grpo.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    rollout, rows = legacy
    for source_name, scope in [("delta", "segment"), ("mc_local", "selected_token")]:
        source_rows = [dict(row, delta_last_chunk=value) for row, value in zip(rows, [0.1, -0.3], strict=True)]
        expected, mask = original.state_advantage_vectors(rollout, source_rows, source_name)
        cfg = replace(
            config, policy_advantage_scope=scope, advantage_source="critic" if source_name == "delta" else "mc_label"
        )
        actual = state_advantages(example, cfg, raw_predictions={1: 0.1, 4: -0.3} if source_name == "delta" else None)
        assert actual.values == pytest.approx(expected)
        assert actual.mask == tuple(mask)
        stats = fit_normalization([actual], population="state_advantages")
        expected_stats = original.state_advantage_stats([{"state_advantages": expected, "state_mask": mask}])
        expected_tuple = tuple(expected_stats[k] for k in ("count", "mean", "std"))
        assert (stats.count, stats.mean, stats.std) == pytest.approx(expected_tuple)
    # Evaluate source definitions in isolation: the delta branch has no model dependencies.
    tree = ast.parse((root / "05_train_token_scalar_model_accelerate_local.py").read_text())
    names = {"DenseTokenScalarDataset", "_target_stats", "_mask_for_row"}
    subset = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
    namespace = {"bisect_right": bisect_right, "VALUE_CENTERING_MODES": {"none"}, "math": math}
    exec(compile(subset, "source_delta_dataset", "exec"), namespace)
    for label_mode in ("selected_segment", "paired_next_state"):
        for mask_mode in ("selected_state", "response"):
            dataset = namespace["DenseTokenScalarDataset"]([rollout], rows, "delta", mask_mode, label_mode)
            expected = dataset[0]
            targets, signal = critic_targets(example, replace(config, label_mode=label_mode, loss_mask=mask_mode))
            assert targets.values == pytest.approx(expected["token_targets"])
            assert targets.mask == tuple(expected["token_loss_mask"])
            assert signal == tuple(expected["token_signal_mask"])
            stats = fit_normalization([targets], population="critic_targets")
            expected_stats = namespace["_target_stats"](dataset)
            expected_tuple = tuple(expected_stats[k] for k in ("count", "mean", "std"))
            assert (stats.count, stats.mean, stats.std) == pytest.approx(expected_tuple)
