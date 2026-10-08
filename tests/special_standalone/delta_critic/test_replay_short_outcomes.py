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

import math

import pytest

from examples.delta_critic.contracts import NormalizationStats
from examples.delta_critic.replay_short_outcomes import (
    ReplayRow,
    compare_short_advantages,
    reconstruct_batch_rows,
    reconstruct_stats_without_short_rows,
    summarize_rows,
)


def _candidate_rows(length):
    return [[{"token_id": 1, "prob": 0.8}, {"token_id": 2, "prob": 0.2}] for _ in range(length)]


def _expanded_fixture():
    parent_uid = "prompt-a_0_0"
    rollout_id = "v0:prompt-a:0"
    continuations = [
        {"continuation_index": 0, "token_ids": [31, 99], "reward": 1.0, "delta_top_logprobs": _candidate_rows(2)},
        {"continuation_index": 1, "token_ids": [32, 99], "reward": 0.0, "delta_top_logprobs": _candidate_rows(2)},
    ]
    return [
        {
            "uid": parent_uid,
            "delta_mc_record": {
                "rollout": {
                    "id": rollout_id,
                    "prompt_token_ids": [1, 2],
                    "response_token_ids": [10, 99],
                    "policy_token_mask": [1, 1],
                    "terminal_reward": 0.0,
                },
                "states": [{"token_index": 1}],
                "labels": [
                    {
                        "rollout_id": rollout_id,
                        "prefix_response_token_ids": [10],
                        "mc_continuations": continuations,
                    }
                ],
            },
        },
        {"uid": "prompt-a_0_1", "delta_mc_record": {}},
        {"uid": "prompt-a_0_2", "delta_mc_record": {}},
    ]


def test_reconstructs_exact_mc_tokens_and_leave_one_out_baselines():
    rows = reconstruct_batch_rows(
        _expanded_fixture(),
        short_response_budget=3,
        default_baseline=0.5,
        marker_ids={"configured_eos_token_id": 99, "im_end_token_id": 100},
        selection_config={"states_per_response": 1, "min_token_gap": 0},
    )
    assert [row.kind for row in rows] == ["original", "continuation", "continuation"]
    first, success, failure = rows
    assert first.response_token_ids == (10, 99)
    assert first.known_expansion_mc_indices == (1,)
    assert success.prompt_token_ids == (1, 2, 10)
    assert success.response_token_ids == (31, 99)
    assert success.baseline == 0.0
    assert success.baseline_source == "leave_one_out_siblings"
    assert failure.baseline == 1.0
    assert success.selected_token_indices == (1,)
    assert success.selection_status == "reconstructed_from_saved_top_logprobs"
    assert all(row.is_short for row in rows)
    assert all(row.trailing_marker == "configured_eos" for row in rows)


def test_new_stats_remove_legacy_reward_tokens_from_the_critic_population():
    values = [1.0, 1.0, 0.0, 0.0, -1.0, 2.0]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    old_stats = NormalizationStats(float(len(values)), mean, std, "state_advantages")
    short = ReplayRow(
        row_id="short",
        parent_id="short",
        kind="original",
        prompt_token_ids=(1,),
        response_token_ids=(2, 3),
        policy_token_mask=(1.0, 1.0),
        reward=1.0,
        baseline=0.5,
        baseline_source="fixed_default",
        short_text_tokens=2,
        response_prefix_tokens=0,
        short_response_budget=2,
    )
    long = ReplayRow(
        row_id="long",
        parent_id="long",
        kind="original",
        prompt_token_ids=(1,),
        response_token_ids=(2, 3),
        policy_token_mask=(1.0, 1.0),
        reward=0.0,
        baseline=0.5,
        baseline_source="fixed_default",
        short_text_tokens=3,
        response_prefix_tokens=0,
        short_response_budget=2,
    )
    new_stats, audit = reconstruct_stats_without_short_rows(old_stats, [short, long])
    assert new_stats.count == 4
    assert new_stats.mean == pytest.approx(0.25)
    assert new_stats.std == pytest.approx(math.sqrt(1.1875))
    assert audit["excluded_short_response_policy_tokens"] == 2
    assert audit["new_critic_population_count"] == 4


def test_old_reward_path_can_make_zero_reward_positive_but_new_outcome_is_negative():
    rows = reconstruct_batch_rows(
        _expanded_fixture(),
        short_response_budget=3,
        default_baseline=0.5,
        marker_ids={"configured_eos_token_id": 99, "im_end_token_id": 100},
        selection_config={"states_per_response": 1, "min_token_gap": 0},
    )
    old_stats = NormalizationStats(10.0, -0.1, 0.5, "state_advantages")
    new_stats = NormalizationStats(8.0, 0.0, 0.4, "state_advantages")
    compared = compare_short_advantages(
        rows,
        old_stats=old_stats,
        new_stats=new_stats,
        advantage_clip=5.0,
        outcome_scale=0.5,
    )
    original_zero = next(row for row in compared if row["kind"] == "original" and row["reward"] == 0.0)
    continuation_zero = next(row for row in compared if row["kind"] == "continuation" and row["reward"] == 0.0)
    assert original_zero["old_advantage_mean"] == pytest.approx(0.2)
    assert original_zero["new_advantage_mean"] == pytest.approx(-1.0)
    assert continuation_zero["baseline"] == 1.0
    assert continuation_zero["new_advantage_mean"] == pytest.approx(-2.0)


def test_eos_audit_reports_markers_without_changing_short_length_rule():
    rows = reconstruct_batch_rows(
        _expanded_fixture(),
        short_response_budget=3,
        default_baseline=0.5,
        marker_ids={"configured_eos_token_id": 99, "im_end_token_id": 100},
        selection_config={"states_per_response": 1, "min_token_gap": 0},
    )
    stats = NormalizationStats(10.0, -0.1, 0.5, "state_advantages")
    comparisons = compare_short_advantages(
        rows, old_stats=stats, new_stats=stats, advantage_clip=5.0, outcome_scale=0.5
    )
    summary = summarize_rows(
        rows,
        marker_ids={"configured_eos_token_id": 99, "im_end_token_id": 100},
        short_comparisons=comparisons,
    )
    assert summary["groups"]["all"]["configured_eos_rows_ending_with_eos"] == 3
    assert summary["groups"]["all"]["short_threshold_flips_if_one_trailing_stop_marker_is_excluded"] == 0
    assert summary["eos_boundary_flip_row_ids"] == []
