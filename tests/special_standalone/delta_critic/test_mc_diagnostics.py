# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy

import pytest

from examples.delta_critic.mc_diagnostics import mc_batch_diagnostics


def record(role, rewards):
    return {
        "rollout": {"terminal_reward": 0.0},
        "critic_role": role,
        "full_selected_indices": [0, 2, 4],
        "labels": [
            {
                "token_index": index,
                "v_prefix": sum(values) / len(values),
                "mc_sampling_config": {"max_tokens": 8, "effective_max_response_tokens": 3},
                "mc_continuations": [{"reward": value, "token_ids": [1, 2, 3]} for value in values],
            }
            for index, values in zip((0, 2), rewards, strict=True)
        ],
    }


def test_signal_support_and_effective_caps_are_split_by_role():
    records = [record("train", ((0, 0), (0, 1))), record("diagnostic", ((1, 1), (1, 1)))]
    before = deepcopy(records)
    metrics = mc_batch_diagnostics(records)
    assert records == before
    assert metrics["delta_mc/all/td_pairs"] == 2
    assert metrics["delta_mc/all/td_target_mean"] == 0.25
    assert metrics["delta_mc/all/td_target_std"] == 0.25
    assert metrics["delta_mc/all/td_target_zero_fraction"] == 0.5
    assert metrics["delta_mc/train/mixed_reward_state_fraction"] == 0.5
    assert metrics["delta_mc/train/all_zero_reward_state_fraction"] == 0.5
    assert metrics["delta_mc/diagnostic/all_positive_reward_state_fraction"] == 1
    assert metrics["delta_mc/all/branches_with_known_cap"] == 8
    assert metrics["delta_mc/all/branch_at_cap_fraction"] == 1
    assert metrics["delta_mc/all/value_reward_mean_abs_error"] == 0


def test_nonadjacent_missing_targets_and_unknown_caps_are_not_invented():
    row = record("train", ((0, 1), (0, 0)))
    row["full_selected_indices"] = [0, 1, 2, 4]
    for label in row["labels"]:
        label.pop("mc_sampling_config")
    metrics = mc_batch_diagnostics([row])
    assert metrics["delta_mc/all/td_pairs"] == 0
    assert metrics["delta_mc/all/branches_with_known_cap"] == 0
    assert metrics["delta_mc/diagnostic/states"] == 0
    assert metrics["delta_mc/all/branches"] == 4


def test_reward_only_exports_and_unlabelled_originals():
    row = record("train", ((0, 1), (0, 0)))
    for label in row["labels"]:
        label["prefix_continuation_rewards"] = [item["reward"] for item in label.pop("mc_continuations")]
    metrics = mc_batch_diagnostics([row, {"rollout": {"terminal_reward": 1.0}, "labels": []}])
    assert metrics["delta_mc/all/original_reward_mean"] == 0.5
    assert metrics["delta_mc/all/td_target_negative_fraction"] == 1
    assert metrics["delta_mc/all/branches"] == 4
    assert metrics["delta_mc/all/branches_with_tokens"] == 0


def test_rejects_nonfinite_rewards():
    row = record("train", ((0, 1), (0, 0)))
    row["rollout"]["terminal_reward"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        mc_batch_diagnostics([row])
