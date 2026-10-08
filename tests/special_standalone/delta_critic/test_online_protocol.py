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

import numpy as np
import pytest

from examples.delta_critic.online_protocol import (
    annotate_protocol_batch,
    choose_expansion_prompts,
    select_mc_state_indices,
    stable_prompt_id,
)


def test_stable_prompt_hash_ignores_numpy_container_and_mapping_order():
    prompt = [{"role": "user", "content": "solve α"}]
    array = np.array([{"content": "solve α", "role": "user"}], dtype=object)
    assert stable_prompt_id(prompt) == stable_prompt_id(array)


def test_pair_selects_adjacent_actor_grid_and_reserves_both_suffix_budgets():
    candidates = [[{"prob": 0.9}]] * 2000
    candidates[32] = [{"prob": 0.4}]
    candidates[1500] = [{"prob": 0.1}]
    # The most uncertain endpoint has no eligible successor: 1600 leaves <512.
    selected = select_mc_state_indices(
        [0, 32, 96, 1500, 1600], candidates, protocol={"sampling_layout": "pair"}, response_cap=2048
    )
    assert selected == [32, 96]
    assert (
        select_mc_state_indices([1500, 1600], candidates, protocol={"sampling_layout": "pair"}, response_cap=2048) == []
    )


def test_single_selects_highest_uncertainty_with_same_suffix_reserve():
    candidates = [[{"prob": 0.9}]] * 2000
    candidates[1500] = [{"prob": 0.1}]
    assert select_mc_state_indices(
        [0, 32, 1500, 1600], candidates, protocol={"sampling_layout": "single"}, response_cap=2048
    ) == [1500]


def test_tied_uncertainty_preserves_existing_preference_for_later_index():
    candidates = [[{"prob": 0.5}]] * 100
    assert select_mc_state_indices(
        [0, 32, 64, 96], candidates, protocol={"sampling_layout": "pair"}, response_cap=2048
    ) == [64, 96]
    assert select_mc_state_indices(
        [0, 32, 64, 96], candidates, protocol={"sampling_layout": "single"}, response_cap=2048
    ) == [96]


def test_role_quota_replaces_ineligible_candidates_inside_same_batch():
    entries = {i: {"role": "diagnostic" if i >= 110 else "train", "eligible": i not in (0, 1, 110)} for i in range(128)}
    selected = choose_expansion_prompts(entries, {}, prompt_count=128, policy_step=1)
    assert selected == set(range(2, 12)) | {111, 112}
    assert len(selected) * 2 * 8 == 192
    single = choose_expansion_prompts(entries, {"sampling_layout": "single"}, prompt_count=128, policy_step=1)
    assert len(single) * 8 == 192


def test_quota_failure_does_not_duplicate_or_add_samples():
    entries = {i: {"role": "diagnostic" if i == 127 else "train", "eligible": True} for i in range(128)}
    with pytest.raises(ValueError, match="diagnostic has 1 eligible originals, needs 2"):
        choose_expansion_prompts(entries, {}, prompt_count=128, policy_step=1)


def test_annotation_fixed_partition_and_shared_original_manifest(tmp_path):
    from tensordict import NonTensorData, NonTensorStack, TensorDict

    prompts = [[{"role": "user", "content": f"problem {i}"}] for i in range(4096)]
    ids = [stable_prompt_id(prompt) for prompt in prompts]
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"all_prompt_ids": ids, "diagnostic_prompt_ids": sorted(ids)[:512]}))
    expansion = {
        "prompts_per_step": 128,
        "protocol": {
            "enabled": True,
            "partition_path": str(partition),
            "original_cache_dir": str(tmp_path / "originals"),
            "original_cache_mode": "write",
        },
    }

    def batch(prompt_rows):
        return TensorDict({"raw_prompt": NonTensorStack(*[NonTensorData(prompt) for prompt in prompt_rows])}, [128])

    first = annotate_protocol_batch(batch(prompts[:128]), expansion, 1)
    assert list(first["delta_prompt_id"]) == ids[:128]
    assert list(first["delta_critic_role"]) == [
        "diagnostic" if pid in sorted(ids)[:512] else "train" for pid in ids[:128]
    ]
    expansion["protocol"]["original_cache_mode"] = "read"
    second = annotate_protocol_batch(batch(prompts[:128]), expansion, 1)
    assert first["delta_protocol_batch_id"] != second["delta_protocol_batch_id"]
    with pytest.raises(ValueError, match="IDs/order differ"):
        annotate_protocol_batch(batch(prompts[1:129]), expansion, 1)
