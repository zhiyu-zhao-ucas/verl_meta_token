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

from dataclasses import replace

import pytest

from examples.delta_critic.train import initialization_changes
from examples.delta_critic.training_config import ScalarConfig


def _source():
    config = ScalarConfig("unused", max_length=4096)
    normalization = {"enabled": True, "mode": "standardize", "mean": 0.002, "std": 0.018, "count": 100}
    return config, {
        "config": config.as_dict(),
        "normalization": normalization,
        "terminal_stats": None,
        "tokenizer_sha256": "tokenizer",
    }


def test_td_warm_start_adds_terminal_objective_without_changing_local_scale():
    source, meta = _source()
    hybrid = replace(source, objective="hybrid_terminal_composition", terminal_normalization="td")
    changes = initialization_changes(meta, hybrid, meta["normalization"], {"std": 0.2}, "tokenizer")
    assert changes == [
        "Objective expanded from local_td0 to hybrid_terminal_composition",
        "terminal_stats refitted on new train split",
    ]


@pytest.mark.parametrize("key,value", [("mean", 0.01), ("std", 0.1)])
def test_td_warm_start_refuses_to_reinterpret_normalized_head(key, value):
    source, meta = _source()
    hybrid = replace(source, objective="hybrid_terminal_composition")
    with pytest.raises(ValueError, match="preserve raw delta scale"):
        initialization_changes(meta, hybrid, {**meta["normalization"], key: value}, {"std": 0.2}, "tokenizer")


@pytest.mark.parametrize("overrides", [{"loss_mask": "selected_state"}, {"delta_label_mode": "paired_next_state"}])
def test_td_warm_start_rejects_local_contract_changes(overrides):
    source, meta = _source()
    hybrid = replace(source, objective="hybrid_terminal_composition", **overrides)
    with pytest.raises(ValueError, match="semantic mismatch"):
        initialization_changes(meta, hybrid, meta["normalization"], {"std": 0.2}, "tokenizer")


def test_initialization_checks_tokenizer_and_reverse_objective_change():
    source, meta = _source()
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        initialization_changes(meta, source, meta["normalization"], None, "other-tokenizer")
    meta["config"] = replace(source, objective="hybrid_terminal_composition").as_dict()
    with pytest.raises(ValueError, match="objective"):
        initialization_changes(meta, source, meta["normalization"], None, "tokenizer")
