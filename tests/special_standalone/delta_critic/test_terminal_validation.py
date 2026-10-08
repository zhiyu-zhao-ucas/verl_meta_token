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

import pytest

from examples.delta_critic.train import terminal_validation_metrics
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.training_data import collate


def test_terminal_validation_compares_raw_errors_independently_of_optimization_weights():
    rows = [
        {"terminal_comp_valid": True, "terminal_comp_target": 1.0},
        {"terminal_comp_valid": True, "terminal_comp_target": -1.0},
        {"terminal_comp_valid": False, "terminal_comp_target": 999.0},
    ]
    metrics = {
        "critic/terminal_count": 2.0,
        "critic/terminal_raw_squared_error": 0.5,
        "critic/terminal_suffix_count": 8.0,
        "critic/terminal_optimization_loss": 0.0001,
    }
    result = terminal_validation_metrics(metrics, rows)
    assert result["terminal_raw_rmse"] == 0.5
    assert result["terminal_zero_raw_rmse"] == 1.0
    assert result["terminal_mean_suffix_count"] == 4.0
    changed_weight = terminal_validation_metrics({**metrics, "critic/terminal_optimization_loss": 1.0}, rows)
    assert changed_weight["terminal_raw_rmse"] == result["terminal_raw_rmse"]
    with pytest.raises(ValueError, match="count"):
        terminal_validation_metrics({**metrics, "critic/terminal_count": 3}, rows)


def test_local_only_validation_has_no_terminal_error_estimate():
    metrics = {
        f"critic/{name}": 0
        for name in (
            "terminal_count",
            "terminal_raw_squared_error",
            "terminal_suffix_count",
            "terminal_optimization_loss",
        )
    }
    result = terminal_validation_metrics(metrics, [{}])
    assert result["terminal_raw_rmse"] is None
    assert result["terminal_zero_raw_rmse"] is None


def test_terminal_validation_uses_collated_validity_for_padding_and_legacy_windows():
    config = ScalarConfig("unused", objective="hybrid_terminal_composition", max_length=4, window_policy="legacy_tail")
    rows = []
    for length, target, sample_valid in ((5, 9.0, True), (3, 0.5, True), (3, 999.0, False)):
        rows.append(
            {
                "id": str(length),
                "prompt_token_ids": [1],
                "response_token_ids": [2] * length,
                "token_targets": [0.0] * length,
                "token_loss_mask": [0.0] * length,
                "token_signal_mask": [0.0] * length,
                "terminal_suffix_response_indices": [0, 2],
                "terminal_comp_target": target,
                "terminal_comp_valid": True,
                "sample_valid": sample_valid,
            }
        )
    batch = collate(rows, config)
    count = (batch["terminal_comp_valid"] * batch["sample_valid_mask"]).sum().item()
    assert count == 1
    result = terminal_validation_metrics(
        {
            "critic/terminal_count": count,
            "critic/terminal_raw_squared_error": 0.25,
            "critic/terminal_suffix_count": 2.0,
            "critic/terminal_optimization_loss": 0.001,
        },
        rows,
        config=config,
    )
    assert result["terminal_raw_rmse"] == 0.5
    assert result["terminal_zero_raw_rmse"] == 0.5
