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

import math

import pytest

from examples.delta_critic.online_diagnostics import prediction_drift, regression_diagnostics


def test_perfect_predictions_and_opposite_signs_distinguish_baseline_skill():
    targets = [-0.25, 0.0, 0.25]
    perfect = regression_diagnostics(targets, targets)
    inverted = regression_diagnostics([0.25, 0.0, -0.25], targets)
    assert perfect["zero_baseline_skill"] == 1.0
    assert perfect["pearson"] == pytest.approx(1.0)
    assert perfect["nonzero_sign_accuracy"] == 1.0
    assert inverted["zero_baseline_skill"] == -3.0
    assert inverted["pearson"] == pytest.approx(-1.0)
    assert inverted["nonzero_sign_accuracy"] == 0.0
    assert inverted["nonzero_target_count"] == 2
    assert inverted["target_zero_fraction"] == pytest.approx(1 / 3)
    assert inverted["zero_baseline_skill_valid"] == inverted["pearson_valid"] == 1


def test_bias_cannot_hide_behind_perfect_correlation():
    result = regression_diagnostics([0.0, 1.0, 2.0], [-1.0, 0.0, 1.0])
    assert result["pearson"] == pytest.approx(1.0)
    assert result["rmse"] == result["bias"] == 1.0
    assert result["zero_baseline_skill"] == pytest.approx(-0.5)
    assert result["nonzero_sign_accuracy"] == 0.5  # A zero prediction has no correct sign.


@pytest.mark.parametrize("predictions, targets", [([], []), ([0.0], [0.0]), ([0.1, 0.1], [0.0, 0.0])])
def test_empty_or_zero_target_support_never_claims_direction_or_skill(predictions, targets):
    result = regression_diagnostics(predictions, targets)
    assert all(math.isfinite(value) for value in result.values())
    assert result["nonzero_target_count"] == 0
    assert result["nonzero_sign_accuracy_valid"] == 0
    assert result["zero_baseline_skill_valid"] == 0
    assert result["pearson_valid"] == 0
    assert result["statistics_valid"] == int(bool(predictions))


def test_collapsed_prediction_has_baseline_error_but_no_correlation():
    result = regression_diagnostics([0.0, 0.0], [-1.0, 1.0])
    assert result["zero_baseline_skill"] == 0.0
    assert result["zero_baseline_skill_valid"] == 1
    assert result["pearson_valid"] == 0
    assert result["nonzero_sign_accuracy"] == 0.0


def test_nonfinite_outputs_are_counted_and_invalidate_full_sample_statistics():
    result = regression_diagnostics([float("nan"), 1.0, 2.0], [0.0, 1.0, float("inf")])
    assert result["count"] == 3 and result["finite_count"] == 1
    assert result["nonfinite_prediction_count"] == result["nonfinite_target_count"] == 1
    assert result["statistics_valid"] == result["zero_baseline_skill_valid"] == 0
    assert all(math.isfinite(value) for value in result.values())


def test_same_row_prediction_drift_measures_output_change():
    result = prediction_drift([-1.0, 0.0, 1.0], [1.0, 0.0, 2.0])
    assert result["prediction_drift_rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert result["prediction_shift_mean"] == 1.0
    assert result["prediction_drift_abs_max"] == 2.0
    assert result["prediction_sign_flip_fraction"] == pytest.approx(1 / 3)
    assert result["count"] == result["finite_count"] == 3
    assert result["statistics_valid"] == 1
    with pytest.raises(ValueError):
        prediction_drift([0.0], [])
