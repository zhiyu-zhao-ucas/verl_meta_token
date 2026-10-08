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
"""Raw-unit online critic metrics with explicit support for undefined statistics."""

import math


def regression_diagnostics(predictions, targets):
    """Summarize paired outputs, retaining exact raw labels for zero/sign tests.

    Undefined statistics use a finite zero placeholder and an explicit validity
    flag. Nonfinite outputs are counted, and summaries use only finite pairs.
    A zero-baseline skill of one is perfect; below zero is worse than predicting
    zero for every row. It is undefined when all targets are zero.
    """
    pairs = list(zip(predictions, targets, strict=True))
    finite = [(float(p), float(y)) for p, y in pairs if math.isfinite(p) and math.isfinite(y)]
    count = len(finite)
    pred, target = zip(*finite, strict=True) if finite else ((), ())
    p_mean, p_std = _moments(pred)
    y_mean, y_std = _moments(target)
    mse = _mean([(p - y) ** 2 for p, y in finite])
    zero_mse = _mean([y * y for y in target])
    nonzero = [(p, y) for p, y in finite if y != 0.0]
    statistics_valid = count > 0 and count == len(pairs)
    pearson_valid = statistics_valid and count >= 2 and p_std > 0.0 and y_std > 0.0
    pearson = _mean([(p - p_mean) * (y - y_mean) for p, y in finite]) / (p_std * y_std) if pearson_valid else 0.0
    return {
        "count": len(pairs),
        "finite_count": count,
        "nonfinite_prediction_count": sum(not math.isfinite(p) for p, _ in pairs),
        "nonfinite_target_count": sum(not math.isfinite(y) for _, y in pairs),
        "statistics_valid": int(statistics_valid),
        "rmse": math.sqrt(mse),
        "mae": _mean([abs(p - y) for p, y in finite]),
        "bias": _mean([p - y for p, y in finite]),
        "prediction_mean": p_mean,
        "prediction_std": p_std,
        "prediction_abs_max": max((abs(p) for p in pred), default=0.0),
        "target_mean": y_mean,
        "target_std": y_std,
        "target_abs_max": max((abs(y) for y in target), default=0.0),
        "target_zero_fraction": (count - len(nonzero)) / count if count else 0.0,
        "nonzero_target_count": len(nonzero),
        "nonzero_sign_accuracy": _mean([int((p > 0 and y > 0) or (p < 0 and y < 0)) for p, y in nonzero]),
        "nonzero_sign_accuracy_valid": int(statistics_valid and bool(nonzero)),
        "nonzero_rmse": math.sqrt(_mean([(p - y) ** 2 for p, y in nonzero])),
        "nonzero_rmse_valid": int(statistics_valid and bool(nonzero)),
        "zero_baseline_rmse": math.sqrt(zero_mse),
        "zero_baseline_skill": 1.0 - mse / zero_mse if zero_mse > 0.0 else 0.0,
        "zero_baseline_skill_valid": int(statistics_valid and zero_mse > 0.0),
        "pearson": min(1.0, max(-1.0, pearson)),
        "pearson_valid": int(pearson_valid),
    }


def prediction_drift(before, after):
    """Compare outputs on identical ordered rows, independent of target changes."""
    pairs = list(zip(before, after, strict=True))
    finite = [(float(p), float(q)) for p, q in pairs if math.isfinite(p) and math.isfinite(q)]
    changes = [q - p for p, q in finite]
    return {
        "count": len(pairs),
        "finite_count": len(finite),
        "statistics_valid": int(bool(finite) and len(finite) == len(pairs)),
        "prediction_shift_mean": _mean(changes),
        "prediction_drift_rmse": math.sqrt(_mean([change * change for change in changes])),
        "prediction_drift_abs_max": max((abs(change) for change in changes), default=0.0),
        "prediction_sign_flip_fraction": _mean([int((p < 0 < q) or (q < 0 < p)) for p, q in finite]),
    }


def _mean(values):
    return math.fsum(values) / len(values) if values else 0.0


def _moments(values):
    mean = _mean(values)
    return mean, math.sqrt(_mean([(value - mean) ** 2 for value in values]))
