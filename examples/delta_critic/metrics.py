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
"""Pure regression/ranking metrics ported from the source `metrics.py`.

The source computes every critic metric with plain Python so evaluation does not
depend on torch or scikit-learn. These keep that property and the exact
conventions, including population standard deviation and NaN for undefined
correlations.
"""

import math


def mean(values):
    return float(sum(values) / len(values)) if values else float("nan")


def std(values):
    """Population standard deviation; 0.0 for fewer than two values."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / len(values)))


def rank(values):
    """Average ranks with tie handling, so Spearman matches the source."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / math.sqrt(vx * vy))


def spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return pearson(rank(x), rank(y))


def regression_metrics(pred, target):
    errors = [p - t for p, t in zip(pred, target, strict=True)]
    mse = mean([error * error for error in errors])
    mae = mean([abs(error) for error in errors])
    var_target = std(target) ** 2
    return {
        "count": float(len(target)),
        "mse": mse,
        "mae": mae,
        "rmse": float(math.sqrt(mse)) if mse == mse else float("nan"),
        "pearson": pearson(pred, target),
        "spearman": spearman(pred, target),
        "explained_variance": float(1.0 - mse / var_target) if var_target > 0.0 else float("nan"),
        "pred_mean": mean(pred),
        "pred_std": std(pred),
        "target_mean": mean(target),
        "target_std": std(target),
    }
