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

"""Delta-critic evaluation golden values and source differential."""

import ast
import importlib.util
import math
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.delta_critic import metrics as local_metrics
from examples.delta_critic.evaluate import (
    DEFAULT_CLASS_MARGIN,
    attach_predictions,
    classification_metrics,
    delta_sign_diagnostics,
    delta_to_class,
    delta_to_sign,
    evaluate,
    group_normalized_regression_metrics,
    normalized_delta_metrics,
    selected_eval_rows,
    split_half_value,
)

SOURCE_MODULE = "06_eval_token_delta_unified.py"
EXTRACTED = (
    "_mean",
    "_class_name",
    "_delta_to_class",
    "_delta_to_sign",
    "_classification_metrics",
    "_group_normalized_regression_metrics",
    "_delta_sign_diagnostics",
    "_group_mc_labels",
    "_split_half_value",
    "_build_selected_eval_rows",
    "_normalized_delta_metrics",
)


def _nan_free(value):
    """NaN is the source's marker for undefined metrics; make it comparable."""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, dict):
        return {key: _nan_free(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nan_free(item) for item in value]
    return value


def _source_root():
    root = os.environ.get("VALUE_MODEL_ROOT")
    if not root:
        pytest.skip("Set VALUE_MODEL_ROOT for source differential test")
    return Path(root) / "delta_value_llm_exp"


def _load_source():
    """Exec the pure source helpers without importing the training stack."""
    source = _source_root()
    metrics_spec = importlib.util.spec_from_file_location("source_metrics", source / "metrics.py")
    source_metrics = importlib.util.module_from_spec(metrics_spec)
    metrics_spec.loader.exec_module(source_metrics)

    path = source / SOURCE_MODULE
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in EXTRACTED:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CLASS_ID_TO_NAME" for target in node.targets
        ):
            body.append(node)
    namespace = {
        "Any": Any,
        "defaultdict": defaultdict,
        "math": math,
        "regression_metrics": source_metrics.regression_metrics,
        "interval_policy_factor_stats": None,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, source_metrics


ROLLOUTS = [
    {
        "id": "r1",
        "prompt_id": "p1",
        "prompt_token_ids": [1, 2],
        "response_token_ids": [3, 4, 5, 6],
        "terminal_reward": 1.0,
    },
    {
        "id": "r2",
        "prompt_id": "p2",
        "prompt_token_ids": [7],
        "response_token_ids": [8, 9],
        "terminal_reward": -0.5,
    },
]
MC_LABELS = [
    {
        "state_id": "r1:t0",
        "rollout_id": "r1",
        "token_index": 0,
        "v_prefix": 0.2,
        "prefix_continuation_rewards": [0.0, 0.4, 0.6, 1.2],
    },
    {
        "state_id": "r1:t2",
        "rollout_id": "r1",
        "token_index": 2,
        "v_prefix": 0.5,
        "prefix_continuation_rewards": [0.4, 0.6, 0.5, 0.5],
    },
    {
        "state_id": "r2:t0",
        "rollout_id": "r2",
        "token_index": 0,
        "v_prefix": -0.25,
        "prefix_continuation_rewards": [0.0, -0.5],
    },
]


class FakeWorker:
    """Stand-in for FrozenDeltaWorker: returns fixed normalized predictions."""

    def __init__(self, predictions, normalization):
        self.predictions = predictions
        self.metadata = {
            "weights_sha256": "deadbeef",
            "normalization": normalization,
        }
        self.config = SimpleNamespace(window_policy="legacy_tail", max_length=2048)

    def _denormalize(self, value):
        normalization = self.metadata["normalization"]
        if not normalization["enabled"]:
            return value
        return value * normalization["std"] + normalization["mean"]

    def score(self, rows):
        return [
            {
                "rollout_id": row.rollout_id,
                "delta_pred_normalized": self.predictions[row.rollout_id],
                "delta_pred_raw": [self._denormalize(value) for value in self.predictions[row.rollout_id]],
                "critic_signal_mask": [
                    1.0 if index in row.selected_token_indices else 0.0 for index in range(len(row.response_token_ids))
                ],
            }
            for row in rows
        ]


def test_metric_helpers_match_source_metrics_module():
    _, source_metrics = _load_source()
    values = [0.5, -1.25, 3.0, 3.0, 0.0, 2.5]
    others = [1.0, 0.5, -2.0, 4.0, 0.25, -0.75]
    assert local_metrics.mean(values) == source_metrics._mean(values)
    assert local_metrics.std(values) == source_metrics._std(values)
    assert local_metrics.rank(values) == source_metrics._rank(values)
    assert local_metrics.pearson(values, others) == source_metrics.pearson(values, others)
    assert local_metrics.spearman(values, others) == source_metrics.spearman(values, others)
    assert local_metrics.regression_metrics(values, others) == source_metrics.regression_metrics(values, others)
    # Degenerate populations keep the source's NaN/0.0 conventions.
    assert math.isnan(local_metrics.pearson([1.0], [1.0]))
    assert local_metrics.std([5.0]) == 0.0 == source_metrics._std([5.0])


def test_delta_class_and_sign_boundaries():
    margin = DEFAULT_CLASS_MARGIN
    assert margin == 1.0 / 32.0
    assert [delta_to_class(value, margin) for value in (margin, -margin, 0.0)] == [1, 1, 1]
    assert [delta_to_class(value, margin) for value in (margin * 1.01, -margin * 1.01)] == [2, 0]
    assert [delta_to_sign(value, margin) for value in (1.0, 0.0, -1.0)] == [1.0, 0.0, -1.0]


def test_classification_metrics_confusion_and_undefined_classes():
    metrics = classification_metrics([0, 1, 2, 2], [1, 1, 2, 1])
    assert metrics["confusion_matrix"] == {
        "neg": {"neg": 0, "neutral": 1, "pos": 0},
        "neutral": {"neg": 0, "neutral": 1, "pos": 0},
        "pos": {"neg": 0, "neutral": 1, "pos": 1},
    }
    assert metrics["support"] == {"neg": 1, "neutral": 1, "pos": 2}
    assert metrics["predicted_count"] == {"neg": 0, "neutral": 3, "pos": 1}
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert math.isnan(metrics["per_class_precision"]["neg"])  # no predictions of that class
    assert metrics["per_class_recall"]["pos"] == pytest.approx(0.5)


def test_split_half_value_rejects_odd_budget():
    assert split_half_value({"prefix_continuation_rewards": [1.0, 3.0, 2.0, 4.0]}) == (2.0, 3.0)
    assert split_half_value({"prefix_continuation_rewards": [1.0, 2.0, 3.0]}) == (None, None)
    assert split_half_value({}) == (None, None)


def test_selected_eval_rows_anchor_last_state_to_terminal_reward():
    rows = selected_eval_rows(ROLLOUTS, MC_LABELS)
    assert [row["state_id"] for row in rows] == ["r1:t0", "r1:t2", "r2:t0"]
    first, last, single = rows
    assert first["gt_next_value"] == 0.5  # next selected state
    assert first["gt_delta"] == pytest.approx(0.3)
    assert first["value_position"] == 2  # len(prompt) + token_index
    assert len(first["input_ids"]) == 3  # prompt + response[:t+1]
    assert last["gt_next_value"] == 1.0  # terminal reward anchors the final state
    assert last["gt_delta"] == pytest.approx(0.5)
    assert single["gt_next_value"] == -0.5
    assert single["gt_delta"] == pytest.approx(-0.25)
    assert [row["query_id"] for row in rows] == ["p1", "p1", "p2"]
    assert first["prev_state_id"] is None and last["prev_state_id"] == "r1:t0"


def test_group_normalized_regression_z_scores_within_each_group():
    rows = [
        {"rollout_id": "a", "pred": 1.0, "target": 2.0},
        {"rollout_id": "a", "pred": 2.0, "target": 4.0},
        {"rollout_id": "b", "pred": 5.0, "target": 1.0},
        {"rollout_id": "b", "pred": 5.0, "target": 3.0},
    ]
    metrics = group_normalized_regression_metrics(rows, "pred", "target", "rollout_id")
    assert metrics["groups"] == 2.0
    assert metrics["constant_pred_groups"] == 1.0
    # Group "a" z-scores to [-1, 1] and the constant group "b" to [0, 0], so the
    # pooled prediction std is sqrt(0.5) rather than 1.
    assert metrics["pred_std"] == pytest.approx(math.sqrt(0.5))
    assert metrics["pred_mean"] == pytest.approx(0.0)
    assert metrics["pearson"] == pytest.approx(0.7071067811865476)


def test_delta_sign_diagnostics_reports_flip_and_neutral_rates():
    rows = [
        {"pred_delta_raw": 0.5, "gt_delta": 0.5},
        {"pred_delta_raw": -0.5, "gt_delta": 0.5},
        {"pred_delta_raw": 0.0, "gt_delta": 0.5},
        {"pred_delta_raw": 0.5, "gt_delta": 0.0},
    ]
    metrics = delta_sign_diagnostics(rows, "pred_delta_raw", "gt_delta", DEFAULT_CLASS_MARGIN)
    assert metrics["count"] == 4.0
    assert metrics["nonzero_target_count"] == 3.0
    # Only row 0 agrees on sign; the neutral target of row 3 is excluded from the
    # nonzero denominators, and row 2 counts as neutral rather than a flip.
    assert metrics["sign_accuracy"] == pytest.approx(0.25)
    assert metrics["nonzero_sign_accuracy"] == pytest.approx(1.0 / 3.0)
    assert metrics["nonzero_flip_rate"] == pytest.approx(0.5)
    assert metrics["pred_neutral_when_target_nonzero_rate"] == pytest.approx(1.0 / 3.0)


def test_evaluate_reports_raw_units_and_denormalizes():
    normalization = {"enabled": True, "mode": "standardize", "mean": 0.5, "std": 2.0}
    worker = FakeWorker({"r1": [0.1, 0.0, 0.2, 0.0], "r2": [-0.3, 0.0]}, normalization)
    report, rows = evaluate(worker, ROLLOUTS, MC_LABELS, split="test")
    assert report["split"] == "test"
    assert report["selected_rows"] == 3.0
    assert report["rollouts"] == 2.0
    assert report["target_normalization"]["std"] == 2.0
    # Raw delta must be the denormalized prediction: 0.1 * 2.0 + 0.5 at r1:t0.
    assert rows[0]["pred_delta_norm"] == pytest.approx(0.1)
    assert rows[0]["pred_delta_raw"] == pytest.approx(0.7)
    # Raw predictions are 0.7, 0.9 and -0.1 after denormalization.
    assert report["delta_regression"]["pred_mean"] == pytest.approx((0.7 + 0.9 - 0.1) / 3)
    assert set(report["normalized_delta_regression"]) == {"response", "query"}
    assert report["discrete"]["count"] == 3.0
    assert report["delta_sign_diagnostics"]["class_margin"] == DEFAULT_CLASS_MARGIN


def test_attach_predictions_rejects_unscored_selected_state():
    rows = selected_eval_rows(ROLLOUTS, MC_LABELS)
    scored = [
        {
            "rollout_id": "r1",
            "delta_pred_normalized": [0.0] * 4,
            "delta_pred_raw": [0.0] * 4,
            "critic_signal_mask": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "rollout_id": "r2",
            "delta_pred_normalized": [0.0] * 2,
            "delta_pred_raw": [0.0] * 2,
            "critic_signal_mask": [1.0, 0.0],
        },
    ]
    with pytest.raises(ValueError, match="was not scored"):
        attach_predictions(rows, scored)


def test_evaluation_matches_source_delta_blocks():
    source, _ = _load_source()
    for value in (0.04, 0.03125, -0.03125, 0.0, -0.9):
        assert delta_to_class(value, DEFAULT_CLASS_MARGIN) == source["_delta_to_class"](value, DEFAULT_CLASS_MARGIN)
        assert delta_to_sign(value, DEFAULT_CLASS_MARGIN) == source["_delta_to_sign"](value, DEFAULT_CLASS_MARGIN)

    rows = selected_eval_rows(ROLLOUTS, MC_LABELS)
    for row, expected in zip(rows, source["_build_selected_eval_rows"](ROLLOUTS, MC_LABELS), strict=True):
        # interval_policy_factor_stats is only used for the policy-gradient path.
        assert {key: value for key, value in row.items()} == expected

    predictions = {"r1": [0.1, 0.0, 0.2, 0.0], "r2": [-0.3, 0.0]}
    scored = FakeWorker(predictions, {"enabled": False, "mode": "none", "mean": None, "std": None})
    report, rows = evaluate(scored, ROLLOUTS, MC_LABELS)
    for row in rows:
        row["pred_delta_norm"] = row["pred_delta_raw"]

    true_labels = [row["gt_delta_class"] for row in rows]
    pred_labels = [row["pred_delta_class"] for row in rows]
    assert _nan_free(classification_metrics(true_labels, pred_labels)) == _nan_free(
        source["_classification_metrics"](true_labels, pred_labels)
    )
    assert _nan_free(normalized_delta_metrics(rows, "pred_delta_raw", "gt_delta")) == _nan_free(
        source["_normalized_delta_metrics"](rows, "pred_delta_raw", "gt_delta")
    )
    assert _nan_free(delta_sign_diagnostics(rows, "pred_delta_raw", "gt_delta", DEFAULT_CLASS_MARGIN)) == _nan_free(
        source["_delta_sign_diagnostics"](rows, "pred_delta_raw", "gt_delta", DEFAULT_CLASS_MARGIN)
    )
