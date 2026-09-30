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
"""Offline delta-critic evaluation against MC ground truth.

Ports the delta path of the source ``06_eval_token_delta_unified.py``: score
every selected MC state at its own ``P+t`` position, then report raw regression,
per-response/per-query z-scored regression, the three-class discrete block and
the sign diagnostics. The source's value, distributional, anchored-value and
policy-gradient blocks are out of migration scope and are not reproduced.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import mean, regression_metrics
from .score import FrozenDeltaWorker, ScoringRow
from .train import read_rows

CLASS_ID_TO_NAME = {0: "neg", 1: "neutral", 2: "pos"}
CLASS_NAME_TO_ID = {name: index for index, name in CLASS_ID_TO_NAME.items()}
# The neutral bucket absorbs small MC noise around zero; the same conversion is
# applied to ground-truth labels and to model predictions.
DEFAULT_CLASS_MARGIN = 1.0 / 32.0


def delta_to_class(delta, margin):
    if delta > margin:
        return 2
    if delta < -margin:
        return 0
    return 1


def delta_to_sign(delta, margin):
    class_id = delta_to_class(delta, margin)
    if class_id == 2:
        return 1.0
    if class_id == 0:
        return -1.0
    return 0.0


def group_normalized_regression_metrics(rows, pred_key, target_key, group_key):
    """Compare delta shapes after independently z-scoring prediction and target per group."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get(pred_key) is None or row.get(target_key) is None:
            continue
        grouped[str(row.get(group_key, row["rollout_id"]))].append(row)

    pred_z, target_z = [], []
    constant_pred_groups = constant_target_groups = 0
    for group_rows in grouped.values():
        pred = [float(row[pred_key]) for row in group_rows]
        target = [float(row[target_key]) for row in group_rows]
        pred_mean, target_mean = mean(pred), mean(target)
        pred_std = _population_std(pred, pred_mean)
        target_std = _population_std(target, target_mean)
        constant_pred_groups += int(pred_std < 1e-8)
        constant_target_groups += int(target_std < 1e-8)
        pred_z.extend([0.0 if pred_std < 1e-8 else (value - pred_mean) / pred_std for value in pred])
        target_z.extend([0.0 if target_std < 1e-8 else (value - target_mean) / target_std for value in target])

    metrics = regression_metrics(pred_z, target_z)
    metrics.update(
        {
            "groups": float(len(grouped)),
            "group_key": group_key,
            "constant_pred_groups": float(constant_pred_groups),
            "constant_target_groups": float(constant_target_groups),
        }
    )
    return metrics


def _population_std(values, mu):
    import math

    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def normalized_delta_metrics(rows, pred_key, target_key):
    return {
        "response": group_normalized_regression_metrics(rows, pred_key, target_key, "rollout_id"),
        "query": group_normalized_regression_metrics(rows, pred_key, target_key, "query_id"),
    }


def classification_metrics(true_labels, pred_labels):
    """Three-class block, implemented directly to avoid a scikit-learn dependency."""
    matrix = [[0, 0, 0] for _ in range(3)]
    for true_label, pred_label in zip(true_labels, pred_labels, strict=True):
        matrix[int(true_label)][int(pred_label)] += 1

    support, predicted, recall_by_class, precision_by_class, f1_by_class = {}, {}, {}, {}, {}
    recalls, f1s = [], []
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(3))
    for class_id, class_name in CLASS_ID_TO_NAME.items():
        true_positive = matrix[class_id][class_id]
        true_total = sum(matrix[class_id])
        pred_total = sum(matrix[row_id][class_id] for row_id in range(3))
        support[class_name] = int(true_total)
        predicted[class_name] = int(pred_total)
        recall = float(true_positive / true_total) if true_total else float("nan")
        precision = float(true_positive / pred_total) if pred_total else float("nan")
        if precision == precision and recall == recall and (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = float("nan")
        recall_by_class[class_name] = recall
        precision_by_class[class_name] = precision
        f1_by_class[class_name] = f1
        if recall == recall:
            recalls.append(recall)
        if f1 == f1:
            f1s.append(f1)

    confusion = {
        CLASS_ID_TO_NAME[true_id]: {CLASS_ID_TO_NAME[pred_id]: int(matrix[true_id][pred_id]) for pred_id in range(3)}
        for true_id in range(3)
    }
    return {
        "count": float(total),
        "accuracy": float(correct / total) if total else float("nan"),
        "macro_f1": mean(f1s),
        "balanced_accuracy": mean(recalls),
        "per_class_recall": recall_by_class,
        "per_class_precision": precision_by_class,
        "per_class_f1": f1_by_class,
        "support": support,
        "predicted_count": predicted,
        "confusion_matrix": confusion,
    }


def delta_sign_diagnostics(rows, pred_key, target_key, class_margin):
    valid = [row for row in rows if row.get(pred_key) is not None and row.get(target_key) is not None]
    pred_sign = [delta_to_sign(float(row[pred_key]), class_margin) for row in valid]
    target_sign = [delta_to_sign(float(row[target_key]), class_margin) for row in valid]
    target_delta = [float(row[target_key]) for row in valid]
    oracle_magnitude = [sign * abs(delta) for sign, delta in zip(pred_sign, target_delta, strict=True)]

    nonzero = [index for index, sign in enumerate(target_sign) if sign != 0.0]
    matches = [pred == target for pred, target in zip(pred_sign, target_sign, strict=True)]
    nonzero_matches = [matches[index] for index in nonzero]
    flips = [pred_sign[i] == -target_sign[i] for i in nonzero if pred_sign[i] != 0.0]
    neutral_on_nonzero = [pred_sign[i] == 0.0 for i in nonzero]
    return {
        "count": float(len(valid)),
        "class_margin": float(class_margin),
        "pred_sign_vs_raw_delta": regression_metrics(pred_sign, target_delta),
        "pred_sign_vs_target_sign": regression_metrics(pred_sign, target_sign),
        "oracle_magnitude": regression_metrics(oracle_magnitude, target_delta),
        "sign_accuracy": mean([float(match) for match in matches]),
        "nonzero_target_count": float(len(nonzero)),
        "nonzero_sign_accuracy": mean([float(match) for match in nonzero_matches]),
        "nonzero_flip_rate": mean([float(flip) for flip in flips]),
        "pred_neutral_when_target_nonzero_rate": mean([float(value) for value in neutral_on_nonzero]),
        "pred_sign_mean": mean(pred_sign),
        "target_sign_mean": mean(target_sign),
    }


def group_mc_labels(mc_rows):
    """MC rows are sparse; token_index order defines the next selected state."""
    grouped = defaultdict(list)
    for row in mc_rows:
        grouped[str(row["rollout_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["token_index"]))
    return grouped


def split_half_value(row):
    """Split-prefix MC control: mean of the first and second half of the rewards."""
    rewards = [float(value) for value in row.get("prefix_continuation_rewards") or []]
    half = len(rewards) // 2
    if half == 0 or len(rewards) != 2 * half:
        return None, None
    return float(sum(rewards[:half]) / half), float(sum(rewards[half:]) / half)


def selected_eval_rows(rollouts, mc_rows):
    """One evaluation row per selected MC state, reading the selected token itself."""
    grouped_mc = group_mc_labels(mc_rows)
    rows = []
    for rollout in rollouts:
        rollout_id = str(rollout["id"])
        selected = grouped_mc.get(rollout_id, [])
        if not selected:
            continue
        prompt = list(rollout["prompt_token_ids"])
        response = list(rollout["response_token_ids"])
        terminal_reward = float(rollout["terminal_reward"])
        for index, current in enumerate(selected):
            token_index = int(current["token_index"])
            if not 0 <= token_index < len(response):
                raise ValueError(
                    f"Selected token_index={token_index} is out of range for rollouts={rollout_id} "
                    f"with response length {len(response)}"
                )
            previous = selected[index - 1] if index > 0 else None
            following = selected[index + 1] if index + 1 < len(selected) else None
            value_half_a, value_half_b = split_half_value(current)
            next_half_a, next_half_b = (
                (terminal_reward, terminal_reward) if following is None else split_half_value(following)
            )
            # gt_delta = V(next selected prefix, or terminal) - V(current prefix).
            next_value = terminal_reward if following is None else float(following["v_prefix"])
            gt_v_prefix = float(current["v_prefix"])
            state_input_ids = prompt + response[:token_index]
            rows.append(
                {
                    "rollout_id": rollout["id"],
                    "query_id": rollout.get("prompt_id", rollout.get("query_id", rollout["id"])),
                    "response_index": rollout.get("response_index"),
                    "state_id": str(current["state_id"]),
                    "token_index": token_index,
                    "input_ids": prompt + response[: token_index + 1],
                    "attention_mask": [1] * (len(prompt) + token_index + 1),
                    "value_position": len(prompt) + token_index,
                    "state_input_ids": state_input_ids,
                    "state_value_position": len(state_input_ids) - 1,
                    "gt_v_prefix": gt_v_prefix,
                    "gt_next_value": next_value,
                    "gt_delta": float(next_value - gt_v_prefix),
                    "gt_value_half_a": value_half_a,
                    "gt_value_half_b": value_half_b,
                    "gt_delta_half_a": None
                    if value_half_a is None or next_half_a is None
                    else float(next_half_a - value_half_a),
                    "gt_delta_half_b": None
                    if value_half_b is None or next_half_b is None
                    else float(next_half_b - value_half_b),
                    "prev_state_id": None if previous is None else str(previous["state_id"]),
                    "gt_prev_value": None if previous is None else float(previous["v_prefix"]),
                    "gt_delta_from_prev": None
                    if previous is None
                    else float(gt_v_prefix - float(previous["v_prefix"])),
                    "next_state_id": None if following is None else str(following["state_id"]),
                    "terminal_reward": terminal_reward,
                    "response_length": len(response),
                }
            )
    return rows


def attach_predictions(rows, scored):
    """Attach per-state critic outputs from FrozenDeltaWorker.score to the eval rows."""
    by_rollout = {row["rollout_id"]: row for row in scored}
    for row in rows:
        output = by_rollout[row["rollout_id"]]
        position = row["token_index"]
        if output["critic_signal_mask"][position] != 1.0:
            raise ValueError(f"Selected state {row['state_id']} was not scored")
        row["pred_delta_norm"] = float(output["delta_pred_normalized"][position])
        row["pred_delta_raw"] = float(output["delta_pred_raw"][position])
    return rows


def scoring_rows(rollouts, mc_rows):
    grouped = group_mc_labels(mc_rows)
    rows = []
    for rollout in rollouts:
        selected = grouped.get(str(rollout["id"]))
        if not selected:
            continue
        rows.append(
            ScoringRow(
                rollout_id=str(rollout["id"]),
                prompt_token_ids=tuple(rollout["prompt_token_ids"]),
                response_token_ids=tuple(rollout["response_token_ids"]),
                selected_token_indices=tuple(int(row["token_index"]) for row in selected),
            )
        )
    return rows


def evaluate(worker, rollouts, mc_rows, class_margin=DEFAULT_CLASS_MARGIN, split=None):
    if not 0.0 <= class_margin:
        raise ValueError("class_margin must be non-negative")
    rows = selected_eval_rows(rollouts, mc_rows)
    if not rows:
        raise ValueError("No selected MC states to evaluate")
    attach_predictions(rows, worker.score(scoring_rows(rollouts, mc_rows)))
    for row in rows:
        row["gt_delta_class"] = delta_to_class(row["gt_delta"], class_margin)
        row["pred_delta_class"] = delta_to_class(row["pred_delta_raw"], class_margin)
        row["class_margin"] = float(class_margin)
    return {
        "split": split,
        "class_margin": float(class_margin),
        "selected_rows": float(len(rows)),
        "rollouts": float(len({row["rollout_id"] for row in rows})),
        "critic_checkpoint": worker.metadata["weights_sha256"],
        "target_normalization": {
            key: worker.metadata["normalization"][key] for key in ("enabled", "mode", "mean", "std")
        },
        "critic_window": {"policy": worker.config.window_policy, "max_length": worker.config.max_length},
        "delta_regression": regression_metrics(
            [row["pred_delta_raw"] for row in rows], [row["gt_delta"] for row in rows]
        ),
        "normalized_delta_regression": normalized_delta_metrics(rows, "pred_delta_raw", "gt_delta"),
        "discrete": classification_metrics(
            [row["gt_delta_class"] for row in rows], [row["pred_delta_class"] for row in rows]
        ),
        "delta_sign_diagnostics": delta_sign_diagnostics(rows, "pred_delta_raw", "gt_delta", class_margin),
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Portable critic artifact directory")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--labels", required=True, help="MC label rows carrying v_prefix")
    parser.add_argument("--output", required=True, help="Metrics JSON path")
    parser.add_argument("--rows-output", help="Optional per-state prediction JSONL path")
    parser.add_argument("--split", help="Split name recorded in the report")
    parser.add_argument("--class-margin", type=float, default=DEFAULT_CLASS_MARGIN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--scoring-max-length", type=int)
    args = parser.parse_args()
    worker = FrozenDeltaWorker(
        args.checkpoint,
        device=args.device,
        microbatch=args.microbatch,
        scoring_max_length=args.scoring_max_length,
    )
    metrics, rows = evaluate(
        worker, read_rows(args.rollouts), read_rows(args.labels), args.class_margin, split=args.split
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    if args.rows_output:
        with open(args.rows_output, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
