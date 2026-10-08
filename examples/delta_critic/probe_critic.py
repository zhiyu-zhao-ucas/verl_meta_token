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
"""Fixed, paired offline probes for portable delta-critic checkpoints.

The default probe scores every eval TD rollout and a deterministic 512-row
terminal sample selected round-robin across prefix groups. Use
``--full-terminal-eval`` explicitly for the full held-out terminal set. Each
JSONL output row retains its rollout/branch, anchor and positions so external
group-bootstrap scripts can resample the same units for every checkpoint.

Example::

    python -m examples.delta_critic.probe_critic \\
      --checkpoint outputs/hybrid/checkpoints/step_00000080 \\
      --compare-checkpoint outputs/local_td/runs/balanced/step_00000080 \\
      --rollouts outputs/data/rollouts.jsonl --labels outputs/data/labels.jsonl \\
      --split outputs/data/split.json --continuations outputs/data/continuations.jsonl \\
      --output outputs/probes/step80.json --device cuda

For a local-TD checkpoint evaluated with hybrid loss semantics, pass the hybrid
artifact as ``--loss-config-checkpoint``. Its target normalization and head
metadata are read as-is; this tool never edits either artifact.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path

import torch

from .checkpoint import read_artifact
from .metrics import regression_metrics
from .scalar_loss import _terminal_components, per_sample_loss
from .score import FrozenDeltaWorker, ScoringRow
from .training_config import ScalarConfig
from .training_data import collate, continuation_rows, ordinary_rows

LOCAL_LENGTH_BINS = (
    (1, 32),
    (33, 64),
    (65, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
    (1025, 2048),
    (2049, 4096),
    (4097, 8192),
    (8193, 16384),
    (16385, 32768),
)


def select_terminal_probe_rows(rows, count, *, seed, group_key="prefix_group_id"):
    """Select rows without replacement, interleaving a seeded order of prefix groups.

    The returned order is part of the selection contract and stays nested:
    calling with a smaller count and the same seed returns a prefix of the
    larger selection. This lets the gradient probe use a smaller, fixed subset
    of the scored terminal probe.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        row_id = str(row.get("id", ""))
        group = row.get(group_key)
        if not row_id or row_id in ids:
            raise ValueError("Terminal probe rows require nonempty unique IDs")
        if group is None or not str(group):
            raise ValueError(f"Terminal probe row {row_id!r} is missing {group_key!r}")
        ids.add(row_id)
        groups[str(group)].append(row)

    rng = random.Random(seed)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    queues = {}
    for group in group_names:
        members = sorted(groups[group], key=lambda row: str(row["id"]))
        rng.shuffle(members)
        queues[group] = deque(members)

    selected = []
    active = deque(group_names)
    limit = min(count, len(rows))
    while active and len(selected) < limit:
        group = active.popleft()
        selected.append(queues[group].popleft())
        if queues[group]:
            active.append(group)
    return selected


def group_bootstrap_key(row):
    """Return the resampling unit carried in an exported per-row record."""
    if row.get("kind") == "terminal":
        return str(row["prefix_group_id"])
    return str(row["rollout_id"])


def local_td_metrics(rows):
    """Raw-unit local-TD regression, rank and exact nonzero-target sign metrics."""
    pred = [float(row["prediction_raw"]) for row in rows]
    target = [float(row["target_raw"]) for row in rows]
    nonzero = [(p, y) for p, y in zip(pred, target, strict=True) if y != 0.0]
    sign_correct = [float((p > 0 and y > 0) or (p < 0 and y < 0)) for p, y in nonzero]
    return {
        "raw": regression_metrics(pred, target),
        "nonzero_target_count": len(nonzero),
        "nonzero_raw": regression_metrics([p for p, _ in nonzero], [y for _, y in nonzero]),
        "nonzero_sign_accuracy": _mean(sign_correct),
        "nonzero_target_definition": "target_raw != 0.0",
    }


def terminal_raw_metrics(rows):
    """Raw reward-unit sum error, zero baseline, exact K and fixed length bins."""
    result = _raw_rmse_summary(rows)
    by_k = defaultdict(list)
    by_length = defaultdict(list)
    for row in rows:
        by_k[int(row["suffix_count"])].append(row)
        length = int(row["continuation_length"])
        bucket = _length_bin(length)
        by_length[bucket].append(row)
    result["by_suffix_count_k"] = {str(key): _raw_rmse_summary(group) for key, group in sorted(by_k.items())}
    result["by_continuation_length"] = {
        key: _raw_rmse_summary(group)
        for key, group in sorted(by_length.items(), key=lambda item: _bin_sort_key(item[0]))
    }
    result["length_bins"] = [f"{low}-{high}" for low, high in LOCAL_LENGTH_BINS] + ["32769+"]
    return result


def paired_records(local_rows, terminal_rows, scores_by_checkpoint, checkpoint_labels):
    """Join each artifact's output on the same original anchors and positions."""
    if set(scores_by_checkpoint) != set(checkpoint_labels):
        raise ValueError("Scored checkpoint labels do not match requested labels")
    records = []
    for row in local_rows:
        rollout_id = str(row["id"])
        targets = row["selected_local_deltas"]
        indices = row["selected_response_indices"]
        if len(targets) != len(indices):
            raise ValueError(f"Local TD targets and positions mismatch for {rollout_id}")
        state_ids = row.get("selected_state_ids") or [f"{rollout_id}:t{index}" for index in indices]
        if len(state_ids) != len(indices):
            raise ValueError(f"Local TD anchor IDs and positions mismatch for {rollout_id}")
        for index, target, state_id in zip(indices, targets, state_ids, strict=True):
            predictions = {}
            for label in checkpoint_labels:
                scored = scores_by_checkpoint[label]["local"][rollout_id]
                predictions[label] = float(scored["delta_pred_raw"][index])
            records.append(
                {
                    "kind": "local_td",
                    "rollout_id": rollout_id,
                    "query_id": row.get("prompt_id", rollout_id),
                    "response_index": row.get("response_index"),
                    "anchor_state_id": str(state_id),
                    "token_index": int(index),
                    "positions": [int(index)],
                    "target_raw": float(target),
                    "prediction_raw": predictions,
                    "bootstrap_group": rollout_id,
                }
            )

    for row in terminal_rows:
        row_id = str(row["id"])
        positions = [int(index) for index in row["terminal_suffix_response_indices"]]
        if not positions:
            raise ValueError(f"Terminal probe row {row_id} has no selected suffix positions")
        position_predictions, sum_predictions = {}, {}
        for label in checkpoint_labels:
            scored = scores_by_checkpoint[label]["terminal"][row_id]
            values = [float(scored["delta_pred_raw"][index]) for index in positions]
            position_predictions[label] = values
            sum_predictions[label] = math.fsum(values)
        records.append(
            {
                "kind": "terminal",
                "rollout_id": str(row["rollout_id"]),
                "query_id": str(row.get("prompt_id", row["rollout_id"])),
                "response_index": row.get("response_index"),
                "branch_id": row_id,
                "prefix_group_id": str(row["prefix_group_id"]),
                "anchor_state_id": str(row["anchor_state_id"]),
                "anchor_token_index": int(row["anchor_token_index"]),
                "positions": positions,
                "suffix_count": len(positions),
                "continuation_length": len(row["response_token_ids"]),
                "target_raw": float(row["terminal_comp_target"]),
                "prediction_raw": sum_predictions,
                "position_prediction_raw": position_predictions,
                "bootstrap_group": str(row["prefix_group_id"]),
            }
        )
    return records


def checkpoint_metrics(records, checkpoint_label):
    local = [row for row in records if row["kind"] == "local_td"]
    terminal = [row for row in records if row["kind"] == "terminal"]
    local_view = [{**row, "prediction_raw": row["prediction_raw"][checkpoint_label]} for row in local]
    terminal_view = [
        {
            **row,
            "prediction_raw": row["prediction_raw"][checkpoint_label],
            "position_prediction_raw": row["position_prediction_raw"][checkpoint_label],
        }
        for row in terminal
    ]
    return {
        "local_td": local_td_metrics(local_view),
        "terminal": terminal_raw_metrics(terminal_view),
    }


def gradient_similarity(td_vector, terminal_vector):
    """Compute true cosine and norms for two gradients in the same parameter subspace."""
    td_vector = torch.as_tensor(td_vector, dtype=torch.float64).reshape(-1)
    terminal_vector = torch.as_tensor(terminal_vector, dtype=torch.float64).reshape(-1)
    if td_vector.shape != terminal_vector.shape:
        raise ValueError("Gradient vectors must have matching shapes")
    td_norm = torch.linalg.vector_norm(td_vector)
    terminal_norm = torch.linalg.vector_norm(terminal_vector)
    dot = torch.dot(td_vector, terminal_vector)
    denominator = td_norm * terminal_norm
    cosine = float((dot / denominator).item()) if denominator.item() > 0 else None
    return {
        "td_gradient_norm": float(td_norm.item()),
        "terminal_gradient_norm": float(terminal_norm.item()),
        "gradient_dot_product": float(dot.item()),
        "gradient_cosine": cosine,
        "gradient_dimension": int(td_vector.numel()),
    }


def head_gradient_probe(
    model,
    td_rows,
    terminal_rows,
    *,
    source_normalization,
    loss_config,
    loss_normalization,
    terminal_stats,
    pad_token_id,
    device,
    batch_size=1,
):
    """Measure separate-sample TD/terminal gradients on the trainable scalar head.

    The head is a small, exactly measured parameter subspace. The report names
    this scope explicitly; it must not be interpreted as a full-backbone norm.
    """
    if loss_config.objective != "hybrid_terminal_composition" or loss_config.hybrid_reduction != "separate_samples":
        raise ValueError("Gradient probe requires hybrid_terminal_composition with separate_samples reduction")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    head = getattr(model, "scalar_head", None)
    if head is None or not list(head.parameters()):
        raise ValueError("Gradient probe model must expose a trainable scalar_head")
    required = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
        td_rows = [row for row in td_rows if any(float(value) > 0 for value in row["token_loss_mask"])]
        terminal_rows = [row for row in terminal_rows if row.get("terminal_comp_valid", False)]
        if not td_rows or not terminal_rows:
            raise ValueError("Gradient probe requires supervised rows for both objectives")

        td_grad, td_count = _objective_head_gradient(
            model,
            td_rows,
            objective="td",
            parameters=parameters,
            config=loss_config,
            source_normalization=source_normalization,
            loss_normalization=loss_normalization,
            terminal_stats=terminal_stats,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size,
        )
        terminal_grad, terminal_count = _objective_head_gradient(
            model,
            terminal_rows,
            objective="terminal",
            parameters=parameters,
            config=loss_config,
            source_normalization=source_normalization,
            loss_normalization=loss_normalization,
            terminal_stats=terminal_stats,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size,
        )
        result = gradient_similarity(td_grad, terminal_grad)
        result.update(
            {
                "parameter_scope": "scalar_head",
                "parameters": [name for name, _ in model.named_parameters() if name.startswith("scalar_head.")],
                "td_rows": td_count,
                "terminal_rows": terminal_count,
                "terminal_weight_included": float(loss_config.terminal_weight),
                "td_weight_included": float(loss_config.td_weight),
            }
        )
        return result
    finally:
        model.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.requires_grad_(required[id(parameter)])


def _objective_head_gradient(
    model,
    rows,
    *,
    objective,
    parameters,
    config,
    source_normalization,
    loss_normalization,
    terminal_stats,
    pad_token_id,
    device,
    batch_size,
):
    model.zero_grad(set_to_none=True)
    if objective == "td":
        denominator = len(rows)
    elif objective == "terminal":
        denominator = len(rows)
    else:
        raise ValueError(f"Unsupported gradient objective: {objective}")
    device = torch.device(device)
    autocast_dtype = _training_autocast_dtype(config)
    autocast = (
        torch.autocast(device.type, dtype=autocast_dtype)
        if device.type in {"cuda", "cpu"} and autocast_dtype is not None
        else nullcontext()
    )
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = collate(batch_rows, config, loss_normalization, pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast:
            prediction = model(batch["input_ids"], batch["attention_mask"])
            prediction = _reproject_normalized(prediction, source_normalization, loss_normalization)
            _, valid, td, terminal = per_sample_loss(prediction, batch, config, loss_normalization, terminal_stats)
            if objective == "td":
                active = valid & (batch["target_mask"].sum(-1) > 0)
                loss = config.td_weight * (td * active).sum() / denominator
            else:
                _, _, active, _suffix_count, length_weight = _terminal_components(
                    prediction, batch, config, loss_normalization, terminal_stats, valid
                )
                loss = config.terminal_weight * (terminal * length_weight * active).sum() / denominator
        loss.backward()
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            gradients.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            gradients.append(parameter.grad.detach().float().reshape(-1).cpu())
    vector = torch.cat(gradients) if gradients else torch.empty(0, dtype=torch.float32)
    if not torch.isfinite(vector).all():
        raise ValueError(f"Nonfinite {objective} scalar-head gradient")
    return vector, denominator


def _reproject_normalized(prediction, source_normalization, loss_normalization):
    """Express a checkpoint's normalized output in the hybrid loss's TD scale."""
    if source_normalization.get("enabled", False):
        raw = prediction * float(source_normalization["std"]) + float(source_normalization["mean"])
    else:
        raw = prediction
    if loss_normalization.get("enabled", False):
        return (raw - float(loss_normalization["mean"])) / float(loss_normalization["std"])
    return raw


def _training_autocast_dtype(config):
    setting = config.training_autocast
    if setting == "model_dtype":
        setting = config.dtype
    if setting == "none":
        return None
    return getattr(torch, setting)


def _read_matching_rows(path, *, id_field, accepted_ids):
    """Read only rows for held-out rollout IDs, avoiding loading the train pool."""
    path = Path(path)
    accepted_ids = set(map(str, accepted_ids))
    if path.suffix == ".parquet":
        from .train import read_rows

        return [row for row in read_rows(path) if str(row.get(id_field, "")) in accepted_ids]
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get(id_field, "")) in accepted_ids:
                rows.append(row)
    return rows


def _build_terminal_rows(continuations, labels, config):
    rows, selection_meta = continuation_rows(continuations, labels, config)
    labels_by_state = {str(row["state_id"]): row for row in labels if row.get("state_id") is not None}
    labels_by_token = {
        (str(row["rollout_id"]), int(row["token_index"])): row
        for row in labels
        if row.get("rollout_id") is not None and row.get("token_index") is not None
    }
    sources = {
        str(row.get("continuation_id") or ""): row for row in continuations if row.get("continuation_id") is not None
    }
    for row in rows:
        source = sources.get(str(row["id"]))
        if source is None:
            raise ValueError(f"Cannot recover source continuation metadata for {row['id']!r}")
        label = labels_by_state.get(str(source.get("state_id")))
        if label is None:
            label = labels_by_token.get((str(source.get("rollout_id")), int(source["token_index"])))
        if label is None:
            raise ValueError(f"Cannot recover anchor label for continuation {row['id']!r}")
        row["prefix_group_id"] = str(label.get("state_id") or f"{label['rollout_id']}:t{label['token_index']}")
        row["anchor_state_id"] = str(label.get("state_id") or row["prefix_group_id"])
        row["anchor_token_index"] = int(label["token_index"])
        row["rollout_id"] = str(source["rollout_id"])
        row["prompt_id"] = label.get("prompt_id", source.get("prompt_id"))
        row["response_index"] = source.get("response_index")
        row["continuation_index"] = source.get("continuation_index")
    return rows, selection_meta


def _score_rows(worker, local_rows, terminal_rows):
    local_inputs = [
        ScoringRow(
            str(row["id"]),
            tuple(row["prompt_token_ids"]),
            tuple(row["response_token_ids"]),
            tuple(row["selected_response_indices"]),
            metadata=row,
        )
        for row in local_rows
    ]
    terminal_inputs = [
        ScoringRow(
            str(row["id"]),
            tuple(row["prompt_token_ids"]),
            tuple(row["response_token_ids"]),
            tuple(row["terminal_suffix_response_indices"]),
            metadata=row,
        )
        for row in terminal_rows
    ]
    # The pilot uses full causal windows. A single full-sequence forward per
    # rollout returns every selected position and avoids replaying the same
    # prefix once for each MC anchor/suffix point. Preserve the worker's exact
    # legacy-tail implementation for artifacts that require prefix-specific
    # cropping; those rows do not satisfy this equivalence.
    local_scores = _score_full_sequences(worker, local_inputs)
    terminal_scores = _score_full_sequences(worker, terminal_inputs)
    return {
        "local": {row["rollout_id"]: row for row in local_scores},
        "terminal": {row["rollout_id"]: row for row in terminal_scores},
    }


def _score_full_sequences(worker, rows):
    """Score all selected positions with one causal forward per full row."""
    if worker.config.window_policy != "full" or worker.engine_worker is not None:
        return worker.score(rows)
    norm = worker.metadata["normalization"]
    output = {}
    to_score = []
    for row in rows:
        length = len(row.response_token_ids)
        output[row.rollout_id] = {
            "rollout_id": row.rollout_id,
            "delta_pred_normalized": [0.0] * length,
            "delta_pred_raw": [0.0] * length,
            "critic_signal_mask": [0.0] * length,
            "critic_checkpoint": worker.metadata["weights_sha256"],
            "critic_stats_version": worker.metadata["stats_version"],
            "critic_source_sha256": worker.metadata.get("source_sha256"),
            "critic_window": {
                "policy": "full",
                "max_length": worker.config.max_length,
                "semantic_changes": [],
                "prefix_shifts": {str(index): 0 for index in row.selected_token_indices},
            },
        }
        if row.selected_token_indices:
            if len(row.prompt_token_ids) + length > worker.config.max_length:
                raise ValueError(f"Selected sequence {row.rollout_id} exceeds full window {worker.config.max_length}")
            to_score.append(row)

    to_score.sort(key=lambda row: (len(row.prompt_token_ids) + len(row.response_token_ids), row.rollout_id))
    worker.model.eval()
    for start in range(0, len(to_score), worker.microbatch):
        batch = to_score[start : start + worker.microbatch]
        sequences = [list(row.prompt_token_ids + row.response_token_ids) for row in batch]
        width = max(map(len, sequences))
        tokens = torch.full((len(batch), width), worker.pad_token_id, dtype=torch.long, device=worker.device)
        mask = torch.zeros_like(tokens)
        for index, sequence in enumerate(sequences):
            tokens[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=worker.device)
            mask[index, : len(sequence)] = 1
        context = (
            torch.autocast(worker.device.type, dtype=torch.bfloat16)
            if worker.config.scoring_autocast == "bfloat16"
            else nullcontext()
        )
        with torch.inference_mode(), context:
            predictions = worker.model(tokens, mask).float()
        for row_index, row in enumerate(batch):
            destination = output[row.rollout_id]
            prompt_length = len(row.prompt_token_ids)
            for response_index in row.selected_token_indices:
                value = float(predictions[row_index, prompt_length + response_index].item())
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite critic prediction for {row.rollout_id}:{response_index}")
                raw = value * norm["std"] + norm["mean"] if norm["enabled"] else value
                destination["delta_pred_normalized"][response_index] = value
                destination["delta_pred_raw"][response_index] = raw
                destination["critic_signal_mask"][response_index] = 1.0
    return [output[row.rollout_id] for row in rows]


def _compatible_artifacts(left, right):
    for key in ("tokenizer_sha256",):
        if left.get(key) != right.get(key):
            raise ValueError(f"Paired checkpoints differ in {key}")
    left_config, right_config = left["config"], right["config"]
    for key in (
        "model_path",
        "tokenizer_path",
        "value_layer",
        "max_length",
        "window_policy",
        "dtype",
        "attention_implementation",
        "scoring_autocast",
    ):
        if left_config.get(key) != right_config.get(key):
            raise ValueError(f"Paired checkpoints differ in scoring semantic {key}")


def _artifact_label(metadata):
    config = metadata["config"]
    state = metadata.get("training_state", {})
    step = state.get("global_step", metadata.get("source_step", "unknown"))
    return f"{config['objective']}-step{step}-{metadata['weights_sha256'][:8]}"


def _raw_rmse_summary(rows):
    errors = [float(row["prediction_raw"]) - float(row["target_raw"]) for row in rows]
    targets = [float(row["target_raw"]) for row in rows]
    mse = _mean([error * error for error in errors])
    zero_mse = _mean([target * target for target in targets])
    return {
        "count": len(rows),
        "raw_mse": mse,
        "raw_rmse": math.sqrt(mse) if math.isfinite(mse) else None,
        "zero_raw_mse": zero_mse,
        "zero_raw_rmse": math.sqrt(zero_mse) if math.isfinite(zero_mse) else None,
        "improvement_vs_zero_mse_fraction": 1.0 - mse / zero_mse if zero_mse > 0 and math.isfinite(mse) else None,
        "mean_prediction_raw": _mean([float(row["prediction_raw"]) for row in rows]),
        "mean_target_raw": _mean(targets),
    }


def _mean(values):
    return float(sum(values) / len(values)) if values else float("nan")


def _length_bin(length):
    for low, high in LOCAL_LENGTH_BINS:
        if low <= length <= high:
            return f"{low}-{high}"
    return "32769+"


def _bin_sort_key(name):
    return int(name.split("-", 1)[0]) if "-" in name else 32769


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run_probe(args):
    from .legacy_adapter import adapt_legacy_rows

    artifact_paths = [args.checkpoint]
    if args.compare_checkpoint:
        if Path(args.checkpoint).resolve() == Path(args.compare_checkpoint).resolve():
            raise ValueError("--checkpoint and --compare-checkpoint must be different artifacts")
        artifact_paths.append(args.compare_checkpoint)
    artifact_metadata = {path: read_artifact(path) for path in artifact_paths}
    for index in range(1, len(artifact_paths)):
        _compatible_artifacts(artifact_metadata[artifact_paths[0]], artifact_metadata[artifact_paths[index]])

    loss_path = args.loss_config_checkpoint
    if loss_path is None:
        for path in artifact_paths:
            if artifact_metadata[path]["config"]["objective"] == "hybrid_terminal_composition":
                loss_path = path
                break
    if loss_path is None:
        raise ValueError(
            "Terminal probes require a hybrid artifact; pass it as --checkpoint, --compare-checkpoint, "
            "or --loss-config-checkpoint"
        )
    loss_metadata = read_artifact(loss_path)
    loss_config = ScalarConfig(**loss_metadata["config"])
    if loss_config.objective != "hybrid_terminal_composition":
        raise ValueError("--loss-config-checkpoint must contain hybrid_terminal_composition loss semantics")
    if args.gradients and loss_config.hybrid_reduction != "separate_samples":
        raise ValueError("--gradients requires the hybrid artifact to use hybrid_reduction=separate_samples")

    split = json.loads(Path(args.split).read_text())
    if not isinstance(split, dict) or any(value not in {"train", "eval"} for value in split.values()):
        raise ValueError("Split must map rollout IDs to train/eval")
    eval_ids = {str(row_id) for row_id, value in split.items() if value == "eval"}
    if not eval_ids:
        raise ValueError("Split contains no eval rollout IDs")
    rollouts = _read_matching_rows(args.rollouts, id_field="id", accepted_ids=eval_ids)
    labels = _read_matching_rows(args.labels, id_field="rollout_id", accepted_ids=eval_ids)
    observed_ids = {str(row["id"]) for row in rollouts}
    if observed_ids != eval_ids:
        raise ValueError(
            "Eval rollout coverage mismatch: "
            f"missing={len(eval_ids - observed_ids)}, extra={len(observed_ids - eval_ids)}"
        )
    examples, diagnostics = adapt_legacy_rows(rollouts, labels)
    td_rows = ordinary_rows(examples, loss_config)
    for example, row in zip(examples, td_rows, strict=True):
        row["selected_state_ids"] = [
            state.state_id or f"{state.rollout_id}:t{state.token_index}" for state in example.states
        ]
        row["prompt_id"] = example.rollout.prompt_id
        row["response_index"] = example.rollout.metadata.get("response_index")

    continuations = _read_matching_rows(args.continuations, id_field="rollout_id", accepted_ids=eval_ids)
    terminal_candidates, terminal_selection = _build_terminal_rows(continuations, labels, loss_config)
    if not terminal_candidates:
        raise ValueError("Hybrid loss config and held-out data produced no terminal probe rows")
    if args.full_terminal_eval:
        terminal_rows = terminal_candidates
        terminal_selection.update(
            {"mode": "full_terminal_eval", "seed": args.seed, "eligible_rows": len(terminal_candidates)}
        )
    else:
        requested = 512 if args.terminal_count is None else args.terminal_count
        terminal_rows = select_terminal_probe_rows(terminal_candidates, requested, seed=args.seed)
        terminal_selection.update(
            {
                "mode": "fixed_prefix_group_probe",
                "seed": args.seed,
                "requested_rows": requested,
                "eligible_rows": len(terminal_candidates),
                "selected_rows": len(terminal_rows),
            }
        )
    selected_ids = [str(row["id"]) for row in terminal_rows]
    terminal_selection.update(
        {
            "selected_prefix_groups": len({str(row["prefix_group_id"]) for row in terminal_rows}),
            "selected_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        }
    )
    if args.gradient_terminal_count < 1:
        raise ValueError("--gradient-terminal-count must be positive")
    if args.full_terminal_eval:
        gradient_terminal_rows = select_terminal_probe_rows(
            terminal_candidates,
            min(args.gradient_terminal_count, len(terminal_candidates)),
            seed=args.seed,
        )
    else:
        gradient_terminal_rows = terminal_rows[: args.gradient_terminal_count]
    gradient_selection = {
        "seed": args.seed,
        "selected_rows": len(gradient_terminal_rows),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in gradient_terminal_rows).encode()
        ).hexdigest(),
    }

    labels_by_checkpoint = {}
    scores = {}
    gradients_by_checkpoint = {}
    for path in artifact_paths:
        metadata = artifact_metadata[path]
        label = _artifact_label(metadata)
        if label in labels_by_checkpoint.values():
            raise ValueError(f"Checkpoint label collision: {label}")
        labels_by_checkpoint[path] = label
        worker = FrozenDeltaWorker(path, device=args.device, microbatch=args.microbatch)
        scores[label] = _score_rows(worker, td_rows, terminal_rows)
        if args.gradients:
            gradients_by_checkpoint[label] = head_gradient_probe(
                worker.model,
                td_rows,
                gradient_terminal_rows,
                source_normalization=artifact_metadata[path]["normalization"],
                loss_config=loss_config,
                loss_normalization=loss_metadata["normalization"],
                terminal_stats=loss_metadata["terminal_stats"],
                pad_token_id=worker.pad_token_id,
                device=args.device,
                batch_size=args.gradient_batch_size,
            )
        del worker

    checkpoint_labels = [labels_by_checkpoint[path] for path in artifact_paths]
    records = paired_records(td_rows, terminal_rows, scores, checkpoint_labels)
    summary = {
        "probe_version": 1,
        "selection": terminal_selection,
        "td_rollouts": len(td_rows),
        "td_selected_positions": sum(len(row["selected_response_indices"]) for row in td_rows),
        "terminal_rows": len(terminal_rows),
        "terminal_candidate_rows": len(terminal_candidates),
        "loss_config": {
            "checkpoint": str(Path(loss_path).resolve()),
            "weights_sha256": loss_metadata["weights_sha256"],
            "objective": loss_config.objective,
            "hybrid_reduction": loss_config.hybrid_reduction,
            "td_weight": loss_config.td_weight,
            "terminal_weight": loss_config.terminal_weight,
            "terminal_normalization": loss_config.terminal_normalization,
            "terminal_length_weighting": loss_config.terminal_length_weighting,
            "normalization": loss_metadata["normalization"],
            "terminal_stats": loss_metadata["terminal_stats"],
        },
        "checkpoints": {},
        "diagnostics": diagnostics,
        "outputs": {
            "per_row": str(Path(args.output).with_suffix(".perrow.jsonl")),
        },
    }
    if args.compare_checkpoint:
        summary["paired_comparison"] = _paired_metric_deltas(records, checkpoint_labels)
    if args.gradients:
        summary["gradient_selection"] = gradient_selection
    for path in artifact_paths:
        label = labels_by_checkpoint[path]
        artifact = artifact_metadata[path]
        checkpoint_summary = {
            "path": str(Path(path).resolve()),
            "weights_sha256": artifact["weights_sha256"],
            "objective": artifact["config"]["objective"],
            **checkpoint_metrics(records, label),
        }
        if args.gradients:
            checkpoint_summary["gradient_probe"] = gradients_by_checkpoint[label]
        summary["checkpoints"][label] = checkpoint_summary

    _atomic_jsonl(args.output.with_suffix(".perrow.jsonl"), records)
    _atomic_json(args.output, summary)
    return summary


def _paired_metric_deltas(records, checkpoint_labels):
    if len(checkpoint_labels) != 2:
        raise ValueError("Paired metric deltas require exactly two checkpoints")
    first, second = checkpoint_labels
    output = {}
    for kind in ("local_td", "terminal"):
        rows = [row for row in records if row["kind"] == kind]
        deltas = [float(row["prediction_raw"][second]) - float(row["prediction_raw"][first]) for row in rows]
        targets = [float(row["target_raw"]) for row in rows]
        first_errors = [float(row["prediction_raw"][first]) - target for row, target in zip(rows, targets, strict=True)]
        second_errors = [
            float(row["prediction_raw"][second]) - target for row, target in zip(rows, targets, strict=True)
        ]
        paired_mse_change = [
            second_error * second_error - first_error * first_error
            for first_error, second_error in zip(first_errors, second_errors, strict=True)
        ]
        output[kind] = {
            "first_checkpoint": first,
            "second_checkpoint": second,
            "second_minus_first_prediction_raw": regression_metrics(deltas, [0.0] * len(deltas)),
            "first_raw_mse": _mean([error * error for error in first_errors]),
            "second_raw_mse": _mean([error * error for error in second_errors]),
            "paired_raw_mse_change": _mean(paired_mse_change),
            "paired_group_key": "prefix_group_id" if kind == "terminal" else "rollout_id",
        }
    return output


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Score paired portable critic checkpoints on fixed held-out TD and terminal probes.",
        epilog="Default selection keeps every eval TD rollout and samples 512 terminal branches by prefix group.",
    )
    parser.add_argument("--checkpoint", required=True, help="Portable artifact to score")
    parser.add_argument("--compare-checkpoint", help="Optional second portable artifact scored on identical rows")
    parser.add_argument(
        "--loss-config-checkpoint",
        help="Hybrid artifact defining local/terminal loss and continuation semantics; inferred from paired artifacts",
    )
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--continuations", required=True)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Summary JSON path; per-row JSONL uses .perrow.jsonl",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--terminal-count", type=int, help="Fixed terminal row count (default: 512)")
    selection.add_argument(
        "--full-terminal-eval",
        action="store_true",
        help="Score every held-out terminal continuation (the full evaluation entry point)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for prefix-group and within-group selection")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument(
        "--gradients",
        action="store_true",
        help="Also probe separate TD/terminal scalar-head gradients",
    )
    parser.add_argument("--gradient-terminal-count", type=int, default=64)
    parser.add_argument("--gradient-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.terminal_count is not None and args.terminal_count < 1:
        parser.error("--terminal-count must be positive")
    if args.microbatch < 1:
        parser.error("--microbatch must be positive")
    if args.gradient_batch_size < 1 or args.gradient_terminal_count < 1:
        parser.error("gradient batch size and terminal count must be positive")
    return args


def main():
    args = _parse_args()
    summary = run_probe(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
