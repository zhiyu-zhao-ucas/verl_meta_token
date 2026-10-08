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
"""Logical batch size one loss; physical batching never changes denominators."""

import math

import torch


def _normalization_values(normalization):
    enabled = normalization is not None and normalization.get("enabled", False)
    if not enabled:
        return False, 1.0, 0.0
    std, mean = float(normalization["std"]), float(normalization["mean"])
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Enabled TD normalization requires a finite positive std")
    if not math.isfinite(mean):
        raise ValueError("Enabled TD normalization requires a finite mean")
    return True, std, mean


def _terminal_components(pred, batch, config, normalization, terminal_stats, sample_valid):
    """Return unweighted normalized loss, raw residual, and suffix counts per row."""
    if terminal_stats is None and config.terminal_normalization == "independent":
        raise ValueError("Independent terminal normalization requires terminal statistics")
    terminal_valid = batch["terminal_comp_valid"].to(device=pred.device).bool() & sample_valid
    if terminal_valid.shape != (pred.shape[0],):
        raise ValueError("terminal_comp_valid must have shape [batch]")

    positions = batch["terminal_suffix_positions"].to(device=pred.device).long()
    suffix_mask = batch["terminal_suffix_mask"].to(device=pred.device).float()
    if positions.shape != suffix_mask.shape or positions.shape[0] != pred.shape[0]:
        raise ValueError("terminal suffix positions and masks must have matching [batch, suffix] shapes")
    if terminal_valid.any():
        active_values = suffix_mask[terminal_valid]
        if not torch.isfinite(active_values).all() or ((active_values != 0) & (active_values != 1)).any():
            raise ValueError("terminal_suffix_mask must contain finite binary values")

    row_mask = terminal_valid.unsqueeze(-1)
    row_suffix_mask = torch.where(row_mask, suffix_mask, torch.zeros_like(suffix_mask))
    active_suffix = row_suffix_mask > 0
    out_of_range = (positions < 0) | (positions >= pred.shape[1])
    if (active_suffix & out_of_range).any():
        raise ValueError("Active terminal suffix position is outside the model output")
    suffix_count = row_suffix_mask.sum(-1)
    if (terminal_valid & (suffix_count == 0)).any():
        raise ValueError("A valid terminal composition row must have at least one suffix position")

    safe_positions = positions.clamp(min=0, max=pred.shape[1] - 1)
    batch_indices = torch.arange(pred.shape[0], device=pred.device).unsqueeze(1).expand_as(safe_positions)
    raw_suffix = pred[batch_indices, safe_positions]
    normalization_enabled, td_std, td_mean = _normalization_values(normalization)
    if normalization_enabled:
        # The terminal target is a raw sum, so each selected position receives
        # the same inverse transform as the local TD scalar, including K * mean.
        raw_suffix = raw_suffix * td_std + td_mean
    raw_suffix = torch.where(active_suffix, raw_suffix, torch.zeros_like(raw_suffix))
    predicted_sum = (raw_suffix * row_suffix_mask).sum(-1)

    terminal_target = batch["terminal_comp_target"].to(device=pred.device).float()
    if terminal_target.shape != (pred.shape[0],):
        raise ValueError("terminal_comp_target must have shape [batch]")
    if terminal_valid.any() and not torch.isfinite(terminal_target[terminal_valid]).all():
        raise ValueError("Valid terminal composition targets must be finite")
    safe_target = torch.where(terminal_valid, terminal_target, torch.zeros_like(terminal_target))
    residual = torch.where(terminal_valid, predicted_sum, torch.zeros_like(predicted_sum)) - safe_target

    if config.terminal_normalization == "td":
        terminal_std = td_std if normalization_enabled else 1.0
    else:
        terminal_std = float(terminal_stats.get("std", 1.0))
        if not math.isfinite(terminal_std):
            raise ValueError("Independent terminal normalization requires a finite std")
        terminal_std = max(terminal_std, 1e-8)
    terminal = 0.5 * (residual / terminal_std).square()
    terminal_length_weight = terminal_valid.float()
    if config.terminal_length_weighting == "inverse_count":
        terminal_length_weight = torch.where(
            terminal_valid, suffix_count.clamp_min(1).reciprocal(), torch.zeros_like(suffix_count)
        )
    return terminal, residual, terminal_valid, suffix_count, terminal_length_weight


def per_sample_loss(pred, batch, config, normalization, terminal_stats=None):
    if pred.ndim != 2:
        raise ValueError(f"pred must have shape [batch, sequence], got {tuple(pred.shape)}")
    if batch["target"].shape != pred.shape or batch["target_mask"].shape != pred.shape:
        raise ValueError("prediction, target and target_mask shapes must match")
    pred = pred.float()
    sample_valid = batch.get("sample_valid_mask")
    if sample_valid is None:
        sample_valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    else:
        sample_valid = sample_valid.to(device=pred.device).bool()
    if sample_valid.shape != (pred.shape[0],):
        raise ValueError("sample_valid_mask must have shape [batch]")
    _normalization_values(normalization)  # Validation also applies to local TD batches.
    target_mask = batch["target_mask"].to(device=pred.device).float()
    mask = torch.where(sample_valid.unsqueeze(-1), target_mask, torch.zeros_like(target_mask))
    signal = batch["signal_mask"].to(device=pred.device).bool()
    if signal.shape != pred.shape:
        raise ValueError("signal_mask must have the same shape as predictions")
    active_targets = mask > 0
    safe_pred = torch.where(active_targets, pred, torch.zeros_like(pred))
    target = batch["target"].to(device=pred.device).float()
    safe_target = torch.where(active_targets, target, torch.zeros_like(target))
    error = 0.5 * (safe_pred - safe_target).square()
    count = mask.sum(-1)
    td = (error * mask).sum(-1) / count.clamp_min(1)
    if config.loss_type == "nonzero_balanced_mse":
        sm, bg = mask * signal, mask * ~signal
        sc, bc = sm.sum(-1), bg.sum(-1)
        balanced = 0.5 * ((error * sm).sum(-1) / sc.clamp_min(1) + (error * bg).sum(-1) / bc.clamp_min(1))
        td = torch.where((sc > 0) & (bc > 0), balanced, td)
    elif config.loss_type != "mse":
        raise ValueError(f"Unsupported loss {config.loss_type}")
    terminal = pred.new_zeros(pred.shape[0])
    if config.objective == "hybrid_terminal_composition":
        terminal, _, _, _, terminal_length_weight = _terminal_components(
            pred, batch, config, normalization, terminal_stats, sample_valid
        )
        losses = config.td_weight * td + config.terminal_weight * terminal * terminal_length_weight
    else:
        # The source local-TD objective uses one optimizer contribution per real
        # dataset row, including zero-loss rows with an empty selected mask.
        losses = td
    losses = losses * sample_valid
    return losses, sample_valid, td, terminal


def delta_loss(model_output, data, dp_group=None):
    """V1 callback: sum real-row losses / global real rows, compensating DP AVG."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric.utils import AggregationType, Metric

    from .training_config import ScalarConfig

    config = ScalarConfig(**tu.get(data, "delta_config"))
    normalization = tu.get(data, "normalization")
    terminal_stats = tu.get(data, "terminal_stats")
    losses, valid, td, terminal = per_sample_loss(
        model_output["delta_scalar"],
        data,
        config,
        normalization,
        terminal_stats,
    )
    dp_size = int(tu.get(data, "dp_size"))
    global_valid_rows = max(int(tu.get(data, "valid_sample_count")), 1)
    scale = dp_size / global_valid_rows
    sample_valid = data["sample_valid_mask"].float() if "sample_valid_mask" in data.keys() else valid.float()
    td_valid = valid & (data["target_mask"].sum(-1) > 0)
    if config.objective == "hybrid_terminal_composition":
        _, terminal_residual, terminal_valid, terminal_suffix_count, terminal_length_weight = _terminal_components(
            model_output["delta_scalar"].float(), data, config, normalization, terminal_stats, valid
        )
    else:
        terminal_residual = model_output["delta_scalar"].new_zeros(model_output["delta_scalar"].shape[0])
        terminal_valid = torch.zeros_like(valid)
        terminal_suffix_count = torch.zeros_like(valid, dtype=torch.float32)
        terminal_length_weight = terminal_suffix_count
    if config.objective == "hybrid_terminal_composition" and config.hybrid_reduction == "separate_samples":
        # Counts cover the whole distributed update, before microbatch splitting.
        td_scale = dp_size / max(int(tu.get(data, "td_sample_count")), 1)
        terminal_scale = dp_size / max(int(tu.get(data, "terminal_sample_count")), 1)
        td_total = (td * td_valid).sum() * td_scale
        terminal_total = (terminal * terminal_valid).sum() * terminal_scale
        terminal_optimization_total = (
            config.terminal_weight * (terminal * terminal_length_weight * terminal_valid).sum() * terminal_scale
        )
        total = config.td_weight * td_total + terminal_optimization_total
    else:
        td_total = (td * valid).sum() * scale
        terminal_total = (terminal * terminal_valid).sum() * scale
        terminal_optimization_total = (
            config.terminal_weight * (terminal * terminal_length_weight * terminal_valid).sum() * scale
        )
        total = (losses * valid).sum() * scale
    row_mask = sample_valid.unsqueeze(-1)
    target_mask = data["target_mask"].float() * row_mask
    signal_mask = data["signal_mask"].float() * row_mask
    metrics = {
        # SUM metrics are averaged across DP ranks and summed across gradient
        # accumulation microbatches, matching the logical-sample loss scaling.
        "critic/total_loss": Metric(value=total, aggregation=AggregationType.SUM),
        "critic/td_loss": Metric(value=td_total, aggregation=AggregationType.SUM),
        "critic/terminal_loss": Metric(value=terminal_total, aggregation=AggregationType.SUM),
        "critic/terminal_optimization_loss": Metric(value=terminal_optimization_total, aggregation=AggregationType.SUM),
        "critic/td_rows": Metric(value=td_valid.float().sum() * dp_size, aggregation=AggregationType.SUM),
        "critic/terminal_rows": Metric(value=terminal_valid.float().sum() * dp_size, aggregation=AggregationType.SUM),
        "critic/terminal_count": Metric(value=terminal_valid.float().sum() * dp_size, aggregation=AggregationType.SUM),
        "critic/terminal_suffix_count": Metric(
            value=(terminal_suffix_count * terminal_valid).sum() * dp_size, aggregation=AggregationType.SUM
        ),
        "critic/terminal_raw_squared_error": Metric(
            value=(terminal_residual.square() * terminal_valid).sum() * dp_size, aggregation=AggregationType.SUM
        ),
        "critic/valid_rows": Metric(value=valid.float().sum() * dp_size, aggregation=AggregationType.SUM),
        "critic/target_tokens": Metric(value=target_mask.sum() * dp_size, aggregation=AggregationType.SUM),
        "critic/signal_tokens": Metric(
            value=(target_mask * signal_mask).sum() * dp_size, aggregation=AggregationType.SUM
        ),
        "critic/background_tokens": Metric(
            value=(target_mask * (1.0 - signal_mask)).sum() * dp_size, aggregation=AggregationType.SUM
        ),
        "critic/input_tokens": Metric(
            value=(data["attention_mask"].float() * row_mask).sum() * dp_size,
            aggregation=AggregationType.SUM,
        ),
    }
    return total, metrics
