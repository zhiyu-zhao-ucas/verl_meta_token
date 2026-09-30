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

import torch


def per_sample_loss(pred, batch, config, normalization, terminal_stats=None):
    if pred.ndim != 2:
        raise ValueError(f"pred must have shape [batch, sequence], got {tuple(pred.shape)}")
    if batch["target"].shape != pred.shape or batch["target_mask"].shape != pred.shape:
        raise ValueError("prediction, target and target_mask shapes must match")
    pred = pred.float()
    mask, signal = batch["target_mask"].float(), batch["signal_mask"].bool()
    if signal.shape != pred.shape:
        raise ValueError("signal_mask must have the same shape as predictions")
    error = 0.5 * (pred - batch["target"].float()).square()
    count = mask.sum(-1)
    td = (error * mask).sum(-1) / count.clamp_min(1)
    if config.loss_type == "nonzero_balanced_mse":
        sm, bg = mask * signal, mask * ~signal
        sc, bc = sm.sum(-1), bg.sum(-1)
        balanced = 0.5 * ((error * sm).sum(-1) / sc.clamp_min(1) + (error * bg).sum(-1) / bc.clamp_min(1))
        td = torch.where((sc > 0) & (bc > 0), balanced, td)
    elif config.loss_type != "mse":
        raise ValueError(f"Unsupported loss {config.loss_type}")
    terminal = pred.sum(-1) * 0
    if config.objective == "hybrid_terminal_composition":
        if terminal_stats is None:
            raise ValueError("Hybrid loss requires independent terminal statistics")
        positions = batch["terminal_suffix_positions"].long()
        suffix_mask = batch["terminal_suffix_mask"].float()
        if positions.shape != suffix_mask.shape or positions.shape[0] != pred.shape[0]:
            raise ValueError("terminal suffix positions and masks must have matching [batch, suffix] shapes")
        active_suffix = suffix_mask > 0
        if active_suffix.any() and (
            (positions[active_suffix] < 0).any() or (positions[active_suffix] >= pred.shape[1]).any()
        ):
            raise ValueError("Active terminal suffix position is outside the model output")
        terminal_valid = batch["terminal_comp_valid"].bool()
        has_suffix = active_suffix.any(-1)
        if (terminal_valid & ~has_suffix).any():
            raise ValueError("A valid terminal composition row must have at least one suffix position")
        safe_positions = positions.clamp(min=0, max=pred.shape[1] - 1)
        batch_indices = torch.arange(pred.shape[0], device=pred.device).unsqueeze(1).expand_as(safe_positions)
        raw_suffix = pred[batch_indices, safe_positions]
        if normalization is not None and normalization.get("enabled", False):
            raw_suffix = raw_suffix * float(normalization["std"]) + float(normalization["mean"])
        predicted_sum = (raw_suffix * suffix_mask).sum(-1)
        terminal_std = max(float(terminal_stats.get("std", 1.0)), 1e-8)
        terms = (predicted_sum - batch["terminal_comp_target"].float()) / terminal_std
        terminal = 0.5 * terms.square() * terminal_valid
        losses = config.td_weight * td + config.terminal_weight * terminal
    else:
        # The source local-TD objective uses one optimizer contribution per real
        # dataset row, including zero-loss rows with an empty selected mask.
        losses = td
    sample_valid = batch.get("sample_valid_mask")
    if sample_valid is None:
        sample_valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    else:
        sample_valid = sample_valid.to(device=pred.device).bool()
    if sample_valid.shape != (pred.shape[0],):
        raise ValueError("sample_valid_mask must have shape [batch]")
    losses = losses * sample_valid
    return losses, sample_valid, td, terminal


def delta_loss(model_output, data, dp_group=None):
    """V1 callback: sum real-row losses / global real rows, compensating DP AVG."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric.utils import AggregationType, Metric

    from .training_config import ScalarConfig

    config = ScalarConfig(**tu.get(data, "delta_config"))
    losses, valid, td, terminal = per_sample_loss(
        model_output["delta_scalar"],
        data,
        config,
        tu.get(data, "normalization"),
        tu.get(data, "terminal_stats"),
    )
    dp_size = int(tu.get(data, "dp_size"))
    global_valid_rows = max(int(tu.get(data, "valid_sample_count")), 1)
    scale = dp_size / global_valid_rows
    sample_valid = data["sample_valid_mask"].float() if "sample_valid_mask" in data.keys() else valid.float()
    row_mask = sample_valid.unsqueeze(-1)
    target_mask = data["target_mask"].float() * row_mask
    signal_mask = data["signal_mask"].float() * row_mask
    metrics = {
        # SUM metrics are averaged across DP ranks and summed across gradient
        # accumulation microbatches, matching the logical-sample loss scaling.
        "critic/total_loss": Metric(value=(losses * valid).sum() * scale, aggregation=AggregationType.SUM),
        "critic/td_loss": Metric(value=(td * valid).sum() * scale, aggregation=AggregationType.SUM),
        "critic/terminal_loss": Metric(value=(terminal * valid).sum() * scale, aggregation=AggregationType.SUM),
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
    return (losses * valid).sum() * scale, metrics
