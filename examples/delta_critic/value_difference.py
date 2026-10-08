# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""MC prefix values and selected-segment differences share explicit boundaries.

Boundary i means prompt + response[:i], BEFORE response token i. Its scalar
is read at causal sequence position len(prompt) + i - 1. Boundary n is the
stopped/capped full response, whose training value is the stored terminal
reward. It is not the zero value of an absorbing state after reward payment.
No value is fabricated for unsampled intermediate prefixes.
"""

import math

SEMANTICS = {
    "version": 1,
    "state": "prompt + response[:i], before token i",
    "sequence_position": "prompt_length + i - 1",
    "value_target": "stored v_prefix: mean terminal reward of collector MC continuations",
    "terminal_value_target": "stored terminal_reward at full response boundary n",
    "delta_target": "V(next selected boundary) - V(current selected boundary)",
    "terminal_delta_target": "0 at boundary n (no remaining selected segment)",
    "policy_value_delta": "predicted V(next boundary) - predicted V(current boundary), including predicted V(n)",
    "support": "selected prefix boundaries plus full-response boundary; no interpolated values",
}


def boundary_positions(prompt_length, response_length, selected):
    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("A nonempty prompt is required")
    if type(response_length) is not int or response_length < 1:
        raise ValueError("A nonempty response is required")
    selected = tuple(selected)
    if any(type(i) is not int or not 0 <= i < response_length for i in selected):
        raise ValueError("Selected boundaries must be response token indices")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("Selected boundaries must be sorted and unique")
    return tuple(prompt_length + i - 1 for i in (*selected, response_length))


def differences(values):
    values = tuple(float(v) for v in values)
    if len(values) < 1 or not all(math.isfinite(v) for v in values):
        raise ValueError("Boundary values must be nonempty and finite")
    return tuple(right - left for left, right in zip(values[:-1], values[1:], strict=True))


def boundary_rows(examples, target):
    """Both arms have identical inputs, supervision positions and masks.

    Direct-delta has an extra zero terminal anchor so its support also includes
    the full-response boundary. These are controlled baselines, not historical
    response-wide zero-background / balanced-MSE TD80 training.
    """
    if target not in {"value", "direct_delta"}:
        raise ValueError("target must be value or direct_delta")
    rows = []
    for example in examples:
        rollout = example.rollout
        indices = tuple(s.token_index for s in example.states)
        positions = boundary_positions(len(rollout.prompt_token_ids), len(rollout.response_token_ids), indices)
        values = tuple(s.v_prefix for s in example.states) + (rollout.terminal_reward,)
        deltas = differences(values) + (0.0,)
        rows.append(
            {
                "id": rollout.rollout_id,
                "prompt_token_ids": list(rollout.prompt_token_ids),
                "response_token_ids": list(rollout.response_token_ids),
                "token_targets": [0.0] * len(rollout.response_token_ids),
                "token_loss_mask": [0.0] * len(rollout.response_token_ids),
                "token_signal_mask": [0.0] * len(rollout.response_token_ids),
                "boundary_positions": positions,
                "boundary_targets": values if target == "value" else deltas,
                "selected_response_indices": indices,
            }
        )
    return rows


def boundary_batch(rows, config, pad_token_id):
    """Use the same V1 dense engine, overriding only target values/positions."""
    from .training_data import training_batch

    norm = {"enabled": False, "mode": "none", "mean": None, "std": None}
    batch = training_batch(rows, config, norm, None, pad_token_id)
    for index, row in enumerate(rows):
        for position, value in zip(row["boundary_positions"], row["boundary_targets"], strict=True):
            if not math.isfinite(value):
                raise ValueError("Nonfinite boundary target")
            batch["target"][index, position] = value
            batch["target_mask"][index, position] = 1.0
            batch["signal_mask"][index, position] = 1.0
    batch["loss_mask"] = batch["target_mask"]
    return batch


def boundary_mse_loss(model_output, data, dp_group=None):
    """Same unnormalized per-row MSE and DP reduction for both learning targets."""
    import torch

    from verl.utils import tensordict_utils as tu
    from verl.utils.metric.utils import AggregationType, Metric

    prediction = model_output["delta_scalar"].float()
    valid = data["sample_valid_mask"].bool()
    mask = data["target_mask"].bool() & valid[:, None]
    residual = torch.where(mask, prediction - data["target"].float(), torch.zeros_like(prediction))
    squared = residual.square()
    count = mask.sum(-1)
    per_row = squared.sum(-1) / count.clamp_min(1)
    dp_size = int(tu.get(data, "dp_size"))
    loss = per_row.sum() * dp_size / max(int(tu.get(data, "valid_sample_count")), 1)
    metrics = {
        "boundary/loss": Metric(value=loss, aggregation=AggregationType.SUM),
        "boundary/squared_error": Metric(value=squared.sum() * dp_size, aggregation=AggregationType.SUM),
        "boundary/points": Metric(value=count.sum() * dp_size, aggregation=AggregationType.SUM),
        "boundary/rows": Metric(value=valid.float().sum() * dp_size, aggregation=AggregationType.SUM),
    }
    return loss, metrics
