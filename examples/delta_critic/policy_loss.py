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
"""Delta state-only PG and KL losses for offline and V1 actor updates."""

from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch

from .policy_config import DeltaPolicyConfig


def _config(value: DeltaPolicyConfig | Mapping[str, Any]) -> DeltaPolicyConfig:
    return value if isinstance(value, DeltaPolicyConfig) else DeltaPolicyConfig(**dict(value))


def per_sample_policy_losses(
    current_logprobs: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    config: DeltaPolicyConfig | Mapping[str, Any],
    *,
    reference_logprobs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-row PG/KL losses in FP32, before row weights/global reduction."""
    config = _config(config)
    if not isinstance(current_logprobs, torch.Tensor) or current_logprobs.is_nested:
        raise ValueError("current_logprobs must be a dense response-aligned [batch, tokens] tensor")
    current = current_logprobs.float()
    old = batch["old_log_probs"].to(device=current.device, dtype=torch.float32)
    advantages = batch.get("advantages", batch.get("state_advantages"))
    if advantages is None:
        raise ValueError("policy batch requires advantages")
    advantages = advantages.to(device=current.device, dtype=torch.float32)
    policy_mask = batch["policy_loss_mask"].to(device=current.device, dtype=torch.float32)
    if current.ndim != 2 or any(x.shape != current.shape for x in (old, advantages, policy_mask)):
        raise ValueError("current logprobs, old logprobs, advantages, and policy_loss_mask must match [batch, tokens]")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all() or not torch.isfinite(advantages).all():
        raise ValueError("current logprobs, old logprobs, and advantages must be finite")
    if not torch.isfinite(policy_mask).all() or ((policy_mask != 0) & (policy_mask != 1)).any():
        raise ValueError("policy_loss_mask must be finite and binary")

    log_ratio = (current - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
    pg_terms = -torch.minimum(unclipped, clipped)
    pg_denominator = policy_mask.sum(dim=-1).clamp_min(1.0)
    pg_per_row = (pg_terms * policy_mask).sum(dim=-1) / pg_denominator

    kl_mask = batch.get("kl_mask")
    if kl_mask is None:
        if config.kl_mask_scope == "response":
            kl_mask = batch.get("response_mask")
            if kl_mask is None:
                raise ValueError("kl_mask_scope='response' requires response_mask when kl_mask is absent")
        else:
            kl_mask = policy_mask
    if not isinstance(kl_mask, torch.Tensor) or kl_mask.is_nested:
        raise ValueError("kl_mask must be a dense response-aligned tensor")
    kl_mask = kl_mask.to(device=current.device, dtype=torch.float32)
    if kl_mask.shape != current.shape:
        raise ValueError("kl_mask must match current_logprobs")
    if not torch.isfinite(kl_mask).all() or ((kl_mask != 0) & (kl_mask != 1)).any():
        raise ValueError("kl_mask must be finite and binary")

    if config.kl_coef > 0.0:
        if config.kl_reference == "behavior":
            reference = old
        else:
            reference = reference_logprobs
            if reference is None:
                reference = batch.get("ref_log_prob")
            if reference is None:
                raise ValueError("reference KL requires fixed ref_log_prob values")
            reference = reference.to(device=current.device, dtype=torch.float32)
        if reference.shape != current.shape or not torch.isfinite(reference).all():
            raise ValueError("KL reference logprobs must be finite and match current_logprobs")
        delta = (reference - current).clamp(-20.0, 20.0)
        kl_terms = delta.exp() - delta - 1.0
        if config.kl_estimator == "low_var_kl":
            # Match PPO's low_var_kl/k3 path: clamp the final estimate as well.
            kl_terms = kl_terms.clamp(-10.0, 10.0)
        kl_per_row = (kl_terms * kl_mask).sum(dim=-1) / kl_mask.sum(dim=-1).clamp_min(1.0)
    else:
        kl_per_row = current.sum(dim=-1) * 0.0

    valid = batch.get("sample_valid_mask")
    if valid is None:
        valid = torch.ones(current.shape[0], dtype=torch.float32, device=current.device)
    else:
        valid = valid.to(device=current.device, dtype=torch.float32)
    if valid.shape != (current.shape[0],) or not torch.isfinite(valid).all():
        raise ValueError("sample_valid_mask must be finite with shape [batch]")
    if ((valid != 0) & (valid != 1)).any():
        raise ValueError("sample_valid_mask must be binary")

    row_weight = batch.get("row_weight")
    if row_weight is None:
        row_weight = torch.ones_like(valid)
    else:
        row_weight = row_weight.to(device=current.device, dtype=torch.float32)
    if row_weight.shape != (current.shape[0],) or not torch.isfinite(row_weight).all() or (row_weight < 0).any():
        raise ValueError("row_weight must be finite, nonnegative, and have shape [batch]")
    row_weight = row_weight * valid
    total_per_row = pg_per_row + float(config.kl_coef) * kl_per_row
    return {
        "loss": total_per_row,
        "pg_loss": pg_per_row,
        "kl_loss": kl_per_row,
        "row_weight": row_weight,
        "policy_token_count": policy_mask.sum(dim=-1) * valid,
        "kl_token_count": kl_mask.sum(dim=-1) * valid,
    }


def _check_nested_offsets(data, *, expected_key: str, fields: list[str]) -> None:
    """Reject jagged policy arrays whose row boundaries differ from their tokens."""
    reference = data.get(expected_key)
    if not isinstance(reference, torch.Tensor) or not reference.is_nested:
        return
    expected_offsets = reference.offsets()
    for name in fields:
        value = data.get(name)
        if isinstance(value, torch.Tensor) and value.is_nested:
            offsets = value.offsets().to(device=expected_offsets.device)
            if offsets.shape != expected_offsets.shape or not torch.equal(offsets, expected_offsets):
                raise ValueError(f"Nested {name} boundaries do not align with {expected_key}")


def _check_full_sequence_offsets(current, data) -> None:
    """Check nested actor outputs against each input prompt+response before slicing."""
    if not current.is_nested:
        return
    prompts = data.get("prompts")
    responses = data.get("responses")
    if not isinstance(prompts, torch.Tensor) or not prompts.is_nested:
        raise ValueError("Nested actor logprobs require nested prompt and response tensors")
    if not isinstance(responses, torch.Tensor) or not responses.is_nested:
        raise ValueError("Nested actor logprobs require nested prompt and response tensors")
    prompt_lengths = prompts.offsets().diff()
    response_lengths = responses.offsets().diff().to(device=prompt_lengths.device)
    if prompt_lengths.shape != response_lengths.shape:
        raise ValueError("Nested prompt and response batches have different row counts")
    sequence_lengths = prompt_lengths + response_lengths
    expected_offsets = torch.cat([sequence_lengths.new_zeros(1), sequence_lengths.cumsum(dim=0)])
    output_offsets = current.offsets().to(device=expected_offsets.device)
    if output_offsets.shape != expected_offsets.shape or not torch.equal(output_offsets, expected_offsets):
        raise ValueError("Nested actor logprob boundaries do not align with prompt+response sequences")


def delta_policy_loss(model_output, data, dp_group=None):
    """V1 actor callback using a real-row mean and explicit DP compensation.

    ``global_valid_sample_weight`` is the sum of real row weights across the
    logical update on every rank. The local loss is multiplied by DP size because
    FSDP's data-parallel gradient reduction averages rank gradients.
    """
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric.utils import AggregationType, Metric

    config = _config(tu.get(data, "delta_policy_config"))
    current = model_output["log_probs"]
    if not isinstance(current, torch.Tensor):
        raise ValueError("Actor model_output['log_probs'] must be a tensor")
    if current.is_nested or current.ndim != 2:
        # V1 no-padding actor outputs contain full-sequence logprobs. Slice the
        # response first; per-row loss math below only accepts dense [B, R] data.
        if "prompts" not in data.keys() or "responses" not in data.keys():
            raise ValueError("Actor logprobs are not dense and V1 prompt/response fields are unavailable")
        _check_full_sequence_offsets(current, data)
        from verl.workers.utils.padding import no_padding_2_padding

        current = no_padding_2_padding(current, data)
    advantage_key = "advantages" if "advantages" in data.keys() else "state_advantages"
    if advantage_key not in data.keys():
        raise ValueError("Policy batch requires advantages or state_advantages")
    fields = ["old_log_probs", advantage_key, "policy_loss_mask"]
    for optional in ("kl_mask", "response_mask", "ref_log_prob", "sample_valid_mask", "row_weight"):
        if optional in data.keys():
            fields.append(optional)
    _check_nested_offsets(data, expected_key="responses", fields=fields)
    # Convert response-aligned policy fields only after extracting the actor's
    # response logprobs so nested full-sequence offsets cannot be misaligned.
    selected = data.select(*fields).to_padded_tensor()
    loss_batch = {key: selected[key] for key in fields}
    if advantage_key != "advantages":
        loss_batch["advantages"] = loss_batch[advantage_key]
    if current.shape != loss_batch["policy_loss_mask"].shape:
        raise ValueError(
            f"Actor logprobs {tuple(current.shape)} do not align with policy mask "
            f"{tuple(loss_batch['policy_loss_mask'].shape)}"
        )
    losses = per_sample_policy_losses(current, loss_batch, config)
    raw_dp_size = tu.get(data, "dp_size", 1)
    dp_size = int(raw_dp_size)
    if dp_size < 1 or dp_size != raw_dp_size:
        raise ValueError("dp_size must be a positive integer")
    denominator = tu.get(data, "global_valid_sample_weight", None)
    if denominator is None:
        raise ValueError("Policy loss requires global_valid_sample_weight for the full logical update")
    if isinstance(denominator, torch.Tensor):
        if denominator.numel() != 1:
            raise ValueError("global_valid_sample_weight must be one scalar for the full update")
        denominator = denominator.item()
    denominator = float(denominator)
    if denominator < 0.0 or not isfinite(denominator):
        raise ValueError("global_valid_sample_weight must be finite and nonnegative")
    if denominator == 0.0:
        # An all-padding local microbatch may be part of an otherwise valid update.
        if losses["row_weight"].sum().item() != 0.0:
            raise ValueError("A nonempty local batch cannot have zero global row weight")
        scale = 0.0
    else:
        local_weight = float(losses["row_weight"].sum().item())
        if local_weight > denominator + max(1e-6, abs(denominator) * 1e-6):
            raise ValueError("Local row weight exceeds global_valid_sample_weight")
        scale = dp_size / denominator
    row_weights = losses["row_weight"]
    total = (losses["loss"] * row_weights).sum() * scale
    pg = (losses["pg_loss"] * row_weights).sum() * scale
    kl = (losses["kl_loss"] * row_weights).sum() * scale
    metrics = {
        "actor/delta_policy_loss": Metric(value=total.detach(), aggregation=AggregationType.SUM),
        "actor/delta_pg_loss": Metric(value=pg.detach(), aggregation=AggregationType.SUM),
        "actor/delta_kl_loss": Metric(value=kl.detach(), aggregation=AggregationType.SUM),
        "actor/delta_rows": Metric(value=row_weights.sum().detach() * dp_size, aggregation=AggregationType.SUM),
        "actor/delta_pg_tokens": Metric(
            value=(losses["policy_token_count"] * row_weights).sum().detach() * dp_size,
            aggregation=AggregationType.SUM,
        ),
        "actor/delta_kl_tokens": Metric(
            value=(losses["kl_token_count"] * row_weights).sum().detach() * dp_size,
            aggregation=AggregationType.SUM,
        ),
    }
    # Observe the forward pass that produced this minibatch's loss. These are
    # not post-update measurements: a single first minibatch has ratio == 1.
    # Match the loss's row mean, global denominator and DP reduction exactly;
    # rows without PG tokens contribute zero (see active_pg_row_fraction).
    with torch.no_grad():
        policy_mask = loss_batch["policy_loss_mask"].to(device=current.device, dtype=torch.float32)
        advantages = loss_batch["advantages"].to(device=current.device, dtype=torch.float32)
        raw_log_ratio = current.float() - loss_batch["old_log_probs"].to(current.device).float()
        log_ratio = raw_log_ratio.clamp(-20.0, 20.0)
        ratio = log_ratio.exp()
        tokens = policy_mask.sum(dim=-1)
        clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
        observations = {
            "ratio_row_mean": ratio,
            "ratio_abs_deviation_row_mean": (ratio - 1.0).abs(),
            "log_ratio_row_mean": raw_log_ratio,
            "abs_log_ratio_row_mean": raw_log_ratio.abs(),
            "old_kl_row_mean": torch.expm1(-log_ratio) + log_ratio,
            "clip_fraction_row_mean": (ratio * advantages > clipped * advantages).float(),
            "ratio_outside_clip_fraction_row_mean": (ratio != clipped).float(),
            "log_ratio_clamp_fraction_row_mean": (raw_log_ratio != log_ratio).float(),
        }
        for name, values in observations.items():
            row_values = (values * policy_mask).sum(dim=-1) / tokens.clamp_min(1.0)
            metrics[f"actor/delta_forward_{name}"] = Metric(
                value=(row_values * row_weights).sum() * scale, aggregation=AggregationType.SUM
            )
        metrics["actor/delta_forward_active_pg_row_fraction"] = Metric(
            value=((tokens > 0).float() * row_weights).sum() * scale,
            aggregation=AggregationType.SUM,
        )
    return total, metrics
