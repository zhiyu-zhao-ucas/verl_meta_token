# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
"""Pure VinePPO estimators and the official microbatch loss."""

import torch
import torch.distributed as dist


def compute_advantages(
    token_level_rewards,
    response_mask,
    vineppo_step_ids,
    vineppo_values,
    vineppo_mc_counts,
    vineppo_num_steps,
    config,
    **kwargs,
):
    """Broadcast step TD residuals. Whitening is deliberately deferred to the actor."""
    vine = config.vineppo
    k = vine.mc_samples
    cap = vine.max_step_for_value_estimation
    cap = cap + 1 if cap is not None else (26 if vine.splitter == "gsm8k" else None)
    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)
    for row in range(len(advantages)):
        mask = response_mask[row].bool()
        n = int(vineppo_num_steps[row])
        if n < 1 or n > vineppo_values.shape[1] or not mask.any():
            raise ValueError("VinePPO missing steps, values, or response tokens")
        ids = vineppo_step_ids[row, mask].long()
        if (ids < 0).any() or (ids >= n).any() or (ids[1:] < ids[:-1]).any():
            raise ValueError("VinePPO invalid token-to-step mapping")
        values = vineppo_values[row, :n]
        counts = vineppo_mc_counts[row, :n]
        sampled = min(n, cap) if cap is not None else n
        if not torch.all(counts[:sampled] == k) or not torch.all(counts[sampled:] == 0):
            raise ValueError("VinePPO requires exactly K MC samples for every selected state")
        reward_tokens = token_level_rewards[row, mask]
        if not torch.isfinite(reward_tokens).all() or not torch.all(reward_tokens[:-1] == 0):
            raise ValueError("VinePPO requires finite terminal-only task rewards")
        reward = reward_tokens[-1]
        if not torch.isfinite(values).all() or not torch.all(values[sampled:] == reward):
            raise ValueError("VinePPO invalid MC value or tail-return fill")
        next_values = torch.cat((values[1:], values.new_zeros(1)))
        step_rewards = torch.zeros_like(values)
        step_rewards[-1] = reward
        step_adv = step_rewards + next_values - values
        advantages[row, mask] = step_adv[ids]
        returns[row, mask] = reward
    return advantages, returns


@torch.no_grad()
def whiten_microbatch(advantages, mask, dp_group=None):
    """Two-pass, token-weighted, unbiased variance across the explicit DP group."""
    x = advantages.float()
    mask = mask.bool()
    # A missing group means single-process computation, never implicit WORLD/SP reduction.
    distributed = dp_group is not None and dist.is_initialized()
    stats = torch.stack((torch.where(mask, x, 0).sum(), mask.sum().float()))
    if distributed:
        dist.all_reduce(stats, group=dp_group)
    count = stats[1]
    if count.item() < 2 or not torch.isfinite(stats).all():
        raise ValueError("VinePPO whitening requires at least two finite valid tokens across DP")
    mean = stats[0] / count
    squared = torch.where(mask, (x - mean).square(), 0).sum()
    if distributed:
        dist.all_reduce(squared, group=dp_group)
    variance = squared / (count - 1)
    if not torch.isfinite(variance):
        raise ValueError("Nonfinite VinePPO advantage variance")
    return torch.where(mask, (x - mean) * torch.rsqrt(variance + 1e-8), 0)


def microbatch_loss(
    old_log_prob,
    log_prob,
    raw_advantages,
    response_mask,
    *,
    ref_log_prob=None,
    clip_low=0.2,
    clip_high=0.2,
    kl_coef=1e-4,
    num_micro_batches=1,
    dp_group=None,
):
    """Official local token-mean PG and sequence-sum/batch-mean KL.

    FSDP averages gradients across DP. Divide by accumulation steps here (the engine
    otherwise sums microbatch gradients). Do not apply minibatch token weighting again.
    """
    mask = response_mask.bool()
    invalid = (~mask.any(dim=-1)).any().to(dtype=torch.int32)
    if dp_group is not None and dist.is_initialized():
        # Every rank must fail before FSDP backward if any rank has an empty row.
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=dp_group)
    if invalid.item():
        raise ValueError("VinePPO actor microbatch contains an empty response")
    advantages = whiten_microbatch(raw_advantages, mask, dp_group)
    if num_micro_batches < 1:
        raise ValueError("VinePPO requires an explicit positive accumulation count")
    log_ratio = torch.where(mask, log_prob - old_log_prob, 0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    pg = torch.maximum(-ratio * advantages, -clipped * advantages)[mask].mean()
    kl = log_prob.new_zeros(())
    if kl_coef:
        if ref_log_prob is None:
            raise ValueError("VinePPO reference KL requires reference log-probabilities")
        d = torch.where(mask, ref_log_prob - log_prob, 0)
        kl = torch.where(mask, (torch.exp(d) - d - 1).clamp(0, 10), 0).sum(-1).mean()
    loss = (pg + kl_coef * kl) / num_micro_batches
    return loss, {
        "actor/pg_loss": pg.detach() / num_micro_batches,
        "actor/kl_loss": kl.detach() / num_micro_batches,
        "actor/pg_clipfrac": (-clipped * advantages > -ratio * advantages)[mask].float().mean().detach(),
        "actor/vineppo_raw_adv_mean": raw_advantages[mask].float().mean().detach(),
        "actor/vineppo_whitened_adv_mean": advantages[mask].mean(),
        "actor/vineppo_whitened_adv_second_moment": advantages[mask].square().mean(),
    }
