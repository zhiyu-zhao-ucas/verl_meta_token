# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.vineppo.config import VinePPOConfig
from verl.trainer.ppo.vineppo.core import compute_advantages, microbatch_loss, whiten_microbatch


def test_step_td_terminal_eos_and_capped_tail():
    config = SimpleNamespace(vineppo=VinePPOConfig(mc_samples=2, max_step_for_value_estimation=1))
    args = dict(
        token_level_rewards=torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        response_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
        vineppo_step_ids=torch.tensor([[0, 0, 1, 2, 2, 0]]),
        vineppo_values=torch.tensor([[0.25, 0.5, 1.0]]),
        vineppo_mc_counts=torch.tensor([[2, 2, 0]]),
        vineppo_num_steps=torch.tensor([3]),
        config=config,
    )
    adv, returns = compute_advantages(**args)
    torch.testing.assert_close(adv, torch.tensor([[0.25, 0.25, 0.5, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(returns, args["response_mask"].float())
    # A raw zero tail is still part of official microbatch whitening.
    assert whiten_microbatch(adv, args["response_mask"])[0, 3] != 0
    args["vineppo_mc_counts"][0, 0] = 1
    with pytest.raises(ValueError, match="exactly K"):
        compute_advantages(**args)


def test_whitening_uses_unbiased_valid_token_variance():
    x = torch.tensor([[1.0, 3.0, float("nan")], [5.0, 9.0, 10.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    torch.testing.assert_close(whiten_microbatch(x, mask), torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert torch.equal(whiten_microbatch(torch.ones(2, 2), torch.ones(2, 2)), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="at least two"):
        whiten_microbatch(torch.ones(1, 1), torch.ones(1, 1))


def test_official_gradient_accumulation_and_separate_kl_reduction():
    # Unequal response lengths reveal global-minibatch token weighting errors.
    theta = torch.tensor(0.03, requires_grad=True)
    fixtures = [
        (torch.tensor([[1.0, 2.0, 4.0], [3.0, 0.0, 0.0]]), torch.tensor([[1, 1, 1], [1, 0, 0]])),
        (torch.tensor([[2.0, 5.0, 0.0], [7.0, 9.0, 0.0]]), torch.tensor([[1, 1, 0], [1, 1, 0]])),
    ]
    observed = 0
    for raw, mask in fixtures:
        logp = theta * torch.arange(1.0, 7.0).reshape(2, 3)
        loss, _ = microbatch_loss(
            torch.zeros_like(raw), logp, raw, mask, ref_log_prob=torch.zeros_like(raw), kl_coef=0.1, num_micro_batches=2
        )
        observed = observed + loss
    observed.backward()
    observed_grad = theta.grad.clone()
    theta.grad = None
    oracle = 0
    for raw, mask in fixtures:
        valid = mask.bool()
        flat = raw[valid]
        adv = (raw - flat.mean()) / (flat.var(unbiased=True) + 1e-8).sqrt()
        logp = theta * torch.arange(1.0, 7.0).reshape(2, 3)
        ratio = logp.exp()
        pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(0.8, 1.2))[valid].mean()
        kl = ((torch.exp(-logp) + logp - 1).clamp(0, 10) * mask).sum(-1).mean()
        oracle = oracle + (pg + 0.1 * kl) / 2
    oracle.backward()
    torch.testing.assert_close(observed, oracle)
    torch.testing.assert_close(observed_grad, theta.grad)


def test_kl_updates_when_all_advantages_are_zero_and_clamps():
    logp = torch.tensor([[-1.0, -20.0]], requires_grad=True)
    loss, _ = microbatch_loss(
        torch.zeros_like(logp),
        logp,
        torch.zeros_like(logp),
        torch.ones_like(logp),
        ref_log_prob=torch.zeros_like(logp),
        kl_coef=0.1,
    )
    loss.backward()
    assert logp.grad[0, 0] < 0
    assert logp.grad[0, 1] == 0
    assert loss > 1


def _distributed_whitening(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        raw = torch.tensor([[1.0, 3.0, 100.0]]) if rank == 0 else torch.tensor([[5.0, 0.0, 0.0]])
        mask = torch.tensor([[1, 1, 0]]) if rank == 0 else torch.tensor([[1, 0, 0]])
        actual = whiten_microbatch(raw, mask, dist.group.WORLD)
        expected = torch.tensor([[-1.0, 0.0, 0.0]]) if rank == 0 else torch.tensor([[1.0, 0.0, 0.0]])
        torch.testing.assert_close(actual, expected)
        theta = torch.tensor(0.0, requires_grad=True)
        weights = torch.tensor([[1.0, 2.0, 0.0]]) if rank == 0 else torch.tensor([[3.0, 0.0, 0.0]])
        loss, _ = microbatch_loss(
            torch.zeros_like(raw),
            theta * weights,
            raw,
            mask,
            kl_coef=0,
            num_micro_batches=2,
            dp_group=dist.group.WORLD,
        )
        loss.backward()
        dist.all_reduce(theta.grad)
        theta.grad /= 2  # FSDP's DP gradient averaging.
        torch.testing.assert_close(theta.grad, torch.tensor(-0.625))
        empty_on_one_rank = mask if rank == 0 else torch.zeros_like(mask)
        with pytest.raises(ValueError, match="empty response"):
            microbatch_loss(
                torch.zeros_like(raw),
                theta * weights,
                raw,
                empty_on_one_rank,
                kl_coef=0,
                dp_group=dist.group.WORLD,
            )
    finally:
        dist.destroy_process_group()


def test_whitening_across_two_dp_ranks(tmp_path):
    torch.multiprocessing.spawn(_distributed_whitening, args=(f"file://{tmp_path}/dist",), nprocs=2, join=True)
