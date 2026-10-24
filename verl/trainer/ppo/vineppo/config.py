# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
"""Algorithm configuration and supported-path checks for native VinePPO."""

import math
from dataclasses import dataclass

from verl.base_config import BaseConfig


@dataclass
class VinePPOConfig(BaseConfig):
    mc_samples: int = 9
    mc_max_new_tokens: int = 1024
    # None uses the rollout engine's context limit, not the remaining response budget.
    mc_context_length: int | None = None
    splitter: str = "math"
    # Variant, not upstream: fold steps holding only a closing \] / $$ / \end{...} into the previous step.
    merge_closing_steps: bool = False
    # Variant, not upstream: merge consecutive official steps until each has at least this many tokens.
    min_step_tokens: int = 0
    # None selects the official task default: all for MATH, root + 25 for GSM8K.
    max_step_for_value_estimation: int | None = None
    concurrency: int = 128
    max_retries: int = 2
    retry_delay: float = 1.0
    unfinished_reward: float = 0.0
    # Upstream grades a verifier error or timeout as incorrect. Errors are counted in the
    # vineppo/*score_error_rate metrics and validation's score_error. None aborts the round instead.
    score_error_reward: float | None = 0.0
    kl_coef: float = 1e-4

    def __post_init__(self):
        for name in ("mc_samples", "mc_max_new_tokens", "concurrency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"VinePPO {name} must be a positive integer")
        for name in ("mc_context_length", "max_step_for_value_estimation"):
            value = getattr(self, name)
            minimum = 0 if name == "max_step_for_value_estimation" else 1
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise ValueError(f"VinePPO {name} must be None or an integer >= {minimum}")
        for name in ("max_retries", "min_step_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"VinePPO {name} must be a nonnegative integer")
        if self.splitter not in ("math", "gsm8k"):
            raise ValueError("VinePPO splitter must be math or gsm8k")
        for name in ("retry_delay", "kl_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"VinePPO {name} must be finite and nonnegative")
        if not math.isfinite(self.unfinished_reward):
            raise ValueError("VinePPO unfinished_reward must be finite")
        if self.score_error_reward is not None and not math.isfinite(self.score_error_reward):
            raise ValueError("VinePPO score_error_reward must be None or finite")

    @property
    def state_cap(self):
        if self.max_step_for_value_estimation is not None:
            return self.max_step_for_value_estimation + 1
        return 26 if self.splitter == "gsm8k" else None


def configure_vineppo(config):
    """Select official loss semantics before reference/critic workers are constructed.

    Called by the standard entry point. Unsupported combinations fail before launching Ray.
    Other algorithms are untouched.
    """
    if config.algorithm.adv_estimator != "vineppo":
        return
    from omegaconf import open_dict

    from verl.utils.config import omega_conf_to_dataclass

    vine = omega_conf_to_dataclass(config.algorithm.vineppo)
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    requirements = {
        "V1 synchronous trainer": config.trainer.use_v1 and config.trainer.v1.trainer_mode == "sync",
        "FSDP/FSDP2 actor": actor.strategy in ("fsdp", "fsdp2"),
        "vLLM rollout": rollout.name == "vllm",
        "fixed actor microbatches": not actor.use_dynamic_bsz,
        "fixed log-prob microbatches": not rollout.log_prob_use_dynamic_bsz,
        "single-turn agent": rollout.agent.default_agent_loop == "single_turn_agent" and not rollout.multi_turn.enable,
        "no critic": config.critic.enable is not True,
        "gamma=1": config.algorithm.gamma == 1.0,
        "task rewards without KL shaping": not config.algorithm.use_kl_in_reward,
        "no preference feedback": not config.algorithm.use_pf_ppo,
        "no group filtering": not config.algorithm.filter_groups.enable,
        "no delta-policy substitution": not config.algorithm.get("delta_policy", {}).get("enabled", False),
        "standard replay sampler": not config.trainer.v1.sampler.custom_sampler.get("path"),
        "no failed-group refill": not config.trainer.v1.sampler.get("sync_refill_failed_groups", False),
        "rule reward": not config.reward.reward_model.enable,
        "error-propagating naive reward manager": config.reward.reward_manager.name == "naive"
        and config.reward.reward_manager.source == "register",
        "no entropy bonus": actor.entropy_coeff == 0,
        "vanilla PPO clipping": actor.policy_loss.loss_mode == "vanilla",
        "no custom loss scaling": actor.loss_scale_factor is None,
        "complete actor checkpoints": {"model", "optimizer", "extra"}.issubset(actor.checkpoint.save_contents)
        and {"model", "optimizer", "extra"}.issubset(actor.checkpoint.load_contents),
        "no distillation": not config.get("distillation", {}).get("enabled", False),
    }
    correction = config.algorithm.get("rollout_correction")
    if correction:
        requirements["no off-policy weighting/rejection"] = not any(
            correction.get(key) for key in ("rollout_is", "rollout_rs", "bypass_mode")
        )
    failed = [name for name, valid in requirements.items() if not valid]
    if failed:
        raise ValueError("VinePPO requires: " + "; ".join(failed))
    with open_dict(config):
        actor.vineppo = True
        actor.use_kl_loss = vine.kl_coef > 0
        actor.kl_loss_coef = vine.kl_coef
        config.critic.enable = False
        config.algorithm.rollout_correction = None
