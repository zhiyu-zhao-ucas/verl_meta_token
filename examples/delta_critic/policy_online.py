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
"""Online delta-policy data preparation for synchronous V1 PPO."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import isfinite, log
from pathlib import Path
from typing import Any

from .contracts import DeltaExample, NormalizationStats, Rollout, SelectedState
from .policy_config import DeltaPolicyConfig, delta_policy_config_from_mapping
from .state_selection import select_spaced_states


def online_policy_config(values: Mapping[str, Any]) -> DeltaPolicyConfig:
    """Parse the shared policy loss config from ``algorithm.delta_policy``."""
    return delta_policy_config_from_mapping(values, mode="online_ppo")


def validate_online_v1_settings(config: Mapping[str, Any]) -> DeltaPolicyConfig:
    """Fail early for online options that cannot preserve the PPO data contract."""
    policy = config.get("algorithm", {}).get("delta_policy", {})
    if not policy or not policy.get("enabled", False):
        raise ValueError("algorithm.delta_policy.enabled must be true")
    loss_config = online_policy_config(policy)
    from .policy_mc import online_mc_config

    online_mc_config(policy, config["actor_rollout_ref"]["rollout"], 0)
    stats_mode = policy.get("advantage_stats_mode", "per_rollout")
    if stats_mode not in {"per_rollout", "initial_rollout"}:
        raise ValueError("advantage_stats_mode must be per_rollout or initial_rollout")
    if stats_mode == "initial_rollout" and loss_config.advantage_normalization != "standardize":
        raise ValueError("initial_rollout statistics require advantage_normalization=standardize")
    if not policy.get("critic_artifact"):
        raise ValueError("algorithm.delta_policy.critic_artifact is required")
    if not policy.get("reference_policy_id"):
        raise ValueError("algorithm.delta_policy.reference_policy_id is required")
    if config.get("trainer", {}).get("v1", {}).get("trainer_mode") != "sync":
        raise ValueError("Online delta policy currently requires trainer.v1.trainer_mode=sync")
    if config.get("trainer", {}).get("v1", {}).get("sync", {}).get("parameter_sync_step", 1) != 1:
        raise ValueError("Online delta policy requires parameter_sync_step=1 so statistics use one full round")
    if config.get("critic", {}).get("enable") is not False:
        raise ValueError("Online delta policy requires critic.enable=false; its delta scorer is frozen")
    if config.get("algorithm", {}).get("use_kl_in_reward", False):
        raise ValueError("Online delta policy does not support KL-in-reward")
    rollout_correction = config.get("algorithm", {}).get("rollout_correction")
    if rollout_correction and (
        rollout_correction.get("rollout_is") is not None
        or rollout_correction.get("rollout_rs") is not None
        or rollout_correction.get("bypass_mode", False)
    ):
        raise ValueError("Online delta policy does not support rollout correction")
    if not config.get("actor_rollout_ref", {}).get("actor", {}).get("use_kl_loss", False):
        raise ValueError("Online delta policy needs actor_rollout_ref.actor.use_kl_loss=true for the fixed reference")
    if float(config["actor_rollout_ref"]["actor"].get("entropy_coeff", 0.0)) != 0.0:
        raise ValueError("Online delta policy does not support entropy regularization")
    if config.get("distillation", {}).get("enable", False):
        raise ValueError("Online delta policy does not support policy distillation")
    if config.get("actor_rollout_ref", {}).get("rollout", {}).get("name") != "vllm":
        raise ValueError("Uncertainty state selection currently requires the V1 vLLM delta_top_logprobs path")
    actor = config.get("actor_rollout_ref", {}).get("actor", {})
    if actor.get("strategy") != "fsdp2":
        raise ValueError("Online delta policy currently supports the V1 FSDP2 actor backend")
    rollout = config.get("actor_rollout_ref", {}).get("rollout", {})
    agent = rollout.get("agent", {})
    if agent.get("default_agent_loop", "single_turn_agent") != "single_turn_agent":
        raise ValueError("Online uncertainty selection currently requires single_turn_agent")
    selection = policy.get("selection", {})
    if selection.get("strategy", "uncertainty") != "uncertainty":
        raise ValueError("Online V1 currently supports source uncertainty state selection only")
    if int(selection.get("states_per_response", 64)) < 1:
        raise ValueError("states_per_response must be positive")
    if int(selection.get("top_k_logprobs", 20)) < 1:
        raise ValueError("top_k_logprobs must be positive")
    if int(selection.get("max_candidates", 5)) > int(selection.get("top_k_logprobs", 20)):
        raise ValueError("selection.max_candidates cannot exceed selection.top_k_logprobs")
    if loss_config.mode != "online_ppo":
        raise ValueError("Online V1 must use mode=online_ppo")
    if loss_config.behavior_logprob_source != "actor_snapshot":
        raise ValueError(
            "Synchronous V1 recomputes old logprobs from the pre-update actor snapshot; "
            "set behavior_logprob_source=actor_snapshot to record that provenance"
        )
    if loss_config.advantage_source != "critic":
        raise ValueError("Online V1 delta policy currently requires advantage_source=critic")
    expansion = policy.get("expansion", {})
    if expansion.get("enabled", False):
        prompts = expansion.get("prompts_per_step")
        states = expansion.get("states_per_step")
        continuations = policy.get("mc", {}).get("continuations_per_state")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (prompts, states, continuations)):
            raise ValueError("Expansion prompt, state, and continuation budgets must be positive integers")
        if states > prompts:
            raise ValueError("Expansion requires at most one state per prompt")
        if config["data"]["train_batch_size"] != prompts + states * continuations:
            raise ValueError("Expansion train_batch_size must exactly equal originals plus continuations")
        if config["data"].get("gen_batch_size") != prompts:
            raise ValueError("Expansion gen_batch_size must equal prompts_per_step")
        if rollout.get("n", 1) != 1:
            raise ValueError("Expansion requires rollout.n=1")
        if policy.get("rollout_mode") != "selected_prefix_mc":
            raise ValueError("Expansion requires selected_prefix_mc")
        if policy.get("mc", {}).get("max_response_total_tokens") != config["data"]["max_response_length"]:
            raise ValueError("Expansion MC response cap must match original response cap")
        reserve = policy.get("mc", {}).get("min_continuation_room", 1)
        if (
            isinstance(reserve, bool)
            or not isinstance(reserve, int)
            or not 1 <= reserve <= config["data"]["max_response_length"]
        ):
            raise ValueError("Expansion min_continuation_room must be within response cap")
        short_response = expansion.get("short_response_tokens")
        if short_response is not None and (
            isinstance(short_response, bool) or not isinstance(short_response, int) or short_response < 1
        ):
            raise ValueError("Expansion short_response_tokens must be a positive integer or absent")
    if loss_config.kl_mask_scope != "policy":
        raise ValueError("Online V1 currently requires KL to use the selected policy-token mask")
    return loss_config


def initial_advantage_stats(
    path, *, reference_policy_id, critic_weights_sha256, label_mode, stats=None, advantage_contract=None
):
    """Load or atomically persist the immutable first-rollout statistics.

    Kept beside the run's checkpoints so recovery never silently recalibrates
    against a later policy. Bind the file to the reference and critic identities.
    """
    path = Path(path)
    identity = {
        "format_version": 1,
        "reference_policy_id": reference_policy_id,
        "critic_weights_sha256": critic_weights_sha256,
        "label_mode": label_mode,
        "recipe": "raw_delta_segment_broadcast_before_policy_mask_population_std",
    }
    if advantage_contract is not None:
        identity["advantage_contract"] = advantage_contract
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Initial advantage statistics belong to a different policy/critic contract")
        restored = NormalizationStats(**saved["stats"])
        if restored.population != "state_advantages":
            raise ValueError("Initial advantage statistics must describe state_advantages")
        if stats is not None and restored != stats:
            raise ValueError("Cannot overwrite fixed initial advantage statistics")
        return restored
    if stats is None:
        return None
    if stats.population != "state_advantages":
        raise ValueError("Initial advantage statistics must describe state_advantages")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({**identity, "stats": asdict(stats)}, indent=2) + "\n")
    temporary.replace(path)
    return stats


def validate_online_scorer_contract(scorer, *, tokenizer_sha256: str, label_mode: str) -> None:
    """Reject critic artifacts that do not score the actor's token/label contract."""
    metadata = getattr(scorer, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Frozen delta scorer is missing artifact metadata")
    if metadata.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("Online actor and frozen delta scorer tokenizer fingerprints differ")
    scorer_config = getattr(scorer, "checkpoint_config", None)
    scorer_label_mode = getattr(scorer_config, "delta_label_mode", None)
    if scorer_label_mode != label_mode:
        raise ValueError(
            f"Online delta label_mode={label_mode!r} does not match frozen critic label_mode={scorer_label_mode!r}"
        )


def select_uncertainty_indices(
    candidates_by_token: Sequence[Sequence[Mapping[str, Any]]],
    policy_token_mask: Sequence[float],
    *,
    states_per_response: int = 64,
    min_token_gap: int = 32,
    entropy_weight: float = 0.0,
    low_top1_weight: float = 1.0,
    final_window_tokens: int = 64,
    final_window_weight: float = 0.0,
    max_candidates: int = 5,
) -> list[int]:
    """Apply the value_model uncertainty score to generated response tokens."""
    if len(candidates_by_token) != len(policy_token_mask):
        raise ValueError(
            "V1 top-logprob rows must align with response tokens after continuous-token merging; "
            "cannot infer a token index mapping"
        )
    if states_per_response < 0 or min_token_gap < 0 or max_candidates < 1:
        raise ValueError("invalid online state-selection budget")
    if final_window_tokens < 0:
        raise ValueError("final_window_tokens must be nonnegative")
    length = len(candidates_by_token)
    scored: list[tuple[float, int]] = []
    for index, (candidates, allowed) in enumerate(zip(candidates_by_token, policy_token_mask, strict=True)):
        if not isfinite(float(allowed)) or float(allowed) not in (0.0, 1.0):
            raise ValueError("policy_token_mask must contain finite binary values")
        if not allowed:
            continue
        if not candidates:
            continue
        probs = []
        for candidate in list(candidates)[:max_candidates]:
            probability = float(candidate["prob"])
            if not isfinite(probability) or probability < 0.0:
                raise ValueError("top-logprob candidate probabilities must be finite and nonnegative")
            probs.append(probability)
        if not probs:
            continue
        entropy = -sum(prob * log(max(prob, 1e-12)) for prob in probs)
        top1_prob = max(probs)
        score = (
            float(entropy_weight) * entropy
            + float(low_top1_weight) * (1.0 - top1_prob)
            + float(index >= max(0, length - final_window_tokens)) * float(final_window_weight)
        )
        if not isfinite(score):
            raise ValueError("uncertainty selection score must be finite")
        scored.append((score, index))
    return sorted(select_spaced_states(scored, states_per_response, min_token_gap))


def delta_examples_from_rows(
    rollout_ids: Sequence[str],
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    policy_masks: Sequence[Sequence[float]],
    selected_indices: Sequence[Sequence[int]],
    *,
    actor_version: str | None = None,
) -> list[DeltaExample]:
    """Build validated response-indexed delta contracts from unpadded V1 rows."""
    lengths = {len(rollout_ids), len(prompts), len(responses), len(policy_masks), len(selected_indices)}
    if len(lengths) != 1:
        raise ValueError("Online V1 row arrays must have the same batch length")
    examples = []
    for row_id, prompt, response, mask, selected in zip(
        rollout_ids, prompts, responses, policy_masks, selected_indices, strict=True
    ):
        prompt_ids = tuple(int(token) for token in prompt)
        response_ids = tuple(int(token) for token in response)
        policy_mask = tuple(float(value) for value in mask)
        metadata = {"actor_version": actor_version} if actor_version is not None else {}
        rollout = Rollout(
            rollout_id=str(row_id),
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            terminal_reward=0.0,
            policy_token_mask=policy_mask,
            metadata=metadata,
        )
        states = tuple(
            SelectedState(rollout_id=str(row_id), token_index=int(index), v_prefix=0.0)
            for index in sorted(int(index) for index in selected)
        )
        examples.append(DeltaExample(rollout=rollout, states=states))
    return examples


def create_frozen_delta_scorer(policy: Mapping[str, Any]):
    """Load the configured scalar artifact for real, frozen prefix inference."""
    from .score import FrozenDeltaWorker

    worker_class = FrozenDeltaWorker
    extra = {}
    if policy.get("rollout_mode", "critic_prediction") == "selected_prefix_mc" and not policy.get("expansion", {}).get(
        "enabled", False
    ):
        from .online_critic import OnlineDeltaWorker

        worker_class = OnlineDeltaWorker
        extra["update_config"] = policy.get("critic_update", {})
    return worker_class(
        policy["critic_artifact"],
        device=policy.get("scorer_device", "cpu"),
        microbatch=int(policy.get("scorer_microbatch", 1)),
        scoring_max_length=policy.get("scoring_max_length"),
        **extra,
    )
