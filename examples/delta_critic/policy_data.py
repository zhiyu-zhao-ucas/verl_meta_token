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
"""Prepare and align delta advantages, behavior logprobs, masks, and policy inputs."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .contracts import DeltaConfig, DeltaExample, NormalizationStats, TokenVector
from .policy_config import DeltaPolicyConfig
from .target_ops import clip_advantages, fit_normalization, normalize, state_advantages


def _finite_sequence(values: Sequence[float], *, name: str, expected: int) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(f"{name} length mismatch: {len(values)} vs {expected}")
    result = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _binary_sequence(values: Sequence[float], *, name: str, expected: int) -> tuple[float, ...]:
    result = _finite_sequence(values, name=name, expected=expected)
    if any(value not in (0.0, 1.0) for value in result):
        raise ValueError(f"{name} must contain only binary values")
    return result


def _behavior_values(
    example: DeltaExample,
    behavior_logprobs_by_id: Mapping[str, Sequence[float]] | None,
    source: str,
) -> tuple[float, ...]:
    rollout = example.rollout
    if behavior_logprobs_by_id is not None and rollout.rollout_id in behavior_logprobs_by_id:
        values = behavior_logprobs_by_id[rollout.rollout_id]
    else:
        if source in {"precomputed", "actor_snapshot"}:
            raise ValueError(f"Missing {source} behavior logprobs for rollout_id={rollout.rollout_id}")
        metadata = rollout.metadata
        if source == "stored" and metadata.get("tokens") is not None:
            tokens = metadata["tokens"]
            expected = len(rollout.response_token_ids)
            if len(tokens) != expected:
                raise ValueError(
                    f"Stored token metadata length mismatch for rollout_id={rollout.rollout_id}: "
                    f"{len(tokens)} vs {expected} response tokens"
                )
            values = []
            for index, (token_id, token_row) in enumerate(zip(rollout.response_token_ids, tokens, strict=True)):
                stored_index = int(token_row.get("token_index", index))
                if stored_index != index:
                    raise ValueError(
                        f"Stored token_index mismatch for rollout_id={rollout.rollout_id}: "
                        f"expected {index}, got {stored_index}"
                    )
                stored_token_id = token_row.get("token_id")
                if stored_token_id is not None and int(stored_token_id) != token_id:
                    raise ValueError(
                        f"Stored token_id mismatch for rollout_id={rollout.rollout_id} at token_index={index}"
                    )
                sampled_logprob = token_row.get("sampled_logprob")
                if sampled_logprob is None:
                    raise ValueError(
                        f"Missing sampled_logprob for rollout_id={rollout.rollout_id} at token_index={index}"
                    )
                values.append(sampled_logprob)
        elif source == "stored":
            values = metadata.get("behavior_logprobs")
        else:
            values = metadata.get("rollout_log_probs")
    if values is None:
        raise ValueError(f"Missing {source} behavior logprobs for rollout_id={rollout.rollout_id}")
    return _finite_sequence(
        values, name=f"behavior logprobs for {rollout.rollout_id}", expected=len(rollout.response_token_ids)
    )


def _reference_values(
    rollout_id: str,
    reference_logprobs_by_id: Mapping[str, Sequence[float]] | None,
    expected: int,
) -> tuple[float, ...] | None:
    if reference_logprobs_by_id is None or rollout_id not in reference_logprobs_by_id:
        return None
    return _finite_sequence(
        reference_logprobs_by_id[rollout_id], name=f"reference logprobs for {rollout_id}", expected=expected
    )


def prepare_policy_rows(
    examples: Sequence[DeltaExample],
    raw_predictions_by_id: Mapping[str, Mapping[int, float]] | None,
    behavior_logprobs_by_id: Mapping[str, Sequence[float]] | None,
    config: DeltaPolicyConfig,
    *,
    reference_logprobs_by_id: Mapping[str, Sequence[float]] | None = None,
    reference_policy_id: str | None = None,
    row_weights_by_id: Mapping[str, float] | None = None,
    stats: NormalizationStats | None = None,
    fit_stats: bool = False,
    reward_deltas_by_id: Mapping[str, float] | None = None,
    outcome_advantages_by_id: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], NormalizationStats | None]:
    """Build response-aligned rows and optionally fit training-set statistics.

    The caller passes only the training split when ``fit_stats=True``. Statistics
    are fitted on broadcast state tokens before policy-mask intersection, padding,
    or a possible legacy tail window. Evaluation rows should receive those saved
    statistics with ``stats=...`` and ``fit_stats=False``.

    ``reward_deltas_by_id`` is the legacy raw-reward replacement API. Its values
    still join the critic fit population and normalization. New short-outcome
    callers should pass ``outcome_advantages_by_id`` instead: those calibrated
    advantages cover the whole response, bypass critic normalization, and are
    excluded from fitted critic statistics.
    """
    if not examples:
        raise ValueError("At least one example is required")
    rollout_ids = [example.rollout.rollout_id for example in examples]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise ValueError("Policy examples must have unique rollout IDs")
    if fit_stats and stats is not None:
        raise ValueError("Pass either existing stats or fit_stats=True, not both")
    if reward_deltas_by_id is not None and outcome_advantages_by_id is not None:
        raise ValueError("Do not mix legacy raw reward deltas with calibrated outcome advantages")
    if config.advantage_source == "mc_local_oracle" and config.label_mode != "paired_next_state":
        raise ValueError("mc_local_oracle requires label_mode='paired_next_state'")
    if config.mode == "online_ppo" and config.kl_coef > 0.0 and not reference_policy_id:
        raise ValueError("Online reference KL requires the immutable initial reference_policy_id")

    rollout_ids_set = set(rollout_ids)
    unknown_outcome_ids = set(outcome_advantages_by_id or {}) - rollout_ids_set
    if unknown_outcome_ids:
        raise ValueError(f"Outcome advantages name unknown rollout IDs: {sorted(unknown_outcome_ids)}")
    calibrated_outcomes = {rollout_id: float(value) for rollout_id, value in (outcome_advantages_by_id or {}).items()}
    if any(not isfinite(value) for value in calibrated_outcomes.values()):
        raise ValueError("outcome_advantages_by_id must contain only finite values")

    delta_config = DeltaConfig(
        label_mode=config.label_mode,
        loss_mask="selected_state",
        advantage_source="critic" if config.advantage_source == "critic" else "mc_label",
        policy_advantage_scope="segment" if config.advantage_source == "critic" else "selected_token",
        target_normalization="none",
        advantage_normalization=config.advantage_normalization,
    )
    unnormalized = []
    for example in examples:
        predictions = None
        if config.advantage_source == "critic":
            if raw_predictions_by_id is None:
                raise ValueError("critic advantage source requires raw_predictions_by_id")
            predictions = raw_predictions_by_id.get(example.rollout.rollout_id)
            if predictions is None:
                raise ValueError(f"Missing critic predictions for rollout_id={example.rollout.rollout_id}")
        unnormalized.append(
            state_advantages(
                example,
                delta_config,
                raw_predictions=predictions,
                reward_delta=(reward_deltas_by_id or {}).get(example.rollout.rollout_id),
            )
        )

    if config.advantage_normalization == "standardize":
        if fit_stats:
            # TransferQueue padding rows carry zero weight and never enter the
            # training-population statistics even if they duplicate real data.
            # Calibrated whole-response outcome advantages are already in policy
            # advantage units and never enter the critic-delta population.
            fit_vectors = [
                vector
                for example, vector in zip(examples, unnormalized, strict=True)
                if example.rollout.rollout_id not in calibrated_outcomes
                and any(vector.mask)
                and float((row_weights_by_id or {}).get(example.rollout.rollout_id, 1.0)) > 0.0
            ]
            if fit_vectors:
                stats = fit_normalization(fit_vectors, population="state_advantages")
        if stats is None:
            if any(
                any(vector.mask)
                and example.rollout.rollout_id not in calibrated_outcomes
                and float((row_weights_by_id or {}).get(example.rollout.rollout_id, 1.0)) > 0.0
                for example, vector in zip(examples, unnormalized, strict=True)
            ):
                raise ValueError("standardize requires fitted state-advantage stats for critic rows")
            # An all-short batch has no critic population. Keep stats absent so a
            # later batch with critic rows can fit the proper population.
            advantages = list(unnormalized)
        else:
            advantages = [
                normalize(vector, mode="standardize", population="state_advantages", stats=stats)
                for vector in unnormalized
            ]
    else:
        if fit_stats:
            raise ValueError("fit_stats=True requires advantage_normalization='standardize'")
        if stats is not None:
            raise ValueError("advantage_normalization='none' cannot consume normalization stats")
        advantages = unnormalized

    for index, example in enumerate(examples):
        rollout_id = example.rollout.rollout_id
        if rollout_id in calibrated_outcomes:
            value = calibrated_outcomes[rollout_id]
            length = len(example.rollout.response_token_ids)
            advantages[index] = TokenVector((value,) * length, (1.0,) * length)

    clip_limit = None if config.advantage_clip is None else float(config.advantage_clip)
    rows: list[dict[str, Any]] = []
    for example, advantage in zip(examples, advantages, strict=True):
        clamped = 0
        if clip_limit is not None:
            advantage, clamped = clip_advantages(advantage, clip_limit)
        rollout = example.rollout
        length = len(rollout.response_token_ids)
        behavior = _behavior_values(example, behavior_logprobs_by_id, config.behavior_logprob_source)
        ref = _reference_values(rollout.rollout_id, reference_logprobs_by_id, length)
        if config.kl_coef > 0.0 and config.kl_reference == "reference" and ref is None:
            raise ValueError(f"Missing fixed reference logprobs for rollout_id={rollout.rollout_id}")
        if config.kl_reference == "behavior" and ref is not None:
            raise ValueError("Do not pass separate reference logprobs when KL uses behavior")
        weight = float((row_weights_by_id or {}).get(rollout.rollout_id, 1.0))
        if not isfinite(weight) or weight < 0.0:
            raise ValueError(f"row weight for {rollout.rollout_id} must be finite and nonnegative")
        state_mask = tuple(float(value) for value in advantage.mask)
        policy_mask = tuple(float(value) for value in rollout.policy_token_mask)
        response_mask = (1.0,) * length
        policy_loss_mask = tuple(
            state * policy * valid for state, policy, valid in zip(state_mask, policy_mask, response_mask, strict=True)
        )
        row = {
            "id": rollout.rollout_id,
            "rollout_id": rollout.rollout_id,
            "prompt_id": rollout.prompt_id,
            "prompt_token_ids": tuple(rollout.prompt_token_ids),
            "response_token_ids": tuple(rollout.response_token_ids),
            "old_logprobs": behavior,
            "advantages": tuple(advantage.values),
            "state_advantages": tuple(advantage.values),
            "state_mask": state_mask,
            "policy_token_mask": policy_mask,
            "response_mask": response_mask,
            "policy_loss_mask": policy_loss_mask,
            "row_weight": weight,
            "sample_valid": weight > 0.0,
            # Diagnostic only: how many active tokens the clip bound changed.
            "advantage_clip_count": clamped,
        }
        if ref is not None:
            row["ref_logprobs"] = ref
            row["reference_policy_id"] = reference_policy_id
        rows.append(row)
    return rows, stats


def policy_training_batch(
    rows: Sequence[Mapping[str, Any]],
    pad_token_id: int,
    config: DeltaPolicyConfig,
    *,
    width: int | None = None,
) -> dict[str, Any]:
    """Collate rows against next-token outputs, reproducing source tail windows.

    A response token aligned to the first retained input token has no preceding
    logit and is deliberately omitted from the output action arrays. All other
    arrays have sequence length ``input_width - 1``.
    """
    import torch

    if not rows:
        raise ValueError("Cannot collate an empty policy batch")
    full_lengths = [len(row["prompt_token_ids"]) + len(row["response_token_ids"]) for row in rows]
    if any(len(row["prompt_token_ids"]) == 0 for row in rows):
        raise ValueError("Every row needs prompt tokens to score its first response token")
    needed_width = max(full_lengths)
    sequence_length = needed_width if width is None else int(width)
    if sequence_length < 2:
        raise ValueError("Policy input width must be at least 2")
    if config.max_length is not None and sequence_length > config.max_length:
        if config.window == "legacy_tail":
            sequence_length = config.max_length
        else:
            raise ValueError(f"Sequence length {sequence_length} exceeds max_length={config.max_length}")
    if width is not None and sequence_length < min(needed_width, config.max_length or needed_width):
        if config.window == "error":
            raise ValueError("Explicit width would truncate a policy row")
    if config.window == "error" and any(length > sequence_length for length in full_lengths):
        raise ValueError("Sequence exceeds configured policy window")

    input_ids, attention_mask = [], []
    old_log_probs, advantages, state_mask = [], [], []
    policy_token_mask, policy_loss_mask, response_mask, ref_log_prob = [], [], [], []
    row_weights, sample_valid, ids = [], [], []
    action_width = sequence_length - 1
    has_reference = config.kl_reference == "reference" and config.kl_coef > 0.0
    for row in rows:
        prompt_ids = tuple(int(value) for value in row["prompt_token_ids"])
        response_ids = tuple(int(value) for value in row["response_token_ids"])
        row_length = len(response_ids)
        aligned_fields = {
            "old_logprobs": row["old_logprobs"],
            "advantages": row["advantages"] if "advantages" in row else row["state_advantages"],
            "state_mask": row["state_mask"],
            "policy_token_mask": row["policy_token_mask"],
            "policy_loss_mask": row["policy_loss_mask"],
            "response_mask": row.get("response_mask", (1.0,) * row_length),
        }
        aligned = {}
        for name, values in aligned_fields.items():
            field_name = f"{name} for {row.get('id', '<unknown>')}"
            validator = (
                _binary_sequence
                if name in {"state_mask", "policy_token_mask", "policy_loss_mask", "response_mask"}
                else _finite_sequence
            )
            aligned[name] = validator(values, name=field_name, expected=row_length)
        reference = None
        if has_reference:
            if "ref_logprobs" not in row:
                raise ValueError(f"Missing fixed reference logprobs for rollout_id={row.get('id')}")
            reference = _finite_sequence(
                row["ref_logprobs"], name=f"reference logprobs for {row.get('id')}", expected=row_length
            )

        full_ids = prompt_ids + response_ids
        shift = max(0, len(full_ids) - sequence_length)
        kept_ids = full_ids[shift:]
        row_arrays = {
            name: [0.0] * action_width
            for name in ("old", "advantage", "state", "policy", "pg_mask", "response", "kl", "reference")
        }
        response_start = len(prompt_ids)
        response_end = response_start + row_length
        for local_position in range(1, len(kept_ids)):
            original_position = shift + local_position
            if response_start <= original_position < response_end:
                response_index = original_position - response_start
                action_index = local_position - 1
                row_arrays["old"][action_index] = aligned["old_logprobs"][response_index]
                row_arrays["advantage"][action_index] = aligned["advantages"][response_index]
                row_arrays["state"][action_index] = aligned["state_mask"][response_index]
                row_arrays["policy"][action_index] = aligned["policy_token_mask"][response_index]
                row_arrays["pg_mask"][action_index] = aligned["policy_loss_mask"][response_index]
                row_arrays["response"][action_index] = aligned["response_mask"][response_index]
                row_arrays["kl"][action_index] = (
                    aligned["policy_loss_mask"][response_index]
                    if config.kl_mask_scope == "policy"
                    else aligned["response_mask"][response_index]
                )
                if reference is not None:
                    row_arrays["reference"][action_index] = reference[response_index]

        pad_len = sequence_length - len(kept_ids)
        input_ids.append(list(kept_ids) + [int(pad_token_id)] * pad_len)
        attention_mask.append([1] * len(kept_ids) + [0] * pad_len)
        old_log_probs.append(row_arrays["old"])
        advantages.append(row_arrays["advantage"])
        state_mask.append(row_arrays["state"])
        policy_token_mask.append(row_arrays["policy"])
        policy_loss_mask.append(row_arrays["pg_mask"])
        response_mask.append(row_arrays["response"])
        ref_log_prob.append(row_arrays["reference"])
        row_weight = float(row.get("row_weight", 1.0))
        if not isfinite(row_weight) or row_weight < 0.0:
            raise ValueError(f"row weight for {row.get('id')} must be finite and nonnegative")
        row_is_valid = row.get("sample_valid", True)
        if not isinstance(row_is_valid, bool):
            raise ValueError(f"sample_valid for {row.get('id')} must be boolean")
        row_weights.append(row_weight)
        sample_valid.append(float(row_is_valid))
        ids.append(str(row.get("id", row.get("rollout_id", ""))))

    row_weight_tensor = torch.tensor(row_weights, dtype=torch.float32)
    valid_tensor = torch.tensor(sample_valid, dtype=torch.float32)
    effective_weights = row_weight_tensor * valid_tensor
    global_weight = float(effective_weights.sum().item())
    result: dict[str, Any] = {
        "ids": ids,
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "state_advantages": torch.tensor(advantages, dtype=torch.float32),
        "state_mask": torch.tensor(state_mask, dtype=torch.float32),
        "policy_token_mask": torch.tensor(policy_token_mask, dtype=torch.float32),
        "policy_loss_mask": torch.tensor(policy_loss_mask, dtype=torch.float32),
        "response_mask": torch.tensor(response_mask, dtype=torch.float32),
        "kl_mask": torch.tensor(
            policy_loss_mask if config.kl_mask_scope == "policy" else response_mask, dtype=torch.float32
        ),
        "row_weight": row_weight_tensor,
        "sample_valid_mask": valid_tensor,
        "dp_size": 1,
        "global_valid_sample_weight": global_weight,
        "delta_policy_config": config.as_dict(),
    }
    if has_reference:
        result["ref_log_prob"] = torch.tensor(ref_log_prob, dtype=torch.float32)
    return result


def policy_response_batch(
    rows: Sequence[Mapping[str, Any]], config: DeltaPolicyConfig, *, width: int | None = None
) -> dict[str, Any]:
    """Pad policy fields by response position for merging into a V1 actor batch.

    V1 already owns prompt/response/input padding and returns response-aligned
    logprobs from its actor worker. This adapter supplies matching policy tensors
    while preserving the V1 batch's prompts, responses, position IDs, and masks.
    """
    import torch

    if not rows:
        raise ValueError("Cannot collate an empty policy batch")
    response_lengths = [len(row["response_token_ids"]) for row in rows]
    response_width = max(response_lengths) if width is None else int(width)
    if response_width < max(response_lengths):
        raise ValueError("policy_response_batch cannot truncate response arrays")
    names = ("old_log_probs", "advantages", "state_mask", "policy_token_mask", "policy_loss_mask", "response_mask")
    values: dict[str, list[list[float]]] = {name: [] for name in names}
    values["kl_mask"] = []
    values["ref_log_prob"] = []
    weights, valid, ids = [], [], []
    has_reference = config.kl_reference == "reference" and config.kl_coef > 0.0
    reference_ids = set()
    for row, length in zip(rows, response_lengths, strict=True):
        source_fields = {
            "old_log_probs": row["old_logprobs"],
            "advantages": row["advantages"] if "advantages" in row else row["state_advantages"],
            "state_mask": row["state_mask"],
            "policy_token_mask": row["policy_token_mask"],
            "policy_loss_mask": row["policy_loss_mask"],
            "response_mask": row.get("response_mask", (1.0,) * length),
        }
        for name, source in source_fields.items():
            validator = (
                _binary_sequence
                if name in {"state_mask", "policy_token_mask", "policy_loss_mask", "response_mask"}
                else _finite_sequence
            )
            aligned = validator(source, name=f"{name} for {row.get('id')}", expected=length)
            values[name].append(list(aligned) + [0.0] * (response_width - length))
        kl_values = (
            source_fields["policy_loss_mask"] if config.kl_mask_scope == "policy" else source_fields["response_mask"]
        )
        values["kl_mask"].append(list(kl_values) + [0.0] * (response_width - length))
        if has_reference:
            if "ref_logprobs" not in row:
                raise ValueError(f"Missing fixed reference logprobs for rollout_id={row.get('id')}")
            ref = _finite_sequence(row["ref_logprobs"], name=f"reference logprobs for {row.get('id')}", expected=length)
            values["ref_log_prob"].append(list(ref) + [0.0] * (response_width - length))
            reference_ids.add(row.get("reference_policy_id"))
        weight = float(row.get("row_weight", 1.0))
        if not isfinite(weight) or weight < 0.0:
            raise ValueError(f"row weight for {row.get('id')} must be finite and nonnegative")
        row_is_valid = row.get("sample_valid", True)
        if not isinstance(row_is_valid, bool):
            raise ValueError(f"sample_valid for {row.get('id')} must be boolean")
        weights.append(weight)
        valid.append(float(row_is_valid))
        ids.append(str(row.get("id", row.get("rollout_id", ""))))
    if has_reference and (None in reference_ids or len(reference_ids) != 1):
        raise ValueError("A policy batch must use one explicit, fixed reference_policy_id")

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    valid_tensor = torch.tensor(valid, dtype=torch.float32)
    result: dict[str, Any] = {
        "ids": ids,
        **{
            name: torch.tensor(items, dtype=torch.float32)
            for name, items in values.items()
            if name != "ref_log_prob" or has_reference
        },
        "row_weight": weight_tensor,
        "sample_valid_mask": valid_tensor,
        "dp_size": 1,
        "global_valid_sample_weight": float((weight_tensor * valid_tensor).sum().item()),
        "delta_policy_config": config.as_dict(),
    }
    return result
