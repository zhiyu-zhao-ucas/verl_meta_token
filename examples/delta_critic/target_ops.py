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
"""Pure scalar operations matching value_model's local delta and final GRPO paths."""

import math
from collections.abc import Iterable, Mapping

from .contracts import DeltaConfig, DeltaExample, NormalizationStats, TokenVector, finite


def value_model_final_grpo(*, label_mode: str, loss_mask: str, target_normalization: str) -> DeltaConfig:
    """July completed critic-driven runs: learned delta, segment broadcast, standardize.

    Critic choices remain explicit: the policy run alone does not select its loss
    or distinguish local TD labels from selected-segment labels.
    """
    return DeltaConfig(label_mode, loss_mask, "critic", "segment", target_normalization, "standardize")


def selected_labels(example: DeltaExample, label_mode: str) -> tuple[float, ...]:
    if label_mode not in {"selected_segment", "paired_next_state"}:
        raise ValueError(f"Unsupported label_mode={label_mode!r}")
    result = []
    for index, state in enumerate(example.states):
        if label_mode == "paired_next_state":
            if state.delta is not None:
                result.append(state.delta)
                continue
            if state.v_next is None:
                raise ValueError("paired_next_state requires delta or v_next")
            next_value = state.v_next
        else:
            next_value = (
                example.states[index + 1].v_prefix
                if index + 1 < len(example.states)
                else example.rollout.terminal_reward
            )
        result.append(next_value - state.v_prefix)
    return tuple(result)


def critic_targets(example: DeltaExample, config: DeltaConfig) -> tuple[TokenVector, tuple[float, ...]]:
    """Return raw targets with loss mask, and a separate selected-state signal mask."""
    length = len(example.rollout.response_token_ids)
    targets, signal = [0.0] * length, [0.0] * length
    for state, label in zip(example.states, selected_labels(example, config.label_mode), strict=True):
        targets[state.token_index] = label
        signal[state.token_index] = 1.0
    mask = [1.0] * length if config.loss_mask == "response" else signal
    return TokenVector(tuple(targets), tuple(mask)), tuple(signal)


def state_advantages(
    example: DeltaExample, config: DeltaConfig, *, raw_predictions: Mapping[int, float] | None = None
) -> TokenVector:
    """Predictions are already denormalized scalars keyed by response token index.

    Require every selected prediction; never silently substitute an MC label for
    a missing model output. Original labels remain available in the example.
    """
    if config.advantage_source == "critic":
        indices = {state.token_index for state in example.states}
        if raw_predictions is None or set(raw_predictions) != indices:
            raise ValueError("raw_predictions must match all selected token indices exactly")
        values = [finite(raw_predictions[state.token_index], "prediction") for state in example.states]
    else:
        if raw_predictions is not None:
            raise ValueError("mc_label advantage source does not consume critic predictions")
        values = selected_labels(example, config.label_mode)
    length = len(example.rollout.response_token_ids)
    advantages, mask = [0.0] * length, [0.0] * length
    for index, (state, value) in enumerate(zip(example.states, values, strict=True)):
        end = state.token_index + 1
        if config.policy_advantage_scope == "segment":
            end = example.states[index + 1].token_index if index + 1 < len(example.states) else length
        for token_index in range(state.token_index, end):
            advantages[token_index], mask[token_index] = value, 1.0
    return TokenVector(tuple(advantages), tuple(mask))


def fit_normalization(vectors: Iterable[TokenVector], *, population: str) -> NormalizationStats:
    """Fit on the train split BEFORE padding/truncation and policy-mask intersection.

    Call separately for critic targets and broadcast state advantages. Both use
    population std; arithmetic order matches the corresponding source function.
    """
    values = [value for vector in vectors for value, keep in zip(vector.values, vector.mask, strict=True) if keep]
    if not values:
        raise ValueError("Cannot fit normalization on an empty active population")
    mean = sum(values) / len(values)
    if population == "critic_targets":
        variance = max(0.0, sum(value * value for value in values) / len(values) - mean * mean)
    elif population == "state_advantages":
        variance = sum((value - mean) ** 2 for value in values) / len(values)
    else:
        raise ValueError(f"Unsupported population={population!r}")
    return NormalizationStats(float(len(values)), mean, math.sqrt(variance), population)


def normalize(
    vector: TokenVector, *, mode: str, population: str, stats: NormalizationStats | None = None
) -> TokenVector:
    if population not in {"critic_targets", "state_advantages"}:
        raise ValueError(f"Unsupported population={population!r}")
    if mode == "none":
        if stats is not None:
            raise ValueError("mode=none must not consume normalization stats")
        return vector
    if mode != "standardize" or stats is None or stats.population != population:
        raise ValueError("standardize requires stats from the matching population")
    # Source critic wrapper transforms ALL targets, including masked background.
    # Source policy normalization instead leaves inactive state tokens at zero.
    values = tuple(
        (value - stats.mean) / stats.std if keep or population == "critic_targets" else 0.0
        for value, keep in zip(vector.values, vector.mask, strict=True)
    )
    return TokenVector(values, vector.mask)


def denormalize_predictions(
    values: Iterable[float], *, mode: str, stats: NormalizationStats | None = None
) -> tuple[float, ...]:
    values = tuple(finite(value, "prediction") for value in values)
    if mode == "none":
        if stats is not None:
            raise ValueError("mode=none must not consume normalization stats")
        return values
    if mode != "standardize" or stats is None or stats.population != "critic_targets":
        raise ValueError("standardize predictions requires critic target stats")
    return tuple(value * stats.std + stats.mean for value in values)
