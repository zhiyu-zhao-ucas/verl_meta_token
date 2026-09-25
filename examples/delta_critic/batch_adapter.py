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
"""Untruncated index/padding adapters; independent of torch and TransferQueue."""

from .contracts import DeltaConfig, DeltaExample, TokenVector
from .target_ops import critic_targets


def read_positions(prompt_length: int, response_length: int, *, left_padding: int = 0) -> dict[str, tuple[int, ...]]:
    """Delta scalar: P+t; policy next-token logits: P+t-1 (plus left padding)."""
    dimensions = (
        ("prompt_length", prompt_length), ("response_length", response_length), ("left_padding", left_padding)
    )
    for name, value in dimensions:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if prompt_length == 0:
        raise ValueError("At least one prompt token is required to score response token zero")
    start = prompt_length + left_padding
    return {
        "delta_scalar": tuple(range(start, start + response_length)),
        "policy_logprob": tuple(range(start - 1, start + response_length - 1)),
    }


def selected_inference_rows(example: DeltaExample) -> list[dict]:
    """One prefix ending WITH the selected token, exactly as the source critic."""
    rollout = example.rollout
    return [
        {
            "rollout_id": rollout.rollout_id,
            "token_index": state.token_index,
            "input_ids": rollout.prompt_token_ids + rollout.response_token_ids[: state.token_index + 1],
            "value_position": len(rollout.prompt_token_ids) + state.token_index,
        }
        for state in example.states
    ]


def pad_response_rows(
    examples: list[DeltaExample], config: DeltaConfig, advantages: list[TokenVector], *, width: int
) -> list[dict]:
    """Right-pad response vectors; width cannot discard real tokens.

    This defines the tensor boundary, not a trainer integration. Masks stay
    independent: a zero state advantage does not remove a policy-loss token.
    policy_loss_mask is the final state_only PG/KL mask and denominator mask.
    It does not affect the earlier training-population normalization statistics.
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 0:
        raise ValueError("width must be a nonnegative integer")
    if len(examples) != len(advantages):
        raise ValueError("One advantage vector is required per example")
    rows = []
    for example, advantage in zip(examples, advantages, strict=True):
        rollout = example.rollout
        length = len(rollout.response_token_ids)
        if width < length:
            raise ValueError("Padding width cannot truncate a response; window policy remains unselected")
        if len(advantage.values) != length:
            raise ValueError("Advantage length must equal response length")
        targets, signal = critic_targets(example, config)
        positions = read_positions(len(rollout.prompt_token_ids), length)

        def pad(values, fill=0.0):
            return tuple(values) + (fill,) * (width - length)

        rows.append(
            {
                "rollout_id": rollout.rollout_id,
                "raw_critic_targets": pad(targets.values),
                "critic_target_mask": pad(targets.mask),
                "critic_signal_mask": pad(signal),
                "state_advantages": pad(advantage.values),
                "state_advantage_mask": pad(advantage.mask),
                "response_valid_mask": pad((1.0,) * length),
                "policy_token_mask": pad(rollout.policy_token_mask),
                "policy_loss_mask": pad(
                    tuple(a * b for a, b in zip(advantage.mask, rollout.policy_token_mask, strict=True))
                ),
                "delta_scalar_positions": pad(positions["delta_scalar"], -1),
                "policy_logprob_positions": pad(positions["policy_logprob"], -1),
            }
        )
    return rows
