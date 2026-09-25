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
"""Response-indexed data contracts; no tokenization, truncation or model loading."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

SCHEMA_VERSION = 1


def finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def binary_mask(values: tuple[float, ...], length: int, name: str) -> None:
    if len(values) != length or any(value not in (0, 1) for value in values):
        raise ValueError(f"{name} must contain {length} binary entries")


@dataclass(frozen=True)
class DeltaConfig:
    # Explicit choices prevent a library default from silently selecting an experiment.
    label_mode: Literal["selected_segment", "paired_next_state"]
    loss_mask: Literal["response", "selected_state", "selected_delta"]
    advantage_source: Literal["critic", "mc_label"]
    policy_advantage_scope: Literal["segment", "selected_token"]
    target_normalization: Literal["none", "standardize"]
    advantage_normalization: Literal["none", "standardize"]

    def __post_init__(self):
        choices = {
            "label_mode": {"selected_segment", "paired_next_state"},
            "loss_mask": {"response", "selected_state", "selected_delta"},
            "advantage_source": {"critic", "mc_label"},
            "policy_advantage_scope": {"segment", "selected_token"},
            "target_normalization": {"none", "standardize"},
            "advantage_normalization": {"none", "standardize"},
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unsupported {name}={getattr(self, name)!r}")


@dataclass(frozen=True)
class Rollout:
    rollout_id: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    terminal_reward: float
    policy_token_mask: tuple[float, ...]
    prompt_id: str | None = None
    finish_reason: str | None = None
    # Original row preserves policy/tokenizer versions, split, token logprobs, etc.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if not self.rollout_id or not self.prompt_token_ids:
            raise ValueError("rollout_id and nonempty prompt_token_ids are required")
        for token in self.prompt_token_ids + self.response_token_ids:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("token IDs must be nonnegative integers")
        finite(self.terminal_reward, "terminal_reward")
        binary_mask(self.policy_token_mask, len(self.response_token_ids), "policy_token_mask")


@dataclass(frozen=True)
class SelectedState:
    rollout_id: str
    token_index: int
    v_prefix: float
    v_next: float | None = None
    delta: float | None = None
    state_id: str | None = None
    mc_num_samples: int | None = None
    mc_next_num_samples: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if isinstance(self.token_index, bool) or not isinstance(self.token_index, int):
            raise ValueError("token_index must be an integer")
        for name in ("v_prefix", "v_next", "delta"):
            if getattr(self, name) is not None:
                finite(getattr(self, name), name)
        for name in ("mc_num_samples", "mc_next_num_samples"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class DeltaExample:
    rollout: Rollout
    states: tuple[SelectedState, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version={self.schema_version}")
        indices = []
        state_ids = set()
        for state in self.states:
            if state.rollout_id != self.rollout.rollout_id:
                raise ValueError("Selected state rollout_id mismatch")
            if not 0 <= state.token_index < len(self.rollout.response_token_ids):
                raise ValueError(f"token_index={state.token_index} out of range")
            if state.state_id is not None:
                if state.state_id in state_ids:
                    raise ValueError(f"Duplicate state_id={state.state_id}")
                state_ids.add(state.state_id)
            indices.append(state.token_index)
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate selected token_index")
        if indices != sorted(indices):
            raise ValueError("Selected states must be sorted by token_index")


@dataclass(frozen=True)
class TokenVector:
    values: tuple[float, ...]
    mask: tuple[float, ...]

    def __post_init__(self):
        binary_mask(self.mask, len(self.values), "mask")
        for value in self.values:
            finite(value, "token value")


@dataclass(frozen=True)
class NormalizationStats:
    count: float
    mean: float
    std: float
    population: Literal["critic_targets", "state_advantages"]

    def __post_init__(self):
        for name in ("count", "mean", "std"):
            finite(getattr(self, name), name)
        if self.count <= 0 or self.std < 1e-8:
            raise ValueError("Normalization requires count > 0 and std >= 1e-8")
        if self.population not in {"critic_targets", "state_advantages"}:
            raise ValueError(f"Unsupported population={self.population!r}")
