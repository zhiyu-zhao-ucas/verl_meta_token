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
"""Runtime settings for the standalone delta policy trainer."""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class PolicyTrainingConfig:
    model_path: str
    tokenizer_path: str | None = None
    reference_model_path: str | None = None
    reference_policy_id: str | None = None
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    trust_remote_code: bool = False
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    rl_epochs: int = 1
    save_every_steps: int = 100
    seed: int = 42

    def __post_init__(self):
        if not self.model_path:
            raise ValueError("model_path is required")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        if self.attention_implementation not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("Unsupported attention_implementation")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("rl_epochs", "save_every_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PolicyTrainingConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{name: values[name] for name in allowed if name in values})
