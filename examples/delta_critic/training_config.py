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
"""Explicit scalar training semantics, shared by trainer, importer and scorer."""

import math
from dataclasses import asdict, dataclass

from .contracts import DeltaConfig


@dataclass(frozen=True)
class ScalarConfig:
    model_path: str
    tokenizer_path: str | None = None
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    training_autocast: str = "model_dtype"
    scoring_autocast: str = "none"
    value_layer: int = -1
    objective: str = "local_td0"
    loss_type: str = "mse"
    loss_mask: str = "response"
    delta_label_mode: str = "selected_segment"
    target_normalization: str = "standardize"
    # ``None`` means use the backbone's native max_position_embeddings. The
    # resolved value is written into artifacts so inference can validate the
    # exact training window. Legacy checkpoints always carry an explicit cap.
    max_length: int | None = None
    window_policy: str = "full"
    td_weight: float = 1.0
    terminal_weight: float = 1.0
    terminal_normalization: str = "independent"
    terminal_length_weighting: str = "none"
    hybrid_reduction: str = "all_samples"
    continuation_selection: str = "uniform"
    continuation_max_candidates: int = 5
    continuation_state_count: int = 64
    continuation_min_gap: int = 32
    continuation_budget: int | None = None
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    seed: int = 42

    def __post_init__(self):
        self.contract()
        if self.terminal_normalization not in {"independent", "td"}:
            raise ValueError("terminal_normalization must be independent or td")
        if self.terminal_length_weighting not in {"none", "inverse_count"}:
            raise ValueError("terminal_length_weighting must be none or inverse_count")
        if self.hybrid_reduction not in {"all_samples", "separate_samples"}:
            raise ValueError("hybrid_reduction must be all_samples or separate_samples")
        if self.continuation_selection not in {"uniform", "uncertainty"}:
            raise ValueError("continuation_selection must be uniform or uncertainty")
        if self.objective not in {"local_td0", "hybrid_terminal_composition"}:
            raise ValueError(f"Unsupported objective: {self.objective}")
        if self.loss_type not in {"mse", "nonzero_balanced_mse"}:
            raise ValueError(f"Unsupported scalar loss: {self.loss_type}")
        if self.training_autocast not in {"model_dtype", "none", "bfloat16"} or self.scoring_autocast not in {
            "none",
            "bfloat16",
        }:
            raise ValueError("Unsupported autocast semantics")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("Only float32 and bfloat16 are supported")
        if self.attention_implementation not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("attention_implementation must be sdpa, eager or flash_attention_2")
        if self.window_policy not in {"full", "legacy_tail"}:
            raise ValueError("Invalid context/window policy")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError("max_length must be positive when specified")
        if self.window_policy == "legacy_tail" and self.max_length is None:
            raise ValueError("legacy_tail requires an explicit max_length")
        if not self.model_path:
            raise ValueError("Model path is required")
        if type(self.continuation_state_count) is not int or self.continuation_state_count < 1:
            raise ValueError("continuation_state_count must be a positive integer")
        if type(self.continuation_min_gap) is not int or self.continuation_min_gap < 0:
            raise ValueError("continuation_min_gap must be a nonnegative integer")
        if type(self.continuation_max_candidates) is not int or self.continuation_max_candidates < 1:
            raise ValueError("continuation_max_candidates must be a positive integer")
        if self.continuation_budget is not None and (
            type(self.continuation_budget) is not int or self.continuation_budget < 1
        ):
            raise ValueError("continuation_budget must be a positive integer")
        for key in ("td_weight", "terminal_weight", "learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if self.objective == "local_td0" and (self.td_weight != 1.0 or self.terminal_weight != 1.0):
            raise ValueError("local_td0 uses fixed unit weights; td_weight/terminal_weight apply only to hybrid")

    def contract(self):
        return DeltaConfig(
            self.delta_label_mode, self.loss_mask, "critic", "segment", self.target_normalization, "none"
        )

    def as_dict(self):
        return asdict(self)
