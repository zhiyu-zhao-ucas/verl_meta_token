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
"""Explicit configuration for delta driven policy updates."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class DeltaPolicyConfig:
    """Select either the historical offline objective or the online PPO variant.

    The online reference KL is against the fixed initial actor. The behavior
    policy still supplies the per-round clipping anchor; callers must not replace
    the reference model when actor checkpoints advance between rounds.
    """

    mode: Literal["offline_legacy", "online_ppo"]
    advantage_source: Literal["critic", "mc_local_oracle"] = "critic"
    label_mode: Literal["selected_segment", "paired_next_state"] = "selected_segment"
    advantage_normalization: Literal["none", "standardize"] = "standardize"
    behavior_logprob_source: Literal["stored", "rollout", "precomputed", "actor_snapshot"] | None = None
    clip_range: float = 0.2
    kl_coef: float | None = None
    kl_reference: Literal["behavior", "reference"] | None = None
    kl_mask_scope: Literal["policy", "response"] | None = None
    kl_estimator: Literal["source_sampled_reverse", "low_var_kl"] | None = None
    train_logprob_temperature: float = 1.0
    max_length: int | None = None
    window: Literal["legacy_tail", "error"] | None = None
    # Symmetric bound applied to state advantages AFTER normalization. It covers
    # every row, so one outlier delta cannot dominate an update.
    advantage_clip: float | None = None
    # Binary outcome advantage calibration for short responses. A per-row
    # continuation baseline may override the fallback baseline, but the scale is
    # always shared so outcomes remain independent of critic normalization.
    short_outcome_baseline: float = 0.5
    short_outcome_scale: float = 0.5
    # ``row_mean`` gives each logical row equal mass regardless of response
    # length. row_weight supports source sampler repeat/subsample weights.
    aggregation: Literal["row_mean"] = "row_mean"

    def __post_init__(self):
        choices = {
            "mode": {"offline_legacy", "online_ppo"},
            "advantage_source": {"critic", "mc_local_oracle"},
            "label_mode": {"selected_segment", "paired_next_state"},
            "advantage_normalization": {"none", "standardize"},
            "behavior_logprob_source": {"stored", "rollout", "precomputed", "actor_snapshot"},
            "kl_reference": {"behavior", "reference"},
            "kl_mask_scope": {"policy", "response"},
            "kl_estimator": {"source_sampled_reverse", "low_var_kl"},
            "window": {"legacy_tail", "error"},
            "aggregation": {"row_mean"},
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise ValueError(f"Unsupported {name}={value!r}")

        defaults = {
            "offline_legacy": {
                "kl_coef": 0.01,
                "kl_reference": "behavior",
                "kl_mask_scope": "policy",
                "kl_estimator": "source_sampled_reverse",
                "behavior_logprob_source": "stored",
                "max_length": 2048,
                "window": "legacy_tail",
            },
            "online_ppo": {
                "kl_coef": 0.001,
                "kl_reference": "reference",
                "kl_mask_scope": "policy",
                "kl_estimator": "low_var_kl",
                "behavior_logprob_source": "actor_snapshot",
                "max_length": None,
                "window": "error",
            },
        }[self.mode]
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)

        if self.mode == "offline_legacy" and self.kl_reference != "behavior":
            raise ValueError("offline_legacy must use the behavior policy as its KL reference")
        if self.mode == "online_ppo" and self.kl_reference != "reference":
            raise ValueError("online_ppo must use a fixed reference policy for KL")
        if self.mode == "online_ppo" and self.behavior_logprob_source in {"rollout", "stored"}:
            raise ValueError("online_ppo old logprobs require actor_snapshot or provenance-checked precomputed values")
        if self.mode == "offline_legacy" and self.kl_estimator != "source_sampled_reverse":
            raise ValueError("offline_legacy must use the unclamped source sampled reverse KL")
        if self.mode == "online_ppo" and self.kl_estimator != "low_var_kl":
            raise ValueError("online_ppo must use PPO low_var_kl")

        for name in ("clip_range", "train_logprob_temperature", "kl_coef"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "clip_range" and not 0.0 < value < 1.0:
                raise ValueError("clip_range must be in (0, 1)")
            if name == "train_logprob_temperature" and value <= 0.0:
                raise ValueError("train_logprob_temperature must be positive")
            if name == "kl_coef" and value < 0.0:
                raise ValueError("kl_coef must be nonnegative")
        baseline = float(self.short_outcome_baseline)
        scale = float(self.short_outcome_scale)
        if not isfinite(baseline) or not 0.0 <= baseline <= 1.0:
            raise ValueError("short_outcome_baseline must be finite and in [0, 1]")
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("short_outcome_scale must be positive and finite")
        if self.max_length is not None and (
            isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or self.max_length < 2
        ):
            raise ValueError("max_length must be an integer at least 2 or None")
        if self.advantage_clip is not None:
            clip = float(self.advantage_clip)
            if not isfinite(clip) or clip <= 0.0:
                raise ValueError("advantage_clip must be a positive finite value or None")

    def as_dict(self) -> dict:
        """Return a plain mapping suitable for TensorDict non-tensor metadata."""
        return asdict(self)

    @classmethod
    def historical_offline(cls, **overrides) -> "DeltaPolicyConfig":
        """July value_model final-GRPO policy defaults."""
        return cls(mode="offline_legacy", **overrides)

    @classmethod
    def online(cls, **overrides) -> "DeltaPolicyConfig":
        """Online delta PG with per-round behavior clipping and fixed-ref KL."""
        return cls(mode="online_ppo", **overrides)


def delta_policy_config_from_mapping(values: Mapping, *, mode: str | None = None) -> DeltaPolicyConfig:
    """Build the shared config from a plain/OmegaConf delta-policy mapping."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(values):
            values = OmegaConf.to_container(values, resolve=True)
    except ImportError:
        pass
    if not isinstance(values, Mapping):
        raise TypeError("delta policy config must be a mapping")
    if "delta_policy" in values:
        values = values["delta_policy"]
    if not isinstance(values, Mapping):
        raise TypeError("delta_policy must be a mapping")
    allowed = {item.name for item in fields(DeltaPolicyConfig)}
    supplied = {key: value for key, value in values.items() if key in allowed}
    if mode is not None:
        if "mode" in supplied and supplied["mode"] != mode:
            raise ValueError(f"Config mode {supplied['mode']!r} does not match requested mode={mode!r}")
        supplied["mode"] = mode
    if "mode" not in supplied:
        raise ValueError("delta policy mapping must specify mode or receive mode=")
    return DeltaPolicyConfig(**supplied)
