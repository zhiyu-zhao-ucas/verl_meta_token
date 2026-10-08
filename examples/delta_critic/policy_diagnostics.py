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
"""CPU diagnostics for the already materialized, pre-update O-TD batch.

All rows passed here are real trajectories, including actor-disabled rows.
Rollout summaries count each row once; advantage summaries include only final
PG-mask tokens. Raw deltas count selected critic states once, excluding outcome
substitutions and actor-disabled rows. Thus segment broadcast cannot inflate raw
state counts. Clip fractions use the pre-policy-intersection state-token scope
of prepare_policy_rows' exact clip counts. Every empty statistic is zero with
an explicit zero denominator; it is not evidence of a healthy signal.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            key: 0.0
            for key in (
                "count",
                "mean",
                "std",
                "abs_mean",
                "max_abs",
                "positive_fraction",
                "negative_fraction",
                "zero_fraction",
            )
        }
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "abs_mean": float(np.abs(array).mean()),
        "max_abs": float(np.abs(array).max()),
        "positive_fraction": float((array > 0.0).mean()),
        "negative_fraction": float((array < 0.0).mean()),
        "zero_fraction": float((array == 0.0).mean()),
    }


def policy_batch_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Describe rollout health and the precise signal supplied to policy loss.

    Reward-conditioned advantages describe association, not an unbiased test of
    critic correctness: negative local deltas can occur in successful responses.
    ``short_outcome`` and ``critic`` partition the configured advantage source;
    original/continuation and reward partitions overlap those source groups.
    """
    groups = {
        "all": list(rows),
        "original": [row for row in rows if not row["is_continuation"]],
        "continuation": [row for row in rows if row["is_continuation"]],
        "short_outcome": [row for row in rows if row["is_outcome"]],
        "critic": [row for row in rows if not row["is_outcome"]],
        "positive_reward": [row for row in rows if row["reward"] > 0.0],
        "zero_reward": [row for row in rows if row["reward"] == 0.0],
    }
    metrics = {}
    for name, group in groups.items():
        lengths, rewards, advantages, raw_deltas, row_advantages = [], [], [], [], []
        state_tokens = clip_tokens = policy_tokens = active_rows = zero_rows = selected_states = 0
        for row in group:
            response_mask = np.asarray(row["response_mask"], dtype=bool)
            policy_mask = np.asarray(row["policy_loss_mask"], dtype=bool) & response_mask
            advantage = np.asarray(row["advantages"], dtype=np.float64)[policy_mask]
            lengths.append(int(response_mask.sum()))
            rewards.append(float(row["reward"]))
            selected_states += len(row["raw_selected_deltas"])
            policy_tokens += int(policy_mask.sum())
            if not row["actor_enabled"]:
                continue
            state_tokens += int(sum(row["state_mask"]))
            clip_tokens += int(row["advantage_clip_count"])
            if not row["is_outcome"]:
                raw_deltas.extend(row["raw_selected_deltas"])
            if advantage.size:
                active_rows += 1
                zero_rows += int(not np.any(advantage))
                advantages.extend(advantage.tolist())
                row_advantages.append(float(advantage.mean()))
        prefix = f"delta_policy/health/{name}"
        response_tokens = sum(lengths)
        summaries = {
            "rows": len(group),
            "actor_enabled_rows": sum(bool(row["actor_enabled"]) for row in group),
            "active_pg_rows": active_rows,
            "zero_advantage_rows": zero_rows,
            "response_tokens": response_tokens,
            "response_length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "positive_reward_fraction": sum(value > 0.0 for value in rewards) / max(len(rewards), 1),
            "selected_states": selected_states,
            "policy_tokens": policy_tokens,
            "policy_token_coverage": policy_tokens / max(response_tokens, 1),
            "state_advantage_tokens": state_tokens,
            "state_advantage_clipped_tokens": clip_tokens,
            "state_advantage_clip_fraction": clip_tokens / max(state_tokens, 1),
            "row_advantage_mean": float(np.mean(row_advantages)) if row_advantages else 0.0,
            "positive_row_advantage_fraction": sum(value > 0.0 for value in row_advantages) / max(active_rows, 1),
        }
        metrics.update({f"{prefix}/{key}": float(value) for key, value in summaries.items()})
        for source, values in (("raw_delta", raw_deltas), ("advantage", advantages)):
            metrics.update({f"{prefix}/{source}_{key}": value for key, value in _distribution(values).items()})
    return metrics
