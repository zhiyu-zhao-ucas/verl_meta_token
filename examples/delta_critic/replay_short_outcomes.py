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
"""Replay old and calibrated short-outcome advantages on one saved rollout batch.

This is a CPU-only *advantage replay*. It reads saved token IDs and labels, calls
``prepare_policy_rows`` for both advantage paths, and reconstructs the new
critic-population statistics by removing the old short-reward contributions
from the persisted initial statistics. It does not run an actor forward, PPO/KL
loss, optimizer step, or GPU operation, so its output must not be described as
an actor update.

Run from the repository root with::

    python examples/delta_critic/replay_short_outcomes.py

Use ``--export-batch`` to write the exact prompt/response token IDs for a later
model-scoring experiment. Continuations are recovered from the original MC
label records, never re-tokenized from their displayed text.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.delta_critic.contracts import DeltaExample, NormalizationStats, Rollout
from examples.delta_critic.policy_config import DeltaPolicyConfig, delta_policy_config_from_mapping
from examples.delta_critic.policy_data import prepare_policy_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/delta_policy_4b_expansion_short_reward_hybrid_20261002"


@dataclass(frozen=True)
class ReplayRow:
    """One original or MC continuation in the fixed actor batch."""

    row_id: str
    parent_id: str
    kind: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    policy_token_mask: tuple[float, ...]
    reward: float
    baseline: float
    baseline_source: str
    short_text_tokens: int
    response_prefix_tokens: int
    continuation_index: int | None = None
    eos_token_count: int = 0
    trailing_token_id: int | None = None
    trailing_marker: str = "none"
    selected_token_indices: tuple[int, ...] | None = None
    known_expansion_mc_indices: tuple[int, ...] = ()
    selection_status: str = "unavailable"

    @property
    def policy_tokens(self) -> int:
        return len(self.response_token_ids)

    @property
    def is_short(self) -> bool:
        return self.short_text_tokens <= self.short_response_budget  # set by report construction

    # The budget is attached after parsing to keep each row self-contained when
    # exporting it. It is assigned in ``row_as_dict`` rather than stored here.
    short_response_budget: int = 100


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a one-dimensional token-ID list")
    result = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{name} must contain nonnegative integer token IDs")
        result.append(int(token))
    return tuple(result)


def _float_tuple(value: Any, name: str, expected: int) -> tuple[float, ...]:
    if value is None:
        return (1.0,) * expected
    if not isinstance(value, list | tuple) or len(value) != expected:
        raise ValueError(f"{name} must have one entry per response token")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item not in (0.0, 1.0) for item in result):
        raise ValueError(f"{name} must contain finite binary values")
    return result


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rollout rows found in {path}")
    return rows


def _marker_table(tokenizer_dir: str | Path) -> dict[str, int | None]:
    """Read local tokenizer JSON only; never resolve or download a model."""
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    config_path = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Local tokenizer metadata is missing: {tokenizer_path}")
    tokenizer = json.loads(tokenizer_path.read_text())
    added = tokenizer.get("added_tokens", [])
    by_content = {
        str(item["content"]): int(item["id"])
        for item in added
        if isinstance(item, dict) and "content" in item and "id" in item
    }
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    eos = config.get("eos_token")
    if isinstance(eos, dict):
        eos = eos.get("content")
    eos_id = by_content.get(str(eos)) if eos is not None else None
    return {
        "configured_eos_token_id": eos_id,
        "im_end_token_id": by_content.get("<|im_end|>"),
    }


def _marker_for(tokens: tuple[int, ...], marker_ids: dict[str, int | None]) -> tuple[int | None, str]:
    if not tokens:
        return None, "empty"
    trailing = tokens[-1]
    for name, key in (("configured_eos", "configured_eos_token_id"), ("im_end", "im_end_token_id")):
        if marker_ids.get(key) is not None and trailing == marker_ids[key]:
            return trailing, name
    return trailing, "other"


def _row_from_tokens(
    *,
    row_id: str,
    parent_id: str,
    kind: str,
    prompt_ids: Any,
    response_ids: Any,
    policy_mask: Any,
    reward: Any,
    baseline: float,
    baseline_source: str,
    short_text_tokens: int,
    prefix_tokens: int,
    budget: int,
    marker_ids: dict[str, int | None],
    continuation_index: int | None = None,
    selected_token_indices: tuple[int, ...] | None = None,
    known_expansion_mc_indices: tuple[int, ...] = (),
    selection_status: str = "unavailable",
) -> ReplayRow:
    prompt = _int_tuple(prompt_ids, f"{row_id}.prompt_token_ids")
    response = _int_tuple(response_ids, f"{row_id}.response_token_ids")
    if not prompt:
        raise ValueError(f"{row_id} has an empty prompt")
    if not response:
        raise ValueError(f"{row_id} has an empty response")
    mask = _float_tuple(policy_mask, f"{row_id}.policy_token_mask", len(response))
    if short_text_tokens < 1 or prefix_tokens < 0 or short_text_tokens != prefix_tokens + len(response):
        raise ValueError(f"Inconsistent full-response token count for {row_id}")
    reward_value = _finite_float(reward, f"{row_id}.reward")
    if not 0.0 <= reward_value <= 1.0:
        raise ValueError(f"{row_id}.reward must be in [0, 1]")
    trailing_id, trailing_marker = _marker_for(response, marker_ids)
    eos_id = marker_ids.get("configured_eos_token_id")
    return ReplayRow(
        row_id=row_id,
        parent_id=parent_id,
        kind=kind,
        prompt_token_ids=prompt,
        response_token_ids=response,
        policy_token_mask=mask,
        reward=reward_value,
        baseline=float(baseline),
        baseline_source=baseline_source,
        short_text_tokens=short_text_tokens,
        response_prefix_tokens=prefix_tokens,
        continuation_index=continuation_index,
        eos_token_count=sum(token == eos_id for token in response) if eos_id is not None else 0,
        trailing_token_id=trailing_id,
        trailing_marker=trailing_marker,
        short_response_budget=budget,
        selected_token_indices=selected_token_indices,
        known_expansion_mc_indices=known_expansion_mc_indices,
        selection_status=selection_status,
    )


def reconstruct_batch_rows(
    source_rows: list[dict[str, Any]],
    *,
    short_response_budget: int,
    default_baseline: float,
    marker_ids: dict[str, int | None] | None = None,
    selection_config: dict[str, Any] | None = None,
) -> list[ReplayRow]:
    """Rebuild all 128 originals and their 24 x 8 child trajectories.

    The saved JSONL has one top-level line for every actor row. Original
    ``delta_mc_record`` labels contain the exact continuation token IDs; their
    displayed text is deliberately ignored here.
    """
    if short_response_budget < 1:
        raise ValueError("short_response_budget must be positive")
    marker_ids = marker_ids or {"configured_eos_token_id": None, "im_end_token_id": None}
    selection_config = selection_config or {}
    lines_by_uid: dict[str, dict[str, Any]] = {}
    original_records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for row in source_rows:
        row_uid = row.get("uid")
        if not isinstance(row_uid, str) or not row_uid:
            raise ValueError("Every fixed-batch rollout line must have a nonempty uid")
        if row_uid in lines_by_uid:
            raise ValueError(f"Duplicate rollout uid={row_uid}")
        lines_by_uid[row_uid] = row
        record = row.get("delta_mc_record")
        if isinstance(record, dict) and isinstance(record.get("rollout"), dict):
            original_records.append((row, record, record["rollout"]))

    if not original_records:
        raise ValueError("No original rows with delta_mc_record.rollout were found")

    samples_by_uid: dict[str, ReplayRow] = {}
    for source, record, rollout in original_records:
        parent_uid = source["uid"]
        rollout_id = str(rollout.get("id") or parent_uid)
        prompt_ids = _int_tuple(rollout.get("prompt_token_ids"), f"{rollout_id}.prompt_token_ids")
        response_ids = _int_tuple(rollout.get("response_token_ids"), f"{rollout_id}.response_token_ids")
        mask = rollout.get("policy_token_mask", [1.0] * len(response_ids))
        known_mc_indices = tuple(
            sorted(
                int(state["token_index"])
                for state in record.get("states", [])
                if isinstance(state, dict) and "token_index" in state
            )
        )
        original = _row_from_tokens(
            row_id=parent_uid,
            parent_id=parent_uid,
            kind="original",
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            policy_mask=mask,
            reward=rollout.get("terminal_reward"),
            baseline=default_baseline,
            baseline_source="fixed_default",
            short_text_tokens=len(response_ids),
            prefix_tokens=0,
            budget=short_response_budget,
            marker_ids=marker_ids,
            known_expansion_mc_indices=known_mc_indices,
            selection_status="incomplete_original_indices_not_saved",
        )
        samples_by_uid[parent_uid] = original

        labels = record.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"MC labels for {rollout_id} must be a list")
        if len(labels) > 1:
            raise ValueError(f"Expected at most one expansion label for original {parent_uid}")
        parent_stem = parent_uid.rsplit("_", 1)[0]
        for label in labels:
            if str(label.get("rollout_id")) != rollout_id:
                raise ValueError(f"MC label rollout_id does not match original {parent_uid}")
            completions = label.get("mc_continuations", [])
            if not isinstance(completions, list) or not completions:
                raise ValueError(f"Expansion label for {parent_uid} has no continuations")
            by_index: dict[int, dict[str, Any]] = {}
            for completion in completions:
                index = completion.get("continuation_index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in by_index:
                    raise ValueError(f"Invalid or repeated continuation index for {parent_uid}")
                by_index[index] = completion
            if sorted(by_index) != list(range(len(by_index))):
                raise ValueError(f"Continuation indices for {parent_uid} must be contiguous from zero")
            prefix_ids = _int_tuple(label.get("prefix_response_token_ids", []), f"{parent_uid}.prefix")
            all_rewards = {
                index: _finite_float(item.get("reward"), f"{parent_uid}.reward[{index}]")
                for index, item in by_index.items()
            }
            for index, completion in sorted(by_index.items()):
                child_uid = f"{parent_stem}_{index + 1}"
                if child_uid not in lines_by_uid:
                    raise ValueError(f"Saved actor batch is missing continuation row {child_uid}")
                child_line = lines_by_uid[child_uid]
                if isinstance(child_line.get("delta_mc_record"), dict) and child_line["delta_mc_record"].get("rollout"):
                    raise ValueError(f"Continuation uid unexpectedly points to an original: {child_uid}")
                child_reward = all_rewards[index]
                siblings = [reward for other_index, reward in all_rewards.items() if other_index != index]
                baseline = sum(siblings) / len(siblings) if siblings else default_baseline
                baseline_source = "leave_one_out_siblings" if siblings else "fixed_default_singleton"
                response_ids = _int_tuple(completion.get("token_ids"), f"{child_uid}.token_ids")
                top_logprobs = completion.get("delta_top_logprobs")
                selected_indices = None
                selection_status = "not_saved"
                if top_logprobs is not None:
                    if not isinstance(top_logprobs, list) or len(top_logprobs) != len(response_ids):
                        raise ValueError(f"Saved continuation top-logprobs do not align for {child_uid}")
                    from examples.delta_critic.policy_online import select_uncertainty_indices

                    selected_indices = tuple(
                        select_uncertainty_indices(
                            top_logprobs,
                            [1.0] * len(response_ids),
                            states_per_response=int(selection_config.get("states_per_response", 64)),
                            min_token_gap=int(selection_config.get("min_token_gap", 32)),
                            entropy_weight=float(selection_config.get("entropy_weight", 0.0)),
                            low_top1_weight=float(selection_config.get("low_top1_weight", 1.0)),
                            final_window_tokens=int(selection_config.get("final_window_tokens", 64)),
                            final_window_weight=float(selection_config.get("final_window_weight", 0.0)),
                            max_candidates=int(selection_config.get("max_candidates", 5)),
                        )
                    )
                    selection_status = "reconstructed_from_saved_top_logprobs"
                child = _row_from_tokens(
                    row_id=child_uid,
                    parent_id=parent_uid,
                    kind="continuation",
                    prompt_ids=prompt_ids + prefix_ids,
                    response_ids=response_ids,
                    policy_mask=[1.0] * len(response_ids),
                    reward=child_reward,
                    baseline=baseline,
                    baseline_source=baseline_source,
                    short_text_tokens=len(prefix_ids) + len(response_ids),
                    prefix_tokens=len(prefix_ids),
                    budget=short_response_budget,
                    marker_ids=marker_ids,
                    continuation_index=index,
                    selected_token_indices=selected_indices,
                    selection_status=selection_status,
                )
                if child_uid in samples_by_uid:
                    raise ValueError(f"Duplicate reconstructed actor row {child_uid}")
                samples_by_uid[child_uid] = child

    batch_uids = [row["uid"] for row in source_rows]
    if set(batch_uids) != set(samples_by_uid):
        missing = sorted(set(batch_uids) - set(samples_by_uid))[:5]
        extra = sorted(set(samples_by_uid) - set(batch_uids))[:5]
        raise ValueError(f"Reconstructed rows do not match saved actor batch; missing={missing}, extra={extra}")
    return [samples_by_uid[uid] for uid in batch_uids]


def reconstruct_stats_without_short_rows(
    old_stats: NormalizationStats,
    rows: list[ReplayRow],
) -> tuple[NormalizationStats, dict[str, float | int]]:
    """Remove the old raw-reward values from persisted population moments.

    In the legacy path, every short row contributed its raw reward once per
    response token. ``old_stats`` stores population mean/std, so those exact
    contributions can be subtracted without re-scoring the frozen critic.
    """
    short_rows = [row for row in rows if row.is_short]
    excluded_count = sum(row.policy_tokens for row in short_rows)
    excluded_sum = sum(row.policy_tokens * row.reward for row in short_rows)
    excluded_sum_squares = sum(row.policy_tokens * row.reward**2 for row in short_rows)
    total_count = int(round(old_stats.count))
    if not math.isclose(old_stats.count, total_count, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Saved normalization count must be an integer token population")
    retained_count = total_count - excluded_count
    if retained_count <= 0:
        raise ValueError("Removing short outcomes leaves no critic advantage population")
    old_sum = total_count * old_stats.mean
    old_sum_squares = total_count * (old_stats.std**2 + old_stats.mean**2)
    new_mean = (old_sum - excluded_sum) / retained_count
    variance = (old_sum_squares - excluded_sum_squares) / retained_count - new_mean**2
    if variance < -1e-10:
        raise ValueError(f"Reconstructed critic variance is negative: {variance}")
    new_std = math.sqrt(max(0.0, variance))
    if new_std < 1e-8:
        raise ValueError("Reconstructed critic population has near-zero standard deviation")
    new_stats = NormalizationStats(
        count=float(retained_count), mean=float(new_mean), std=float(new_std), population="state_advantages"
    )
    audit = {
        "old_population_count": total_count,
        "excluded_short_response_policy_tokens": excluded_count,
        "new_critic_population_count": retained_count,
        "excluded_short_reward_sum": excluded_sum,
        "excluded_short_reward_sum_squares": excluded_sum_squares,
        "moment_reconstruction_precision": "derived from persisted mean/std, exact up to their saved decimal precision",
    }
    return new_stats, audit


def compare_short_advantages(
    rows: list[ReplayRow],
    *,
    old_stats: NormalizationStats,
    new_stats: NormalizationStats,
    advantage_clip: float | None,
    outcome_scale: float,
) -> list[dict[str, Any]]:
    """Use the production row-preparation code for both short-outcome paths."""
    short_rows = [row for row in rows if row.is_short]
    if not short_rows:
        return []
    examples = []
    dummy_behavior: dict[str, tuple[float, ...]] = {}
    raw_predictions: dict[str, dict[int, float]] = {}
    rewards: dict[str, float] = {}
    outcome_advantages: dict[str, float] = {}
    row_by_id: dict[str, ReplayRow] = {}
    for row in short_rows:
        rollout = Rollout(
            rollout_id=row.row_id,
            prompt_token_ids=row.prompt_token_ids,
            response_token_ids=row.response_token_ids,
            terminal_reward=row.reward,
            policy_token_mask=row.policy_token_mask,
            prompt_id=row.parent_id,
        )
        examples.append(DeltaExample(rollout=rollout, states=()))
        # All these rows are outcome-overridden. Empty critic maps satisfy the
        # existing contracts without pretending a critic prediction was read.
        raw_predictions[row.row_id] = {}
        dummy_behavior[row.row_id] = (0.0,) * row.policy_tokens
        rewards[row.row_id] = row.reward
        outcome_advantages[row.row_id] = (row.reward - row.baseline) / outcome_scale
        row_by_id[row.row_id] = row

    policy_config = DeltaPolicyConfig.online(advantage_clip=advantage_clip, kl_coef=0.0)
    old_prepared, _ = prepare_policy_rows(
        examples,
        raw_predictions,
        dummy_behavior,
        policy_config,
        stats=old_stats,
        reward_deltas_by_id=rewards,
    )
    new_prepared, _ = prepare_policy_rows(
        examples,
        raw_predictions,
        dummy_behavior,
        policy_config,
        stats=new_stats,
        outcome_advantages_by_id=outcome_advantages,
    )
    new_by_id = {row["id"]: row for row in new_prepared}
    comparisons = []
    for old in old_prepared:
        sample = row_by_id[old["id"]]
        new = new_by_id[old["id"]]
        old_values = old["advantages"]
        new_values = new["advantages"]
        active = [i for i, value in enumerate(old["policy_loss_mask"]) if value]
        if not active:
            raise ValueError(f"Short outcome {sample.row_id} has no active policy tokens")
        old_mean = sum(old_values[index] for index in active) / len(active)
        new_mean = sum(new_values[index] for index in active) / len(active)
        comparisons.append(
            {
                "row_id": sample.row_id,
                "parent_id": sample.parent_id,
                "kind": sample.kind,
                "continuation_index": sample.continuation_index,
                "short_text_tokens_including_prefix": sample.short_text_tokens,
                "prefix_tokens_conditioning_only": sample.response_prefix_tokens,
                "response_tokens_trained": sample.policy_tokens,
                "reward": sample.reward,
                "baseline": sample.baseline,
                "baseline_source": sample.baseline_source,
                "old_advantage_mean": old_mean,
                "new_advantage_mean": new_mean,
                "old_advantage_first_token": old_values[0],
                "new_advantage_first_token": new_values[0],
                "active_policy_tokens": len(active),
                "configured_eos_token_count_in_response": sample.eos_token_count,
                "trailing_token_id": sample.trailing_token_id,
                "trailing_marker": sample.trailing_marker,
                "eos_is_policy_active": bool(
                    sample.trailing_marker == "configured_eos" and sample.policy_token_mask[-1] == 1.0
                ),
            }
        )
    return comparisons


def summarize_rows(
    rows: list[ReplayRow],
    *,
    marker_ids: dict[str, int | None],
    short_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison_by_id = {row["row_id"]: row for row in short_comparisons}
    stop_ids = {value for value in marker_ids.values() if value is not None}
    groups: dict[str, list[ReplayRow]] = {"all": rows, "original": [], "continuation": [], "short": [], "long": []}
    groups["original"] = [row for row in rows if row.kind == "original"]
    groups["continuation"] = [row for row in rows if row.kind == "continuation"]
    groups["short"] = [row for row in rows if row.is_short]
    groups["long"] = [row for row in rows if not row.is_short]
    group_metrics = {}
    eos_boundary_flips = []
    for name, group in groups.items():
        trailing = Counter(row.trailing_marker for row in group)
        with_eos_removed = []
        for row in group:
            excludes_one_marker = row.short_text_tokens - int(row.trailing_token_id in stop_ids)
            if (row.short_text_tokens <= row.short_response_budget) != (
                excludes_one_marker <= row.short_response_budget
            ):
                with_eos_removed.append(row)
        if name == "all":
            eos_boundary_flips = [row.row_id for row in with_eos_removed]
        rewards = [row.reward for row in group]
        group_metrics[name] = {
            "rows": len(group),
            "policy_response_tokens": sum(row.policy_tokens for row in group),
            "short_by_full_token_rule": sum(row.is_short for row in group),
            "positive_reward_rows": sum(row.reward > 0.0 for row in group),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "configured_eos_rows_ending_with_eos": trailing["configured_eos"],
            "rows_ending_with_im_end": trailing["im_end"],
            "rows_ending_with_other_token": trailing["other"],
            "empty_response_rows": trailing["empty"],
            "eos_tokens_in_policy_responses": sum(row.eos_token_count for row in group),
            "short_threshold_flips_if_one_trailing_stop_marker_is_excluded": len(with_eos_removed),
        }
    short = groups["short"]
    old_values = [comparison_by_id[row.row_id]["old_advantage_mean"] for row in short]
    new_values = [comparison_by_id[row.row_id]["new_advantage_mean"] for row in short]
    token_count = sum(comparison_by_id[row.row_id]["active_policy_tokens"] for row in short)
    old_token_sum = sum(
        comparison_by_id[row.row_id]["old_advantage_mean"] * comparison_by_id[row.row_id]["active_policy_tokens"]
        for row in short
    )
    new_token_sum = sum(
        comparison_by_id[row.row_id]["new_advantage_mean"] * comparison_by_id[row.row_id]["active_policy_tokens"]
        for row in short
    )
    false_positive_zero_reward = [
        row for row in short if row.reward == 0.0 and comparison_by_id[row.row_id]["old_advantage_mean"] > 0.0
    ]
    return {
        "groups": group_metrics,
        "eos_boundary_flip_row_ids": eos_boundary_flips,
        "short_advantage_comparison": {
            "rows": len(short),
            "response_policy_tokens": token_count,
            "old_mean_row_advantage": sum(old_values) / len(old_values) if old_values else 0.0,
            "new_mean_row_advantage": sum(new_values) / len(new_values) if new_values else 0.0,
            "old_mean_token_advantage": old_token_sum / token_count if token_count else 0.0,
            "new_mean_token_advantage": new_token_sum / token_count if token_count else 0.0,
            "zero_reward_rows_with_positive_old_advantage": len(false_positive_zero_reward),
            "zero_reward_positive_old_advantage_row_ids": [row.row_id for row in false_positive_zero_reward],
        },
    }


def export_replay_batch(path: str | Path, rows: list[ReplayRow], comparisons: list[dict[str, Any]]) -> None:
    """Write lossless actor inputs plus outcome metadata for later model scoring."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_by_id = {row["row_id"]: row for row in comparisons}
    with path.open("w") as stream:
        for row in rows:
            item = {
                "row_id": row.row_id,
                "parent_id": row.parent_id,
                "kind": row.kind,
                "continuation_index": row.continuation_index,
                "prompt_token_ids": list(row.prompt_token_ids),
                "response_token_ids": list(row.response_token_ids),
                "policy_token_mask": list(row.policy_token_mask),
                "short_response_tokens": row.short_response_budget,
                "short_text_tokens_including_prefix": row.short_text_tokens,
                "prefix_tokens_conditioning_only": row.response_prefix_tokens,
                "reward": row.reward,
                "baseline": row.baseline,
                "baseline_source": row.baseline_source,
                "is_short": row.is_short,
                "selected_token_indices": (
                    list(row.selected_token_indices) if row.selected_token_indices is not None else None
                ),
                "known_expansion_mc_indices": list(row.known_expansion_mc_indices),
                "selected_token_indices_status": row.selection_status,
                "advantage_replay": comparison_by_id.get(row.row_id),
            }
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading the saved YAML config requires PyYAML in the active environment") from exc
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _load_step_metrics(path: str | Path, step: int = 1) -> dict[str, Any]:
    for row in read_jsonl(path):
        if row.get("step") == step:
            data = row.get("data")
            if not isinstance(data, dict):
                raise ValueError(f"Metrics step {step} has no data mapping")
            return data
    raise ValueError(f"Metrics file has no step={step} row: {path}")


def _load_old_stats(path: str | Path) -> tuple[NormalizationStats, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    stats_payload = payload.get("stats")
    if not isinstance(stats_payload, dict):
        raise ValueError("Saved initial advantage statistics have no fitted population")
    stats = NormalizationStats(**stats_payload)
    if stats.population != "state_advantages":
        raise ValueError("Saved run statistics are not state_advantages")
    return stats, payload


def replay_from_artifacts(
    *,
    rollouts_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    stats_path: str | Path,
    metrics_path: str | Path,
    eos_token_id: int | None = None,
    im_end_token_id: int | None = None,
    export_batch_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    config = _load_yaml(config_path)
    policy_mapping = config.get("algorithm", {}).get("delta_policy", {})
    if not isinstance(policy_mapping, dict):
        raise ValueError("Saved config must include algorithm.delta_policy")
    policy_config = delta_policy_config_from_mapping(policy_mapping, mode="online_ppo")
    expansion = policy_mapping.get("expansion", {})
    budget = expansion.get("short_response_tokens", manifest.get("short_response_tokens"))
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("Could not resolve the saved short-response token budget")
    if policy_config.advantage_normalization != "standardize":
        raise ValueError("This saved replay expects standardize advantage normalization")
    if policy_config.advantage_clip is None:
        raise ValueError("This saved replay expects an explicit advantage clip bound")
    marker_ids = _marker_table(manifest["actor"])
    if eos_token_id is not None:
        marker_ids["configured_eos_token_id"] = eos_token_id
    if im_end_token_id is not None:
        marker_ids["im_end_token_id"] = im_end_token_id

    source_rows = read_jsonl(rollouts_path)
    replay_rows = reconstruct_batch_rows(
        source_rows,
        short_response_budget=budget,
        default_baseline=policy_config.short_outcome_baseline,
        marker_ids=marker_ids,
        selection_config=policy_mapping.get("selection", {}),
    )
    originals = sum(row.kind == "original" for row in replay_rows)
    continuations = sum(row.kind == "continuation" for row in replay_rows)
    metrics = _load_step_metrics(metrics_path)
    if originals != int(manifest.get("rollout_prompt_batch_size", originals)):
        raise ValueError(f"Original rows do not match manifest: {originals}")
    if continuations != int(manifest.get("continuation_rows_per_step", continuations)):
        raise ValueError(f"Continuation rows do not match manifest: {continuations}")
    short_rows = [row for row in replay_rows if row.is_short]
    if len(short_rows) != int(metrics.get("delta_policy/short_response_rows", -1)):
        raise ValueError("Reconstructed short-row count disagrees with first-step metrics")
    positive_short = sum(row.reward > 0.0 for row in short_rows)
    if positive_short != int(metrics.get("delta_policy/short_response_positive_rows", -1)):
        raise ValueError("Reconstructed positive short-row count disagrees with first-step metrics")
    mean_short_reward = sum(row.reward for row in short_rows) / len(short_rows) if short_rows else 0.0
    if not math.isclose(
        mean_short_reward,
        float(metrics.get("delta_policy/short_response_mean_reward", math.nan)),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("Reconstructed short-row reward mean disagrees with first-step metrics")

    old_stats, old_stats_payload = _load_old_stats(stats_path)
    if old_stats_payload.get("reference_policy_id") != manifest.get("reference_policy_id"):
        raise ValueError("Saved normalization statistics use a different reference policy")
    if old_stats_payload.get("critic_weights_sha256") != manifest.get("critic_weights_sha256"):
        raise ValueError("Saved normalization statistics use a different frozen critic")
    if old_stats_payload.get("label_mode") != policy_config.label_mode:
        raise ValueError("Saved normalization statistics use a different delta label mode")
    if not math.isclose(old_stats.count, float(metrics["delta_policy/advantage_stats_count"]), abs_tol=1e-6):
        raise ValueError("Saved normalization count disagrees with first-step metrics")
    contract = str(old_stats_payload.get("advantage_contract", ""))
    if "short_reward" not in contract:
        raise ValueError("Saved first-run stats do not identify the legacy short_reward contract")

    new_stats, stats_audit = reconstruct_stats_without_short_rows(old_stats, replay_rows)
    comparisons = compare_short_advantages(
        replay_rows,
        old_stats=old_stats,
        new_stats=new_stats,
        advantage_clip=policy_config.advantage_clip,
        outcome_scale=policy_config.short_outcome_scale,
    )
    summary = summarize_rows(replay_rows, marker_ids=marker_ids, short_comparisons=comparisons)
    old_zero_reward_advantage = min(
        policy_config.advantage_clip,
        max(-policy_config.advantage_clip, (0.0 - old_stats.mean) / old_stats.std),
    )
    new_default_zero_advantage = min(
        policy_config.advantage_clip,
        max(
            -policy_config.advantage_clip,
            (0.0 - policy_config.short_outcome_baseline) / policy_config.short_outcome_scale,
        ),
    )

    if export_batch_path is not None:
        export_replay_batch(export_batch_path, replay_rows, comparisons)

    return {
        "experiment": {
            "name": manifest.get("experiment", Path(manifest_path).parent.name),
            "manifest": str(Path(manifest_path)),
            "config": str(Path(config_path)),
            "rollouts": str(Path(rollouts_path)),
            "metrics": str(Path(metrics_path)),
            "initial_advantage_stats": str(Path(stats_path)),
            "actor": manifest.get("actor"),
            "critic_artifact": manifest.get("critic_artifact"),
            "critic_training_step": manifest.get("critic_training_step"),
            "fixed_first_batch": True,
        },
        "replay_scope": {
            "kind": "advantage_only",
            "actor_forward_performed": False,
            "ppo_or_kl_loss_performed": False,
            "optimizer_step_performed": False,
            "gpu_operation_performed": False,
            "later_actor_update_requirements": [
                "recompute exact actor selected-token indices for original rows from the cached actor snapshot",
                "compute chunked actor behavior/reference response logprobs on the recovered token IDs",
                "run both variants from identical initial actor weights and optimizer state with sample-mean PPO+KL",
            ],
        },
        "batch": {
            "source_lines": len(source_rows),
            "original_rows": originals,
            "continuation_rows_reconstructed_from_mc_labels": continuations,
            "expanded_originals": len({row.parent_id for row in replay_rows if row.kind == "continuation"}),
            "continuation_rows_with_reconstructed_selected_indices": sum(
                row.kind == "continuation" and row.selected_token_indices is not None for row in replay_rows
            ),
            "original_rows_with_full_selected_indices": sum(
                row.kind == "original" and row.selected_token_indices is not None for row in replay_rows
            ),
            "original_rows_with_only_expansion_mc_indices": sum(
                row.kind == "original" and bool(row.known_expansion_mc_indices) for row in replay_rows
            ),
            "selected_index_limitation": (
                "The saved shard omits complete selected_token_indices for original rows; only their MC expansion "
                "state indices are saved. Full paired actor updates must recompute original selections from the "
                "cached actor snapshot."
            ),
            "short_response_tokens": budget,
            "outcome_baseline": policy_config.short_outcome_baseline,
            "outcome_scale": policy_config.short_outcome_scale,
            "advantage_clip": policy_config.advantage_clip,
            "reference_policy_id": manifest.get("reference_policy_id"),
            "critic_weights_sha256": manifest.get("critic_weights_sha256"),
        },
        "source_consistency": {
            "first_step_short_rows_metric": int(metrics["delta_policy/short_response_rows"]),
            "first_step_positive_short_rows_metric": int(metrics["delta_policy/short_response_positive_rows"]),
            "reconstructed_short_reward_mean": mean_short_reward,
            "old_stats_count_metric": float(metrics["delta_policy/advantage_stats_count"]),
            "old_stats_contract": contract,
            "eos_marker_ids_from_local_tokenizer": marker_ids,
        },
        "normalization": {
            "old_legacy_stats": {
                "count": old_stats.count,
                "mean": old_stats.mean,
                "std": old_stats.std,
            },
            "new_stats_excluding_short_outcome_rows": {
                "count": new_stats.count,
                "mean": new_stats.mean,
                "std": new_stats.std,
            },
            "reconstruction": stats_audit,
            "critic_advantage_mapping": {
                "old": "clip((raw critic delta - old mean) / old std, [-advantage_clip, +advantage_clip])",
                "new": "clip((raw critic delta - new mean) / new std, [-advantage_clip, +advantage_clip])",
            },
            "zero_reward_short_outcome_advantage": {
                "legacy_reward_delta_after_old_normalization": old_zero_reward_advantage,
                "new_original_with_default_baseline": new_default_zero_advantage,
                "note": "Continuation zero-reward advantages use their recorded leave-one-out sibling baseline.",
            },
        },
        "summary": summary,
        "short_rows": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--step", type=int, default=1, help="First actor update step to cross-check in metrics")
    parser.add_argument("--eos-token-id", type=int, help="Override the EOS ID read from local tokenizer files")
    parser.add_argument("--im-end-token-id", type=int, help="Override the assistant-turn end ID")
    parser.add_argument("--output", type=Path, help="Write the JSON report here; stdout is used by default")
    parser.add_argument("--export-batch", type=Path, help="Optionally export exact token rows for later actor scoring")
    args = parser.parse_args()
    run_dir = args.run_dir
    paths = {
        "rollouts_path": args.rollouts or run_dir / "rollouts/1.jsonl",
        "config_path": args.config or run_dir / "config.yaml",
        "manifest_path": args.manifest or run_dir / "manifest.json",
        "stats_path": args.stats or run_dir / "checkpoints/delta_advantage_stats.json",
        "metrics_path": args.metrics or run_dir / "metrics.jsonl",
        "eos_token_id": args.eos_token_id,
        "im_end_token_id": args.im_end_token_id,
        "export_batch_path": args.export_batch,
    }
    report = replay_from_artifacts(**paths)
    if args.step != 1:
        raise ValueError("The current replay is pinned to first actor update step 1")
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
