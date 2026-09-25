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
"""Select response-token states while preserving source indices and provenance."""

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

def select_spaced_states(scored: list[tuple[float, int]], count: int, min_token_gap: int) -> list[int]:
    """Source tie break: higher score, then later token index."""
    if count < 0 or min_token_gap < 0:
        raise ValueError("count and min_token_gap must be nonnegative")
    chosen = []
    for score, index in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True):
        if not isfinite(score):
            raise ValueError("selection scores must be finite")
        if all(abs(index - previous) >= min_token_gap for previous in chosen):
            chosen.append(index)
            if len(chosen) == count:
                break
    return chosen if count else []


def _top_mass(candidates: list[dict], threshold: float, limit: int) -> tuple[list[dict], float]:
    selected = []
    mass = 0.0
    for candidate in sorted(candidates, key=lambda item: float(item.get("prob", 0)), reverse=True)[:limit]:
        probability = float(candidate.get("prob", 0))
        if not isfinite(probability) or probability < 0:
            raise ValueError("candidate probability must be finite and nonnegative")
        selected.append(candidate)
        mass += probability
        if mass >= threshold:
            break
    return selected, mass


def select_states(
    rollouts: Iterable[dict[str, Any]], *, strategy: str, states_per_response: int,
    min_token_gap: int = 0, indices_by_rollout: Mapping[str, Iterable[int]] | None = None,
    top_mass: float = 0.9, max_candidates: int = 20, entropy_weight: float = 1.0,
    low_top1_weight: float = 1.0, final_window_tokens: int = 0, final_window_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Use source uncertainty scoring or caller supplied token positions for V1.

    V1 does not export a top-k distribution. Explicit positions avoid silently
    substituting a different score for the source uncertainty strategy.
    """
    if strategy not in {"uncertainty", "indices"}:
        raise ValueError(f"Unsupported selection strategy={strategy!r}")
    if states_per_response < 0 or min_token_gap < 0 or max_candidates <= 0:
        raise ValueError("invalid state selection budget")
    if not 0 < top_mass <= 1:
        raise ValueError("top_mass must be in (0, 1]")
    if strategy == "indices" and indices_by_rollout is None:
        raise ValueError("indices strategy requires indices_by_rollout")
    rows = []
    seen = set()
    for rollout in rollouts:
        rid = str(rollout["id"])
        if rid in seen:
            raise ValueError(f"Duplicate rollout_id={rid}")
        seen.add(rid)
        response = list(rollout["response_token_ids"])
        length = len(response)
        if strategy == "indices":
            indices = list(indices_by_rollout.get(rid, ()))
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate requested token index for rollout_id={rid}")
            scored = [(0.0, index) for index in indices]
            details = {index: ([], 0.0) for index in indices}
        else:
            tokens = list(rollout.get("tokens") or [])
            if len(tokens) != length:
                raise ValueError("uncertainty selection requires one top-k token row per response token")
            scored, details = [], {}
            for expected, token in enumerate(tokens):
                index = token.get("token_index")
                if index != expected:
                    raise ValueError("stored token_index does not align with response")
                candidates, mass = _top_mass(token.get("top_candidates") or [], top_mass, max_candidates)
                if not candidates:
                    continue
                score = (float(token.get("entropy", 0)) * entropy_weight
                         + (1 - float(token.get("top1_prob", 0))) * low_top1_weight
                         + float(expected >= max(0, length - final_window_tokens)) * final_window_weight)
                scored.append((score, expected))
                details[expected] = (candidates, mass)
        for _, index in scored:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < length:
                raise ValueError(f"token_index={index} outside rollout_id={rid}")
        selected = select_spaced_states(scored, states_per_response, min_token_gap)
        score_by_index = {index: score for score, index in scored}
        for rank, index in enumerate(selected):
            candidates, mass = details[index]
            rows.append({
                "state_id": f"{rid}:t{index}", "rollout_id": rid,
                "split": rollout.get("split"), "prompt_id": rollout.get("prompt_id"),
                "response_index": rollout.get("response_index"),
                "token_index": index, "selection_rank": rank,
                "selection_score": float(score_by_index[index]),
                "selection_strategy": strategy,
                "prompt_token_ids": list(rollout["prompt_token_ids"]),
                "prefix_response_token_ids": response[:index],
                "sampled_token_id": response[index],
                "candidates": candidates, "candidate_mass": mass,
                "actor_version": rollout.get("actor_version"),
                "gold_answer": rollout.get("gold_answer"),
            })
    if strategy == "indices" and set(indices_by_rollout) - seen:
        raise ValueError("indices_by_rollout references unknown rollouts")
    # Validate identity and prefix without requiring MC values yet.
    return rows
