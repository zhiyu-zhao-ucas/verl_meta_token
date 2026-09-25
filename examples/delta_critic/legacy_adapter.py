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
"""Import already loaded value_model JSONL rows without modifying their labels."""

from collections.abc import Iterable
from copy import deepcopy
from math import isclose
from typing import Any

from .contracts import DeltaExample, Rollout, SelectedState


def _required(row: dict[str, Any], key: str):
    if key not in row or row[key] is None:
        raise ValueError(f"Missing required field {key!r}")
    return row[key]


def adapt_legacy_rows(
    rollouts: Iterable[dict[str, Any]], mc_rows: Iterable[dict[str, Any]]
) -> tuple[list[DeltaExample], list[str]]:
    """Return sorted examples and compatibility diagnostics; preserve source metadata.

    Absent policy masks mean all original response tokens, as in value_model.
    Absent state IDs/versions/budgets remain unknown. No split is recomputed.
    """
    by_id = {}
    for row in rollouts:
        rollout_id = str(_required(row, "id"))
        if rollout_id in by_id:
            raise ValueError(f"Duplicate rollout_id={rollout_id}")
        response = tuple(_required(row, "response_token_ids"))
        by_id[rollout_id] = Rollout(
            rollout_id=rollout_id,
            prompt_token_ids=tuple(_required(row, "prompt_token_ids")),
            response_token_ids=response,
            terminal_reward=float(_required(row, "terminal_reward")),
            policy_token_mask=tuple(row.get("policy_token_mask", [1.0] * len(response))),
            prompt_id=str(row["prompt_id"]) if row.get("prompt_id") is not None else None,
            finish_reason=row.get("finish_reason"),
            metadata=deepcopy(row),
        )
    states = {key: [] for key in by_id}
    diagnostics = []
    state_ids = set()
    for row in mc_rows:
        rollout_id = str(_required(row, "rollout_id"))
        if rollout_id not in by_id:
            raise ValueError(f"Selected state references missing rollout_id={rollout_id}")
        rollout = by_id[rollout_id]
        state = SelectedState(
            rollout_id=rollout_id,
            token_index=_required(row, "token_index"),
            v_prefix=float(_required(row, "v_prefix")),
            v_next=float(row["v_next"]) if row.get("v_next") is not None else None,
            delta=float(row["delta"]) if row.get("delta") is not None else None,
            state_id=str(row["state_id"]) if row.get("state_id") is not None else None,
            mc_num_samples=row.get("mc_num_samples"),
            mc_next_num_samples=row.get("mc_next_num_samples"),
            metadata=deepcopy(row),
        )
        if state.state_id is not None:
            if state.state_id in state_ids:
                raise ValueError(f"Duplicate state_id={state.state_id}")
            state_ids.add(state.state_id)
        if "prompt_token_ids" in row and tuple(row["prompt_token_ids"]) != rollout.prompt_token_ids:
            raise ValueError(f"prompt_token_ids mismatch for rollout_id={rollout_id}")
        if row.get("prompt_id") is not None and str(row["prompt_id"]) != rollout.prompt_id:
            raise ValueError(f"prompt_id mismatch for rollout_id={rollout_id}")
        if "prefix_response_token_ids" in row:
            if tuple(row["prefix_response_token_ids"]) != rollout.response_token_ids[: state.token_index]:
                raise ValueError(f"prefix_response_token_ids mismatch at token_index={state.token_index}")
        if state.delta is not None and state.v_next is not None:
            if not isclose(state.delta, state.v_next - state.v_prefix, rel_tol=1e-9, abs_tol=1e-9):
                diagnostics.append(
                    f"rollout_id={rollout_id} token_index={state.token_index}: "
                    "delta differs from v_next-v_prefix; paired_next_state uses stored delta"
                )
        states[rollout_id].append(state)
    examples = [
        DeltaExample(rollout, tuple(sorted(states[key], key=lambda state: state.token_index)))
        for key, rollout in by_id.items()
    ]
    return examples, diagnostics


def terminal_after_last_token(
    finish_reason: str | None, *, treat_length_truncation_as_terminal: bool, assume_legacy_last_token_terminal: bool
) -> bool:
    """value_model's MC terminal shortcut, with both choices explicit.

    Existing label imports do not require this function or invent finish_reason.
    The source defaults were length=True, missing-finish-reason=False.
    """
    if finish_reason in {"stop", "eos"}:
        return True
    if finish_reason == "length":
        return treat_length_truncation_as_terminal
    if finish_reason is not None:
        return False
    if assume_legacy_last_token_terminal:
        return True
    raise ValueError("Missing finish_reason: choose assume_legacy_last_token_terminal explicitly")
