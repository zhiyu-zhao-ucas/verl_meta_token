# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strict adjacent-pair online labels; unobserved positions never become zeros."""

from dataclasses import replace
from math import isclose, isfinite

from .legacy_adapter import adapt_legacy_rows
from .training_data import continuation_rows


def paired_training_config(config, *, terminal_weight=0.003, state_count=64, min_gap=32, max_candidates=5):
    return replace(
        config,
        objective="hybrid_terminal_composition",
        loss_type="mse",
        loss_mask="selected_state",
        window_policy="full",
        td_weight=1.0,
        terminal_weight=terminal_weight,
        terminal_normalization="td",
        terminal_length_weighting="inverse_count",
        hybrid_reduction="separate_samples",
        continuation_selection="uncertainty",
        continuation_state_count=state_count,
        continuation_min_gap=min_gap,
        continuation_max_candidates=max_candidates,
        continuation_budget=None,
    )


def paired_rows(records, config, *, branches_per_state=8):
    """Build one local label and 2K terminal rows for each complete pair.

    Terminal targets use a leave-one-out baseline from the same fixed-K group.
    Record roles, full actor selection, state IDs and branch budgets are checked
    before any row can enter the optimizer buffer. No data is silently dropped.
    """
    result = {role: {"td": [], "terminal": []} for role in ("train", "diagnostic")}
    seen_ids = set()
    for record in records:
        labels = record.get("labels", [])
        if not labels:
            if record.get("states"):
                raise ValueError("Unlabelled online record contains selected MC states")
            continue
        role = record.get("critic_role", record["rollout"].get("critic_role"))
        if role not in result:
            raise ValueError("Paired online record requires critic_role=train or diagnostic")
        examples, diagnostics = adapt_legacy_rows([record["rollout"]], labels)
        if diagnostics:
            raise ValueError(f"Paired online label diagnostics: {diagnostics}")
        example = examples[0]
        rid = example.rollout.rollout_id
        if rid in seen_ids:
            raise ValueError(f"Duplicate paired rollout {rid}")
        seen_ids.add(rid)
        indices = [state.token_index for state in example.states]
        if len(indices) != 2 or indices != sorted(state["token_index"] for state in record["states"]):
            raise ValueError("Each online pair requires exactly two selected states and matching labels")
        full = record.get("full_selected_indices")
        if not isinstance(full, list | tuple) or list(full) != sorted(set(full)):
            raise ValueError("Paired records require the full sorted actor selected-state sequence")
        if any(index not in full for index in indices) or full.index(indices[1]) != full.index(indices[0]) + 1:
            raise ValueError("Online TD endpoints must be adjacent in the full actor selected-state sequence")
        if record.get("valid_td_indices", [indices[0]]) != [indices[0]]:
            raise ValueError("Only the first endpoint has a valid local TD target")
        response = list(example.rollout.response_token_ids)
        target = [0.0] * len(response)
        mask = [0.0] * len(response)
        target[indices[0]] = example.states[1].v_prefix - example.states[0].v_prefix
        mask[indices[0]] = 1.0
        common = {
            "prompt_stable_id": record.get("prompt_stable_id", example.rollout.prompt_id),
            "actor_version": record["rollout"].get("actor_version"),
        }
        result[role]["td"].append(
            {
                **common,
                "id": rid,
                "prompt_token_ids": list(example.rollout.prompt_token_ids),
                "response_token_ids": response,
                "token_targets": target,
                "token_loss_mask": mask,
                "token_signal_mask": mask.copy(),
                "selected_response_indices": [indices[0]],
                "selected_mc_values": [example.states[0].v_prefix],
            }
        )
        for label in labels:
            completions = label.get("mc_continuations", [])
            if len(completions) != branches_per_state or branches_per_state < 2:
                raise ValueError(f"Each online anchor must have exactly {branches_per_state} saved continuations")
            branch_ids = [completion["continuation_index"] for completion in completions]
            if sorted(branch_ids) != list(range(branches_per_state)):
                raise ValueError("Online continuation indices must cover the complete fixed-K group")
            rewards = [float(completion["reward"]) for completion in completions]
            if not all(isfinite(reward) for reward in rewards):
                raise ValueError("Online continuation rewards must be finite")
            mean = sum(rewards) / branches_per_state
            if not isclose(float(label["v_prefix"]), mean, abs_tol=1e-7, rel_tol=1e-7):
                raise ValueError("Anchor MC value does not equal the saved branch reward mean")
            normalized_label = {
                **label,
                "prompt_token_ids": list(example.rollout.prompt_token_ids),
                "prefix_response_token_ids": response[: label["token_index"]],
            }
            continuations = []
            for completion in completions:
                if not completion.get("token_ids"):
                    raise ValueError("Paired online training requires nonempty continuations")
                continuations.append(
                    {
                        "rollout_id": rid,
                        "state_id": label["state_id"],
                        "token_index": label["token_index"],
                        "continuation_index": completion["continuation_index"],
                        "continuation_id": f"{label['state_id']}:k{completion['continuation_index']}",
                        "continuation_token_ids": completion["token_ids"],
                        "delta_top_logprobs": completion.get("delta_top_logprobs"),
                        "reward": completion["reward"],
                    }
                )
            rows, counts = continuation_rows(continuations, [normalized_label], config)
            if counts["included"] != branches_per_state:
                raise ValueError("Online terminal row construction changed the fixed branch budget")
            by_id = {row["continuation_id"]: row for row in continuations}
            for row in rows:
                reward = float(by_id[row["id"]]["reward"])
                row["terminal_comp_target"] = reward - (sum(rewards) - reward) / (branches_per_state - 1)
                row["terminal_anchor_id"] = str(label["state_id"])
                row.update(common)
            result[role]["terminal"].extend(rows)
    return result
