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
"""MC batch health from saved rewards, without additional sampling or scoring."""

import math


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _summary(records):
    original_rewards, rewards, lengths, cap_hits, values, targets, gaps = [], [], [], [], [], [], []
    branch_counts, mixed, all_zero, all_positive, binary = [], [], [], [], []
    label_errors = []
    for record in records:
        rollout = record["rollout"]
        if rollout.get("terminal_reward") is not None:
            original_rewards.append(float(rollout["terminal_reward"]))
        labels = sorted(record.get("labels", []), key=lambda label: label["token_index"])
        for label in labels:
            values.append(float(label["v_prefix"]))
            completions = label.get("mc_continuations", [])
            # Old exports can retain rewards while omitting continuation text/tokens.
            branch_rewards = (
                [float(branch["reward"]) for branch in completions]
                if completions
                else [float(value) for value in label.get("prefix_continuation_rewards", [])]
            )
            if branch_rewards:
                rewards.extend(branch_rewards)
                branch_counts.append(len(branch_rewards))
                mixed.append(len(set(branch_rewards)) > 1)
                all_zero.append(all(reward == 0 for reward in branch_rewards))
                all_positive.append(all(reward > 0 for reward in branch_rewards))
                binary.extend(reward in (0.0, 1.0) for reward in branch_rewards)
                label_errors.append(abs(values[-1] - _mean(branch_rewards)))
            sampling = label.get("mc_sampling_config", {})
            caps = [
                int(sampling[key])
                for key in ("max_tokens", "effective_max_tokens", "effective_max_response_tokens")
                if sampling.get(key) is not None
            ]
            cap = min(caps) if caps else None
            for branch in completions:
                if branch.get("token_ids") is not None:
                    length = len(branch["token_ids"])
                    lengths.append(length)
                    if cap is not None:
                        cap_hits.append(length >= cap)
        # Only a pair adjacent on the full actor grid has an observed local TD
        # target. Never interpret an unobserved endpoint as a zero target.
        full = record.get("full_selected_indices", [])
        if len(labels) == 2:
            left, right = (label["token_index"] for label in labels)
            if left in full and right in full and full.index(right) == full.index(left) + 1:
                targets.append(float(labels[1]["v_prefix"]) - float(labels[0]["v_prefix"]))
                gaps.append(right - left)
    for population in (original_rewards, rewards, values, targets):
        if not all(math.isfinite(value) for value in population):
            raise ValueError("MC diagnostics require finite saved rewards and values")
    mean = _mean(targets)
    return {
        "original_rows": len(records),
        "original_reward_rows": len(original_rewards),
        "original_reward_mean": _mean(original_rewards),
        "original_positive_reward_fraction": _mean([reward > 0 for reward in original_rewards]),
        "states": len(values),
        "states_with_rewards": len(branch_counts),
        "branches": len(rewards),
        "branches_per_state_mean": _mean(branch_counts),
        "branch_reward_mean": _mean(rewards),
        "branch_positive_reward_fraction": _mean([reward > 0 for reward in rewards]),
        "branch_binary_reward_fraction": _mean(binary),
        "all_zero_reward_state_fraction": _mean(all_zero),
        "all_positive_reward_state_fraction": _mean(all_positive),
        "mixed_reward_state_fraction": _mean(mixed),
        "value_mean": _mean(values),
        "value_reward_mean_abs_error": _mean(label_errors),
        "branches_with_tokens": len(lengths),
        "branch_length_mean": _mean(lengths),
        "empty_branch_fraction": _mean([length == 0 for length in lengths]),
        "branches_with_known_cap": len(cap_hits),
        "branch_at_cap_fraction": _mean(cap_hits),
        "td_pairs": len(targets),
        "td_target_mean": mean,
        "td_target_std": math.sqrt(_mean([(value - mean) ** 2 for value in targets])),
        "td_target_abs_mean": _mean([abs(value) for value in targets]),
        "td_target_zero_fraction": _mean([value == 0 for value in targets]),
        "td_target_positive_fraction": _mean([value > 0 for value in targets]),
        "td_target_negative_fraction": _mean([value < 0 for value in targets]),
        "td_pair_gap_mean": _mean(gaps),
    }


def mc_batch_diagnostics(records):
    """Return all/train/diagnostic metrics with explicit population counts.

    Zero fractions with zero population are placeholders. A branch reaching its
    effective cap is not proof of truncation; an all-zero finite MC group is not
    proof that the underlying state's success probability is zero.
    """
    records = list(records)
    groups = {"all": records}
    for role in ("train", "diagnostic"):
        groups[role] = [
            record for record in records if record.get("critic_role", record["rollout"].get("critic_role")) == role
        ]
    return {f"delta_mc/{role}/{key}": value for role, rows in groups.items() for key, value in _summary(rows).items()}
