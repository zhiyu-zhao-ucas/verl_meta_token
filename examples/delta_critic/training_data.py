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
"""Full-context scalar rows and source continuation matching; no silent truncation."""

import math
from collections import Counter

import torch
from torch.utils.data import Dataset

from .target_ops import critic_targets, selected_labels


def resolve_max_length(requested_max_length, backbone_config):
    """Resolve a run's context cap against the model's native positional limit.

    New configs should leave ``requested_max_length`` unset so the full native
    context is used. Explicit smaller caps remain useful for controlled legacy
    reproduction, but values above the backbone limit are always rejected.
    """
    native_limit = getattr(backbone_config, "max_position_embeddings", None)
    if isinstance(native_limit, bool) or not isinstance(native_limit, int) or native_limit < 1:
        raise ValueError("backbone config must define a positive max_position_embeddings")
    if requested_max_length is None:
        return native_limit
    if isinstance(requested_max_length, bool) or not isinstance(requested_max_length, int):
        raise ValueError("max_length must be a positive integer or None")
    if requested_max_length < 1:
        raise ValueError("max_length must be positive")
    if requested_max_length > native_limit:
        raise ValueError(
            f"Configured max_length={requested_max_length} exceeds backbone max_position_embeddings={native_limit}"
        )
    return requested_max_length


class ScalarCriticDataset(Dataset):
    """PyTorch dataset over unpadded critic rows.

    Rows retain response-indexed targets and masks until batch collation, where
    they are mapped to their causal sequence positions. Holdout splitting must
    happen before this dataset's training rows are passed to the statistics
    helpers below.
    """

    def __init__(self, rows):
        self.rows = list(rows)

    @classmethod
    def from_examples(cls, examples, config):
        return cls(ordinary_rows(examples, config))

    @classmethod
    def from_continuations(cls, continuations, labels, config, excluded_rollout_ids=()):
        rows, counts = continuation_rows(continuations, labels, config, excluded_rollout_ids)
        return cls(rows), counts

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def ordinary_rows(examples, config):
    rows = []
    for example in examples:
        target, signal = critic_targets(example, config.contract())
        rows.append(
            {
                "id": example.rollout.rollout_id,
                "prompt_token_ids": list(example.rollout.prompt_token_ids),
                "response_token_ids": list(example.rollout.response_token_ids),
                "token_targets": list(target.values),
                "token_loss_mask": list(target.mask),
                "token_signal_mask": list(signal),
                "selected_response_indices": [s.token_index for s in example.states],
                "selected_mc_values": [s.v_prefix for s in example.states],
                "selected_local_deltas": list(selected_labels(example, config.delta_label_mode)),
                "terminal_reward": example.rollout.terminal_reward,
            }
        )
    return rows


def continuation_rows(continuations, labels, config, excluded_rollout_ids=()):
    by_state = {str(r["state_id"]): r for r in labels if r.get("state_id") is not None}
    by_token = {
        (str(r["rollout_id"]), int(r["token_index"])): r
        for r in labels
        if r.get("rollout_id") is not None and r.get("token_index") is not None
    }
    excluded = set(map(str, excluded_rollout_ids))
    groups, counts, rows = {}, Counter(), []
    counts["input"] = len(continuations)
    for row in continuations:
        if str(row.get("rollout_id")) in excluded:
            counts["heldout"] += 1
            continue
        state = str(row.get("state_id"))
        if state in by_state:
            label, key = by_state[state], ("state", state)
        else:
            try:
                token_key = (str(row["rollout_id"]), int(row["token_index"]))
            except (KeyError, TypeError, ValueError):
                counts["unmatched"] += 1
                continue
            key = token_key
            label = by_token.get(key)
        if label is None:
            counts["unmatched"] += 1
            continue
        if str(label["rollout_id"]) in excluded:
            counts["heldout"] += 1
            continue
        counts["matched"] += 1
        groups.setdefault(key, []).append((row, label))
    for items in groups.values():
        items.sort(key=lambda pair: int(pair[0].get("continuation_index", len(rows))))
        counts["eligible"] += sum(bool(row.get("continuation_token_ids")) for row, _ in items)
        counts["empty"] += sum(not bool(row.get("continuation_token_ids")) for row, _ in items)
        if config.continuation_budget is not None:
            counts["over_budget"] += sum(
                bool(row.get("continuation_token_ids")) for row, _ in items[config.continuation_budget :]
            )
            items = items[: config.continuation_budget]
        for row, label in items:
            response = list(row.get("continuation_token_ids") or [])
            if not response:
                continue
            prompt = list(label.get("prompt_token_ids") or row.get("prompt_token_ids") or [])
            prefix = list(label.get("prefix_response_token_ids") or row.get("prefix_response_token_ids") or [])
            if not prompt:
                raise ValueError("Continuation requires original prompt token IDs")
            step = max(1, config.continuation_min_gap, math.ceil(len(response) / config.continuation_state_count))
            selected = list(range(0, len(response), step))[: config.continuation_state_count]
            rows.append(
                {
                    "id": row.get("continuation_id") or f"{label.get('state_id')}:{len(rows)}",
                    "prompt_token_ids": prompt + prefix,
                    "response_token_ids": response,
                    "token_targets": [0.0] * len(response),
                    "token_loss_mask": [0.0] * len(response),
                    "token_signal_mask": [0.0] * len(response),
                    "terminal_suffix_response_indices": selected,
                    "terminal_comp_target": float(row["reward"]) - float(label["v_prefix"]),
                    "terminal_comp_valid": True,
                }
            )
    counts["included"] = len(rows)
    if config.continuation_budget is None and counts["included"] != counts["eligible"]:
        raise RuntimeError(
            "Unlimited hybrid continuation budget must include every matched nonempty continuation: "
            f"eligible={counts['eligible']} included={counts['included']}"
        )
    return rows, dict(counts)


def scan_lengths(rows, config):
    if config.max_length < 1:
        raise ValueError("max_length must be resolved to a positive integer before scanning rows")
    maxima = {"rollout": 0, "selected_prefix": 0, "hybrid": 0}
    for row in rows:
        p, r = len(row["prompt_token_ids"]), len(row["response_token_ids"])
        kind = "hybrid" if row.get("terminal_comp_valid") else "rollout"
        maxima[kind] = max(maxima[kind], p + r)
        for index in row.get("selected_response_indices", []):
            if not 0 <= index < r:
                raise ValueError(f"Invalid selected position in {row['id']}")
            maxima["selected_prefix"] = max(maxima["selected_prefix"], p + index + 1)
        if config.window_policy == "full" and p + r > config.max_length:
            raise ValueError(f"{kind} {row['id']}: length {p + r} exceeds {config.max_length}; truncation forbidden")
    return maxima


def fit_stats(rows, *, terminal=False):
    values = []
    for row in rows:
        if terminal:
            if row.get("terminal_comp_valid"):
                values.append(float(row["terminal_comp_target"]))
        else:
            values.extend(float(v) for v, m in zip(row["token_targets"], row["token_loss_mask"], strict=True) if m)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Statistics require a nonempty finite training population")
    mean = sum(values) / len(values)
    std = math.sqrt(max(0, sum(v * v for v in values) / len(values) - mean * mean))
    if std < 1e-8:
        if not terminal:
            raise ValueError("Target normalization std must be >= 1e-8")
        std = 1.0
    return {"count": len(values), "mean": mean, "std": std}


def normalization_metadata(rows, config):
    if config.target_normalization == "none":
        return {"enabled": False, "mode": "none", "mean": None, "std": None}
    return {"enabled": True, "mode": "standardize", "stats_source": "train_split_after_holdout", **fit_stats(rows)}


def collate(rows, config, normalization=None, pad_token_id=0):
    if not rows:
        raise ValueError("Cannot collate an empty critic batch")
    scan_lengths(rows, config)
    width = max(len(r["prompt_token_ids"]) + len(r["response_token_ids"]) for r in rows)
    if config.window_policy == "legacy_tail":
        width = min(width, config.max_length)
    output = {
        k: []
        for k in (
            "input_ids",
            "attention_mask",
            "target",
            "target_mask",
            "signal_mask",
        )
    }
    terminal_entries = []
    for row in rows:
        prompt, response = row["prompt_token_ids"], row["response_token_ids"]
        full = prompt + response
        if not prompt or not full or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in full):
            raise ValueError("Invalid token IDs or empty prompt")
        p, n = len(prompt), len(full)
        shift = max(0, n - width)
        ids = full[shift:]
        output["input_ids"].append(ids + [pad_token_id] * (width - len(ids)))
        output["attention_mask"].append([1] * len(ids) + [0] * (width - len(ids)))
        for dest, source in (
            ("target", "token_targets"),
            ("target_mask", "token_loss_mask"),
            ("signal_mask", "token_signal_mask"),
        ):
            values = list(row[source])
            if len(values) != len(response):
                raise ValueError(f"{source} length mismatch")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{source} must contain finite values")
            if dest != "target" and any(float(value) not in (0.0, 1.0) for value in values):
                raise ValueError(f"{source} must be a binary mask")
            if dest == "target" and normalization is not None and normalization.get("enabled", False):
                values = [(v - normalization["mean"]) / normalization["std"] for v in values]
            values = ([0.0] * p + list(values))[shift:]
            output[dest].append(values + [0.0] * (width - len(values)))
        valid = bool(row.get("terminal_comp_valid", False))
        selected = list(row.get("terminal_suffix_response_indices", []))
        suffix_positions = []
        for index in selected:
            if not 0 <= index < len(response):
                raise ValueError("Invalid terminal suffix index")
            position = p + index - shift
            if position < 0:
                valid = False  # exact legacy collator behavior
            else:
                suffix_positions.append(position)
        target = float(row.get("terminal_comp_target", 0.0))
        if not math.isfinite(target):
            raise ValueError("terminal_comp_target must be finite")
        terminal_entries.append(
            {"positions": suffix_positions, "target": target, "valid": bool(valid and bool(selected))}
        )
    batch = {
        k: torch.tensor(v, dtype=torch.long if k in {"input_ids", "attention_mask"} else torch.float32)
        for k, v in output.items()
    }
    max_suffix = max((len(row["positions"]) for row in terminal_entries), default=0)
    suffix_positions, suffix_masks = [], []
    for row in terminal_entries:
        pad = max_suffix - len(row["positions"])
        suffix_positions.append(row["positions"] + [-1] * pad)
        suffix_masks.append([1.0] * len(row["positions"]) + [0.0] * pad)
    batch["terminal_suffix_positions"] = torch.tensor(suffix_positions, dtype=torch.long)
    batch["terminal_suffix_mask"] = torch.tensor(suffix_masks, dtype=torch.float32)
    batch["terminal_comp_target"] = torch.tensor([row["target"] for row in terminal_entries], dtype=torch.float32)
    batch["terminal_comp_valid"] = torch.tensor([float(row["valid"]) for row in terminal_entries], dtype=torch.float32)
    sample_valid = [float(row.get("sample_valid", 1.0)) for row in rows]
    if any(not math.isfinite(value) or value not in (0.0, 1.0) for value in sample_valid):
        raise ValueError("sample_valid must be a binary per-row value")
    batch["sample_valid_mask"] = torch.tensor(sample_valid, dtype=torch.float32)
    return batch


def training_batch(rows, config, normalization, terminal_stats, pad_token_id):
    """Dense right padding is inside our engine; V1 scheduling sees jagged IDs."""
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    batch = collate(rows, config, normalization, pad_token_id)
    lengths = batch["attention_mask"].sum(-1).tolist()
    ids = [ids[:length] for ids, length in zip(batch.pop("input_ids"), lengths, strict=True)]
    batch["input_ids"] = torch.nested.as_nested_tensor(ids, layout=torch.jagged)
    batch["loss_mask"] = batch["target_mask"]
    td = TensorDict(batch, batch_size=[len(rows)])
    tu.assign_non_tensor(
        td,
        delta_config=config.as_dict(),
        normalization=normalization,
        terminal_stats=terminal_stats,
        global_token_num=lengths,
        pad_token_id=pad_token_id,
        update_lr_scheduler=True,
        use_dynamic_bsz=False,
        return_model_output=False,
    )
    return td
