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
"""Budget and identity contracts for matched frozen/online critic experiments."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from functools import lru_cache
from hashlib import sha256
from pathlib import Path


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def canonical_json(value):
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def stable_prompt_id(raw_prompt):
    """Hash prompt content, independent of runtime UUIDs and numpy containers."""
    return sha256(canonical_json(raw_prompt).encode()).hexdigest()


@lru_cache(maxsize=8)
def _partition(path):
    payload = json.loads(Path(path).read_text())
    all_ids = payload["all_prompt_ids"]
    diagnostic_ids = payload["diagnostic_prompt_ids"]
    if len(all_ids) != 4096 or len(set(all_ids)) != 4096:
        raise ValueError("Matched critic protocol requires exactly 4096 distinct prompt IDs")
    if len(diagnostic_ids) != 512 or len(set(diagnostic_ids)) != 512:
        raise ValueError("Matched critic protocol requires exactly 512 diagnostic prompt IDs")
    if diagnostic_ids != sorted(all_ids)[:512]:
        raise ValueError("Diagnostic partition must contain the lowest 512 stable prompt hashes")
    return frozenset(all_ids), frozenset(diagnostic_ids)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(canonical_json(payload) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def annotate_protocol_batch(batch, expansion, global_steps):
    """Attach stable roles before rollout; eligible prompt quotas are filled afterwards."""
    import torch
    from tensordict import NonTensorData, NonTensorStack

    protocol = expansion.get("protocol", {})
    if not protocol.get("enabled", False):
        raise ValueError("annotate_protocol_batch requires expansion.protocol.enabled")
    all_ids, diagnostic_ids = _partition(str(protocol["partition_path"]))
    prompts = batch["raw_prompt"]
    ids = []
    for index in range(len(batch)):
        prompt = prompts[index]
        if isinstance(prompt, NonTensorData):
            prompt = prompt.data
        ids.append(stable_prompt_id(prompt))
    if len(ids) != int(expansion["prompts_per_step"]):
        raise ValueError("Original batch size differs from the matched protocol")
    if len(ids) != len(set(ids)) or any(prompt_id not in all_ids for prompt_id in ids):
        raise ValueError("Original batch has duplicate prompts or a prompt outside the fixed partition")
    roles = ["diagnostic" if prompt_id in diagnostic_ids else "train" for prompt_id in ids]
    # The runtime batch UUID deliberately differs between independent runs.
    batch["delta_protocol_batch_id"] = NonTensorData(uuid.uuid4().hex)
    batch["delta_prompt_id"] = NonTensorStack(*[NonTensorData(value) for value in ids])
    batch["delta_critic_role"] = NonTensorStack(*[NonTensorData(value) for value in roles])
    batch["delta_protocol_index"] = torch.arange(len(ids), dtype=torch.int64)
    batch["delta_expand"] = torch.zeros(len(ids), dtype=torch.bool)
    cache_dir = protocol.get("original_cache_dir")
    if int(global_steps) == 1 and cache_dir:
        manifest_path = Path(cache_dir) / "first_batch_prompt_ids.json"
        manifest = {"prompt_ids": ids}
        mode = protocol.get("original_cache_mode", "read")
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Shared first-batch original prompt IDs/order differ")
        elif mode == "write":
            atomic_json(manifest_path, manifest)
        else:
            raise ValueError("Shared first-batch originals are not ready")
    return batch


def select_mc_state_indices(
    full_selected_indices, candidates, *, protocol, response_cap, prompt_length=0, total_cap=None, max_candidates=5
):
    """Select one state or an adjacent pair without changing the actor's complete grid."""
    indices = [int(index) for index in full_selected_indices]
    if indices != sorted(set(indices)):
        raise ValueError("Actor selected-state grid must be sorted and unique")
    reserve = int(protocol.get("min_continuation_room", 512))

    def eligible(index):
        return response_cap - index >= reserve and (total_cap is None or total_cap - prompt_length - index >= reserve)

    def uncertainty(index):
        probs = [float(item["prob"]) for item in candidates[index][:max_candidates]]
        if not probs or any(not math.isfinite(prob) or not 0 <= prob <= 1 for prob in probs):
            raise ValueError("MC state selection requires finite top-token probabilities")
        return 1.0 - max(probs)

    layout = protocol.get("sampling_layout", "pair")
    if layout == "pair":
        pairs = [
            (left, right)
            for left, right in zip(indices, indices[1:], strict=False)
            if eligible(left) and eligible(right)
        ]
        if not pairs:
            return []
        return list(max(pairs, key=lambda pair: (uncertainty(pair[0]), pair[0])))
    if layout == "single":
        eligible_indices = [index for index in indices if eligible(index)]
        return [max(eligible_indices, key=lambda index: (uncertainty(index), index))] if eligible_indices else []
    raise ValueError("sampling_layout must be pair or single")


def choose_expansion_prompts(entries, protocol, *, prompt_count, policy_step):
    """Fill role quotas from all generated originals, never adding or duplicating data."""
    if set(entries) != set(range(prompt_count)):
        raise ValueError("Expansion selection requires every original prompt exactly once")
    start = (int(policy_step) - 1) * 24 % prompt_count
    order = [(start + offset) % prompt_count for offset in range(prompt_count)]
    defaults = (10, 2) if protocol.get("sampling_layout", "pair") == "pair" else (20, 4)
    selected = set()
    for role, key, default in (
        ("train", "train_prompts_per_step", defaults[0]),
        ("diagnostic", "diagnostic_prompts_per_step", defaults[1]),
    ):
        quota = int(protocol.get(key, default))
        candidates = [index for index in order if entries[index]["role"] == role and entries[index]["eligible"]]
        if len(candidates) < quota:
            raise ValueError(f"MC quota failure: {role} has {len(candidates)} eligible originals, needs {quota}")
        selected.update(candidates[:quota])
    return selected


def first_original_cache_path(protocol, prompt_id, session_id):
    directory = protocol.get("original_cache_dir")
    return Path(directory) / f"{prompt_id}_session_{int(session_id)}.json" if directory else None


def original_cache_contract(config, sampling_params, prompt_id):
    """Sampling and model identity shared across arms; expansion settings are intentionally absent."""
    actor = config["actor_rollout_ref"]
    rollout = actor["rollout"]
    return {
        "format_version": 1,
        "prompt_id": prompt_id,
        "model_path": str(actor["model"]["path"]),
        "prompt_length": int(rollout["prompt_length"]),
        "response_length": int(rollout["response_length"]),
        "sampling_params": dict(sampling_params),
        "actor_version": 0,
    }
