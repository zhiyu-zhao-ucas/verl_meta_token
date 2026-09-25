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
"""Export unpadded V1 AgentLoop/TransferQueue trajectories as delta rollouts."""

from collections.abc import Iterable, Mapping
from hashlib import sha256
from math import isfinite
from typing import Any

from .contracts import Rollout


def prompt_split(prompt_id: str, *, eval_fraction: float, seed: str) -> str:
    """Stable prompt-level split for new experiments; never rewrite legacy splits."""
    if not 0 <= eval_fraction < 1 or not isfinite(eval_fraction):
        raise ValueError("eval_fraction must be finite and in [0, 1)")
    if not prompt_id:
        raise ValueError("prompt_id is required for a new split")
    digest = sha256(f"{seed}\0{prompt_id}".encode()).digest()
    return "eval" if int.from_bytes(digest[:8], "big") / 2**64 < eval_fraction else "train"


def _list(value: Any, name: str) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a one-dimensional sequence")
    result = list(value)
    if any(isinstance(item, (list, tuple)) for item in result):
        raise ValueError(f"{name} must be a one-dimensional sequence")
    return result


def export_v1_rollouts(
    rows: Iterable[Mapping[str, Any]], *, actor_version: str, sampling_config: Mapping[str, Any],
    eval_fraction: float, split_seed: str,
) -> list[dict[str, Any]]:
    """Convert raw V1 TQ fields (prompts/responses/response_mask) without retokenizing.

    Rows must be unpadded, as emitted by AgentLoopOutput.as_dict before trainer
    collation. ``rollout_log_probs`` is the generation behavior logprob, not a
    subsequently recomputed old-policy or reference logprob.
    """
    if not actor_version or not sampling_config:
        raise ValueError("actor_version and sampling_config are required")
    exported = []
    seen = set()
    for row in rows:
        uid = str(row.get("uid") or "")
        prompt_id = str(row.get("prompt_id") or "")
        if not uid or not prompt_id:
            raise ValueError("V1 row requires uid and prompt_id")
        session = row.get("session_id", 0)
        output_index = row.get("output_index", 0)
        global_steps = row.get("global_steps")
        if row.get("rollout_id") is None and global_steps is None:
            raise ValueError("V1 row requires global_steps or an explicit rollout_id")
        rollout_id = str(row.get("rollout_id") or f"{actor_version}:{global_steps}:{uid}:{session}:{output_index}")
        if rollout_id in seen:
            raise ValueError(f"Duplicate rollout_id={rollout_id}")
        seen.add(rollout_id)
        prompt_ids = _list(row.get("prompts"), "prompts")
        response_ids = _list(row.get("responses"), "responses")
        mask = _list(row.get("response_mask"), "response_mask")
        if len(response_ids) != len(mask):
            raise ValueError("response_mask length mismatch")
        logprobs = row.get("rollout_log_probs")
        if logprobs is not None:
            logprobs = _list(logprobs, "rollout_log_probs")
            if len(logprobs) != len(response_ids) or any(not isfinite(float(x)) for x in logprobs):
                raise ValueError("rollout_log_probs must align with response and be finite")
        reward = row.get("reward_score")
        if reward is None:
            scores = row.get("rm_scores")
            if scores is not None:
                scores = _list(scores, "rm_scores")
                if len(scores) != len(response_ids) or not scores:
                    raise ValueError("rm_scores must align with nonempty response")
                reward = scores[-1]
        if reward is None:
            raise ValueError("V1 row requires reward_score or rm_scores")
        finish_reason = row.get("finish_reason")
        if finish_reason is None:
            finish_reason = (row.get("extra_fields") or {}).get("finish_reason")
        result = {
            "id": rollout_id,
            "prompt_id": prompt_id,
            "split": prompt_split(prompt_id, eval_fraction=eval_fraction, seed=split_seed),
            "prompt_token_ids": prompt_ids,
            "response_token_ids": response_ids,
            "policy_token_mask": mask,
            "terminal_reward": float(reward),
            "finish_reason": finish_reason,
            "behavior_logprobs": logprobs,
            "actor_version": actor_version,
            "sampling_config": dict(sampling_config),
            "source_uid": uid,
            "session_id": session,
            "response_index": session,
            "output_index": output_index,
            "global_steps": global_steps,
            "stop_reason": row.get("stop_reason"),
        }
        if row.get("prompt") is not None:
            result["prompt"] = row["prompt"]
        if row.get("gold_answer") is not None:
            result["gold_answer"] = row["gold_answer"]
        Rollout(
            rollout_id=rollout_id, prompt_token_ids=tuple(prompt_ids), response_token_ids=tuple(response_ids),
            terminal_reward=float(reward), policy_token_mask=tuple(mask), prompt_id=prompt_id,
            finish_reason=finish_reason,
        )
        exported.append(result)
    return exported
