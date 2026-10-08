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
"""Selected-prefix MC collection on the synchronous online actor server."""

import json
from collections.abc import Mapping
from functools import partial
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from .mc_labeling import MCConfig, label_mc_states_async
from .sampling_io import score_text, score_texts_async
from .state_selection import select_states
from .v1_sampler import sample_v1_continuations


def online_mc_config(policy, rollout, actor_version):
    """Keep continuation sampling explicit and independent of the answer cap."""
    mode = policy.get("rollout_mode", "critic_prediction")
    if mode not in {"critic_prediction", "selected_prefix_mc"}:
        raise ValueError("rollout_mode must be critic_prediction or selected_prefix_mc")
    if mode == "critic_prediction":
        return None
    mc = policy.get("mc", {})
    budget = mc.get("continuations_per_state", 16)
    concurrency = mc.get("concurrency", 16)
    max_tokens = mc.get("max_continuation_tokens")
    max_total_tokens = mc.get("max_total_tokens")
    max_response_total_tokens = mc.get("max_response_total_tokens")
    for name, value in (("concurrency", concurrency), ("max_continuation_tokens", max_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"mc.{name} must be an explicit positive integer")
    sampling = {
        "temperature": mc.get("temperature", rollout["temperature"]),
        "top_p": mc.get("top_p", rollout["top_p"]),
        "top_k": mc.get("top_k", rollout["top_k"]),
        "min_p": mc.get("min_p", 0.0),
        "max_tokens": max_tokens,
    }
    selection = policy.get("selection", {})
    save_continuations = mc.get("save_continuations", True)
    top_k_logprobs = selection.get("top_k_logprobs")
    if policy.get("expansion", {}).get("enabled", False) or (save_continuations and top_k_logprobs is not None):
        top_k_logprobs = int(top_k_logprobs if top_k_logprobs is not None else 20)
        if top_k_logprobs < 1:
            raise ValueError("selection.top_k_logprobs must be positive when continuation metadata is requested")
        sampling["delta_top_logprobs"] = top_k_logprobs
    if policy.get("label_mode", "selected_segment") != "selected_segment":
        raise ValueError("selected_prefix_mc requires label_mode=selected_segment")
    return MCConfig(
        mode="prefix_only",
        continuations_per_state=budget,
        sampling_config=sampling,
        actor_version=str(actor_version),
        continuation_skip_special_tokens=True,
        save_individual_rewards=True,
        save_continuations=save_continuations,
        max_total_tokens=max_total_tokens,
        max_response_total_tokens=max_response_total_tokens,
        require_nonempty=bool(policy.get("expansion", {}).get("enabled", False)),
    )


class _VersionedClient:
    def __init__(self, client, version):
        self.client, self.version = client, version

    async def generate(self, **kwargs):
        output = await self.client.generate(**kwargs)
        metadata = output.extra_fields or {}
        for name in ("global_steps", "min_global_steps", "max_global_steps"):
            if metadata.get(name, metadata.get("global_steps")) != self.version:
                raise ValueError("MC continuation actor version differs from its original rollout")
        return output


async def collect_online_mc(
    output,
    selected_indices,
    *,
    policy,
    rollout_config,
    server_client,
    tokenizer,
    uid,
    session_id,
    sample_kwargs,
    request_semaphore,
):
    """Reuse collector contracts without retokenizing the selected prefixes."""
    version = output.extra_fields["global_steps"]
    config = online_mc_config(policy, rollout_config, version)
    reward_model = sample_kwargs.get("reward_model")
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
        raise ValueError("selected_prefix_mc requires reward_model.ground_truth")
    if any(value != 1 for value in output.response_mask):
        raise ValueError("selected_prefix_mc requires a single-turn generated response")
    reward_config = dict(policy.get("mc", {}).get("reward", {}))
    score_batch = partial(score_texts_async, reward_config=reward_config)
    terminal_reward = (
        await score_batch(
            [tokenizer.decode(output.response_ids, skip_special_tokens=True)],
            {"gold_answer": reward_model["ground_truth"]},
        )
    )[0]
    rid = f"{version}:{uid}:{session_id}"
    rollout = {
        "id": rid,
        "prompt_id": sha256(json.dumps(list(output.prompt_ids)).encode()).hexdigest(),
        "split": "train",
        "prompt_token_ids": list(output.prompt_ids),
        "response_token_ids": list(output.response_ids),
        "policy_token_mask": list(output.response_mask),
        "terminal_reward": terminal_reward,
        "trainer_reward": output.reward_score,
        "reward_config": reward_config,
        "finish_reason": output.extra_fields.get("finish_reason"),
        "actor_version": str(version),
        "gold_answer": reward_model["ground_truth"],
        "response_index": session_id,
        "sampling_config": {key: rollout_config[key] for key in ("temperature", "top_p", "top_k")},
    }
    states = select_states(
        [rollout],
        strategy="indices",
        states_per_response=len(selected_indices),
        indices_by_rollout={rid: list(selected_indices)},
    )
    labels = await label_mc_states_async(
        [rollout],
        states,
        config,
        sample=partial(
            sample_v1_continuations,
            server_client=_VersionedClient(server_client, version),
            tokenizer=tokenizer,
            request_semaphore=request_semaphore,
        ),
        decode_prefix=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        score=lambda text, row: score_text(text, row["gold_answer"], reward_config),
        request_concurrency=policy.get("mc", {}).get("concurrency", 16),
        score_batch_async=score_batch,
    )
    return {"rollout": rollout, "states": states, "labels": labels}


def mc_record_rows(value):
    """TensorDict unwraps dict-valued NonTensorStacks to iterable LinkedLists."""
    return value.tolist() if hasattr(value, "tolist") else list(value)


def export_mc_records(records, directory):
    """Write the same reusable JSONL contracts as the original critic collector."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {"rollouts_train_regen": [], "states_train": [], "mc_labels_train": [], "continuations_train": []}
    for record in records:
        tables["rollouts_train_regen"].append(record["rollout"])
        tables["states_train"].extend(record["states"])
        for original in record["labels"]:
            label = dict(original)
            for completion in label.pop("mc_continuations", []):
                tokens = completion["token_ids"]
                index = completion["continuation_index"]
                tables["continuations_train"].append(
                    {
                        "state_id": label["state_id"],
                        "rollout_id": label["rollout_id"],
                        "prompt_id": label["prompt_id"],
                        "split": label["split"],
                        "response_index": record["rollout"]["response_index"],
                        "token_index": label["token_index"],
                        "continuation_index": index,
                        "continuation_id": f"{label['state_id']}:k{index}",
                        "continuation_token_ids": tokens,
                        "continuation_text": completion["text"],
                        "continuation_num_tokens": len(tokens),
                        "empty_continuation": not tokens,
                        "reward": completion["reward"],
                        "actor_version": label["mc_actor_version"],
                        "sampling_config": label["mc_sampling_config"],
                        **(
                            {"delta_top_logprobs": completion["delta_top_logprobs"]}
                            if completion.get("delta_top_logprobs") is not None
                            else {}
                        ),
                    }
                )
            tables["mc_labels_train"].append(label)
    for name, rows in tables.items():
        path = directory / f"{name}.jsonl"
        contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if path.exists():
            if path.read_text() != contents:
                raise ValueError(f"Refusing to overwrite different MC data: {path}")
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(contents)
            temporary.replace(path)


def export_mc_attempt(records, directory):
    """Atomically publish one sampling attempt without blocking checkpoint retries.

    A repeated step may have different UUIDs/tokens after restoring an earlier
    checkpoint. Keep both batches; identical records reuse the same directory.
    Legacy step-level files are preserved and never interpreted as a checkpoint.
    """
    records = list(records)
    digest = sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"attempt_{digest}"
    if destination.exists():
        export_mc_records(records, destination)  # Verify the immutable batch.
        return destination
    with TemporaryDirectory(prefix=".mc-", dir=directory) as temporary:
        batch = Path(temporary) / "batch"
        export_mc_records(records, batch)
        try:
            batch.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            export_mc_records(records, destination)
    return destination
