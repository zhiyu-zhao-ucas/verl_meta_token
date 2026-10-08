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
"""Batch rollout -> uncertainty states -> MC labels on verl's Ray V1 server."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

from .legacy_adapter import adapt_legacy_rows
from .mc_labeling import MCConfig, label_mc_states_async
from .sampling_io import load_prompt_rows, score_text, score_texts_async
from .state_selection import select_states
from .v1_sampler import sample_v1_continuations


def sampling_settings(config: dict, mode: str) -> tuple[dict, MCConfig]:
    rollout = config["rollout"]
    selection = config["state_selection"]
    mc = config["mc"]
    if int(config["smoke"]["states_per_response"]) <= 0:
        raise ValueError("states_per_response must be positive")
    if int(rollout["top_k_logprobs"]) <= 0:
        raise ValueError("top_k_logprobs must be positive for uncertainty selection")
    generation = {
        "temperature": float(rollout["temperature"]),
        "top_p": float(rollout["top_p"]),
        "max_tokens": int(rollout["max_tokens"]),
        "delta_top_logprobs": int(rollout["top_k_logprobs"]),
    }
    if rollout.get("top_k") is not None:
        generation["top_k"] = int(rollout["top_k"])
    for key in ("repetition_penalty", "presence_penalty", "frequency_penalty", "min_p"):
        if rollout.get(key) is not None:
            generation[key] = float(rollout[key])
    if rollout.get("stop"):
        generation["stop"] = list(rollout["stop"])
    if rollout.get("include_stop_str_in_output") is not None:
        generation["include_stop_str_in_output"] = bool(rollout["include_stop_str_in_output"])
    mc_sampling = {
        "temperature": float(mc["temperature"]),
        "top_p": float(mc["top_p"]),
        "max_tokens": int(mc["max_continuation_tokens"]),
    }
    save_continuations = bool(mc.get("save_continuations", False))
    if save_continuations and rollout.get("top_k_logprobs") is not None:
        mc_sampling["delta_top_logprobs"] = int(rollout["top_k_logprobs"])
    for key in ("top_k", "min_p"):
        if mc.get(key) is not None:
            mc_sampling[key] = int(mc[key]) if key == "top_k" else float(mc[key])
    mc_config = MCConfig(
        mode=mode,
        continuations_per_state=int(mc["continuations_per_state"]),
        sampling_config=mc_sampling,
        actor_version=str(config["model"]["actor_model"]),
        continuation_skip_special_tokens=True,
        treat_length_truncation_as_terminal=bool(mc.get("treat_length_truncation_as_terminal", True)),
        assume_legacy_last_token_terminal=False,
        save_individual_rewards=bool(mc.get("save_individual_rewards", False)),
        save_continuations=save_continuations,
    )
    if int(selection["max_candidates"]) > int(rollout["top_k_logprobs"]):
        raise ValueError("max_candidates cannot exceed top_k_logprobs")
    return generation, mc_config


def rollout_from_output(
    prompt: dict,
    prompt_ids: list[int],
    output,
    *,
    tokenizer,
    split: str,
    response_index: int,
    actor_version: str,
    sampling: dict,
) -> dict:
    token_ids = list(output.token_ids)
    if not token_ids:
        raise ValueError(f"Empty rollout for prompt_id={prompt['prompt_id']}")
    top_rows = (output.extra_fields or {}).get("delta_top_logprobs")
    if top_rows is None or len(top_rows) != len(token_ids):
        raise ValueError("V1 backend must return one delta_top_logprobs row per response token")
    if output.log_probs is None or len(output.log_probs) != len(token_ids):
        raise ValueError("V1 backend must return sampled-token logprobs")
    finish_reason = (output.extra_fields or {}).get("finish_reason")
    if finish_reason is None:
        raise ValueError("V1 backend must return raw finish_reason")
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    tokens = []
    for index, (token_id, candidates, sampled_lp) in enumerate(zip(token_ids, top_rows, output.log_probs, strict=True)):
        candidates = [candidate for candidate in candidates if math.isfinite(float(candidate["logprob"]))]
        if not candidates:
            raise ValueError("Every generated token needs top-k candidates")
        if not math.isfinite(float(sampled_lp)):
            raise ValueError(f"Nonfinite sampled logprob at token_index={index}")
        tokens.append(
            {
                "token_index": index,
                "token_id": token_id,
                "sampled_logprob": float(sampled_lp),
                "top1_prob": float(candidates[0]["prob"]),
                "entropy": -sum(float(item["prob"]) * math.log(max(float(item["prob"]), 1e-12)) for item in candidates),
                "top_candidates": candidates,
            }
        )
    row = {
        "id": f"{split}-{prompt['prompt_id']}-{response_index}",
        "split": split,
        "prompt_id": prompt["prompt_id"],
        "prompt": prompt["prompt"],
        "gold_answer": prompt["gold_answer"],
        "response_index": response_index,
        "response": text,
        "finish_reason": finish_reason,
        "stop_reason": output.stop_reason,
        "terminal_reward": score_text(text, prompt["gold_answer"], sampling["reward"]),
        "prompt_token_ids": prompt_ids,
        "response_token_ids": token_ids,
        "tokens": tokens,
        "behavior_logprobs": list(output.log_probs),
        "policy_token_mask": [1] * len(token_ids),
        "actor_version": actor_version,
        "sampling_config": {key: value for key, value in sampling.items() if key != "reward"},
    }
    if "raw_prompt" in prompt:
        row["raw_prompt"] = prompt["raw_prompt"]
        row["data_source"] = prompt["data_source"]
    return row


def _write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _read_partial_rows(path: Path) -> list[dict]:
    """Read a journal, tolerating only a truncated final line from interruption."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def _rewrite_rows(path: Path, rows) -> None:
    replacement = path.with_name(path.name + ".recovering")
    with replacement.open("w", encoding="utf-8") as stream:
        for row in rows:
            _write_row(stream, row)
        stream.flush()
        os.fsync(stream.fileno())
    replacement.replace(path)


def _recover_partial_outputs(
    temporary: dict[str, Path], *, continuations_per_state: int, save_continuations: bool, mc_mode: str
) -> tuple[dict, dict]:
    """Canonicalize committed state journals and build rollout resume records.

    A label is the commit record for one state. Continuations without a matching
    label can be left by an interrupted write and are deliberately discarded.
    """
    rollouts = {str(row["id"]): row for row in _read_partial_rows(temporary["rollouts"])}
    states = _read_partial_rows(temporary["states"])
    labels = _read_partial_rows(temporary["mc_labels"])
    continuations = _read_partial_rows(temporary["continuations"])
    states_by_rollout = {}
    for state in states:
        states_by_rollout.setdefault(str(state["rollout_id"]), {})[str(state["state_id"])] = state
    valid_rollouts = {
        rollout_id: rollout for rollout_id, rollout in rollouts.items() if states_by_rollout.get(rollout_id)
    }
    valid_state_ids = {state_id for rollout_id in valid_rollouts for state_id in states_by_rollout[rollout_id]}
    continuations_by_state = {}
    for row in continuations:
        continuations_by_state.setdefault(str(row["state_id"]), {})[int(row["continuation_index"])] = row
    labels_by_state = {}
    for label in labels:
        state_id = str(label["state_id"])
        continuation_indices = set(continuations_by_state.get(state_id, {}))
        expected_indices = set(range(continuations_per_state))
        mode_matches = (mc_mode == "paired_next_state") == ("v_next" in label and "delta" in label)
        if (
            state_id in valid_state_ids
            and mode_matches
            and label.get("mc_num_samples") == continuations_per_state
            and (not save_continuations or continuation_indices == expected_indices)
        ):
            labels_by_state[state_id] = label

    if mc_mode == "paired_next_state":
        # A partially committed paired rollout may share a sampled prefix between
        # adjacent labels. Resample the entire label set to preserve v_next/v_prefix identity.
        for rollout_id in valid_rollouts:
            rollout_state_ids = set(states_by_rollout[rollout_id])
            if not rollout_state_ids.issubset(labels_by_state):
                for state_id in rollout_state_ids:
                    labels_by_state.pop(state_id, None)

    rollout_rows = list(valid_rollouts.values())
    state_rows = [state for rollout_id in valid_rollouts for state in states_by_rollout[rollout_id].values()]
    label_rows = [labels_by_state[state_id] for state_id in valid_state_ids if state_id in labels_by_state]
    continuation_rows = (
        [
            continuations_by_state[state_id][index]
            for state_id in valid_state_ids
            if state_id in labels_by_state
            for index in sorted(continuations_by_state.get(state_id, {}))
        ]
        if save_continuations
        else []
    )
    canonical = {
        "rollouts": rollout_rows,
        "states": state_rows,
        "mc_labels": label_rows,
        "continuations": continuation_rows,
    }
    for name, rows in canonical.items():
        _rewrite_rows(temporary[name], rows)
    resume = {}
    for rollout_id, rollout in valid_rollouts.items():
        rollout_states = list(states_by_rollout[rollout_id].values())
        resume[rollout_id] = {
            "rollout": rollout,
            "states": rollout_states,
            "labels": [
                labels_by_state[str(state["state_id"])]
                for state in rollout_states
                if str(state["state_id"]) in labels_by_state
            ],
        }
    counts = {
        "rollouts": len(rollout_rows),
        "states": len(state_rows),
        "mc_labels": len(label_rows),
        "continuations": len(continuation_rows),
    }
    return resume, counts


async def sample_labeled_rollout(
    prompt: dict,
    prompt_ids: list[int],
    output,
    *,
    tokenizer,
    client,
    config: dict,
    split: str,
    response_index: int,
    actor_version: str,
    generation: dict,
    mc_config: MCConfig,
    mc_concurrency: int,
    mc_request_semaphore: asyncio.Semaphore | None = None,
    score_semaphore: asyncio.Semaphore | None = None,
    mc_native_n_batch_size: int | None = None,
    resumed_rollout: dict | None = None,
    resumed_states: list[dict] | None = None,
    completed_labels: list[dict] | None = None,
    emit_rollout=None,
    emit_label=None,
) -> tuple[dict, list[dict], list[dict]]:
    if resumed_rollout is None:
        rollout = rollout_from_output(
            prompt,
            prompt_ids,
            output,
            tokenizer=tokenizer,
            split=split,
            response_index=response_index,
            actor_version=actor_version,
            sampling={**generation, "reward": config.get("reward") or {}},
        )
        selection = config["state_selection"]
        states = select_states(
            [rollout],
            strategy="uncertainty",
            states_per_response=int(config["smoke"]["states_per_response"]),
            min_token_gap=int(selection.get("min_token_gap", 0)),
            top_mass=float(config["smoke"]["top_mass"]),
            max_candidates=int(selection["max_candidates"]),
            entropy_weight=float(selection["entropy_weight"]),
            low_top1_weight=float(selection["low_top1_weight"]),
            final_window_tokens=int(selection["final_window_tokens"]),
            final_window_weight=float(selection["final_window_weight"]),
        )
        if emit_rollout is not None:
            emit_rollout(rollout, states)
    else:
        rollout = resumed_rollout
        states = list(resumed_states or [])
        if not states:
            raise ValueError(f"Resumed rollout {rollout['id']} has no states")
    print(json.dumps({"event": "mc_start", "prompt_id": prompt["prompt_id"], "states": len(states)}), flush=True)
    labels_by_state = {str(label["state_id"]): label for label in completed_labels or []}
    pending_states = [state for state in states if str(state["state_id"]) not in labels_by_state]

    def on_label(label):
        if emit_label is not None:
            emit_label(rollout, label)
        labels_by_state[str(label["state_id"])] = label

    if pending_states:
        new_labels = await label_mc_states_async(
            [rollout],
            pending_states,
            mc_config,
            sample=partial(
                sample_v1_continuations,
                server_client=client,
                tokenizer=tokenizer,
                request_semaphore=mc_request_semaphore,
                native_n_batch_size=mc_native_n_batch_size,
            ),
            decode_prefix=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
            score=lambda response, row: score_text(response, row["gold_answer"], config.get("reward") or {}),
            request_concurrency=mc_concurrency,
            score_batch_async=partial(score_texts_async, reward_config=config.get("reward") or {}),
            score_semaphore=score_semaphore,
            on_label=on_label,
        )
        for label in new_labels:
            labels_by_state[str(label["state_id"])] = label
    labels = [labels_by_state[str(state["state_id"])] for state in states]
    _, diagnostics = adapt_legacy_rows([rollout], labels)
    if diagnostics:
        raise ValueError(f"MC label diagnostics: {diagnostics}")
    print(json.dumps({"event": "mc_complete", "prompt_id": prompt["prompt_id"], "states": len(states)}), flush=True)
    return rollout, states, labels


async def collect_labeled_rollouts(
    prompts: list[dict],
    *,
    tokenizer,
    client,
    config: dict,
    split: str,
    actor_version: str,
    generation: dict,
    mc_config: MCConfig,
    rollout_concurrency: int,
    mc_concurrency: int,
    emit,
    mc_sampling_mode: str = "native",
    mc_native_batch_size: int | None = None,
    resume: dict | None = None,
    emit_rollout=None,
    emit_label=None,
) -> None:
    """Stream rollout outputs into MC workers with a global sequence budget."""
    response_count = int(config["smoke"]["responses_per_prompt"])
    total_jobs = len(prompts) * response_count
    if total_jobs == 0:
        return
    if mc_sampling_mode not in {"native", "requests"}:
        raise ValueError(f"Unknown MC sampling mode: {mc_sampling_mode}")
    if mc_native_batch_size is not None and not 1 <= mc_native_batch_size <= mc_concurrency:
        raise ValueError("mc_native_batch_size must be between 1 and mc_concurrency")
    mc_native_n_batch_size = (
        min(mc_native_batch_size or mc_concurrency, mc_config.continuations_per_state)
        if mc_sampling_mode == "native"
        else None
    )
    if mc_config.save_continuations and mc_config.sampling_config.get("delta_top_logprobs") is not None:
        # Native n-sample responses expose token IDs only; individual requests
        # are needed to retain one actor top-k row per continuation token.
        mc_native_n_batch_size = None
    mc_request_limit = (
        max(1, mc_concurrency // mc_native_n_batch_size) if mc_native_n_batch_size is not None else mc_concurrency
    )
    mc_request_semaphore = asyncio.Semaphore(mc_request_limit)
    score_semaphore = (
        asyncio.Semaphore(min(4, os.cpu_count() or 1))
        if (config.get("reward") or {}).get("method", "math_verify") == "math_verify"
        else None
    )
    generated = asyncio.Queue(maxsize=rollout_concurrency)

    def jobs():
        for prompt in prompts:
            prompt_ids = prompt.get("prompt_token_ids")
            if prompt_ids is None:
                prompt_ids = tokenizer.encode(prompt["prompt"], add_special_tokens=False)
            if not prompt_ids:
                raise ValueError(f"Empty prompt tokenization for {prompt['prompt_id']}")
            for response_index in range(response_count):
                yield prompt, prompt_ids, response_index

    pending_jobs = iter(jobs())

    async def rollout_worker():
        while True:
            job = next(pending_jobs, None)
            if job is None:
                return
            prompt, prompt_ids, response_index = job
            rollout_id = f"{split}-{prompt['prompt_id']}-{response_index}"
            resumed = (resume or {}).get(rollout_id)
            if resumed is None:
                output = await client.generate(
                    request_id=f"delta-rollout:{split}:{prompt['prompt_id']}:{response_index}",
                    prompt_ids=prompt_ids,
                    sampling_params=dict(generation),
                )
                await generated.put((prompt, prompt_ids, response_index, output, None))
            else:
                await generated.put((prompt, prompt_ids, response_index, None, resumed))

    async def mc_worker():
        while True:
            job = await generated.get()
            if job is None:
                return
            prompt, prompt_ids, response_index, output, resumed = job
            result = await sample_labeled_rollout(
                prompt,
                prompt_ids,
                output,
                tokenizer=tokenizer,
                client=client,
                config=config,
                split=split,
                response_index=response_index,
                actor_version=actor_version,
                generation=generation,
                mc_config=mc_config,
                mc_concurrency=mc_concurrency,
                mc_request_semaphore=mc_request_semaphore,
                score_semaphore=score_semaphore,
                mc_native_n_batch_size=mc_native_n_batch_size,
                resumed_rollout=(resumed or {}).get("rollout"),
                resumed_states=(resumed or {}).get("states"),
                completed_labels=(resumed or {}).get("labels"),
                emit_rollout=emit_rollout,
                emit_label=emit_label,
            )
            emit(*result)

    worker_count = min(rollout_concurrency, total_jobs)
    rollout_tasks = [asyncio.create_task(rollout_worker()) for _ in range(worker_count)]
    mc_tasks = [asyncio.create_task(mc_worker()) for _ in range(worker_count)]

    async def finish_rollouts():
        await asyncio.gather(*rollout_tasks)
        for _ in mc_tasks:
            await generated.put(None)

    finisher = asyncio.create_task(finish_rollouts())
    tasks = [*rollout_tasks, *mc_tasks, finisher]
    try:
        await asyncio.gather(finisher, *mc_tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run(
    config: dict,
    *,
    split: str,
    output_dir: Path,
    prompts_jsonl: str | None,
    limit: int | None,
    model_path: str | None,
    actor_version: str | None,
    mc_mode: str,
    rollout_concurrency: int,
    mc_concurrency: int,
    gpu_memory_utilization: float,
    prompt_source: str = "verl_v1",
    verl_overrides: list[str] | None = None,
    mc_sampling_mode: str = "native",
    mc_native_batch_size: int | None = None,
) -> dict:
    if rollout_concurrency < 1 or mc_concurrency < 1:
        raise ValueError("Concurrency must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1)")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    generation, mc_config = sampling_settings(config, mc_mode)
    import ray
    from hydra import compose, initialize_config_dir

    from verl.workers.rollout.llm_server import LLMServerManager

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        verl_config = compose(config_name="ppo_trainer", overrides=verl_overrides or [])
    verl_config.trainer.n_gpus_per_node = 1
    verl_config.trainer.nnodes = 1
    if model_path is not None:
        verl_config.actor_rollout_ref.model.path = model_path
    model_path = str(verl_config.actor_rollout_ref.model.path)
    actor_version = actor_version or model_path
    mc_config = replace(mc_config, actor_version=actor_version)
    rollout_config = verl_config.actor_rollout_ref.rollout
    rollout_config.name = "vllm"
    rollout_config.mode = "async"
    rollout_config.nnodes = 1
    rollout_config.tensor_model_parallel_size = int(config["model"].get("tensor_parallel_size", 1))
    rollout_config.data_parallel_size = 1
    rollout_config.pipeline_model_parallel_size = int(config["model"].get("pipeline_parallel_size", 1))
    if rollout_config.tensor_model_parallel_size != 1 or rollout_config.pipeline_model_parallel_size != 1:
        raise ValueError("This single-GPU entrypoint requires tensor/pipeline parallel size 1")
    rollout_config.load_format = "auto"
    rollout_config.skip_tokenizer_init = False
    rollout_config.dtype = str(config["model"].get("dtype", "auto"))
    rollout_config.response_length = int(generation["max_tokens"])
    rollout_config.seed = int(config.get("seed", 42))
    rollout_config.gpu_memory_utilization = gpu_memory_utilization
    rollout_config.standalone_gpu_memory_utilization = gpu_memory_utilization
    if prompt_source == "verl_v1":
        if prompts_jsonl is not None:
            raise ValueError("Use data.train_files/data.val_files in --verl-override for V1 prompts")
        if split == "rank_eval":
            raise ValueError("rank_eval is only defined by the value_model prompt source")
        if (
            rollout_config.multi_turn.enable
            or rollout_config.agent.default_agent_loop != "single_turn_agent"
            or rollout_config.agent.agent_loop_config_path is not None
        ):
            raise ValueError("V1 prompt parity requires the built-in single_turn_agent without multi-turn tools")
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config.model import HFModelConfig

        from .v1_prompts import load_v1_prompts

        model_config = omega_conf_to_dataclass(verl_config.actor_rollout_ref.model, HFModelConfig)
        tokenizer = model_config.tokenizer
        prompts = load_v1_prompts(
            verl_config,
            tokenizer=tokenizer,
            processor=model_config.processor,
            hf_model_type=model_config.hf_config.model_type,
            split=split,
            limit=limit,
        )
    elif prompt_source == "value_model":
        from transformers import AutoTokenizer

        from verl.utils.tokenizer.tokenizer import normalize_token_ids

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=bool(config["model"].get("trust_remote_code", True))
        )
        rollout_config.max_model_len = int(config["model"]["max_model_len"])
        rollout_config.prompt_length = int(config["model"]["max_model_len"])
        prompts = load_prompt_rows(config, split, prompts_jsonl=prompts_jsonl, limit=limit)
        if split == "rank_eval":
            train_prompts = load_prompt_rows(config, "train")
            overlap = {p["prompt_id"] for p in prompts} & {p["prompt_id"] for p in train_prompts}
            if overlap:
                raise ValueError(f"rank_eval overlaps train on {len(overlap)} prompt IDs")
        if config.get("prompt", {}).get("use_model_chat_template", False):
            template_kwargs = dict(config["prompt"].get("chat_template_kwargs") or {})
            for prompt in prompts:
                raw_prompt = prompt["prompt"]
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                prompt["raw_prompt"] = raw_prompt
                source = config["data"][f"{split}_hf_dataset"]
                prompt["data_source"] = source.get("repo_id", "") if isinstance(source, dict) else str(source)
                prompt_ids = normalize_token_ids(prompt_ids)
                prompt["prompt_token_ids"] = prompt_ids
                prompt["prompt"] = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    else:
        raise ValueError(f"Unknown prompt source: {prompt_source}")
    if not prompts:
        raise ValueError("No prompts were loaded; V1 train loading drops incomplete batches")
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt_id in input split")
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "rollouts": f"rollouts_{split}_regen.jsonl",
        "states": f"states_{split}.jsonl",
        "mc_labels": f"mc_labels_{split}.jsonl",
        "continuations": f"continuations_{split}.jsonl",
    }
    names = tuple(filenames)
    destinations = {name: output_dir / filename for name, filename in filenames.items()}
    temporary = {name: output_dir / f".{filename}.incomplete" for name, filename in filenames.items()}
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing and len(existing) != len(destinations):
        # Recover a crash during the final sequence of per-file atomic renames.
        for name, destination in destinations.items():
            if destination.exists():
                if temporary[name].exists():
                    raise RuntimeError(f"Both final and temporary output exist for {name}")
                destination.replace(temporary[name])
        existing = []
    if existing:
        raise FileExistsError(f"Sampling output already exists: {existing}")

    checkpoint_path = output_dir / f".checkpoint_{split}.json"
    checkpoint_payload = {
        "version": 1,
        "split": split,
        "actor_version": actor_version,
        "mc_mode": mc_mode,
        "config": config,
        "prompts": prompts,
    }
    checkpoint_fingerprint = hashlib.sha256(
        json.dumps(checkpoint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != checkpoint_fingerprint:
            raise ValueError("Partial output checkpoint does not match the current config, mode, or prompts")
    else:
        nonempty_partial = [str(path) for path in temporary.values() if path.exists() and path.stat().st_size]
        if nonempty_partial:
            raise ValueError(f"Partial outputs have no compatible checkpoint: {nonempty_partial}")
        replacement = checkpoint_path.with_suffix(checkpoint_path.suffix + ".incomplete")
        replacement.write_text(
            json.dumps(
                {
                    "fingerprint": checkpoint_fingerprint,
                    "payload": checkpoint_payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        replacement.replace(checkpoint_path)
    resume, recovered_counts = _recover_partial_outputs(
        temporary,
        continuations_per_state=mc_config.continuations_per_state,
        save_continuations=mc_config.save_continuations,
        mc_mode=mc_mode,
    )
    expected_rollout_ids = {
        f"{split}-{prompt['prompt_id']}-{response_index}"
        for prompt in prompts
        for response_index in range(int(config["smoke"]["responses_per_prompt"]))
    }
    unexpected = set(resume) - expected_rollout_ids
    if unexpected:
        raise ValueError(f"Partial outputs contain unexpected rollout IDs: {sorted(unexpected)[:3]}")
    expected_rollout_sampling = dict(generation)
    for rollout_id, recovered in resume.items():
        rollout = recovered["rollout"]
        if rollout.get("actor_version") != actor_version:
            raise ValueError(f"Partial output actor_version mismatch for {rollout_id}")
        if rollout.get("sampling_config") != expected_rollout_sampling:
            raise ValueError(f"Partial output rollout sampling config mismatch for {rollout_id}")
        for label in recovered["labels"]:
            if label.get("mc_actor_version") != actor_version or label.get("mc_sampling_config") != dict(
                mc_config.sampling_config
            ):
                raise ValueError(f"Partial output MC sampling config mismatch for state {label['state_id']}")
    counts = {"prompts": len(prompts), **recovered_counts}
    expected_rollouts = len(prompts) * int(config["smoke"]["responses_per_prompt"])
    ray.init(
        num_gpus=1,
        num_cpus=min(8, os.cpu_count() or 8),
        object_store_memory=4 * 1024**3,
        include_dashboard=False,
        log_to_driver=False,
    )
    try:
        manager = await LLMServerManager.create(config=verl_config)
        client = manager.get_client()
        with (
            temporary["rollouts"].open("a", encoding="utf-8") as rollout_file,
            temporary["states"].open("a", encoding="utf-8") as state_file,
            temporary["mc_labels"].open("a", encoding="utf-8") as label_file,
            temporary["continuations"].open("a", encoding="utf-8") as continuation_file,
        ):

            def emit_rollout(rollout, states):
                for state in states:
                    _write_row(state_file, state)
                state_file.flush()
                # The rollout row commits the preceding state rows for restart.
                _write_row(rollout_file, rollout)
                rollout_file.flush()
                counts["rollouts"] += 1
                counts["states"] += len(states)

            def emit_label(rollout, label):
                for continuation in label.pop("mc_continuations", []):
                    continuation_index = continuation["continuation_index"]
                    token_ids = continuation["token_ids"]
                    _write_row(
                        continuation_file,
                        {
                            "state_id": label["state_id"],
                            "rollout_id": label["rollout_id"],
                            "prompt_id": label["prompt_id"],
                            "split": split,
                            "response_index": rollout["response_index"],
                            "token_index": label["token_index"],
                            "continuation_index": continuation_index,
                            "continuation_id": f"{label['state_id']}:k{continuation_index}",
                            "continuation_token_ids": token_ids,
                            "continuation_text": continuation["text"],
                            "continuation_num_tokens": len(token_ids),
                            "empty_continuation": not token_ids,
                            "reward": continuation["reward"],
                            **(
                                {"delta_top_logprobs": continuation["delta_top_logprobs"]}
                                if continuation.get("delta_top_logprobs") is not None
                                else {}
                            ),
                            "actor_version": mc_config.actor_version,
                            "sampling_config": dict(mc_config.sampling_config),
                        },
                    )
                    counts["continuations"] += 1
                continuation_file.flush()
                _write_row(label_file, label)
                label_file.flush()
                counts["mc_labels"] += 1
                if counts["mc_labels"] % 8 == 0:
                    print(json.dumps(counts, sort_keys=True), flush=True)

            def emit_complete(rollout, states, labels):
                if counts["rollouts"] == expected_rollouts and counts["mc_labels"] == counts["states"]:
                    print(json.dumps(counts, sort_keys=True), flush=True)

            await collect_labeled_rollouts(
                prompts,
                tokenizer=tokenizer,
                client=client,
                config=config,
                split=split,
                actor_version=actor_version,
                generation=generation,
                mc_config=mc_config,
                rollout_concurrency=rollout_concurrency,
                mc_concurrency=mc_concurrency,
                emit=emit_complete,
                mc_sampling_mode=mc_sampling_mode,
                mc_native_batch_size=mc_native_batch_size,
                resume=resume,
                emit_rollout=emit_rollout,
                emit_label=emit_label,
            )
        expected_states = counts["states"]
        if counts["rollouts"] != expected_rollouts or counts["mc_labels"] != expected_states:
            raise RuntimeError(f"Incomplete collection counts: {counts}")
        if mc_config.save_continuations:
            expected_continuations = expected_states * mc_config.continuations_per_state
            if counts["continuations"] != expected_continuations:
                raise RuntimeError(f"Incomplete continuation counts: {counts}")
        for name in names:
            temporary[name].replace(destinations[name])
    finally:
        ray.shutdown()
    return {"counts": counts, "files": {name: str(path) for name, path in destinations.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/delta_critic/config_sampling_qwen3_8b.yaml")
    parser.add_argument("--split", choices=["train", "test", "rank_eval"], default="train")
    parser.add_argument("--prompts-jsonl", help="Override the configured prompt dataset with a JSONL file")
    parser.add_argument("--prompt-source", choices=["verl_v1", "value_model"], default="verl_v1")
    parser.add_argument(
        "--verl-override",
        action="append",
        default=[],
        help="Hydra override for ppo_trainer, repeatable (e.g. data.train_files=/path/train.parquet)",
    )
    parser.add_argument("--limit", type=int, help="Override the configured prompt count")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", help="Override model.actor_model, e.g. with a local snapshot")
    parser.add_argument("--actor-version", help="Recorded actor checkpoint identifier")
    parser.add_argument("--mc-mode", choices=["prefix_only", "paired_next_state"], default="prefix_only")
    parser.add_argument(
        "--rollout-concurrency", type=int, default=8, help="Maximum concurrent rollout requests and MC workers"
    )
    parser.add_argument(
        "--mc-concurrency",
        type=int,
        default=16,
        help="Maximum concurrent MC continuation sequences across all rollouts",
    )
    parser.add_argument(
        "--mc-sampling-mode",
        choices=["native", "requests"],
        default="native",
        help="Use vLLM n-sample requests or separate V1 requests for MC continuations",
    )
    parser.add_argument(
        "--mc-native-batch-size",
        type=int,
        help="Continuations per native vLLM request; smaller values allow parallel states",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, help="Override model.gpu_memory_utilization (default from config)"
    )
    args = parser.parse_args()
    import yaml

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    has_v1_model_override = any(
        override.lstrip("+").startswith("actor_rollout_ref.model.path=") for override in args.verl_override
    )
    model_path = args.model_path or (
        None if args.prompt_source == "verl_v1" and has_v1_model_override else config["model"]["actor_model"]
    )
    actor_version = args.actor_version
    result = asyncio.run(
        run(
            config,
            split=args.split,
            output_dir=Path(args.output_dir),
            prompts_jsonl=args.prompts_jsonl,
            limit=args.limit,
            model_path=model_path,
            actor_version=actor_version,
            mc_mode=args.mc_mode,
            rollout_concurrency=args.rollout_concurrency,
            mc_concurrency=args.mc_concurrency,
            mc_sampling_mode=args.mc_sampling_mode,
            mc_native_batch_size=args.mc_native_batch_size,
            prompt_source=args.prompt_source,
            verl_overrides=args.verl_override,
            gpu_memory_utilization=(
                args.gpu_memory_utilization
                if args.gpu_memory_utilization is not None
                else float(config["model"].get("gpu_memory_utilization", 0.9))
            ),
        )
    )
    print("DELTA_BATCH_SAMPLING_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
