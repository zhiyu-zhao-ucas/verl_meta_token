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
import json
import math
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

from .legacy_adapter import adapt_legacy_rows
from .mc_labeling import MCConfig, label_mc_states_async
from .sampling_io import load_prompt_rows, score_text
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
    # Match the source scripts' effective SamplingParams. The source YAML's
    # rollout repetition_penalty and stop are not passed to those scripts.
    mc_sampling = {
        "temperature": float(mc["temperature"]),
        "top_p": float(mc["top_p"]),
        "max_tokens": int(mc["max_continuation_tokens"]),
    }
    mc_config = MCConfig(
        mode=mode,
        continuations_per_state=int(mc["continuations_per_state"]),
        sampling_config=mc_sampling,
        actor_version=str(config["model"]["actor_model"]),
        continuation_skip_special_tokens=True,
        treat_length_truncation_as_terminal=bool(mc.get("treat_length_truncation_as_terminal", True)),
        assume_legacy_last_token_terminal=False,
        save_individual_rewards=bool(mc.get("save_individual_rewards", False)),
    )
    if int(selection["max_candidates"]) > int(rollout["top_k_logprobs"]):
        raise ValueError("max_candidates cannot exceed top_k_logprobs")
    return generation, mc_config


def rollout_from_output(prompt: dict, prompt_ids: list[int], output, *, tokenizer, split: str,
                        response_index: int, actor_version: str, sampling: dict) -> dict:
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
        if not candidates:
            raise ValueError("Every generated token needs top-k candidates")
        tokens.append({
            "token_index": index,
            "token_id": token_id,
            "sampled_logprob": float(sampled_lp),
            "top1_prob": float(candidates[0]["prob"]),
            "entropy": -sum(float(item["prob"]) * math.log(max(float(item["prob"]), 1e-12)) for item in candidates),
            "top_candidates": candidates,
        })
    return {
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


def _write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


async def sample_labeled_rollout(
    prompt: dict, prompt_ids: list[int], output, *, tokenizer, client,
    config: dict, split: str, response_index: int, actor_version: str,
    generation: dict, mc_config: MCConfig, mc_concurrency: int,
) -> tuple[dict, list[dict], list[dict]]:
    rollout = rollout_from_output(
        prompt, prompt_ids, output, tokenizer=tokenizer, split=split,
        response_index=response_index, actor_version=actor_version,
        sampling={**generation, "reward": config.get("reward") or {}},
    )
    selection = config["state_selection"]
    states = select_states(
        [rollout], strategy="uncertainty",
        states_per_response=int(config["smoke"]["states_per_response"]),
        min_token_gap=int(selection.get("min_token_gap", 0)),
        top_mass=float(config["smoke"]["top_mass"]),
        max_candidates=int(selection["max_candidates"]),
        entropy_weight=float(selection["entropy_weight"]),
        low_top1_weight=float(selection["low_top1_weight"]),
        final_window_tokens=int(selection["final_window_tokens"]),
        final_window_weight=float(selection["final_window_weight"]),
    )
    labels = await label_mc_states_async(
        [rollout], states, mc_config,
        sample=partial(sample_v1_continuations, server_client=client, tokenizer=tokenizer),
        decode_prefix=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        score=lambda response, row: score_text(response, row["gold_answer"], config.get("reward") or {}),
        request_concurrency=mc_concurrency,
    )
    _, diagnostics = adapt_legacy_rows([rollout], labels)
    if diagnostics:
        raise ValueError(f"MC label diagnostics: {diagnostics}")
    return rollout, states, labels


async def run(config: dict, *, split: str, output_dir: Path, prompts_jsonl: str | None,
              limit: int | None, model_path: str, actor_version: str, mc_mode: str,
              rollout_concurrency: int, mc_concurrency: int, gpu_memory_utilization: float) -> dict:
    if rollout_concurrency < 1 or mc_concurrency < 1:
        raise ValueError("Concurrency must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1)")
    prompts = load_prompt_rows(config, split, prompts_jsonl=prompts_jsonl, limit=limit)
    if not prompts:
        raise ValueError("No prompts were loaded")
    if split == "rank_eval":
        train_prompts = load_prompt_rows(config, "train")
        overlap = {p["prompt_id"] for p in prompts} & {p["prompt_id"] for p in train_prompts}
        if overlap:
            raise ValueError(f"rank_eval overlaps train on {len(overlap)} prompt IDs")
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt_id in input split")
    generation, mc_config = sampling_settings(config, mc_mode)
    mc_config = replace(mc_config, actor_version=actor_version)
    import ray
    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer
    from verl.workers.rollout.llm_server import LLMServerManager

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        verl_config = compose(config_name="ppo_trainer")
    verl_config.trainer.n_gpus_per_node = 1
    verl_config.trainer.nnodes = 1
    verl_config.actor_rollout_ref.model.path = model_path
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
    rollout_config.max_model_len = int(config["model"]["max_model_len"])
    rollout_config.prompt_length = int(config["model"]["max_model_len"])
    rollout_config.response_length = int(generation["max_tokens"])
    rollout_config.seed = int(config.get("seed", 42))
    rollout_config.gpu_memory_utilization = gpu_memory_utilization
    rollout_config.standalone_gpu_memory_utilization = gpu_memory_utilization
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=bool(config["model"].get("trust_remote_code", True))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "rollouts": f"rollouts_{split}_regen.jsonl",
        "states": f"states_{split}.jsonl",
        "mc_labels": f"mc_labels_{split}.jsonl",
    }
    names = tuple(filenames)
    destinations = {name: output_dir / filename for name, filename in filenames.items()}
    temporary = {name: output_dir / f".{filename}.incomplete" for name, filename in filenames.items()}
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Sampling output already exists: {existing}")
    counts = {"prompts": len(prompts), "rollouts": 0, "states": 0, "mc_labels": 0}
    ray.init(num_gpus=1, include_dashboard=False, log_to_driver=False)
    try:
        manager = await LLMServerManager.create(config=verl_config)
        client = manager.get_client()
        with (temporary["rollouts"].open("w", encoding="utf-8") as rollout_file,
              temporary["states"].open("w", encoding="utf-8") as state_file,
              temporary["mc_labels"].open("w", encoding="utf-8") as label_file):
            for start in range(0, len(prompts), rollout_concurrency):
                jobs = []
                for prompt in prompts[start:start + rollout_concurrency]:
                    prompt_ids = tokenizer.encode(prompt["prompt"], add_special_tokens=False)
                    if not prompt_ids:
                        raise ValueError(f"Empty prompt tokenization for {prompt['prompt_id']}")
                    for response_index in range(int(config["smoke"]["responses_per_prompt"])):
                        request_id = f"delta-rollout:{split}:{prompt['prompt_id']}:{response_index}"
                        jobs.append((prompt, prompt_ids, response_index, client.generate(
                            request_id=request_id, prompt_ids=prompt_ids, sampling_params=dict(generation)
                        )))
                outputs = await asyncio.gather(*(job[3] for job in jobs))
                for (prompt, prompt_ids, response_index, _), output in zip(jobs, outputs, strict=True):
                    rollout, states, labels = await sample_labeled_rollout(
                        prompt, prompt_ids, output, tokenizer=tokenizer, client=client,
                        config=config, split=split, response_index=response_index,
                        actor_version=actor_version, generation=generation,
                        mc_config=mc_config, mc_concurrency=mc_concurrency,
                    )
                    _write_row(rollout_file, rollout)
                    for state in states:
                        _write_row(state_file, state)
                    for label in labels:
                        _write_row(label_file, label)
                    counts["rollouts"] += 1
                    counts["states"] += len(states)
                    counts["mc_labels"] += len(labels)
                print(json.dumps(counts, sort_keys=True), flush=True)
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
    parser.add_argument("--limit", type=int, help="Override the configured prompt count")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", help="Override model.actor_model, e.g. with a local snapshot")
    parser.add_argument("--actor-version", help="Recorded actor checkpoint identifier")
    parser.add_argument("--mc-mode", choices=["prefix_only", "paired_next_state"], default="prefix_only")
    parser.add_argument("--rollout-concurrency", type=int, default=8)
    parser.add_argument("--mc-concurrency", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float,
                        help="Override model.gpu_memory_utilization (default from config)")
    args = parser.parse_args()
    import yaml

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model_path = args.model_path or config["model"]["actor_model"]
    actor_version = args.actor_version or model_path
    result = asyncio.run(run(
        config, split=args.split, output_dir=Path(args.output_dir),
        prompts_jsonl=args.prompts_jsonl, limit=args.limit, model_path=model_path,
        actor_version=actor_version, mc_mode=args.mc_mode,
        rollout_concurrency=args.rollout_concurrency, mc_concurrency=args.mc_concurrency,
        gpu_memory_utilization=(args.gpu_memory_utilization if args.gpu_memory_utilization is not None
                                else float(config["model"].get("gpu_memory_utilization", 0.9))),
    ))
    print("DELTA_BATCH_SAMPLING_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
