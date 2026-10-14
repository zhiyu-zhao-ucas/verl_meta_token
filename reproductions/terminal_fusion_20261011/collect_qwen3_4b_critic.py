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
"""Collect initial 512-question critic labels with Qwen3-4B, never Qwen3-4B-Base."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import yaml

MODEL_REPO = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TOKENIZER_FINGERPRINT = "81eb7b4b4670977626cbb842958cb0156cfd88fc3ae99037ea1bb7192d5aad27"
PROMPTS_SHA256 = "c8b0151b68d075f3b97310f81f8c6565ee4b27505d71049b67abc347aabd8a07"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def check_model(model: Path):
    from transformers import AutoTokenizer

    from examples.delta_critic.checkpoint import tokenizer_fingerprint_from_tokenizer

    if not all(
        (model / name).is_file()
        for name in (
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        )
    ):
        raise ValueError("Qwen3-4B snapshot is incomplete")
    config = read(model / "config.json")
    if config.get("model_type") != "qwen3" or config.get("hidden_size") != 2560:
        raise ValueError("Unexpected model architecture")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    actual = tokenizer_fingerprint_from_tokenizer(tokenizer, expected_vocab_size=config["vocab_size"])
    if actual != TOKENIZER_FINGERPRINT:
        raise ValueError(f"Expected {MODEL_REPO} tokenizer {TOKENIZER_FINGERPRINT}; got {actual}")
    return tokenizer


def prepare(args) -> None:
    root = args.output.resolve()
    model = args.model.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    if root.exists():
        raise FileExistsError(f"Use a fresh output directory: {root}")
    expected_archive = read(archive.with_suffix(".sha256.json"))["sha256"]
    if sha256(archive) != expected_archive:
        raise ValueError("Committed handoff archive checksum mismatch")
    tokenizer = check_model(model)
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or not all(x.isdecimal() for x in gpu_ids):
        raise ValueError("--gpus must contain distinct physical GPU IDs")
    root.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(root, filter="data")
    assets = root / "assets"
    prompts_file = assets / "bootstrap/prompts.jsonl"
    if sha256(prompts_file) != PROMPTS_SHA256:
        raise ValueError("Fixed 512 prompt checksum mismatch")
    prompts = [json.loads(line) for line in prompts_file.read_text().splitlines()]
    if len(prompts) != 512:
        raise ValueError("Expected exactly 512 fixed critic prompts")
    for index, prompt in enumerate(prompts):
        if tokenizer.encode(prompt["prompt"], add_special_tokens=False) != prompt["prompt_token_ids"]:
            raise ValueError(f"Qwen3-4B prompt token IDs differ at row {index}")
    reference = yaml.safe_load((assets / "bootstrap/sampling.yaml").read_text())
    reference["prompt"]["use_model_chat_template"] = False
    sampling = yaml.safe_load((assets / "bootstrap/sampling.yaml").read_text())
    sampling["model"]["actor_model"] = str(model)
    sampling["prompt"]["use_model_chat_template"] = False
    if args.smoke:
        sampling["smoke"]["states_per_response"] = 1
        sampling["mc"]["continuations_per_state"] = 1
        sampling["rollout"]["max_tokens"] = 512
        sampling["mc"]["max_continuation_tokens"] = 512
    else:
        expected = yaml.safe_load(yaml.safe_dump(reference))
        expected["model"]["actor_model"] = str(model)
        if sampling != expected:
            raise ValueError("Sampling parameters differ from the frozen initial critic collection")
    (root / "sampling.yaml").write_text(yaml.safe_dump(sampling, sort_keys=False))
    shutil.copy2(prompts_file, root / "prompts.jsonl")
    write(
        root / "collection_plan.json",
        {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_path": str(model),
            "model_config_sha256": sha256(model / "config.json"),
            "model_tokenizer_sha256": sha256(model / "tokenizer.json"),
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "prompt_sha256": PROMPTS_SHA256,
            "archive_sha256": expected_archive,
            "sampling_sha256": sha256(root / "sampling.yaml"),
            "source": str(assets / "source_online"),
            "gpus": gpu_ids,
            "questions": 1 if args.smoke else 512,
            "smoke": args.smoke,
            "continuations_per_state": sampling["mc"]["continuations_per_state"],
            "states_per_response": sampling["smoke"]["states_per_response"],
            "sampling_parameters_same_as_initial_critic": not args.smoke,
            "reference_effective_sampling_sha256": hashlib.sha256(
                json.dumps(reference, sort_keys=True).encode()
            ).hexdigest(),
            "runtime_sampling": {
                "rollout_concurrency_per_gpu": 8,
                "mc_concurrency_per_gpu": 128,
                "gpu_memory_utilization": 0.9,
                "mc_mode": "prefix_only",
                "mc_sampling_mode": "requests",
                "mc_native_batch_size": None,
                "prompt_source": "value_model",
                "verl_overrides": [],
            },
        },
    )
    print(f"Prepared {root}; model={MODEL_REPO}@{MODEL_REVISION}; GPUs={','.join(gpu_ids)}")


def check_gpus(ids: list[str]) -> None:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    used = {i.strip(): int(memory.strip()) for i, memory in (line.split(",") for line in rows.splitlines())}
    if not all(i in used and used[i] <= 512 for i in ids):
        raise RuntimeError(f"Selected GPUs are occupied: {used}")


def finite_mc_diagnostics(continuations):
    """Discard impossible zero-probability top-k entries, preserving every sampled token."""
    cleaned = []
    masked = 0
    for continuation in continuations:
        if continuation.delta_top_logprobs is None:
            cleaned.append(continuation)
            continue
        rows = []
        for candidates in continuation.delta_top_logprobs:
            valid = []
            for candidate in candidates:
                probability = float(candidate["prob"])
                logprob = float(candidate["logprob"])
                if math.isfinite(probability) and math.isfinite(logprob):
                    valid.append(candidate)
                elif probability == 0.0 and logprob == float("-inf"):
                    masked += 1
                else:
                    raise ValueError("Nonfinite MC top-k diagnostic has nonzero probability")
            if not valid:
                raise ValueError("MC token has no finite top-k diagnostic candidate")
            rows.append(valid)
        cleaned.append(replace(continuation, delta_top_logprobs=rows))
    return cleaned, masked


def run(args) -> None:
    root = args.output.resolve(strict=True)
    plan = read(root / "collection_plan.json")
    config_file = root / "sampling.yaml"
    if sha256(config_file) != plan["sampling_sha256"] or sha256(root / "prompts.jsonl") != PROMPTS_SHA256:
        raise ValueError("Collection inputs changed after preparation")
    if (root / "collection/result.json").exists():
        raise FileExistsError("Collection already finished")
    gpu_ids = plan["gpus"]
    check_gpus(gpu_ids)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=",".join(gpu_ids),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="2",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        RAY_TMPDIR=f"/tmp/critic512_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}",
    )
    os.environ.pop("RAY_ADDRESS", None)
    source = Path(plan["source"])
    os.environ["PYTHONPATH"] = str(source)
    sys.path.insert(0, str(source))
    os.chdir(source)
    from examples.delta_critic import batch_sample

    original_sample = batch_sample.sample_v1_continuations

    async def sample_with_finite_diagnostics(*sample_args, **sample_kwargs):
        continuations = await original_sample(*sample_args, **sample_kwargs)
        cleaned, masked = finite_mc_diagnostics(continuations)
        if masked:
            request = sample_args[0]
            with (root / "collection/masked_toplogprob_counts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "rollout_id": request.rollout_id,
                            "prefix_tokens": len(request.prefix_response_token_ids),
                            "masked_zero_probability_candidates": masked,
                        }
                    )
                    + "\n"
                )
        return cleaned

    batch_sample.sample_v1_continuations = sample_with_finite_diagnostics

    loader = importlib.util.spec_from_file_location("frozen_online_collect", root / "assets/online_collect.py")
    if loader is None or loader.loader is None:
        raise RuntimeError("Cannot load frozen replica-pool helper")
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)
    helper.ROOT = root
    count = len(gpu_ids)
    (root / "collection").mkdir(exist_ok=True)
    if count > 1:
        helper.configure_replica_pool(count)
    prompts = [json.loads(line) for line in (root / "prompts.jsonl").read_text().splitlines()]
    batch_sample.load_prompt_rows = lambda *unused_args, **unused_kwargs: prompts[: plan["questions"]]
    config = yaml.safe_load(config_file.read_text())
    write(root / "collector_started.json", {"pid": os.getpid(), "gpus": gpu_ids, "model": plan["model_repo"]})
    result = asyncio.run(
        batch_sample.run(
            config,
            split="train",
            output_dir=root / "collection",
            prompts_jsonl="sealed bootstrap prompts",
            limit=plan["questions"],
            model_path=config["model"]["actor_model"],
            actor_version=config["model"]["actor_model"],
            mc_mode="prefix_only",
            rollout_concurrency=8 * count,
            mc_concurrency=128 * count,
            gpu_memory_utilization=0.9,
            prompt_source="value_model",
            verl_overrides=[],
            mc_sampling_mode="requests",
            mc_native_batch_size=None,
        )
    )
    write(root / "collection/result.json", result)
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="local Qwen/Qwen3-4B snapshot; prepare only")
    parser.add_argument("--archive", type=Path, help="committed handoff_assets.tgz; prepare only")
    parser.add_argument("--gpus", default="0", help="idle physical GPU IDs, comma separated")
    parser.add_argument("--smoke", action="store_true", help="one prompt, one state, one branch")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.model is None or args.archive is None:
            parser.error("prepare requires --model and --archive")
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
