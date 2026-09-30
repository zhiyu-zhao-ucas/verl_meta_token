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
"""Standalone torchrun entry point using the real V1 TrainingWorker and FSDP2."""

import argparse
import json
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from .checkpoint import file_hash, fingerprint, read_artifact, tokenizer_fingerprint, write_artifact
from .legacy_adapter import adapt_legacy_rows
from .training_config import ScalarConfig
from .training_data import (
    continuation_rows,
    fit_stats,
    normalization_metadata,
    ordinary_rows,
    resolve_max_length,
    scan_lengths,
    training_batch,
)


def read_rows(path):
    if Path(path).suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_worker(config, *, max_steps, microbatch, initial_artifact=None, forward_only=False):
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig, TrainingWorkerConfig
    from verl.workers.engine_workers import TrainingWorker

    from . import engine  # noqa: F401 -- register the dedicated engine in this process
    from .scalar_loss import delta_loss

    model = HFModelConfig(
        path=config.model_path,
        tokenizer_path=config.tokenizer_path,
        model_type="delta_scalar",
        trust_remote_code=config.trust_remote_code,
        enable_gradient_checkpointing=config.gradient_checkpointing,
        use_remove_padding=False,
        use_fused_kernels=False,
        override_config={"attn_implementation": config.attention_implementation},
    )
    model.hf_config.delta_scalar_config = config.as_dict()
    model.hf_config.delta_initial_artifact = initial_artifact
    engine_config = FSDPEngineConfig(
        strategy="fsdp2",
        use_remove_padding=False,
        use_fused_kernels=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=microbatch,
        infer_micro_batch_size_per_gpu=microbatch,
        use_torch_compile=False,
        forward_only=forward_only,
        seed=config.seed,
    )
    optimizer = (
        None
        if forward_only
        else FSDPOptimizerConfig(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            clip_grad=config.gradient_clip,
            total_training_steps=max_steps,
        )
    )
    contents = ["model"] if forward_only else ["model", "optimizer", "extra"]
    worker = TrainingWorker(
        TrainingWorkerConfig(
            model_type="delta_scalar",
            model_config=model,
            engine_config=engine_config,
            optimizer_config=optimizer,
            checkpoint_config=CheckpointConfig(save_contents=contents, load_contents=contents),
        )
    )
    worker.reset()
    worker.set_loss_fn(delta_loss)
    # Scoring workers can reuse the same registered model loader while
    # retaining an auditable reference to the portable artifact.
    worker.delta_initial_artifact = initial_artifact
    return worker


class RowIterator:
    """Deterministic permutations independent of DP and physical batching."""

    def __init__(self, size, seed, epoch=0, cursor=0):
        self.size, self.seed, self.epoch, self.cursor = size, seed, epoch, cursor
        if size < 1 or not 0 <= cursor < size:
            raise ValueError("Invalid iterator position")

    def take(self, count):
        result = []
        while len(result) < count:
            order = list(range(self.size))
            random.Random(self.seed + self.epoch).shuffle(order)
            end = min(self.size, self.cursor + count - len(result))
            result.extend(order[self.cursor : end])
            self.cursor = end
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
        return result

    def state(self):
        return {"epoch": self.epoch, "cursor": self.cursor}


def evaluate(worker, rows, config, normalization, terminal_stats, pad, microbatch):
    from verl.utils import tensordict_utils as tu

    coverage = coverage_metrics(rows)
    if coverage["supervised_rows"] == 0:
        raise ValueError("Evaluation data contains no supervised rows")
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    padded = list(rows)
    while len(padded) % (world * microbatch):
        dummy = dict(rows[0])
        dummy.update(
            token_loss_mask=[0.0] * len(dummy["response_token_ids"]),
            terminal_comp_valid=False,
            sample_valid=False,
        )
        padded.append(dummy)
    local = padded[rank::world]
    batch = training_batch(local, config, normalization, terminal_stats, pad)
    tu.assign_non_tensor(batch, compute_loss=True)
    result = worker.infer_batch(batch)
    losses = reduce_engine_metrics(tu.get(result, "metrics"))
    return {
        "loss": losses["critic/total_loss"],
        "td_loss": losses["critic/td_loss"],
        "terminal_loss": losses["critic/terminal_loss"],
        "coverage": coverage,
        "rows": len(rows),
    }


def reduce_engine_metrics(metrics):
    """Reduce FSDP callback Metric objects to exact logical-update totals."""
    from verl.utils.metric.utils import Metric

    def reduce(value):
        if isinstance(value, Metric):
            return float(value.aggregate())
        if isinstance(value, list | tuple):
            if value and all(isinstance(item, Metric) for item in value):
                # SUM metrics are locally accumulated across microbatches and
                # globally averaged across DP ranks by verl's metric contract.
                return float(Metric.aggregate_dp(list(value)))
            return sum(reduce(item) for item in value)
        return float(value)

    return {key: reduce(value) for key, value in metrics.items()}


def coverage_metrics(rows):
    """Coverage from real rows only; evaluation padding rows never count."""
    target_tokens = signal_tokens = terminal_rows = supervised_rows = response_tokens = input_tokens = 0
    for row in rows:
        target_mask = row.get("token_loss_mask", row.get("token_target_mask", []))
        signal_mask = row.get("token_signal_mask", [0.0] * len(target_mask))
        if len(target_mask) != len(row["response_token_ids"]) or len(signal_mask) != len(target_mask):
            raise ValueError(f"Coverage mask length mismatch in row {row.get('id')}")
        td_valid = any(float(value) > 0 for value in target_mask)
        terminal_valid = bool(row.get("terminal_comp_valid", False))
        supervised_rows += int(td_valid or terminal_valid)
        terminal_rows += int(terminal_valid)
        target_tokens += sum(float(value) > 0 for value in target_mask)
        signal_tokens += sum(
            float(mask) > 0 and float(signal) > 0 for mask, signal in zip(target_mask, signal_mask, strict=False)
        )
        response_tokens += len(row["response_token_ids"])
        input_tokens += len(row["prompt_token_ids"]) + len(row["response_token_ids"])
    real_rows = len(rows)
    background_tokens = target_tokens - signal_tokens
    return {
        "real_rows": real_rows,
        "valid_rows": real_rows,
        "supervised_rows": supervised_rows,
        "supervised_row_fraction": supervised_rows / real_rows if real_rows else 0.0,
        "terminal_rows": terminal_rows,
        "terminal_row_fraction": terminal_rows / real_rows if real_rows else 0.0,
        "response_tokens": response_tokens,
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "target_token_fraction": target_tokens / response_tokens if response_tokens else 0.0,
        "signal_tokens": signal_tokens,
        "background_tokens": background_tokens,
        "signal_token_fraction": signal_tokens / target_tokens if target_tokens else 0.0,
    }


def expected_continuation_rows(meta):
    """Rows a hybrid split must contribute after an explicit budget is honored.

    `eligible` counts every matched nonempty continuation, so a configured
    `continuation_budget` deliberately drops the excess into `over_budget`.
    Comparing `included` against `eligible` would therefore reject every legal
    budgeted run.
    """
    return meta["eligible"] - meta.get("over_budget", 0)


def reserve_step_directory(path, *, device=None):
    """Reserve this step's directory exactly once across ranks.

    Rank zero owns creation and broadcasts the verdict; no other rank may test
    the path itself, because that races rank zero's ``mkdir`` and makes a rank
    report an overwrite for a directory rank zero just created.
    """
    if device is None:
        device = torch.cuda.current_device()
    status = torch.zeros((), dtype=torch.int32, device=device)
    if torch.distributed.get_rank() == 0:
        try:
            Path(path).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            status.fill_(1)
    torch.distributed.broadcast(status, src=0)
    if status.item():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {path}")


def save(worker, output, config, normalization, terminal_stats, state, provenance):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    path = Path(output) / f"step_{state['global_step']:08d}"
    reserve_step_directory(path)
    worker.save_checkpoint(str(path / "engine"), global_step=state["global_step"])
    weights = get_model_state_dict(
        worker.engine.module, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if torch.distributed.get_rank() == 0:
        write_artifact(
            path,
            weights,
            config,
            normalization,
            terminal_stats,
            **provenance,
            training_state=state,
            resume_capable=True,
        )
        temporary_state = path / "trainer_state.json.tmp"
        temporary_state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary_state.replace(path / "trainer_state.json")
        (path / "COMPLETE").write_text("complete\n")
    torch.distributed.barrier()
    return path


def resolve_config(config_path, attention_implementation=None):
    """Load the scalar config, optionally overriding the attention backend.

    The override exists so a run can try an optional backend without editing and
    re-freezing a whole config file; the chosen value still lands in the artifact
    and the run signature, so the run stays reproducible.
    """
    config = ScalarConfig(**yaml.safe_load(Path(config_path).read_text()))
    if attention_implementation:
        config = replace(config, attention_implementation=attention_implementation)
    return config


def run(args):
    config = resolve_config(args.config, args.attention_implementation)
    from transformers import AutoConfig

    backbone_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)
    config = replace(config, max_length=resolve_max_length(config.max_length, backbone_config))
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    if args.global_batch < 1 or args.max_steps < 1 or args.microbatch < 1:
        raise ValueError("global_batch, max_steps and microbatch must be positive and explicit")
    if args.global_batch % (world * args.microbatch):
        raise ValueError("global_batch must be divisible by world_size * microbatch")
    if args.resume and args.initialize:
        raise ValueError("Choose resume OR weight initialization")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    rollouts, labels = read_rows(args.rollouts), read_rows(args.labels)
    examples, diagnostics = adapt_legacy_rows(rollouts, labels)
    split = json.loads(Path(args.split).read_text())
    if set(split) != {e.rollout.rollout_id for e in examples} or set(split.values()) != {"train", "eval"}:
        raise ValueError("Split must map every rollout ID exactly once to train or eval, both nonempty")
    train_examples = [e for e in examples if split[e.rollout.rollout_id] == "train"]
    eval_examples = [e for e in examples if split[e.rollout.rollout_id] == "eval"]
    train_rows, eval_rows = ordinary_rows(train_examples, config), ordinary_rows(eval_examples, config)
    hybrid_meta, terminal_stats = {"train": None, "eval": None}, None
    if config.objective == "hybrid_terminal_composition":
        if not args.continuations:
            raise ValueError("Hybrid objective requires --continuations")
        continuations = read_rows(args.continuations)
        train_ids = [e.rollout.rollout_id for e in train_examples]
        eval_ids = [e.rollout.rollout_id for e in eval_examples]
        train_hybrid, hybrid_meta["train"] = continuation_rows(continuations, labels, config, eval_ids)
        eval_hybrid, hybrid_meta["eval"] = continuation_rows(continuations, labels, config, train_ids)
        if hybrid_meta["train"]["included"] != expected_continuation_rows(hybrid_meta["train"]):
            raise ValueError("Training continuation rows are incomplete")
        if hybrid_meta["eval"]["included"] != expected_continuation_rows(hybrid_meta["eval"]):
            raise ValueError("Evaluation continuation rows are incomplete")
        if not eval_hybrid:
            raise ValueError("Hybrid objective requires held-out continuation rows for terminal evaluation")
        terminal_stats = fit_stats(train_hybrid, terminal=True)
        train_rows += train_hybrid
        eval_rows += eval_hybrid
    elif args.continuations:
        raise ValueError("Continuations require hybrid objective")
    lengths = scan_lengths(train_rows + eval_rows, config)
    normalization = normalization_metadata(train_rows, config)
    token_hash, tokenizer = tokenizer_fingerprint(config.tokenizer_path or config.model_path, config.trust_remote_code)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires pad or eos token")
    files = {
        key: file_hash(getattr(args, key))
        for key in ("rollouts", "labels", "split", "continuations")
        if getattr(args, key)
    }
    provenance = {
        "tokenizer_sha256": token_hash,
        "data_fingerprints": files,
        "length_scan": lengths,
        "hybrid_data": hybrid_meta,
        "diagnostics": diagnostics,
        "semantics_changes": [],
        "effective_global_batch": args.global_batch,
        "max_steps": args.max_steps,
        "logical_batch": 1,
        "physical_microbatch": args.microbatch,
        "world_size": world,
        "seed": config.seed,
    }
    signature = fingerprint(
        {
            "config": config.as_dict(),
            "files": files,
            "global_batch": args.global_batch,
            "max_steps": args.max_steps,
            "world_size": world,
            "microbatch": args.microbatch,
            "tokenizer": token_hash,
        }
    )
    state = {
        "global_step": 0,
        "epoch": 0,
        "cursor": 0,
        "signature": signature,
        "iterator_seed": config.seed,
        "train_row_count": len(train_rows),
        "world_size": world,
    }
    initial = args.initialize
    if args.resume:
        checkpoint = Path(args.resume)
        if not (checkpoint / "COMPLETE").exists():
            raise ValueError("Incomplete resume checkpoint")
        meta = read_artifact(checkpoint)
        if not meta.get("resume_capable", False):
            raise ValueError("This portable artifact does not include resumable optimizer/data state")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if (
            state["signature"] != signature
            or meta["normalization"] != normalization
            or meta["terminal_stats"] != terminal_stats
        ):
            raise ValueError("Exact resume requires identical data, configuration, statistics and topology")
        initial = str(checkpoint)
        provenance["semantics_changes"] = meta.get("semantics_changes", [])
        if "initialization" in meta:
            provenance["initialization"] = meta["initialization"]
    elif initial:
        meta = read_artifact(initial)
        for key in (
            "model_path",
            "tokenizer_path",
            "value_layer",
            "objective",
            "loss_type",
            "loss_mask",
            "delta_label_mode",
            "target_normalization",
        ):
            if meta["config"][key] != config.as_dict()[key]:
                # tokenizer_path=None means model_path, so compare effective identities.
                if key == "tokenizer_path" and (meta["config"][key] or meta["config"]["model_path"]) == (
                    config.tokenizer_path or config.model_path
                ):
                    continue
                raise ValueError(f"Initialization semantic mismatch: {key}")
        if meta["tokenizer_sha256"] != token_hash:
            raise ValueError("Initialization tokenizer mismatch")
        provenance["initialization"] = {"weights_sha256": meta["weights_sha256"], "path": initial}
        for key, new in (("normalization", normalization), ("terminal_stats", terminal_stats)):
            if meta[key] != new:
                provenance["semantics_changes"].append(f"{key} refitted on new train split")
        if meta["config"]["max_length"] != config.max_length or meta["config"]["window_policy"] != config.window_policy:
            provenance["semantics_changes"].append("Training window changed from initialization checkpoint")
    worker = build_worker(config, max_steps=args.max_steps, microbatch=args.microbatch, initial_artifact=initial)
    if args.resume:
        worker.load_checkpoint(str(Path(args.resume) / "engine"))
    if state.get("iterator_seed", config.seed) != config.seed or state.get("train_row_count", len(train_rows)) != len(
        train_rows
    ):
        raise ValueError("Resume row-order state does not match the current training data")
    iterator = RowIterator(len(train_rows), config.seed, state["epoch"], state["cursor"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    end = min(args.max_steps, args.stop_after or args.max_steps)
    from verl.utils import tensordict_utils as tu

    for step in range(state["global_step"], end):
        indices = iterator.take(args.global_batch)
        update_rows = [train_rows[i] for i in indices]
        local = update_rows[rank::world]
        batch = training_batch(local, config, normalization, terminal_stats, pad)
        result = worker.train_batch(batch)
        update_metrics = reduce_engine_metrics(tu.get(result, "metrics"))
        metrics = {
            "step": step + 1,
            "train_loss": update_metrics["critic/total_loss"],
            "train_td_loss": update_metrics["critic/td_loss"],
            "train_terminal_loss": update_metrics["critic/terminal_loss"],
            "train_coverage": coverage_metrics(update_rows),
            "provenance": {
                "signature": signature,
                "objective": config.objective,
                "max_length": config.max_length,
                "effective_global_batch": args.global_batch,
                "physical_microbatch": args.microbatch,
                "world_size": world,
                "data_fingerprints": files,
            },
        }
        state.update(global_step=step + 1, **iterator.state())
        if (step + 1) % args.save_every == 0 or step + 1 == end:
            metrics["validation"] = evaluate(
                worker, eval_rows, config, normalization, terminal_stats, pad, args.microbatch
            )
            provenance["validation"] = metrics["validation"]
            save(worker, output, config, normalization, terminal_stats, state, provenance)
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "rollouts", "labels", "split", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--continuations")
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--stop-after", type=int, help="Stop early without changing the LR schedule; useful for resume tests"
    )
    parser.add_argument("--resume")
    parser.add_argument("--initialize", help="Portable artifact: weights only, starts new optimizer and data stream")
    parser.add_argument(
        "--attention-implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        help=(
            "Override config attention_implementation. flash_attention_2 requires the flash-attn "
            "package to be installed in the environment; the reference runs used sdpa."
        ),
    )
    args = parser.parse_args()
    if args.save_every < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error("save-every and stop-after must be positive")
    run(args)


if __name__ == "__main__":
    main()
