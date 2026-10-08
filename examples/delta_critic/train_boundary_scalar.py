# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""One training implementation for value and directly regressed delta controls."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from .checkpoint import file_hash, fingerprint, tokenizer_fingerprint, write_artifact
from .legacy_adapter import adapt_legacy_rows
from .train import RowIterator, build_worker, read_rows, reduce_engine_metrics
from .training_config import ScalarConfig
from .value_difference import SEMANTICS, boundary_batch, boundary_mse_loss, boundary_rows


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def initial_weight_identity(module):
    """Compare actual initial parameter bytes, including the new FP32 head."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(local.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, digest.hexdigest())
    return fingerprint(gathered)


def evaluate(worker, rows, config, pad, microbatch):
    from verl.utils import tensordict_utils as tu

    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    padded = list(rows)
    while len(padded) % (world * microbatch):
        padded.append(dict(rows[0], sample_valid=False))
    batch = boundary_batch(padded[rank::world], config, pad)
    tu.assign_non_tensor(batch, compute_loss=True)
    result = worker.infer_batch(batch)
    metrics = reduce_engine_metrics(tu.get(result, "metrics"))
    count = metrics["boundary/points"]
    mse = metrics["boundary/squared_error"] / count
    if count <= 0 or not math.isfinite(mse):
        raise ValueError("Evaluation requires nonempty finite regression points")
    return {"mse": mse, "row_mean_mse": metrics["boundary/loss"], "points": count, "rows": len(rows)}


def save_best(worker, output, config, step, mse, provenance):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    weights = get_model_state_dict(
        worker.engine.module, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if torch.distributed.get_rank() == 0:
        path = output / f"step_{step:08d}"
        norm = {"enabled": False, "mode": "none", "mean": None, "std": None}
        write_artifact(
            path,
            weights,
            config,
            norm,
            **provenance,
            training_state={"global_step": step},
            resume_capable=False,
            validation={"mse": mse},
        )
        (path / "COMPLETE").write_text("complete\n")
        link = output / "best"
        previous = link.resolve() if link.is_symlink() else None
        temporary = output / "best.tmp"
        temporary.symlink_to(path.name, target_is_directory=True)
        temporary.replace(link)
        write_json(
            output / "selection.json",
            {
                "step": step,
                "mse": mse,
                "criterion": "heldout point-weighted raw MSE",
                "path": str(link),
                "tie_break": "earliest evaluated step",
            },
        )
        if previous is not None:
            shutil.rmtree(previous)
    del weights
    torch.distributed.barrier()


def run(args):
    from verl.utils import tensordict_utils as tu

    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    if args.global_batch % (world * args.microbatch):
        raise ValueError("global_batch must divide evenly into world_size * microbatch")
    config = ScalarConfig(**yaml.safe_load(Path(args.config).read_text()))
    if config.loss_type != "mse" or config.target_normalization != "none" or config.window_policy != "full":
        raise ValueError("Both controls require the same raw MSE and full context")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    examples, diagnostics = adapt_legacy_rows(read_rows(args.rollouts), read_rows(args.labels))
    split = json.loads(Path(args.split).read_text())
    if set(split) != {e.rollout.rollout_id for e in examples} or set(split.values()) != {"train", "eval"}:
        raise ValueError("The exact TD80 split must cover every row")
    train = boundary_rows([e for e in examples if split[e.rollout.rollout_id] == "train"], args.target)
    heldout = boundary_rows([e for e in examples if split[e.rollout.rollout_id] == "eval"], args.target)
    tokenizer_hash, tokenizer = tokenizer_fingerprint(
        config.tokenizer_path or config.model_path, config.trust_remote_code
    )
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires pad/EOS")
    worker = build_worker(config, max_steps=args.max_steps, microbatch=args.microbatch)
    worker.set_loss_fn(boundary_mse_loss)
    identity = initial_weight_identity(worker.engine.module)
    common = {
        "config": config.as_dict(),
        "initial_parameter_sha256": identity,
        "tokenizer_sha256": tokenizer_hash,
        "data_fingerprints": {key: file_hash(getattr(args, key)) for key in ("rollouts", "labels", "split")},
        "global_batch": args.global_batch,
        "microbatch": args.microbatch,
        "world_size": world,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "train_rows": len(train),
        "eval_rows": len(heldout),
        "support_sha256": fingerprint([(r["id"], r["boundary_positions"]) for r in train + heldout]),
        "row_order_sha256": fingerprint([r["id"] for r in train]),
        "boundary_semantics": SEMANTICS,
    }
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "controls.json", common)
    torch.distributed.barrier()
    source = Path(__file__).resolve().parents[2]
    manifest = (
        json.loads((source / "source_manifest.json").read_text()) if (source / "source_manifest.json").exists() else {}
    )
    provenance = {
        "artifact_kind": "boundary_scalar",
        "boundary_target": args.target,
        "boundary_semantics": SEMANTICS,
        "tokenizer_sha256": tokenizer_hash,
        "diagnostics": diagnostics,
        "source_sha256": fingerprint(manifest),
        "controlled_training": common,
    }
    iterator = RowIterator(len(train), config.seed)
    best, order_digest = float("inf"), hashlib.sha256()
    for step in range(1, args.max_steps + 1):
        indices = iterator.take(args.global_batch)
        order_digest.update(json.dumps(indices).encode())
        batch = boundary_batch([train[i] for i in indices[rank::world]], config, pad)
        result = worker.train_batch(batch)
        metrics = reduce_engine_metrics(tu.get(result, "metrics"))
        if not math.isfinite(metrics["boundary/loss"]):
            raise ValueError("Nonfinite training loss")
        record = {"step": step, "target": args.target, "train_mse": metrics["boundary/loss"]}
        if step % args.eval_every == 0 or step == args.max_steps:
            validation = evaluate(worker, heldout, config, pad, args.microbatch)
            record["validation"] = validation
            if validation["mse"] < best:
                best = validation["mse"]
                save_best(worker, output, config, step, best, provenance)
            record["best_mse"] = best
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    if rank == 0:
        write_json(
            output / "complete.json",
            {
                "steps": args.max_steps,
                "batch_schedule_sha256": order_digest.hexdigest(),
                "initial_parameter_sha256": identity,
                "best_mse": best,
            },
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "rollouts", "labels", "split", "output"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--target", choices=["value", "direct_delta"], required=True)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=16)
    args = parser.parse_args()
    if min(args.max_steps, args.eval_every, args.global_batch, args.microbatch) < 1:
        parser.error("Step and batch counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
