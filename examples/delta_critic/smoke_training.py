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
"""Real V1/FSDP2, resume, microbatch and TransferQueue acceptance on a tiny Qwen3.

Run with an installed verl training environment; no source repository or shims.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml


def prepare(directory, dtype):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "tiny"
    torch.manual_seed(3)
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=16384,
            tie_word_embeddings=False,
        )
    ).save_pretrained(model_path)
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="0", pad_token="0", eos_token="1").save_pretrained(
        model_path
    )
    rollouts, labels, continuations, split = [], [], [], {}
    for i in range(6):
        response = list(range(3, 7 + i))
        row_id = str(i)
        rollouts.append(
            dict(id=row_id, prompt_token_ids=[1, 2], response_token_ids=response, terminal_reward=float(i % 2))
        )
        split[row_id] = "train" if i < 4 else "eval"
        for t in (0, len(response) - 1):
            labels.append(
                dict(
                    rollout_id=row_id,
                    state_id=f"{i}:{t}",
                    token_index=t,
                    v_prefix=0.25 * (i % 3),
                    prompt_token_ids=[1, 2],
                    prefix_response_token_ids=response[:t],
                )
            )
        continuations.append(
            dict(
                rollout_id=row_id,
                state_id=f"{i}:0",
                continuation_id=f"c{i}",
                continuation_index=0,
                continuation_token_ids=response,
                reward=float(i % 2),
            )
        )
    for name, data in (("rollouts", rollouts), ("labels", labels), ("continuations", continuations)):
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in data))
    (directory / "split.json").write_text(json.dumps(split))
    config = dict(
        model_path=str(model_path),
        dtype=dtype,
        objective="hybrid_terminal_composition",
        loss_type="nonzero_balanced_mse",
        learning_rate=1e-3,
    )
    (directory / "config.yaml").write_text(yaml.safe_dump(config))


def launch(directory, name, gpus, microbatch, *extra):
    output = directory / name
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m",
        "examples.delta_critic.train",
        "--config",
        str(directory / "config.yaml"),
        "--rollouts",
        str(directory / "rollouts.jsonl"),
        "--labels",
        str(directory / "labels.jsonl"),
        "--continuations",
        str(directory / "continuations.jsonl"),
        "--split",
        str(directory / "split.json"),
        "--output",
        str(output),
        "--global-batch",
        "4",
        "--max-steps",
        "3",
        "--microbatch",
        str(microbatch),
        "--save-every",
        "1",
        *extra,
    ]
    print(" ".join(command), flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    with (directory / f"{name}.log").open("w") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {directory / (name + '.log')}")
    return output / "step_00000003"


def compare(left, right, atol, rtol):
    a = torch.load(left / "model.pt", map_location="cpu", weights_only=True)
    b = torch.load(right / "model.pt", map_location="cpu", weights_only=True)
    maximum = 0.0
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=atol, rtol=rtol)
        maximum = max(maximum, (a[key] - b[key]).abs().max().item())
    return maximum


def transfer_queue_check(artifact):
    import transfer_queue as tq
    from tensordict import TensorDict

    from examples.delta_critic.score import FrozenDeltaWorker

    worker = FrozenDeltaWorker(artifact, microbatch=2)
    before = {key: value.clone() for key, value in worker.model.state_dict().items()}
    tq.init()
    try:
        fields = TensorDict(
            {
                key: torch.nested.as_nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)
                for key, rows in {
                    "prompts": [[1, 2], [2]],
                    "responses": [[3, 4, 5], [3, 4]],
                    "selected_token_indices": [[0, 2], [1]],
                }.items()
            },
            batch_size=[2],
        )
        meta = tq.kv_batch_put(keys=["first", "second"], partition_id="delta-smoke", fields=fields)
        worker.score_transfer_queue(meta)
        actual = tq.kv_batch_get(
            keys=meta.keys,
            partition_id=meta.partition_id,
            select_fields=["critic_signal_mask", "delta_pred_raw", "critic_checkpoint"],
        )
        assert actual["critic_signal_mask"][0].tolist() == [1.0, 0.0, 1.0]
        assert actual["critic_signal_mask"][1].tolist() == [0.0, 1.0]
        for key, value in worker.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert all(p.grad is None for p in worker.model.parameters())
        return "passed"
    finally:
        tq.close()


def v1_scoring_check(directory, artifact):
    """Score through the real V1 forward-only TrainingWorker.

    The other scoring check goes through ``FrozenDeltaWorker``'s direct model
    path, which has a different batch contract from ``TrainingWorker.infer_batch``
    (the V1 batch carries ``loss_mask`` but no ``target_mask``/``signal_mask``,
    and ``global_token_num`` must be per-row). Comparing the two here is what
    actually validates the V1 scoring wiring; a fake worker cannot.
    """
    from dataclasses import replace

    from transformers import AutoConfig

    from .score import FrozenDeltaWorker, ScoringRow
    from .train import build_worker, read_rows
    from .training_config import ScalarConfig
    from .training_data import resolve_max_length

    config = ScalarConfig(**yaml.safe_load((directory / "config.yaml").read_text()))
    backbone_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)
    config = replace(config, max_length=resolve_max_length(config.max_length, backbone_config))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    worker = build_worker(config, max_steps=1, microbatch=1, initial_artifact=str(artifact), forward_only=True)
    rows = [
        ScoringRow(
            rollout_id=str(row["id"]),
            prompt_token_ids=tuple(row["prompt_token_ids"]),
            response_token_ids=tuple(row["response_token_ids"]),
            selected_token_indices=(0, min(1, len(row["response_token_ids"]) - 1)),
        )
        for row in read_rows(directory / "rollouts.jsonl")[:2]
    ]
    via_v1 = FrozenDeltaWorker(str(artifact), microbatch=2, engine_worker=worker).score(rows)
    direct = FrozenDeltaWorker(str(artifact), microbatch=2).score(rows)
    maximum = 0.0
    for v1_row, direct_row in zip(via_v1, direct, strict=True):
        for key in ("delta_pred_normalized", "delta_pred_raw"):
            a, b = torch.tensor(v1_row[key]), torch.tensor(direct_row[key])
            torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)
            maximum = max(maximum, (a - b).abs().max().item())
        assert v1_row["critic_signal_mask"] == direct_row["critic_signal_mask"]
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return {"max_abs_delta": maximum, "rows": len(rows)}


def gradient_check(directory):
    from transformers import AutoConfig

    from examples.delta_critic.legacy_adapter import adapt_legacy_rows
    from examples.delta_critic.train import build_worker, read_rows
    from examples.delta_critic.training_config import ScalarConfig
    from examples.delta_critic.training_data import (
        continuation_rows,
        fit_stats,
        normalization_metadata,
        ordinary_rows,
        resolve_max_length,
        training_batch,
    )
    from verl.utils import tensordict_utils as tu

    config = ScalarConfig(**yaml.safe_load((directory / "config.yaml").read_text()))
    backbone_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)
    config = replace(config, max_length=resolve_max_length(config.max_length, backbone_config))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(config.seed)
    labels = read_rows(directory / "labels.jsonl")
    examples, _ = adapt_legacy_rows(read_rows(directory / "rollouts.jsonl"), labels)
    ordinary = ordinary_rows(examples, config)
    hybrid, _ = continuation_rows(read_rows(directory / "continuations.jsonl"), labels, config)
    norm, terminal = normalization_metadata(ordinary, config), fit_stats(hybrid, terminal=True)
    dummy = dict(
        ordinary[0],
        token_loss_mask=[0.0] * len(ordinary[0]["response_token_ids"]),
        sample_valid=False,
    )
    rows = [ordinary[0], hybrid[0], ordinary[1], dummy]
    worker = build_worker(config, max_steps=3, microbatch=1)
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    results = []
    for microbatch in (1, 2):
        batch = training_batch(rows[rank::world], config, norm, terminal, 0)
        tu.assign_non_tensor(batch, micro_batch_size_per_gpu=microbatch)
        with worker.engine.train_mode():
            worker.engine.optimizer_zero_grad()
            output = worker.engine.forward_backward_batch(batch, worker.loss_fn)
            gradients = {}
            for key, parameter in worker.engine.module.named_parameters():
                if parameter.grad is not None:
                    gradients[key] = parameter.grad.full_tensor().cpu().clone()
            loss = torch.tensor(sum(output["loss"]), device="cuda")
            torch.distributed.all_reduce(loss)
            results.append({"gradient": gradients, "loss": loss.item() / world})
    atol, rtol = (2e-7, 2e-5) if config.dtype == "float32" else (0.03, 0.03)
    for key in results[0]["gradient"]:
        torch.testing.assert_close(results[0]["gradient"][key], results[1]["gradient"][key], atol=atol, rtol=rtol)
    if rank == 0:
        torch.save(results, directory / f"gradients_dp{world}.pt")
        print(
            {
                "world_size": world,
                "losses": [r["loss"] for r in results],
                "gradient_max_abs": max(
                    (results[0]["gradient"][k] - results[1]["gradient"][k]).abs().max().item()
                    for k in results[0]["gradient"]
                ),
            }
        )
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--gradient-worker", action="store_true")
    parser.add_argument(
        "--v1-scoring-worker",
        action="store_true",
        help="Run the V1 forward-only scoring comparison under torchrun against a finished run",
    )
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    if args.gradient_worker:
        gradient_check(directory)
        return
    if args.v1_scoring_worker:
        steps = sorted((directory / "single").glob("step_*"))
        if not steps:
            raise RuntimeError(f"No finished run under {directory / 'single'}; run the smoke test first")
        print(json.dumps(v1_scoring_check(directory, steps[-1])))
        return
    prepare(directory, args.dtype)
    if args.prepare_only:
        return
    baseline = launch(directory, "single", 1, 1)
    launch(directory, "interrupted", 1, 1, "--stop-after", "1")
    resumed = launch(directory, "resumed", 1, 1, "--resume", str(directory / "interrupted" / "step_00000001"))
    batched = launch(directory, "micro2", 1, 2)
    multi = launch(directory, "two_gpu", 2, 1)
    multi2 = launch(directory, "two_gpu_micro2", 2, 2)
    atol, rtol = (2e-7, 2e-5) if args.dtype == "float32" else (2e-3, 2e-2)
    report = {
        "dtype": args.dtype,
        "atol": atol,
        "rtol": rtol,
        "resume_max_abs": compare(baseline, resumed, 0, 0),
        "microbatch_max_abs": compare(baseline, batched, atol, rtol),
        "two_gpu_max_abs": compare(baseline, multi, atol, rtol),
        "two_gpu_microbatch_max_abs": compare(multi, multi2, atol, rtol),
        "transfer_queue": transfer_queue_check(baseline),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
