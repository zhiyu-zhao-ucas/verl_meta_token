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
"""Standalone offline/online delta policy trainer backed by verl V1 FSDP2."""

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from .checkpoint import file_hash, fingerprint, tokenizer_fingerprint
from .legacy_adapter import adapt_legacy_rows
from .policy_config import DeltaPolicyConfig, delta_policy_config_from_mapping
from .policy_data import policy_training_batch, prepare_policy_rows
from .policy_loss import delta_policy_loss
from .policy_training_config import PolicyTrainingConfig


def read_rows(path):
    if Path(path).suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_precomputed_behavior(path, config, *, actor_fingerprint):
    """Load externally scored old-policy values with a checkable scoring identity."""
    values_by_id = {}
    identities = set()
    for row in read_rows(path):
        rollout_id = str(row.get("rollout_id", row.get("id", "")))
        values = row.get("logprobs", row.get("log_probs"))
        provenance = row.get("provenance")
        if not rollout_id or values is None or rollout_id in values_by_id:
            raise ValueError("Precomputed behavior rows require unique rollout_id and logprobs")
        if not isinstance(provenance, dict):
            raise ValueError(f"Precomputed behavior row {rollout_id} requires provenance")
        identity = provenance.get("actor_fingerprint")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"Precomputed behavior row {rollout_id} requires actor_fingerprint")
        if provenance.get("temperature") != config.train_logprob_temperature:
            raise ValueError(f"Precomputed behavior temperature mismatch for {rollout_id}")
        if provenance.get("window") != config.window or provenance.get("max_length") != config.max_length:
            raise ValueError(f"Precomputed behavior context mismatch for {rollout_id}")
        identities.add(identity)
        values_by_id[rollout_id] = values
    if len(identities) != 1:
        raise ValueError("Precomputed behavior rows must share one actor_fingerprint")
    identity = identities.pop()
    if config.mode == "online_ppo" and identity != actor_fingerprint:
        raise ValueError("Online precomputed behavior actor differs from the current actor snapshot")
    return values_by_id, identity


def _path_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return fingerprint({"path": str(path)})
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    hashes = {}
    for item in files:
        rel = item.name if path.is_file() else str(item.relative_to(path))
        hashes[rel] = file_hash(item)
    if not hashes:
        raise ValueError(f"Empty model directory: {path}")
    return fingerprint(hashes)


def _nested(rows, dtype):
    return torch.nested.as_nested_tensor([torch.as_tensor(row, dtype=dtype) for row in rows], layout=torch.jagged)


def build_worker(config, *, max_steps, microbatch, forward_only=False, model_path=None):
    """Build the registered verl V1 actor worker with its native FSDP2 engine."""
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig, TrainingWorkerConfig
    from verl.workers.engine_workers import TrainingWorker

    from . import policy_engine  # noqa: F401 -- register the dedicated actor engine

    model = HFModelConfig(
        path=model_path or config.model_path,
        tokenizer_path=config.tokenizer_path or model_path or config.model_path,
        model_type="delta_policy",
        trust_remote_code=config.trust_remote_code,
        enable_gradient_checkpointing=config.gradient_checkpointing and not forward_only,
        use_remove_padding=False,
        use_fused_kernels=False,
        override_config={"attn_implementation": config.attention_implementation},
    )
    dtype = "bfloat16" if config.dtype == "bfloat16" else "float32"
    engine_config = FSDPEngineConfig(
        strategy="fsdp2",
        model_dtype=dtype,
        mixed_precision={"param_dtype": dtype, "reduce_dtype": "float32", "buffer_dtype": "float32"},
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
            lr_scheduler_type="constant",
        )
    )
    contents = ["model"] if forward_only else ["model", "optimizer", "extra", "hf_model"]
    worker = TrainingWorker(
        TrainingWorkerConfig(
            model_type="delta_policy",
            model_config=model,
            engine_config=engine_config,
            optimizer_config=optimizer,
            checkpoint_config=CheckpointConfig(save_contents=contents, load_contents=contents),
        )
    )
    worker.reset()
    if not forward_only:
        worker.set_loss_fn(delta_policy_loss)
    return worker


class RowIterator:
    """Deterministic global row permutations independent of rank and microbatch."""

    def __init__(self, size, seed, epoch=0, cursor=0):
        if size < 1 or epoch < 0 or not 0 <= cursor < size:
            raise ValueError("Invalid policy iterator position")
        self.size, self.seed, self.epoch, self.cursor = size, seed, epoch, cursor

    def take(self, count):
        if count < 1:
            raise ValueError("count must be positive")
        order = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(order)
        take = min(count, self.size - self.cursor)
        result = order[self.cursor : self.cursor + take]
        self.cursor += take
        if self.cursor == self.size:
            self.epoch += 1
            self.cursor = 0
        return result

    def state(self):
        return {"epoch": self.epoch, "cursor": self.cursor}


def _action_rows(batch):
    """Recover per-row action arrays from the response-aligned dense collator."""
    lengths = batch["attention_mask"].sum(-1).tolist()
    result = []
    for index, sequence_length in enumerate(lengths):
        action_length = int(sequence_length) - 1
        if action_length < 1:
            raise ValueError("Every policy sequence must provide at least one next-token action")
        input_ids = batch["input_ids"][index, :sequence_length].tolist()
        result.append(
            {
                "input_ids": input_ids,
                "actions": input_ids[1:],
                "old_log_probs": batch["old_log_probs"][index, :action_length],
                "advantages": batch["advantages"][index, :action_length],
                "policy_loss_mask": batch["policy_loss_mask"][index, :action_length],
                "response_mask": batch["response_mask"][index, :action_length],
                "kl_mask": batch["kl_mask"][index, :action_length],
                "ref_log_prob": batch["ref_log_prob"][index, :action_length] if "ref_log_prob" in batch else None,
            }
        )
    return result


def actor_tensor_batch(
    rows,
    pad_token_id,
    policy_config: DeltaPolicyConfig,
    training_config: PolicyTrainingConfig,
    *,
    dp_size,
    global_valid_sample_weight,
):
    """Make the real V1 nested actor batch, with source-aligned causal actions.

    The source collator drops the first retained input token because it has no
    preceding logit. Representing that token as a one-token prompt lets V1's
    nested response slicer retain precisely the same action positions, including
    when an offline tail window has removed the original prompt entirely.
    """
    from verl.utils import tensordict_utils as tu

    dense = policy_training_batch(rows, pad_token_id, policy_config)
    actions = _action_rows(dense)
    prompts = [[row["input_ids"][0]] for row in actions]
    action_ids = [row["actions"] for row in actions]
    sequence_ids = [row["input_ids"] for row in actions]
    response_action_fields = {
        "old_log_probs": [row["old_log_probs"].tolist() for row in actions],
        "advantages": [row["advantages"].tolist() for row in actions],
        "policy_loss_mask": [row["policy_loss_mask"].tolist() for row in actions],
        "response_mask": [row["response_mask"].tolist() for row in actions],
        "kl_mask": [row["kl_mask"].tolist() for row in actions],
    }
    if policy_config.kl_coef > 0.0 and policy_config.kl_reference == "reference":
        response_action_fields["ref_log_prob"] = [row["ref_log_prob"].tolist() for row in actions]
    sequence_lengths = [len(ids) for ids in sequence_ids]
    action_loss_mask = response_action_fields["response_mask"]
    tensors = {
        "input_ids": _nested(sequence_ids, torch.long),
        "position_ids": _nested([list(range(length)) for length in sequence_lengths], torch.long),
        "prompts": _nested(prompts, torch.long),
        "responses": _nested(action_ids, torch.long),
        "loss_mask": _nested(action_loss_mask, torch.float32),
        "temperature": torch.full((len(rows),), float(policy_config.train_logprob_temperature), dtype=torch.float32),
        "row_weight": dense["row_weight"],
        "sample_valid_mask": dense["sample_valid_mask"],
        **{name: _nested(values, torch.float32) for name, values in response_action_fields.items()},
    }
    data = TensorDict(tensors, batch_size=[len(rows)])
    tu.assign_non_tensor(
        data,
        delta_policy_config=policy_config.as_dict(),
        dp_size=int(dp_size),
        global_valid_sample_weight=float(global_valid_sample_weight),
        pad_token_id=int(pad_token_id),
        global_token_num=sequence_lengths,
        use_remove_padding=False,
        use_dynamic_bsz=False,
        use_fused_kernels=False,
        update_lr_scheduler=True,
    )
    return data


def language_model_batch(examples, *, pad_token_id, temperature, microbatch):
    """Build a no-padding causal batch for a frozen reference policy."""
    from verl.utils import tensordict_utils as tu

    sequences = [list(example.rollout.prompt_token_ids + example.rollout.response_token_ids) for example in examples]
    prompts = [list(example.rollout.prompt_token_ids) for example in examples]
    responses = [list(example.rollout.response_token_ids) for example in examples]
    lengths = [len(row) for row in sequences]
    data = TensorDict(
        {
            "input_ids": _nested(sequences, torch.long),
            "position_ids": _nested([list(range(length)) for length in lengths], torch.long),
            "prompts": _nested(prompts, torch.long),
            "responses": _nested(responses, torch.long),
            "loss_mask": _nested([[1.0] * length for length in lengths], torch.float32),
            "temperature": torch.full((len(examples),), float(temperature), dtype=torch.float32),
        },
        batch_size=[len(examples)],
    )
    tu.assign_non_tensor(
        data,
        compute_loss=False,
        return_model_output=True,
        pad_token_id=int(pad_token_id),
        global_token_num=lengths,
        use_remove_padding=False,
        use_dynamic_bsz=False,
        use_fused_kernels=False,
        micro_batch_size_per_gpu=int(microbatch),
        infer_micro_batch_size_per_gpu=int(microbatch),
    )
    return data


def reference_logprobs(
    examples,
    model_path,
    config,
    *,
    pad_token_id,
    world,
    rank,
    microbatch,
    max_length,
    temperature,
):
    """Compute fixed-reference response logprobs with the same V1 FSDP2 path."""
    if not examples:
        raise ValueError("Reference scoring requires at least one example")
    padded = list(examples)
    multiple = world * microbatch
    while len(padded) % multiple:
        padded.append(examples[0])
    local = padded[rank::world]
    for example in local:
        length = len(example.rollout.prompt_token_ids) + len(example.rollout.response_token_ids)
        if max_length is not None and length > max_length:
            raise ValueError(
                f"Reference sequence for {example.rollout.rollout_id} exceeds model max_length={max_length}"
            )
    worker = build_worker(config, max_steps=1, microbatch=microbatch, forward_only=True, model_path=model_path)
    result = worker.infer_batch(
        language_model_batch(
            local,
            pad_token_id=pad_token_id,
            temperature=temperature,
            microbatch=microbatch,
        )
    )
    if result is None or "log_probs" not in result:
        raise RuntimeError("Reference FSDP2 worker did not return model log_probs")
    sequence_logprobs = result["log_probs"].unbind()
    local_values = []
    for example, values in zip(local, sequence_logprobs, strict=True):
        prompt_length = len(example.rollout.prompt_token_ids)
        response_length = len(example.rollout.response_token_ids)
        selected = values[prompt_length - 1 : prompt_length + response_length - 1].float().cpu().tolist()
        if len(selected) != response_length:
            raise ValueError(f"Reference logprob alignment failed for rollout_id={example.rollout.rollout_id}")
        local_values.append((example.rollout.rollout_id, selected))
    del result, worker
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, local_values)
    values_by_id = {}
    for rank_values in gathered:
        for rollout_id, values in rank_values:
            previous = values_by_id.setdefault(rollout_id, values)
            if previous != values:
                raise RuntimeError(f"Reference logprobs differ across ranks for rollout_id={rollout_id}")
    expected = {example.rollout.rollout_id for example in examples}
    if set(values_by_id) != expected:
        raise RuntimeError("Reference scoring did not return every rollout")
    return values_by_id


def _pad_eval_rows(rows, world, microbatch):
    padded = list(rows)
    if not padded:
        raise ValueError("Evaluation split is empty")
    multiple = world * microbatch
    while len(padded) % multiple:
        dummy = dict(rows[0])
        dummy["id"] = f"__policy_eval_padding_{len(padded)}"
        dummy["rollout_id"] = dummy["id"]
        dummy["sample_valid"] = False
        dummy["row_weight"] = 0.0
        padded.append(dummy)
    return padded


def _pad_update_rows(rows, world, microbatch):
    """Pad an epoch-tail update with zero-weight copies for equal FSDP shards."""
    if not rows:
        raise ValueError("A policy update cannot contain zero real rows")
    result = list(rows)
    multiple = world * microbatch
    while len(result) % multiple:
        dummy = dict(rows[0])
        dummy["id"] = f"__policy_train_padding_{len(result)}"
        dummy["rollout_id"] = dummy["id"]
        dummy["sample_valid"] = False
        dummy["row_weight"] = 0.0
        result.append(dummy)
    return result


def evaluate(worker, rows, pad_token_id, loss_config, training_config, *, world, rank, microbatch):
    from verl.utils import tensordict_utils as tu

    padded = _pad_eval_rows(rows, world, microbatch)
    local = padded[rank::world]
    denominator = sum(float(row.get("row_weight", 1.0)) for row in rows if row.get("sample_valid", True))
    batch = actor_tensor_batch(
        local,
        pad_token_id,
        loss_config,
        training_config,
        dp_size=world,
        global_valid_sample_weight=denominator,
    )
    tu.assign_non_tensor(batch, compute_loss=True, update_lr_scheduler=False)
    result = worker.infer_batch(batch)
    if result is None:
        raise RuntimeError("Policy evaluation did not return loss metrics")
    from .train import reduce_engine_metrics

    metrics = reduce_engine_metrics(tu.get(result, "metrics"))
    return {
        "loss": metrics.get("actor/delta_policy_loss", metrics.get("loss", 0.0)),
        "pg_loss": metrics.get("actor/delta_pg_loss", 0.0),
        "kl_loss": metrics.get("actor/delta_kl_loss", 0.0),
        "rows": len(rows),
    }


def reserve_checkpoint(path):
    error = [None]
    if torch.distributed.get_rank() == 0:
        try:
            Path(path).mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error[0] = f"Refusing to create policy checkpoint {path}: {exc}"
    torch.distributed.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def export_final_hf(checkpoint, output):
    """Publish an existing complete checkpoint export, including after interruption."""
    error = [None]
    if torch.distributed.get_rank() == 0:
        source = Path(checkpoint) / "huggingface"
        destination = Path(output) / "final_hf"
        temporary = Path(output) / ".final_hf.tmp"
        try:
            if not (Path(checkpoint) / "COMPLETE").exists():
                raise ValueError(f"Cannot export an incomplete policy checkpoint: {checkpoint}")
            if not source.is_dir() or not any(source.glob("*.safetensors")):
                raise RuntimeError(f"FSDP2 actor checkpoint did not export safetensors: {source}")
            if destination.exists():
                if _path_fingerprint(source) != _path_fingerprint(destination):
                    raise FileExistsError(f"Existing final_hf differs from checkpoint: {destination}")
            else:
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(source, temporary)
                temporary.replace(destination)
        except Exception as exc:
            error[0] = str(exc)
    torch.distributed.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    torch.distributed.barrier()


def save_checkpoint(worker, output, state, *, final=False):
    path = Path(output) / f"step_{state['global_step']:08d}"
    reserve_checkpoint(path)
    worker.save_checkpoint(str(path), global_step=state["global_step"])
    if torch.distributed.get_rank() == 0:
        (path / "trainer_state.json.tmp").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        (path / "trainer_state.json.tmp").replace(path / "trainer_state.json")
        (path / "COMPLETE").write_text("complete\n")
    torch.distributed.barrier()
    if final:
        export_final_hf(path, output)
    return path


def _mode_override(mode):
    if mode is None:
        return None
    return {"offline": "offline_legacy", "online": "online_ppo"}[mode]


def run(args):
    raw_config = yaml.safe_load(Path(args.config).read_text())
    training_config = PolicyTrainingConfig.from_mapping(raw_config)
    loss_config = delta_policy_config_from_mapping(raw_config, mode=_mode_override(args.mode))
    if args.initialize:
        training_config = replace(training_config, model_path=args.initialize)

    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.global_batch < 1 or args.microbatch < 1 or args.scoring_microbatch < 1:
        raise ValueError("global_batch, microbatch and scoring_microbatch must be positive")
    if args.global_batch % (world * args.microbatch):
        raise ValueError("global_batch must be divisible by world_size * microbatch")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(training_config.seed)
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)

    from transformers import AutoConfig

    backbone_config = AutoConfig.from_pretrained(
        training_config.model_path, trust_remote_code=training_config.trust_remote_code
    )
    native_length = getattr(backbone_config, "max_position_embeddings", None)
    if native_length is None:
        native_length = getattr(backbone_config, "n_positions", None)
    if loss_config.max_length is not None and native_length is not None and loss_config.max_length > native_length:
        raise ValueError(f"Policy max_length={loss_config.max_length} exceeds model context={native_length}")
    effective_max_length = loss_config.max_length or native_length

    rollouts, labels = read_rows(args.rollouts), read_rows(args.labels)
    examples, diagnostics = adapt_legacy_rows(rollouts, labels)
    split = json.loads(Path(args.split).read_text())
    if set(split) != {example.rollout.rollout_id for example in examples} or set(split.values()) != {"train", "eval"}:
        raise ValueError("Split must map every rollout ID exactly once to train or eval, both nonempty")
    train_examples = [example for example in examples if split[example.rollout.rollout_id] == "train"]
    eval_examples = [example for example in examples if split[example.rollout.rollout_id] == "eval"]

    tokenizer_sha256, tokenizer = tokenizer_fingerprint(
        training_config.tokenizer_path or training_config.model_path,
        training_config.trust_remote_code,
        expected_vocab_size=getattr(backbone_config, "vocab_size", None),
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer requires pad or eos token")
    actor_fingerprint = _path_fingerprint(training_config.model_path)

    files = {name: file_hash(getattr(args, name)) for name in ("rollouts", "labels", "split")}
    raw_predictions = None
    critic_fingerprint = None
    if loss_config.advantage_source == "critic":
        if not args.critic:
            raise ValueError("critic advantage source requires --critic")
        from .checkpoint import read_artifact
        from .score import FrozenDeltaWorker
        from .train import build_worker as build_critic_worker
        from .training_config import ScalarConfig

        critic_meta = read_artifact(args.critic)
        critic_config = ScalarConfig(**critic_meta["config"])
        if critic_config.delta_label_mode != loss_config.label_mode:
            raise ValueError("Policy label_mode differs from the selected critic artifact")
        if critic_meta.get("tokenizer_sha256") != tokenizer_sha256:
            raise ValueError("Policy tokenizer differs from the selected critic artifact")
        critic_fingerprint = _path_fingerprint(args.critic)
        critic_worker = build_critic_worker(
            critic_config,
            max_steps=1,
            microbatch=args.scoring_microbatch,
            initial_artifact=args.critic,
            forward_only=True,
        )
        scorer = FrozenDeltaWorker(
            args.critic, device="cuda", microbatch=args.scoring_microbatch, engine_worker=critic_worker
        )
        scored = scorer.score(examples)
        raw_predictions = {}
        for example, result in zip(examples, scored, strict=True):
            raw_predictions[example.rollout.rollout_id] = {
                state.token_index: float(result["delta_pred_raw"][state.token_index]) for state in example.states
            }
        del scorer, critic_worker
        torch.cuda.empty_cache()
        torch.distributed.barrier()
    elif args.critic:
        raise ValueError("--critic is only used when advantage_source=critic")

    reference_values = None
    reference_id = training_config.reference_policy_id
    reference_fingerprint = None
    if loss_config.kl_coef > 0.0 and loss_config.kl_reference == "reference":
        reference_path = training_config.reference_model_path
        if not reference_path:
            raise ValueError("reference KL requires reference_model_path")
        reference_hash, _ = tokenizer_fingerprint(
            reference_path,
            training_config.trust_remote_code,
            expected_vocab_size=getattr(backbone_config, "vocab_size", None),
        )
        if reference_hash != tokenizer_sha256:
            raise ValueError("Fixed reference policy tokenizer differs from the actor tokenizer")
        reference_fingerprint = _path_fingerprint(reference_path)
        reference_id = reference_id or fingerprint({"policy": reference_fingerprint})
        reference_values = reference_logprobs(
            examples,
            reference_path,
            training_config,
            pad_token_id=pad_token_id,
            world=world,
            rank=rank,
            microbatch=args.scoring_microbatch,
            max_length=effective_max_length,
            temperature=loss_config.train_logprob_temperature,
        )

    behavior_values = None
    behavior_actor_fingerprint = None
    if loss_config.behavior_logprob_source == "actor_snapshot":
        if args.behavior_logprobs:
            raise ValueError("actor_snapshot computes old logprobs itself; omit --behavior-logprobs")
        behavior_actor_fingerprint = actor_fingerprint
        behavior_values = reference_logprobs(
            examples,
            training_config.model_path,
            training_config,
            pad_token_id=pad_token_id,
            world=world,
            rank=rank,
            microbatch=args.scoring_microbatch,
            max_length=effective_max_length,
            temperature=loss_config.train_logprob_temperature,
        )
    elif loss_config.behavior_logprob_source == "precomputed":
        if not args.behavior_logprobs:
            raise ValueError("precomputed behavior requires --behavior-logprobs")
        behavior_values, behavior_actor_fingerprint = read_precomputed_behavior(
            args.behavior_logprobs, loss_config, actor_fingerprint=actor_fingerprint
        )
        files["behavior_logprobs"] = file_hash(args.behavior_logprobs)
    elif args.behavior_logprobs:
        raise ValueError("--behavior-logprobs requires behavior_logprob_source=precomputed")

    fit_stats = loss_config.advantage_normalization == "standardize"
    train_rows, stats = prepare_policy_rows(
        train_examples,
        raw_predictions,
        behavior_values,
        loss_config,
        reference_logprobs_by_id=reference_values,
        reference_policy_id=reference_id,
        fit_stats=fit_stats,
    )
    eval_rows, _ = prepare_policy_rows(
        eval_examples,
        raw_predictions,
        behavior_values,
        loss_config,
        reference_logprobs_by_id=reference_values,
        reference_policy_id=reference_id,
        stats=stats,
    )
    for row in train_rows + eval_rows:
        length = len(row["prompt_token_ids"]) + len(row["response_token_ids"])
        if effective_max_length is not None and loss_config.window == "error" and length > effective_max_length:
            raise ValueError(f"Full policy sequence {row['id']} length {length} exceeds context={effective_max_length}")
    stats_dict = None if stats is None else asdict(stats)
    if args.max_steps is None:
        max_steps = math.ceil(len(train_rows) / args.global_batch) * training_config.rl_epochs
    else:
        max_steps = args.max_steps
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.save_every is not None:
        save_every = args.save_every
    else:
        save_every = training_config.save_every_steps
    if save_every < 1:
        raise ValueError("save_every must be positive")

    signature_payload = {
        "policy_config": asdict(loss_config),
        "training_config": asdict(training_config),
        "files": files,
        "critic": critic_fingerprint,
        "initial_actor": actor_fingerprint,
        "behavior_actor": behavior_actor_fingerprint,
        "reference": reference_fingerprint,
        "tokenizer": tokenizer_sha256,
        "global_batch": args.global_batch,
        "microbatch": args.microbatch,
        "scoring_microbatch": args.scoring_microbatch,
        "max_steps": max_steps,
        "world_size": world,
        "normalization": stats_dict,
        "train_ids": [row["id"] for row in train_rows],
    }
    signature = fingerprint(signature_payload)
    state = {
        "global_step": 0,
        "epoch": 0,
        "cursor": 0,
        "signature": signature,
        "iterator_seed": training_config.seed,
        "train_row_count": len(train_rows),
        "world_size": world,
        "normalization": stats_dict,
        "initial_actor_fingerprint": actor_fingerprint,
        "behavior_actor_fingerprint": behavior_actor_fingerprint,
        "reference_policy_id": reference_id,
    }
    if args.resume:
        checkpoint = Path(args.resume)
        if not (checkpoint / "COMPLETE").exists():
            raise ValueError("Incomplete policy resume checkpoint")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state.get("signature") != signature or state.get("world_size") != world:
            raise ValueError("Exact policy resume requires identical data, config, reference and topology")
        if state["global_step"] > max_steps:
            raise ValueError("Resume step exceeds configured max_steps")

    worker = build_worker(training_config, max_steps=max_steps, microbatch=args.microbatch)
    if args.resume:
        worker.load_checkpoint(str(args.resume))
    if state.get("iterator_seed") != training_config.seed or state.get("train_row_count") != len(train_rows):
        raise ValueError("Resume row-order state differs from current policy data")
    iterator = RowIterator(len(train_rows), training_config.seed, state["epoch"], state["cursor"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if state["global_step"] == max_steps:
        if not args.resume:
            raise RuntimeError("Completed policy state requires a resume checkpoint")
        export_final_hf(args.resume, output)
        torch.distributed.destroy_process_group()
        return
    end = min(max_steps, args.stop_after or max_steps)
    from verl.utils import tensordict_utils as tu

    while state["global_step"] < end:
        update_indices = iterator.take(args.global_batch)
        real_update_rows = [train_rows[index] for index in update_indices]
        update_rows = _pad_update_rows(real_update_rows, world, args.microbatch)
        local_rows = update_rows[rank::world]
        denominator = sum(
            float(row.get("row_weight", 1.0)) * float(row.get("sample_valid", True)) for row in real_update_rows
        )
        batch = actor_tensor_batch(
            local_rows,
            pad_token_id,
            loss_config,
            training_config,
            dp_size=world,
            global_valid_sample_weight=denominator,
        )
        result = worker.train_batch(batch)
        if result is None:
            raise RuntimeError("Policy actor update returned no metrics")
        from .train import reduce_engine_metrics

        update_metrics = reduce_engine_metrics(tu.get(result, "metrics"))
        step = state["global_step"] + 1
        metrics = {
            "step": step,
            "train_loss": update_metrics.get("actor/delta_policy_loss", update_metrics.get("loss")),
            "train_pg_loss": update_metrics.get("actor/delta_pg_loss"),
            "train_kl_loss": update_metrics.get("actor/delta_kl_loss"),
            "train_rows": len(real_update_rows),
            "provenance": {
                "signature": signature,
                "mode": loss_config.mode,
                "reference_policy_id": reference_id,
                "global_batch": args.global_batch,
                "microbatch": args.microbatch,
                "scoring_microbatch": args.scoring_microbatch,
                "world_size": world,
                "tokenizer_sha256": tokenizer_sha256,
                "initial_actor_fingerprint": actor_fingerprint,
                "behavior_actor_fingerprint": behavior_actor_fingerprint,
                "behavior_logprob_source": loss_config.behavior_logprob_source,
                "train_logprob_temperature": loss_config.train_logprob_temperature,
            },
        }
        state.update(global_step=step, **iterator.state())
        final = step == end
        if step % save_every == 0 or final:
            metrics["validation"] = evaluate(
                worker,
                eval_rows,
                pad_token_id,
                loss_config,
                training_config,
                world=world,
                rank=rank,
                microbatch=args.microbatch,
            )
            save_checkpoint(worker, output, state, final=final)
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
    if state["global_step"] < max_steps and args.stop_after:
        # The caller intentionally interrupted a run for a later --resume.
        pass
    elif not (output / "final_hf").is_dir():
        raise RuntimeError("Training ended without a final_hf actor export")
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "rollouts", "labels", "split", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--critic")
    parser.add_argument("--behavior-logprobs")
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--scoring-microbatch", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--initialize")
    parser.add_argument("--mode", choices=["offline", "online"])
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after must be positive")
    run(args)


if __name__ == "__main__":
    main()
