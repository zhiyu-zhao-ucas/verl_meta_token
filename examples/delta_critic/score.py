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
"""Frozen prefix scorer with explicit legacy windows and TransferQueue IO."""

import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path

import torch

from .checkpoint import load_weights, read_artifact, tokenizer_fingerprint
from .contracts import DeltaExample
from .scalar_model import DeltaScalarModel
from .training_config import ScalarConfig


@dataclass(frozen=True)
class ScoringRow:
    """Unpadded response row plus selected response positions, with no label dependency."""

    rollout_id: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    selected_token_indices: tuple[int, ...]
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.rollout_id, str) or not self.rollout_id or not self.prompt_token_ids:
            raise ValueError("Scoring rows require a stable row ID and nonempty prompt")
        for token in self.prompt_token_ids + self.response_token_ids:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("Scoring token IDs must be nonnegative integers")
        indices = self.selected_token_indices
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("Selected response positions must be integers")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate selected response positions for row {self.rollout_id}")
        if tuple(sorted(indices)) != indices:
            raise ValueError(f"Selected response positions must be sorted for row {self.rollout_id}")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.response_token_ids):
                raise ValueError(f"Selected response position {index!r} is out of range for row {self.rollout_id}")


def _as_scoring_row(example):
    if isinstance(example, ScoringRow):
        return example
    if isinstance(example, DeltaExample):
        return ScoringRow(
            rollout_id=example.rollout.rollout_id,
            prompt_token_ids=example.rollout.prompt_token_ids,
            response_token_ids=example.rollout.response_token_ids,
            selected_token_indices=tuple(state.token_index for state in example.states),
            metadata=example.rollout.metadata,
        )
    raise TypeError("score() expects ScoringRow or DeltaExample rows")


def _one_dimensional_list(value, name):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple) or any(isinstance(item, list | tuple) for item in value):
        raise ValueError(f"TransferQueue field {name!r} must contain one-dimensional rows")
    return list(value)


class FrozenDeltaWorker:
    def __init__(self, artifact, *, device="cuda", microbatch=1, scoring_max_length=None, engine_worker=None):
        if microbatch < 1:
            raise ValueError("microbatch must be positive")
        self.metadata = read_artifact(artifact)
        self.config = ScalarConfig(**self.metadata["config"])
        self.checkpoint_config = self.config
        self.window_changes = []
        if scoring_max_length is not None:
            if isinstance(scoring_max_length, bool) or not isinstance(scoring_max_length, int):
                raise ValueError("scoring_max_length must be an integer")
            if scoring_max_length < self.config.max_length:
                raise ValueError("Scoring window override may only expand the checkpoint window")
            if scoring_max_length != self.config.max_length:
                self.window_changes.append(
                    {"change": "expanded_scoring_window", "old": self.config.max_length, "new": scoring_max_length}
                )
                self.config = replace(self.config, max_length=scoring_max_length)
        token_hash, tokenizer = tokenizer_fingerprint(
            self.config.tokenizer_path or self.config.model_path, self.config.trust_remote_code
        )
        if not self.metadata.get("tokenizer_sha256") or token_hash != self.metadata["tokenizer_sha256"]:
            raise ValueError("Tokenizer has changed since the checkpoint was written")
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer requires pad or eos token")
        self.engine_worker = engine_worker
        if engine_worker is None:
            self.model = DeltaScalarModel.from_config(replace(self.config, gradient_checkpointing=False))
            load_weights(self.model, torch.load(Path(artifact) / "model.pt", map_location="cpu", weights_only=True))
            self.model.to(device).eval().requires_grad_(False)
        else:
            self.model = self._validate_engine_worker(engine_worker, artifact)
        model_config = getattr(getattr(self.model, "module", self.model), "config", None)
        native_length = getattr(self.model, "max_position_embeddings", None)
        if native_length is None:
            native_length = getattr(model_config, "max_position_embeddings", None)
        if native_length is not None and self.config.max_length > native_length:
            raise ValueError(
                f"Scoring max_length {self.config.max_length} exceeds backbone native context {native_length}"
            )
        expected_vocab_size = getattr(model_config, "vocab_size", None)
        if expected_vocab_size is None:
            raise ValueError("Critic model does not expose its backbone vocabulary size")
        if max(self.tokenizer.get_vocab().values()) >= expected_vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds the critic backbone vocabulary")
        self.device, self.microbatch = torch.device(device), microbatch

    def _validate_engine_worker(self, worker, artifact):
        """Bind scoring to a V1 TrainingWorker initialized from this exact artifact."""
        configured_artifact = getattr(worker, "delta_initial_artifact", None)
        if configured_artifact is None:
            model_config = getattr(worker, "model_config", None)
            hf_config = getattr(model_config, "hf_config", None)
            configured_artifact = getattr(hf_config, "delta_initial_artifact", None)
        if configured_artifact is None or Path(configured_artifact).resolve() != Path(artifact).resolve():
            raise ValueError("V1 scoring worker must be initialized from the same portable critic artifact")
        worker_config = getattr(worker, "engine_config", None)
        if worker_config is None or not getattr(worker_config, "forward_only", False):
            raise ValueError("V1 frozen scoring requires a forward_only TrainingWorker")
        if getattr(worker, "optimizer_config", None) is not None:
            raise ValueError("V1 frozen scoring worker must not have an optimizer")
        engine = getattr(worker, "engine", None)
        if engine is None or not hasattr(engine, "module"):
            raise ValueError("V1 scoring worker has no initialized engine module")
        if getattr(engine, "optimizer", None) is not None:
            raise ValueError("V1 frozen scoring engine unexpectedly has an optimizer")
        model = engine.module
        saved_config = getattr(
            getattr(getattr(worker, "model_config", None), "hf_config", None), "delta_scalar_config", None
        )
        if saved_config is not None:
            worker_scalar_config = ScalarConfig(**saved_config)
            if worker_scalar_config != self.checkpoint_config:
                raise ValueError("V1 worker scalar configuration differs from the checkpoint")
        model.eval().requires_grad_(False)
        return model

    def score(self, examples):
        """Score selected states, frozen and without building an autograd graph.

        This is deliberately not decorated with ``torch.inference_mode()``: the
        V1 path hands these tensors to FSDP2, whose pre-forward hook calls
        ``_unsafe_preserve_version_counter`` and therefore fails on inference
        tensors ("Inference tensors do not track version counter"). The direct
        path opts into inference mode locally instead, and the V1 engine already
        runs its forward under ``torch.no_grad()`` with ``requires_grad_(False)``
        parameters.
        """
        rows = [_as_scoring_row(example) for example in examples]
        ids = [row.rollout_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scoring row IDs")
        outputs = {}
        prefixes = []
        for row in rows:
            n = len(row.response_token_ids)
            row_id = row.rollout_id
            outputs[row_id] = {
                "rollout_id": row_id,
                "delta_pred_normalized": [0.0] * n,
                "delta_pred_raw": [0.0] * n,
                "critic_signal_mask": [0.0] * n,
                "critic_checkpoint": self.metadata["weights_sha256"],
                "critic_stats_version": self.metadata["stats_version"],
                "critic_source_sha256": self.metadata.get("source_sha256"),
                "critic_window": {
                    "policy": self.config.window_policy,
                    "max_length": self.config.max_length,
                    "semantic_changes": self.window_changes,
                    "prefix_shifts": {},
                },
            }
            for token_index in row.selected_token_indices:
                tokens = row.prompt_token_ids + row.response_token_ids[: token_index + 1]
                shift = max(0, len(tokens) - self.config.max_length)
                if shift and self.config.window_policy == "full":
                    raise ValueError(
                        f"Selected prefix {row_id}:{token_index} length {len(tokens)} exceeds {self.config.max_length}"
                    )
                prefix = {
                    "rollout_id": row_id,
                    "token_index": token_index,
                    "input_ids": tokens[shift:],
                    "value_position": len(tokens) - 1 - shift,
                }
                prefixes.append(prefix)
                outputs[row_id]["critic_window"]["prefix_shifts"][str(token_index)] = shift
        prefixes.sort(key=lambda r: (len(r["input_ids"]), r["rollout_id"], r["token_index"]))
        norm = self.metadata["normalization"]
        self.model.eval()
        for start in range(0, len(prefixes), self.microbatch):
            batch = prefixes[start : start + self.microbatch]
            if self.engine_worker is not None:
                batch_predictions = self._predict_with_v1_worker(batch)
            else:
                width = max(len(r["input_ids"]) for r in batch)
                tokens = torch.full((len(batch), width), self.pad_token_id, dtype=torch.long, device=self.device)
                mask = torch.zeros_like(tokens)
                for i, row in enumerate(batch):
                    length = len(row["input_ids"])
                    tokens[i, :length] = torch.tensor(row["input_ids"], device=self.device)
                    mask[i, :length] = 1
                context = (
                    torch.autocast(self.device.type, dtype=torch.bfloat16)
                    if self.config.scoring_autocast == "bfloat16"
                    else nullcontext()
                )
                with torch.inference_mode(), context:
                    predictions = self.model(tokens, mask).float()
                batch_predictions = [predictions[i, row["value_position"]] for i, row in enumerate(batch)]
            for i, row in enumerate(batch):
                prediction = batch_predictions[i]
                value = float(prediction.item())
                if not isfinite(value):
                    raise ValueError("Nonfinite critic prediction")
                raw = value * norm["std"] + norm["mean"] if norm["enabled"] else value
                dest, index = outputs[row["rollout_id"]], row["token_index"]
                dest["delta_pred_normalized"][index] = value
                dest["delta_pred_raw"][index] = raw
                dest["critic_signal_mask"][index] = 1.0
        return [outputs[row_id] for row_id in ids]

    def _predict_with_v1_worker(self, rows):
        """Run sorted selected prefixes through V1 TrainingWorker.infer_batch."""
        from tensordict import TensorDict

        from verl.utils import tensordict_utils as tu

        microbatch = int(self.engine_worker.engine_config.infer_micro_batch_size_per_gpu)
        if microbatch < 1:
            raise ValueError("V1 infer_micro_batch_size_per_gpu must be positive")
        padded_rows = list(rows)
        while len(padded_rows) % microbatch:
            padded_rows.append(rows[0])
        width = max(len(row["input_ids"]) for row in padded_rows)
        jagged_ids = torch.nested.as_nested_tensor(
            [torch.tensor(row["input_ids"], dtype=torch.long) for row in padded_rows], layout=torch.jagged
        )
        attention_mask = torch.zeros((len(padded_rows), width), dtype=torch.long)
        for i, row in enumerate(padded_rows):
            attention_mask[i, : len(row["input_ids"])] = 1
        data = TensorDict(
            {
                "input_ids": jagged_ids,
                "attention_mask": attention_mask,
                # DeltaFSDPEngine needs these fields to enter the regular infer path;
                # the zero loss mask keeps this scoring batch out of all loss terms.
                "target": torch.zeros((len(padded_rows), width), dtype=torch.float32),
                "loss_mask": torch.zeros((len(padded_rows), width), dtype=torch.float32),
            },
            batch_size=[len(padded_rows)],
        )
        tu.assign_non_tensor(
            data,
            compute_loss=False,
            # The framework's FLOPs counter expects one entry per row
            # (`sum(batch_seqlens)`), not a single total.
            global_token_num=attention_mask.sum(-1).tolist(),
            pad_token_id=self.pad_token_id,
            return_model_output=True,
        )
        result = self.engine_worker.infer_batch(data)
        if hasattr(result, "batch") and result.batch is not None:
            result = result.batch
        predictions = result["delta_scalar"]
        return [predictions[i][row["value_position"]].float() for i, row in enumerate(rows)]

    def score_transfer_queue(self, meta, *, selected_indices_by_key=None, selection_field="selected_token_indices"):
        """Read unpadded V1 rows and write response-indexed predictions to the same TQ keys.

        Indices may be supplied by the caller or already stored under
        ``selection_field``. Queue keys are the stable rollout IDs; all three
        response vectors retain the original jagged row lengths.
        """
        import transfer_queue as tq
        from tensordict import NonTensorStack, TensorDict

        keys = list(meta.keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("TransferQueue scoring requires nonempty unique row keys")
        if not meta.partition_id:
            raise ValueError("TransferQueue metadata must include a partition_id")
        if selected_indices_by_key is not None:
            selected_keys = set(selected_indices_by_key)
            if selected_keys - set(keys):
                raise ValueError("Selected indices include row IDs outside this TransferQueue batch")
            if set(keys) - selected_keys:
                raise ValueError("Selected indices are missing for one or more TransferQueue rows")
        source_fields = ["prompts", "responses"]
        if selected_indices_by_key is None:
            if selection_field not in (meta.fields or []):
                raise ValueError(f"TransferQueue rows must include selection field {selection_field!r}")
            source_fields.append(selection_field)
        data = tq.kv_batch_get_by_meta(meta, select_fields=source_fields)
        examples = []
        for i, key in enumerate(keys):
            prompt = tuple(_one_dimensional_list(data["prompts"][i], "prompts"))
            response = tuple(_one_dimensional_list(data["responses"][i], "responses"))
            selected = (
                selected_indices_by_key[key]
                if selected_indices_by_key is not None
                else _one_dimensional_list(data[selection_field][i], selection_field)
            )
            selected = tuple(selected)
            examples.append(ScoringRow(key, prompt, response, selected))
        scored = self.score(examples)
        fields = {}
        for name in ("delta_pred_normalized", "delta_pred_raw", "critic_signal_mask"):
            fields[name] = torch.nested.as_nested_tensor(
                [torch.tensor(row[name]) for row in scored], layout=torch.jagged
            )
        for name in ("critic_checkpoint", "critic_stats_version", "critic_source_sha256", "critic_window"):
            fields[name] = NonTensorStack(*[row[name] for row in scored])
        fields["critic_row_id"] = NonTensorStack(*keys)
        return tq.kv_batch_put(
            keys=keys, partition_id=meta.partition_id, fields=TensorDict(fields, batch_size=[len(scored)])
        )


def main():
    from .train import read_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument(
        "--selected", required=True, help="MC/selected rows; only rollout_id and token_index are required"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--scoring-max-length", type=int)
    args = parser.parse_args()
    rollouts, selections = read_rows(args.rollouts), read_rows(args.selected)
    rollout_by_id = {}
    for row in rollouts:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in rollout_by_id:
            raise ValueError(f"Missing or duplicate rollout ID {row_id!r}")
        rollout_by_id[row_id] = row
    grouped = {row_id: [] for row_id in rollout_by_id}
    for selected in selections:
        row_id = str(selected.get("rollout_id", ""))
        if row_id not in grouped:
            raise ValueError(f"Selected state references unknown rollout {row_id!r}")
        grouped[row_id].append(selected)
    examples = []
    for row_id, row in rollout_by_id.items():
        prompt = tuple(row["prompt_token_ids"])
        response = tuple(row["response_token_ids"])
        states = grouped[row_id]
        indices = tuple(sorted(state["token_index"] for state in states))
        examples.append(ScoringRow(row_id, prompt, response, indices, metadata=row))
    worker = FrozenDeltaWorker(
        args.checkpoint, device=args.device, microbatch=args.microbatch, scoring_max_length=args.scoring_max_length
    )
    scored = worker.score(examples)
    with open(args.output, "w") as stream:
        for row in scored:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
