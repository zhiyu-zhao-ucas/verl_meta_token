# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Frozen boundary scorer: raw prefix values become selected-segment deltas."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from .checkpoint import load_weights, read_artifact, tokenizer_fingerprint
from .scalar_model import DeltaScalarModel
from .score import FrozenDeltaWorker, _as_scoring_row
from .training_config import ScalarConfig
from .value_difference import SEMANTICS, boundary_positions, differences


class FrozenBoundaryWorker(FrozenDeltaWorker):
    """Reuse the delta TransferQueue contract, with explicit value semantics.

    Full-row causal inference gives the same prefix hidden states as individual
    prefix inference. Both controlled arms use this exact batching algorithm.
    The value arm necessarily reads the later boundary when forming a delta;
    that is part of its prescribed prediction rule, not equal information at
    each individual scalar readout to the direct-delta arm.
    """

    def __init__(self, artifact, *, device="cuda", microbatch=1, scoring_max_length=None):
        self.metadata = read_artifact(artifact)
        self.target = self.metadata.get("boundary_target")
        if self.metadata.get("artifact_kind") != "boundary_scalar" or self.target not in {"value", "direct_delta"}:
            raise ValueError("Expected a controlled boundary-scalar artifact")
        if self.metadata.get("boundary_semantics") != SEMANTICS:
            raise ValueError("Boundary semantics differ from the scoring implementation")
        self.config = self.checkpoint_config = ScalarConfig(**self.metadata["config"])
        if self.config.window_policy != "full" or self.metadata["normalization"]["enabled"]:
            raise ValueError("Boundary scoring requires full context and raw reward units")
        if scoring_max_length is not None and scoring_max_length != self.config.max_length:
            raise ValueError("Boundary scorer forbids changing the training context limit")
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("microbatch must be positive")
        token_hash, self.tokenizer = tokenizer_fingerprint(
            self.config.tokenizer_path or self.config.model_path, self.config.trust_remote_code
        )
        if token_hash != self.metadata["tokenizer_sha256"]:
            raise ValueError("Tokenizer identity differs from the trained artifact")
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token")
        self.model = DeltaScalarModel.from_config(replace(self.config, gradient_checkpointing=False))
        load_weights(self.model, torch.load(Path(artifact) / "model.pt", map_location="cpu", weights_only=True))
        self.model.to(device).eval().requires_grad_(False)
        self.device, self.microbatch = torch.device(device), microbatch
        self.engine_worker = None
        self.window_changes = []
        if max(self.tokenizer.get_vocab().values()) >= self.model.config.vocab_size:
            raise ValueError("Tokenizer exceeds the model vocabulary")

    def score(self, examples):
        rows = [_as_scoring_row(row) for row in examples]
        ids = [row.rollout_id for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate scoring row IDs")
        result = {}
        ordered = sorted(
            rows, key=lambda row: (len(row.prompt_token_ids) + len(row.response_token_ids), row.rollout_id)
        )
        self.model.eval()
        for start in range(0, len(ordered), self.microbatch):
            chunk = ordered[start : start + self.microbatch]
            width = max(len(row.prompt_token_ids) + len(row.response_token_ids) for row in chunk)
            if width > self.config.max_length:
                raise ValueError("Boundary scoring context exceeds training context; truncation forbidden")
            tokens = torch.full((len(chunk), width), self.pad_token_id, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(tokens)
            for index, row in enumerate(chunk):
                sequence = row.prompt_token_ids + row.response_token_ids
                tokens[index, : len(sequence)] = torch.tensor(sequence, device=self.device)
                mask[index, : len(sequence)] = 1
            context = (
                torch.autocast(self.device.type, dtype=torch.bfloat16)
                if self.config.scoring_autocast == "bfloat16"
                else nullcontext()
            )
            with torch.inference_mode(), context:
                prediction = self.model(tokens, mask).float()
            for index, row in enumerate(chunk):
                n = len(row.response_token_ids)
                positions = boundary_positions(len(row.prompt_token_ids), n, row.selected_token_indices)
                raw = prediction[index, list(positions)].cpu().tolist()
                if not all(torch.isfinite(torch.tensor(raw))):
                    raise ValueError("Nonfinite boundary prediction")
                deltas = differences(raw) if self.target == "value" else tuple(raw[:-1])
                vector, signal = [0.0] * n, [0.0] * n
                for selected, delta in zip(row.selected_token_indices, deltas, strict=True):
                    vector[selected], signal[selected] = delta, 1.0
                result[row.rollout_id] = {
                    "rollout_id": row.rollout_id,
                    "delta_pred_normalized": list(vector),
                    "delta_pred_raw": vector,
                    "critic_signal_mask": signal,
                    "critic_checkpoint": self.metadata["weights_sha256"],
                    "critic_stats_version": self.metadata["stats_version"],
                    "critic_source_sha256": self.metadata.get("source_sha256"),
                    "critic_window": {
                        "policy": "full",
                        "max_length": self.config.max_length,
                        "semantic_changes": [],
                        "prefix_shifts": {},
                        "boundary_target": self.target,
                        "boundary_indices": [*row.selected_token_indices, n],
                        "boundary_predictions_raw": raw,
                        "terminal_source": "model_prediction",
                    },
                }
        return [result[row_id] for row_id in ids]
