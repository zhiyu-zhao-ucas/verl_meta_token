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
"""Online delta critic: score before fitting and retain checkpoint normalization."""

from contextlib import nullcontext
from math import isfinite
from pathlib import Path

import torch

from .legacy_adapter import adapt_legacy_rows
from .scalar_loss import per_sample_loss
from .score import FrozenDeltaWorker
from .training_data import collate, ordinary_rows


class OnlineDeltaWorker(FrozenDeltaWorker):
    """Single-device critic with full-context microbatch accumulation."""

    def __init__(self, artifact, *, update_config=None, **kwargs):
        super().__init__(artifact, **kwargs)
        if self.config.objective != "local_td0" or self.config.delta_label_mode != "selected_segment":
            raise ValueError("Online MC critic requires local_td0/selected_segment")
        self.update_config = dict(update_config or {})
        self.epochs = self.update_config.get("epochs", 1)
        self.microbatch_train = self.update_config.get("microbatch", 1)
        self.mini_batch_size = self.update_config.get("mini_batch_size")
        for name, value in (("epochs", self.epochs), ("microbatch", self.microbatch_train)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"critic_update.{name} must be a positive integer")
        if self.mini_batch_size is not None and (
            isinstance(self.mini_batch_size, bool)
            or not isinstance(self.mini_batch_size, int)
            or self.mini_batch_size < 1
        ):
            raise ValueError("critic_update.mini_batch_size must be a positive integer")
        learning_rate = self.update_config.get("learning_rate", self.config.learning_rate)
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("critic_update.learning_rate must be finite and positive")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.update_step = 0
        if self.config.gradient_checkpointing:
            self.model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def score(self, examples):
        outputs = super().score(examples)
        for row in outputs:
            row["critic_update_step"] = self.update_step
        return outputs

    def fit_mc(self, records):
        rollouts = [record["rollout"] for record in records]
        labels = [label for record in records for label in record["labels"]]
        examples, diagnostics = adapt_legacy_rows(rollouts, labels)
        if diagnostics:
            raise ValueError(f"Online MC label diagnostics: {diagnostics}")
        for record, example in zip(records, examples, strict=True):
            indices = [state.token_index for state in example.states]
            if indices != sorted(state["token_index"] for state in record["states"]):
                raise ValueError("MC labels do not cover every selected state")
        rows = ordinary_rows(examples, self.config)
        if not rows:
            raise ValueError("Online critic update requires real trajectories")
        normalization = self.metadata["normalization"]
        # Each mini-batch makes one optimizer update; microbatches retain its
        # sample denominator and the checkpoint's target statistics.
        total_loss = 0.0
        optimizer_steps = 0
        self.model.train().requires_grad_(True)
        try:
            for _ in range(self.epochs):
                mini_size = self.mini_batch_size or len(rows)
                for mini_start in range(0, len(rows), mini_size):
                    mini_rows = rows[mini_start : mini_start + mini_size]
                    self.optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(mini_rows), self.microbatch_train):
                        batch = collate(
                            mini_rows[start : start + self.microbatch_train],
                            self.config,
                            normalization,
                            self.pad_token_id,
                        )
                        batch = {
                            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                            for key, value in batch.items()
                        }
                        context = (
                            torch.autocast(
                                self.device.type,
                                dtype=getattr(
                                    torch,
                                    self.config.dtype if self.config.training_autocast == "model_dtype" else "bfloat16",
                                ),
                            )
                            if self.config.training_autocast != "none" and self.device.type == "cuda"
                            else nullcontext()
                        )
                        with context:
                            prediction = self.model(
                                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                            )
                            losses, _, _, _ = per_sample_loss(prediction, batch, self.config, normalization)
                            loss = losses.sum() / len(mini_rows)
                        if not torch.isfinite(loss):
                            raise ValueError("Online critic loss is not finite")
                        loss.backward()
                        total_loss += loss.detach().item() * len(mini_rows) / len(rows)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True
                    )
                    self.optimizer.step()
                    self.update_step += 1
                    optimizer_steps += 1
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval().requires_grad_(False)
        return {
            "delta_critic/loss": total_loss / self.epochs,
            "delta_critic/grad_norm": float(grad_norm),
            "delta_critic/update_step": self.update_step,
            "delta_critic/optimizer_steps": optimizer_steps,
            "delta_critic/rows": len(rows),
        }

    def save_online(self, path, policy_step):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "initial_weights_sha256": self.metadata["weights_sha256"],
                "stats_version": self.metadata["stats_version"],
                "update_config": self.update_config,
                "policy_step": policy_step,
                "update_step": self.update_step,
                "model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "optimizer": self.optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            },
            temporary,
        )
        temporary.replace(path)

    def load_online(self, path, policy_step):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state["initial_weights_sha256"] != self.metadata["weights_sha256"]
            or state["stats_version"] != self.metadata["stats_version"]
            or state["update_config"] != self.update_config
            or state["policy_step"] != policy_step
        ):
            raise ValueError("Online critic checkpoint identity/configuration/step mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_step = state["update_step"]
        torch.set_rng_state(state["rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)
