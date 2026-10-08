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
from copy import deepcopy
from math import isfinite
from pathlib import Path

import torch

from .legacy_adapter import adapt_legacy_rows
from .online_diagnostics import prediction_drift, regression_diagnostics
from .online_optimizer import FP32MasterAdamW
from .online_update_data import paired_rows, paired_training_config
from .scalar_loss import _terminal_components, per_sample_loss
from .score import FrozenDeltaWorker
from .training_data import collate, ordinary_rows


class OnlineDeltaWorker(FrozenDeltaWorker):
    """Single-device critic with full-context microbatch accumulation."""

    def __init__(self, artifact, *, update_config=None, **kwargs):
        super().__init__(artifact, **kwargs)
        if self.config.objective != "local_td0" or self.config.delta_label_mode != "selected_segment":
            raise ValueError("Online MC critic requires local_td0/selected_segment")
        self.update_config = dict(update_config or {})
        self.paired_protocol = self.update_config.get("protocol") == "paired_fresh_v1"
        if self.update_config.get("protocol") not in {None, "paired_fresh_v1"}:
            raise ValueError("Unsupported online critic update protocol")
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
        learning_rate = self.update_config.get(
            "learning_rate", 2e-6 if self.paired_protocol else self.config.learning_rate
        )
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("critic_update.learning_rate must be finite and positive")
        if self.paired_protocol:
            self._initialize_paired(learning_rate)
        else:
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
        if self.paired_protocol:
            raise ValueError("Paired online updates require buffer_fit_mc(records, policy_step)")
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

    def _initialize_paired(self, learning_rate):
        if self.epochs != 1 or self.mini_batch_size is not None or self.microbatch_train != 1:
            raise ValueError("paired_fresh_v1 requires epochs=1, microbatch=1 and no mini_batch_size")
        self.mode = self.update_config.get("mode", "td")
        if self.mode not in {"td", "hybrid"}:
            raise ValueError("critic_update.mode must be td or hybrid")
        self.interval = self.update_config.get("interval", 4)
        self.train_pairs_per_step = self.update_config.get("train_pairs_per_step", 10)
        self.diagnostic_pairs_per_step = self.update_config.get("diagnostic_pairs_per_step", 2)
        self.branches_per_state = self.update_config.get("branches_per_state", 8)
        for name in ("interval", "train_pairs_per_step", "diagnostic_pairs_per_step", "branches_per_state"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"critic_update.{name} must be a positive integer")
        if self.branches_per_state < 2:
            raise ValueError("LOO terminal labels require at least two branches per state")
        gradient_clip = self.update_config.get("gradient_clip", 1.0)
        if not isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("critic_update.gradient_clip must be finite and positive")
        self.fit_config = paired_training_config(
            self.config,
            terminal_weight=self.update_config.get("terminal_weight", 0.003),
            state_count=self.update_config.get("terminal_state_count", 64),
            min_gap=self.update_config.get("terminal_min_gap", 32),
            max_candidates=self.update_config.get("terminal_max_candidates", 5),
        )
        self.optimizer = FP32MasterAdamW(self.model, learning_rate=learning_rate, gradient_clip=gradient_clip)
        self.pending_windows = []
        self.last_policy_step = 0
        self.prompt_roles = {}

    def _paired_batch(self, rows):
        batch = collate(rows, self.fit_config, self.metadata["normalization"], self.pad_token_id)
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
        }

    def _training_context(self):
        if self.config.training_autocast == "none" or self.device.type != "cuda":
            return nullcontext()
        dtype = self.config.dtype if self.config.training_autocast == "model_dtype" else "bfloat16"
        return torch.autocast(self.device.type, dtype=getattr(torch, dtype))

    @torch.no_grad()
    def _paired_diagnostics(self, rows, *, return_predictions=False):
        self.model.eval()
        norm = self.metadata["normalization"]
        std = float(norm["std"]) if norm.get("enabled") else 1.0
        mean = float(norm["mean"]) if norm.get("enabled") else 0.0
        totals = {
            "td_squared_error": 0.0,
            "terminal_squared_error": 0.0,
            "prediction_sum": 0.0,
            "prediction_squared_sum": 0.0,
            "td_rows": 0,
            "terminal_rows": 0,
        }
        predictions = {"td": [], "terminal": []}
        targets = {"td": [], "terminal": []}
        for kind in ("td", "terminal"):
            for row in rows[kind]:
                batch = self._paired_batch([row])
                context = (
                    torch.autocast(self.device.type, dtype=torch.bfloat16)
                    if self.config.scoring_autocast == "bfloat16"
                    else nullcontext()
                )
                with context:
                    prediction = self.model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                    ).float()
                if kind == "td":
                    mask = batch["target_mask"].bool()
                    raw = prediction[mask] * std + mean
                    residual = (prediction - batch["target"])[mask] * std
                    totals["td_squared_error"] += float(residual.square().sum())
                    totals["prediction_sum"] += float(raw.sum())
                    totals["prediction_squared_sum"] += float(raw.square().sum())
                    totals["td_rows"] += int(mask.sum())
                    predictions[kind].extend(raw.cpu().tolist())
                    # Use original reward-unit labels: undoing float32 target
                    # normalization can turn an exactly zero label into epsilon.
                    targets[kind].extend(
                        float(target)
                        for target, active in zip(row["token_targets"], row["token_loss_mask"], strict=True)
                        if active
                    )
                else:
                    valid = torch.ones(prediction.shape[0], dtype=torch.bool, device=self.device)
                    _, residual, terminal_valid, _, _ = _terminal_components(
                        prediction, batch, self.fit_config, norm, None, valid
                    )
                    totals["terminal_squared_error"] += float(residual[terminal_valid].square().sum())
                    totals["terminal_rows"] += int(terminal_valid.sum())
                    positions = batch["terminal_suffix_positions"].clamp(min=0, max=prediction.shape[1] - 1)
                    raw_suffix = prediction.gather(1, positions) * std + mean
                    # Reconstruct from outputs directly: residual + target can
                    # erase a small prediction through float32 cancellation.
                    raw_sum = torch.where(batch["terminal_suffix_mask"].bool(), raw_suffix, 0.0).sum(-1)
                    predictions[kind].extend(raw_sum[terminal_valid].cpu().tolist())
                    if terminal_valid.item():
                        targets[kind].append(float(row["terminal_comp_target"]))
        count = max(totals["td_rows"], 1)
        average = totals["prediction_sum"] / count
        metrics = {
            **totals,
            "td_rmse": (totals["td_squared_error"] / count) ** 0.5,
            "terminal_rmse": (totals["terminal_squared_error"] / max(totals["terminal_rows"], 1)) ** 0.5,
            "prediction_mean": average,
            "prediction_std": max(0.0, totals["prediction_squared_sum"] / count - average * average) ** 0.5,
        }
        for kind in ("td", "terminal"):
            metrics.update(
                {
                    f"{kind}_{key}": value
                    for key, value in regression_diagnostics(predictions[kind], targets[kind]).items()
                }
            )
        return (metrics, predictions) if return_predictions else metrics

    def buffer_fit_mc(self, records, policy_step):
        """Consume one fresh actor batch, publishing a critic after each full window.

        Call only after the caller has materialized that actor step's advantages.
        Every selected pair has exactly one true TD target. Buffer state survives
        policy checkpoints taken between critic updates.
        """
        if not self.paired_protocol:
            raise ValueError("buffer_fit_mc requires critic_update.protocol=paired_fresh_v1")
        if type(policy_step) is not int or policy_step != self.last_policy_step + 1:
            raise ValueError("Online critic policy steps must be contiguous and begin at one")
        records = list(records)
        expected_version = str(policy_step - 1)
        for record in records:
            if not record.get("labels"):
                continue
            if str(record["rollout"].get("actor_version")) != expected_version:
                raise ValueError("Online critic records must come from the current pre-update actor version")
            for label in record["labels"]:
                if str(label.get("mc_actor_version", expected_version)) != expected_version:
                    raise ValueError("Online critic branch actor version differs from its original rollout")
        rows = paired_rows(records, self.fit_config, branches_per_state=self.branches_per_state)
        next_roles = dict(self.prompt_roles)
        for role, expected in (("train", self.train_pairs_per_step), ("diagnostic", self.diagnostic_pairs_per_step)):
            if (
                len(rows[role]["td"]) != expected
                or len(rows[role]["terminal"]) != expected * 2 * self.branches_per_state
            ):
                raise ValueError(f"Online {role} pair/terminal quota mismatch at policy step {policy_step}")
            for row in rows[role]["td"]:
                prompt_id = row["prompt_stable_id"]
                if prompt_id is None:
                    raise ValueError("Paired online training requires stable prompt IDs for diagnostic isolation")
                if str(prompt_id) in next_roles and next_roles[str(prompt_id)] != role:
                    raise ValueError("A prompt changed its critic training/diagnostic partition")
                next_roles[str(prompt_id)] = role
        self.prompt_roles = next_roles
        self.pending_windows.append({"policy_step": policy_step, "rows": rows})
        self.last_policy_step = policy_step
        metrics = {
            "delta_critic/update_step": self.update_step,
            "delta_critic/optimizer_steps": 0,
            "delta_critic/buffer_policy_steps": len(self.pending_windows),
            "delta_critic/new_td_rows": len(rows["train"]["td"]),
            "delta_critic/new_terminal_rows": len(rows["train"]["terminal"]),
        }
        if len(self.pending_windows) < self.interval:
            return metrics
        if len(self.pending_windows) != self.interval:
            raise ValueError("Online critic buffer exceeded the configured fresh window")
        merged = {
            role: {
                kind: [row for entry in self.pending_windows for row in entry["rows"][role][kind]]
                for kind in ("td", "terminal")
            }
            for role in ("train", "diagnostic")
        }
        train, diagnostic = merged["train"], merged["diagnostic"]
        expected_td = self.interval * self.train_pairs_per_step
        if len(train["td"]) != expected_td:
            raise ValueError("Online critic effective TD batch differs from the configured window")
        prefit, before_predictions = self._paired_diagnostics(diagnostic, return_predictions=True)
        metrics.update({f"delta_critic/diagnostic_prefit/{key}": value for key, value in prefit.items()})
        total_td, total_terminal = 0.0, 0.0
        self.model.train().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for kind in ("td", "terminal") if self.mode == "hybrid" else ("td",):
                for row in train[kind]:
                    batch = self._paired_batch([row])
                    with self._training_context():
                        prediction = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        losses, _, td, _ = per_sample_loss(
                            prediction, batch, self.fit_config, self.metadata["normalization"]
                        )
                        # Each component retains its own global denominator.
                        loss = (td.sum() if kind == "td" else losses.sum()) / len(train[kind])
                    if not torch.isfinite(loss):
                        raise ValueError("Paired online critic loss is not finite")
                    loss.backward()
                    if kind == "td":
                        total_td += float(loss.detach())
                    else:
                        total_terminal += float(loss.detach())
                if kind == "td":
                    metrics["delta_critic/td_grad_norm_before_clip"] = self.optimizer.gradient_norm()
            changes = self.optimizer.step()
            self.update_step += 1
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval().requires_grad_(False)
        metrics.update({f"delta_critic/{key}": value for key, value in changes.items()})
        postfit, after_predictions = self._paired_diagnostics(diagnostic, return_predictions=True)
        metrics.update({f"delta_critic/diagnostic_postfit/{key}": value for key, value in postfit.items()})
        for kind in ("td", "terminal"):
            metrics.update(
                {
                    f"delta_critic/diagnostic_update/{kind}_{key}": value
                    for key, value in prediction_drift(before_predictions[kind], after_predictions[kind]).items()
                }
            )
            for key, valid_key in (
                ("rmse", "statistics_valid"),
                ("bias", "statistics_valid"),
                ("zero_baseline_skill", "zero_baseline_skill_valid"),
                ("nonzero_sign_accuracy", "nonzero_sign_accuracy_valid"),
            ):
                valid = bool(prefit[f"{kind}_{valid_key}"] and postfit[f"{kind}_{valid_key}"])
                metrics[f"delta_critic/diagnostic_update/{kind}_{key}_change"] = (
                    postfit[f"{kind}_{key}"] - prefit[f"{kind}_{key}"] if valid else 0.0
                )
                metrics[f"delta_critic/diagnostic_update/{kind}_{key}_change_valid"] = int(valid)
        metrics.update(
            {
                "delta_critic/loss": total_td + total_terminal,
                "delta_critic/td_loss": total_td,
                "delta_critic/train_unique_prompts": len({row["prompt_stable_id"] for row in train["td"]}),
                "delta_critic/terminal_optimization_loss": total_terminal,
                "delta_critic/update_step": self.update_step,
                "delta_critic/optimizer_steps": 1,
                "delta_critic/td_rows": len(train["td"]),
                "delta_critic/terminal_rows": len(train["terminal"]) if self.mode == "hybrid" else 0,
                "delta_critic/oldest_policy_step": self.pending_windows[0]["policy_step"],
                "delta_critic/newest_policy_step": policy_step,
                "delta_critic/data_reuse_count": 1,
                "delta_critic/buffer_policy_steps": 0,
            }
        )
        self.pending_windows.clear()
        return metrics

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
                "paired_state": (
                    {
                        "pending_windows": self.pending_windows,
                        "last_policy_step": self.last_policy_step,
                        "prompt_roles": self.prompt_roles,
                    }
                    if self.paired_protocol
                    else None
                ),
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
        if self.paired_protocol:
            paired = state.get("paired_state")
            if paired is None or paired["last_policy_step"] != policy_step:
                raise ValueError("Paired online checkpoint buffer/policy step mismatch")
            self.pending_windows = deepcopy(paired["pending_windows"])
            self.last_policy_step = paired["last_policy_step"]
            self.prompt_roles = dict(paired["prompt_roles"])
        torch.set_rng_state(state["rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)
