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
"""Paired one-step actor probe using only the saved short rows.

This is a short-only mechanism probe, not a full 320-row PPO update. It uses
saved token IDs/masks and old/new outcome advantages, reloading the same local
FP32 actor snapshot for each branch. Forward autocast is configurable. No actor
checkpoint is written and model loading is local-files-only.

Response logits are projected through the LM head in bounded chunks, never as
one sequence-wide vocabulary tensor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.delta_critic.policy_config import DeltaPolicyConfig
from examples.delta_critic.policy_loss import per_sample_policy_losses

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = ROOT / "outputs/delta_terminal_corrected_experiments_20261002/replay/batch.jsonl"


@dataclass(frozen=True)
class ShortRow:
    row_id: str
    prompt: tuple[int, ...]
    response: tuple[int, ...]
    mask: tuple[float, ...]
    reward: float
    baseline: float
    baseline_source: str
    old_advantage: float
    new_advantage: float


@dataclass(frozen=True)
class Measure:
    logprobs: tuple[float, ...]
    eos_probability: float | None


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_short_rows(path: str | Path) -> list[ShortRow]:
    """Load only short rows, preserving saved token IDs and policy masks."""
    rows, seen = [], set()
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"line {number} must be a JSON object")
            if raw.get("is_short") is not True:
                continue
            row_id = str(raw.get("row_id", ""))
            if not row_id or row_id in seen:
                raise ValueError(f"line {number} has a missing or duplicate short row_id")
            seen.add(row_id)
            prompt, response = raw.get("prompt_token_ids"), raw.get("response_token_ids")
            if not isinstance(prompt, list) or not prompt or not isinstance(response, list) or not response:
                raise ValueError(f"line {number} requires nonempty saved prompt/response token IDs")
            if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in prompt + response):
                raise ValueError(f"line {number} contains invalid token IDs")
            mask = raw.get("policy_token_mask")
            if not isinstance(mask, list) or len(mask) != len(response):
                raise ValueError(f"line {number} policy_token_mask must align with response_token_ids")
            mask = tuple(_finite(x, "policy_token_mask") for x in mask)
            if any(x not in (0.0, 1.0) for x in mask) or not any(mask):
                raise ValueError(f"line {number} policy_token_mask must be binary with an active token")
            replay = raw.get("advantage_replay")
            if not isinstance(replay, dict):
                raise ValueError(f"line {number} is missing advantage_replay")
            rows.append(
                ShortRow(
                    row_id=row_id,
                    prompt=tuple(prompt),
                    response=tuple(response),
                    mask=mask,
                    reward=_finite(raw.get("reward"), "reward"),
                    baseline=_finite(raw.get("baseline"), "baseline"),
                    baseline_source=str(raw.get("baseline_source", "unknown")),
                    old_advantage=_finite(replay.get("old_advantage_mean"), "old_advantage_mean"),
                    new_advantage=_finite(replay.get("new_advantage_mean"), "new_advantage_mean"),
                )
            )
    if not rows:
        raise ValueError(f"No short rows found in {path}")
    return rows


def chunked_response_logprobs(model, prompt, response, *, chunk_size=32, eos_token_id=None):
    """Return saved-response logprobs and EOS probability at its final decision."""
    if not prompt or not response or chunk_size < 1:
        raise ValueError("prompt/response must be nonempty and chunk_size positive")
    decoder = getattr(model, "model", None) or getattr(model, "base_model", None)
    head = getattr(model, "lm_head", None)
    if decoder is None or head is None:
        raise ValueError("Expected a decoder-only model/base_model and lm_head")
    device = next(model.parameters()).device
    tokens = torch.tensor([[*prompt, *response[:-1]]], dtype=torch.long, device=device)
    output = decoder(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state[0, len(prompt) - 1 :]
    if hidden.shape[0] != len(response):
        raise ValueError("Decoder hidden states do not align with response")
    values, eos_logprob = [], None
    for start in range(0, len(response), chunk_size):
        end = min(start + chunk_size, len(response))
        logits = head(hidden[start:end]).float()
        logprobs = F.log_softmax(logits, dim=-1)
        target = torch.tensor(response[start:end], dtype=torch.long, device=device)
        values.append(logprobs.gather(-1, target[:, None]).squeeze(-1))
        if eos_token_id is not None and start <= len(response) - 1 < end:
            if eos_token_id >= logprobs.shape[-1]:
                raise ValueError("eos_token_id exceeds actor vocabulary size")
            eos_logprob = logprobs[len(response) - 1 - start, eos_token_id]
    return torch.cat(values), eos_logprob


def _autocast(device, kind):
    if kind == "none":
        return nullcontext()
    dtype = torch.bfloat16 if kind == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def measure_rows(model, rows, *, device, chunk_size, eos_token_id, autocast_dtype):
    model.eval()
    measured = []
    with torch.no_grad():
        for row in rows:
            with _autocast(device, autocast_dtype):
                logprobs, eos = chunked_response_logprobs(
                    model, row.prompt, row.response, chunk_size=chunk_size, eos_token_id=eos_token_id
                )
            measured.append(
                Measure(
                    tuple(float(x) for x in logprobs.float().cpu().tolist()),
                    None if eos is None else float(eos.float().exp().cpu()),
                )
            )
    return measured


def perform_one_step(model, rows, behavior_logprobs, *, variant, device, chunk_size, autocast_dtype):
    """Apply one sample-mean PPO+fixed-initial-KL AdamW update."""
    if variant not in ("old", "new") or len(rows) != len(behavior_logprobs) or not rows:
        raise ValueError("Expected old/new variant and one behavior-logprob row per sample")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01, foreach=False)
    config = DeltaPolicyConfig.online(kl_coef=0.001)
    optimizer.zero_grad(set_to_none=True)
    loss_mean = 0.0
    model.eval()
    for row, behavior in zip(rows, behavior_logprobs, strict=True):
        shape = (1, len(row.response))
        if len(behavior) != len(row.response):
            raise ValueError(f"Behavior logprobs do not align with {row.row_id}")
        old = torch.tensor(behavior, device=device, dtype=torch.float32).reshape(shape)
        advantage = row.old_advantage if variant == "old" else row.new_advantage
        advantages = torch.full(shape, advantage, device=device, dtype=torch.float32)
        mask = torch.tensor(row.mask, device=device, dtype=torch.float32).reshape(shape)
        with _autocast(device, autocast_dtype):
            current, _ = chunked_response_logprobs(model, row.prompt, row.response, chunk_size=chunk_size)
        losses = per_sample_policy_losses(
            current.reshape(shape),
            {"old_log_probs": old, "advantages": advantages, "policy_loss_mask": mask},
            config,
            reference_logprobs=old,
        )["loss"][0]
        loss_mean += float(losses.detach().cpu()) / len(rows)
        (losses / len(rows)).backward()
    grad_before = float(torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False, error_if_nonfinite=True).cpu())
    grad_after = grad_before * min(1.0, 1.0 / (grad_before + 1e-6))
    optimizer.step()
    del optimizer
    return {
        "sample_mean_loss_before_step": loss_mean,
        "grad_norm_before_clip": grad_before,
        "grad_norm_after_clip": grad_after,
    }


def group_metrics(rows, measured, eos_token_id):
    groups = defaultdict(list)
    for row, score in zip(rows, measured, strict=True):
        groups[(row.reward, row.baseline_source, row.baseline)].append((row, score))
    result = {}
    for key, members in groups.items():
        token_means = []
        eos_probs, terminal_eos = [], 0
        for row, score in members:
            active = [v for v, m in zip(score.logprobs, row.mask, strict=True) if m]
            token_means.append(sum(active) / len(active))
            if eos_token_id is not None:
                eos_probs.append(score.eos_probability)
                terminal_eos += row.response[-1] == eos_token_id
        result[key] = {
            "rows": len(members),
            "active_policy_tokens": int(sum(sum(row.mask) for row, _ in members)),
            "mean_response_logprob_row_mean": sum(token_means) / len(token_means),
            "mean_eos_probability_at_last_response_decision": (sum(eos_probs) / len(eos_probs) if eos_probs else None),
            "terminal_eos_rows": terminal_eos if eos_token_id is not None else None,
            "terminal_eos_rate": terminal_eos / len(members) if eos_token_id is not None else None,
        }
    return result


def group_changes(before, after):
    output = []
    metrics = ("mean_response_logprob_row_mean", "mean_eos_probability_at_last_response_decision")
    for key, initial in sorted(before.items()):
        final = after[key]
        output.append(
            {
                "reward": key[0],
                "baseline_source": key[1],
                "baseline": key[2],
                "before": initial,
                "after": final,
                "delta": {name: None if initial[name] is None else final[name] - initial[name] for name in metrics},
            }
        )
    return output


def _model(model_path, device):
    from transformers import AutoModelForCausalLM

    actor = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    return actor.to(device).eval()


def _eos_id(model_path, override):
    if override is not None:
        return override
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    value = json.loads(config_path.read_text()).get("eos_token_id")
    return value[0] if isinstance(value, list) and value else value


def run_probe(rows, *, model_path, device, chunk_size, eos_token_id, autocast_dtype):
    actor = _model(model_path, device)
    before = measure_rows(
        actor, rows, device=device, chunk_size=chunk_size, eos_token_id=eos_token_id, autocast_dtype=autocast_dtype
    )
    behavior = [x.logprobs for x in before]
    before_groups = group_metrics(rows, before, eos_token_id)
    old_step = perform_one_step(
        actor, rows, behavior, variant="old", device=device, chunk_size=chunk_size, autocast_dtype=autocast_dtype
    )
    old_after = measure_rows(
        actor, rows, device=device, chunk_size=chunk_size, eos_token_id=eos_token_id, autocast_dtype=autocast_dtype
    )
    old_groups = group_metrics(rows, old_after, eos_token_id)
    del actor, old_after
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Reload the same local snapshot. Initial logprobs stay fixed as behavior/KL reference.
    actor = _model(model_path, device)
    second_initial = measure_rows(
        actor, rows, device=device, chunk_size=chunk_size, eos_token_id=eos_token_id, autocast_dtype=autocast_dtype
    )
    max_initial_logprob_diff = max(
        abs(actual - expected)
        for observed, reference in zip(second_initial, before, strict=True)
        for actual, expected in zip(observed.logprobs, reference.logprobs, strict=True)
    )
    if max_initial_logprob_diff > 1e-5:
        raise RuntimeError(
            "Reloaded actor does not match the initial actor logprobs "
            f"(max absolute difference {max_initial_logprob_diff:.3g})"
        )
    del second_initial
    new_step = perform_one_step(
        actor, rows, behavior, variant="new", device=device, chunk_size=chunk_size, autocast_dtype=autocast_dtype
    )
    new_after = measure_rows(
        actor, rows, device=device, chunk_size=chunk_size, eos_token_id=eos_token_id, autocast_dtype=autocast_dtype
    )
    new_groups = group_metrics(rows, new_after, eos_token_id)
    del actor, new_after
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "short-only paired actor mechanism probe",
        "scope": "Short rows only; this does not replace the complete 320-row PPO update.",
        "rows": len(rows),
        "active_policy_tokens": int(sum(sum(row.mask) for row in rows)),
        "sample_mean_denominator_rows": len(rows),
        "model_path": model_path,
        "paired_initialization": "same local FP32 snapshot reloaded independently for both branches",
        "reload_initial_logprob_max_abs_diff": max_initial_logprob_diff,
        "behavior_and_reference": "fixed initial actor logprobs cached before the old-advantage update",
        "parameter_dtype": "float32",
        "autocast_dtype": autocast_dtype,
        "train_logprob_temperature": 1.0,
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "max_grad_norm": 1.0,
            "foreach": False,
        },
        "objective": {
            "implementation": "examples.delta_critic.policy_loss.per_sample_policy_losses",
            "aggregation": "sample mean of row token-means",
            "ppo_clip_range": 0.2,
            "kl_coef": 0.001,
            "kl_reference": "fixed initial actor",
            "mask": "saved policy_token_mask",
        },
        "logits_chunk_size": chunk_size,
        "eos_token_id": eos_token_id,
        "branches": {
            "old_advantage_mean": {
                **old_step,
                "groups": group_changes(before_groups, old_groups),
            },
            "new_advantage_mean": {
                **new_step,
                "groups": group_changes(before_groups, new_groups),
            },
        },
        "actor_checkpoints_written": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=str(DEFAULT_BATCH))
    parser.add_argument("--model-path", required=True, help="local cached actor snapshot; no downloads")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logits-chunk-size", type=int, default=32)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--autocast-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--output", help="optional small JSON report path")
    args = parser.parse_args()
    if args.logits_chunk_size < 1:
        parser.error("--logits-chunk-size must be positive")
    try:
        rows = load_short_rows(args.batch)
        report = run_probe(
            rows,
            model_path=args.model_path,
            device=torch.device(args.device),
            chunk_size=args.logits_chunk_size,
            eos_token_id=_eos_id(args.model_path, args.eos_token_id),
            autocast_dtype=args.autocast_dtype,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
