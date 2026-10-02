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
"""Validated portable scalar artifacts; optimizer resume lives in FSDP checkpoints."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .training_config import ScalarConfig

FORMAT_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with open(Path(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_normalization(meta):
    if not isinstance(meta, dict):
        raise ValueError("Normalization metadata must be an object")
    if not isinstance(meta.get("enabled"), bool):
        raise ValueError("Normalization metadata must include enabled")
    if meta["enabled"]:
        if meta.get("mode") != "standardize":
            raise ValueError("Unsupported normalization mode")
        if any(
            isinstance(meta.get(k), bool) or not isinstance(meta.get(k), int | float) or not math.isfinite(meta[k])
            for k in ("mean", "std")
        ):
            raise ValueError("Nonfinite/missing normalization statistics")
        if meta["std"] < 1e-8:
            raise ValueError("Normalization std must be >= 1e-8")
    elif (
        meta.get("mode", "none") != "none"
        or meta.get("mean") not in (None, 0, 0.0)
        or meta.get("std") not in (None, 1, 1.0)
    ):
        raise ValueError("Disabled normalization must have None or identity statistics")
    return meta


def validate_terminal_stats(stats):
    if stats is None:
        return None
    if not isinstance(stats, dict):
        raise ValueError("Terminal statistics must be an object or None")
    for name in ("mean", "std"):
        value = stats.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Terminal statistics require finite {name}")
    if stats["std"] < 1e-8:
        raise ValueError("Terminal statistics std must be >= 1e-8")
    if "count" in stats:
        count = stats["count"]
        if isinstance(count, bool) or not isinstance(count, int | float) or not math.isfinite(count) or count <= 0:
            raise ValueError("Terminal statistics count must be finite and positive")
    return stats


def tokenizer_fingerprint_from_tokenizer(tokenizer, *, expected_vocab_size=None):
    """Fingerprint an already loaded tokenizer using the scalar-artifact contract."""
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Tokenizer has no vocabulary")
    token_ids = list(vocab.values())
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise ValueError("Tokenizer has invalid vocabulary IDs")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Tokenizer maps multiple tokens to the same ID")
    if expected_vocab_size is not None and max(token_ids) >= expected_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ID {max(token_ids)} exceeds backbone vocabulary size {expected_vocab_size}"
        )
    return fingerprint({"vocab": vocab, "special_tokens": tokenizer.special_tokens_map})


def tokenizer_fingerprint(path, trust_remote_code=False, *, expected_vocab_size=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    return tokenizer_fingerprint_from_tokenizer(tokenizer, expected_vocab_size=expected_vocab_size), tokenizer


def load_weights(model, state):
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model state must be a tensor mapping")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            f"Checkpoint keys mismatch: missing={set(expected) - set(state)}, extra={set(state) - set(expected)}"
        )
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or not value.is_floating_point():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", type(value).__name__)
            raise ValueError(f"Invalid scalar/backbone tensor {key}: {shape}, {dtype}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor {key} contains nonfinite weights")
        if key.startswith("scalar_head."):
            if value.dtype != torch.float32:
                raise ValueError(f"Scalar head tensor {key} must remain float32, got {value.dtype}")
        elif value.dtype != expected[key].dtype:
            raise ValueError(
                f"Backbone tensor {key} dtype {value.dtype} does not match configured dtype {expected[key].dtype}"
            )
    if "scalar_head.weight" not in state or "scalar_head.bias" not in state:
        raise ValueError("Checkpoint is missing the biased scalar head")
    if state["scalar_head.weight"].shape[0] != 1 or state["scalar_head.bias"].shape != (1,):
        raise ValueError("Only a biased scalar head is supported")
    # assign=True preserves each source tensor's dtype, including the FP32 head.
    model.load_state_dict(state, strict=True, assign=True)


def write_artifact(directory, state, config, normalization, terminal_stats=None, **provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validate_normalization(normalization)
    validate_terminal_stats(terminal_stats)
    reserved = {
        "format_version",
        "config",
        "normalization",
        "terminal_stats",
        "stats_version",
        "weights_sha256",
    }
    overlap = reserved.intersection(provenance)
    if overlap:
        raise ValueError(f"Artifact provenance cannot override reserved metadata fields: {sorted(overlap)}")
    temporary = directory / "model.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "model.pt")
    metadata = {
        "format_version": FORMAT_VERSION,
        "config": config.as_dict(),
        "normalization": normalization,
        "terminal_stats": terminal_stats,
        "stats_version": fingerprint({"target": normalization, "terminal": terminal_stats}),
        "weights_sha256": file_hash(directory / "model.pt"),
        **provenance,
    }
    tmp = directory / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    tmp.replace(directory / "metadata.json")
    return metadata


def read_artifact(directory):
    directory = Path(directory)
    try:
        meta = json.loads((directory / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read portable critic metadata from {directory}") from exc
    if not isinstance(meta, dict) or meta.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported scalar artifact version")
    try:
        config = ScalarConfig(**meta["config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Portable critic artifact has invalid configuration metadata") from exc
    if isinstance(config.max_length, bool) or not isinstance(config.max_length, int) or config.max_length < 1:
        raise ValueError("Portable critic artifact must record its resolved positive max_length")
    normalization = validate_normalization(meta.get("normalization"))
    if normalization["enabled"] != (config.target_normalization == "standardize"):
        raise ValueError("Config and normalization metadata disagree")
    terminal = validate_terminal_stats(meta.get("terminal_stats"))
    if config.objective == "hybrid_terminal_composition":
        if terminal is None or not math.isfinite(terminal["std"]) or terminal["std"] < 1e-8:
            raise ValueError("Hybrid checkpoint requires finite terminal statistics")
    if meta["stats_version"] != fingerprint({"target": normalization, "terminal": terminal}):
        raise ValueError("Checkpoint statistics version mismatch")
    if file_hash(directory / "model.pt") != meta["weights_sha256"]:
        raise ValueError("Checkpoint weight checksum mismatch")
    return meta


def import_legacy(
    path,
    output,
    *,
    expected_model=None,
    expected_tokenizer=None,
    declared_label_mode=None,
    label_mode_source=None,
):
    from .scalar_model import DeltaScalarModel

    path = Path(path)
    try:
        old = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Cannot load source critic checkpoint {path}") from exc
    if not isinstance(old, dict):
        raise ValueError("Source checkpoint must be a metadata mapping")
    if old.get("target_type") != "delta" or old.get("output_granularity") != "token_sequence":
        raise ValueError("Only token_sequence scalar delta checkpoints are supported")
    if isinstance(old.get("step"), bool) or not isinstance(old.get("step"), int) or old["step"] < 0:
        raise ValueError("Source checkpoint must include a nonnegative integer step")
    method, source_config = old.get("training_method"), old.get("config")
    if not isinstance(method, dict) or not isinstance(source_config, dict):
        raise ValueError("Source checkpoint must include training_method and config metadata")
    output_dim = method.get("output_dim", 1)
    if (
        method.get("target_type", old["target_type"]) != "delta"
        or isinstance(output_dim, bool)
        or not isinstance(output_dim, int)
        or output_dim != 1
    ):
        raise ValueError("Unsupported target/head")
    if method.get("value_centering", "none") != "none" or method.get("target_model_enabled", False):
        raise ValueError("Value centering and target-network objectives are unsupported")
    if method.get("scalar_semantics", "local_delta") != "local_delta":
        raise ValueError("Checkpoint is not a local delta predictor")
    objective = method.get("delta_objective")
    if objective not in {"local_td0", "hybrid_terminal_composition"}:
        raise ValueError(f"Unsupported objective: {objective!r}")
    loss_type = method.get("loss_type")
    if loss_type not in {"mse", "nonzero_balanced_mse"}:
        raise ValueError(f"Unsupported scalar loss: {loss_type!r}")
    loss_mask = method.get("loss_mask")
    if loss_mask not in {"response", "selected_state", "selected_delta"}:
        raise ValueError(f"Unsupported scalar supervision mask: {loss_mask!r}")
    delta_label_mode = method.get("delta_label_mode", (source_config.get("critic") or {}).get("delta_label_mode"))
    declared_metadata = {}
    if declared_label_mode is not None:
        if declared_label_mode not in {"selected_segment", "paired_next_state"}:
            raise ValueError(f"Unsupported declared delta label mode: {declared_label_mode!r}")
        if not isinstance(label_mode_source, str) or not label_mode_source.strip():
            raise ValueError("An explicit label mode requires a nonempty label_mode_source")
        if delta_label_mode is not None and delta_label_mode != declared_label_mode:
            raise ValueError("Declared delta label mode conflicts with checkpoint metadata")
        if delta_label_mode is None:
            declared_metadata = {"delta_label_mode": declared_label_mode, "source": label_mode_source}
            delta_label_mode = declared_label_mode
    elif label_mode_source is not None:
        raise ValueError("label_mode_source requires declared_label_mode")
    if delta_label_mode not in {"selected_segment", "paired_next_state"}:
        raise ValueError(f"Unsupported or missing delta label mode: {delta_label_mode!r}")
    model_cfg = source_config.get("model")
    train_cfg = source_config.get("train")
    if not isinstance(model_cfg, dict) or not isinstance(train_cfg, dict):
        raise ValueError("Source config must include model and train objects")
    model_path = model_cfg.get("train_model") or model_cfg.get("actor_model")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("Source config must identify the critic backbone")
    tokenizer_path = model_cfg.get("tokenizer_path") or model_path
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("Source config must identify the tokenizer")
    if expected_model is not None and expected_model != model_path:
        raise ValueError(f"Backbone identity mismatch: {model_path} != {expected_model}")
    if expected_tokenizer is not None and expected_tokenizer != tokenizer_path:
        raise ValueError("Tokenizer identity mismatch")
    norm = validate_normalization(old.get("target_normalization"))
    mode = "standardize" if norm["enabled"] else "none"
    if method.get("target_normalization", mode) != mode:
        raise ValueError("Conflicting normalization metadata")
    dtype = model_cfg.get("dtype", "bfloat16")
    dtype = {"bf16": "bfloat16", "fp32": "float32"}.get(str(dtype).lower(), str(dtype).lower())
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError(f"Unsupported source backbone dtype: {dtype!r}")
    if train_cfg.get("fp16") and not train_cfg.get("bf16"):
        raise ValueError("FP16 autocast checkpoints are unsupported")
    max_length = train_cfg.get("max_length")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("Source config must include a positive training max_length")
    value_layer = model_cfg.get("value_layer", -1)
    if isinstance(value_layer, bool) or not isinstance(value_layer, int):
        raise ValueError("value_layer must be the original integer hidden-state index")
    trust_remote_code = model_cfg.get("trust_remote_code", True)
    if not isinstance(trust_remote_code, bool):
        raise ValueError("trust_remote_code must be a boolean in source config")
    attention_implementation = model_cfg.get("attn_implementation", model_cfg.get("attention_implementation", "sdpa"))
    if not isinstance(attention_implementation, str):
        raise ValueError("attention_implementation must be a string in source config")
    td_weight = method.get("hybrid_td_weight", 1) if objective == "hybrid_terminal_composition" else 1.0
    terminal_weight = method.get("hybrid_terminal_weight", 1) if objective == "hybrid_terminal_composition" else 1.0
    config = ScalarConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        dtype=dtype,
        attention_implementation=attention_implementation,
        training_autocast="bfloat16" if train_cfg.get("bf16", False) else "none",
        scoring_autocast="none",
        value_layer=value_layer,
        objective=objective,
        loss_type=loss_type,
        loss_mask=loss_mask,
        delta_label_mode=delta_label_mode,
        target_normalization=mode,
        max_length=max_length,
        window_policy="legacy_tail",
        continuation_budget=method.get("hybrid_continuation_budget"),
        continuation_state_count=int(method.get("continuation_state_count", 64)),
        continuation_min_gap=int(method.get("continuation_state_min_gap", 32)),
        td_weight=float(td_weight),
        terminal_weight=float(terminal_weight),
        trust_remote_code=trust_remote_code,
    )
    terminal = validate_terminal_stats(method.get("hybrid_terminal_target_stats"))
    if config.objective == "hybrid_terminal_composition":
        if terminal is None or not math.isfinite(terminal["std"]) or terminal["std"] < 1e-8:
            raise ValueError("Hybrid checkpoint requires terminal statistics")
    model = DeltaScalarModel.from_config(config)
    if model.max_position_embeddings is not None and config.max_length > model.max_position_embeddings:
        raise ValueError(
            f"Source training max_length {config.max_length} exceeds backbone native context "
            f"{model.max_position_embeddings}"
        )
    load_weights(model, old["model"])
    token_hash, _ = tokenizer_fingerprint(
        tokenizer_path,
        config.trust_remote_code,
        expected_vocab_size=getattr(model.config, "vocab_size", None),
    )
    return write_artifact(
        output,
        model.state_dict(),
        config,
        norm,
        terminal,
        tokenizer_sha256=token_hash,
        source_path=str(path.resolve()),
        source_sha256=file_hash(path),
        source_step=old["step"],
        source_metadata={k: v for k, v in old.items() if k != "model"},
        resume_capable=False,
        semantics_changes=[],
        declared_metadata=declared_metadata,
        inferred_metadata={},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-tokenizer")
    parser.add_argument("--declared-label-mode", choices=["selected_segment", "paired_next_state"])
    parser.add_argument("--label-mode-source", help="Run config or experiment record establishing a missing label mode")
    args = parser.parse_args()
    import_legacy(
        args.checkpoint,
        args.output,
        expected_model=args.expected_model,
        expected_tokenizer=args.expected_tokenizer,
        declared_label_mode=args.declared_label_mode,
        label_mode_source=args.label_mode_source,
    )


if __name__ == "__main__":
    main()
