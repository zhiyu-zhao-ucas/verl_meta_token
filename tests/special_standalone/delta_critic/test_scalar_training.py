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
"""Scalar semantics, optional source differential and portable checkpoint coverage."""

import ast
import importlib.util
import math
import os
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from examples.delta_critic.checkpoint import import_legacy, read_artifact, tokenizer_fingerprint, write_artifact
from examples.delta_critic.legacy_adapter import adapt_legacy_rows
from examples.delta_critic.scalar_loss import per_sample_loss
from examples.delta_critic.scalar_model import DeltaScalarModel
from examples.delta_critic.score import FrozenDeltaWorker
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.training_data import (
    collate,
    continuation_rows,
    fit_stats,
    normalization_metadata,
    ordinary_rows,
    scan_lengths,
)


@pytest.fixture
def tiny_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(3)
    path = tmp_path / "tiny"
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
    ).save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="0", pad_token="0", eos_token="1").save_pretrained(
        path
    )
    return path


@pytest.fixture
def records():
    rollouts = [
        {"id": "a", "prompt_token_ids": [1, 2], "response_token_ids": [3, 4, 5, 6], "terminal_reward": 1.0},
        {"id": "b", "prompt_token_ids": [2], "response_token_ids": [7, 8], "terminal_reward": 0.0},
        {"id": "c", "prompt_token_ids": [1, 2], "response_token_ids": [3], "terminal_reward": 0.0},
    ]
    labels = [
        {"rollout_id": "a", "state_id": "a0", "token_index": 0, "v_prefix": 0.5, "delta": 0.1},
        {"rollout_id": "a", "state_id": "a3", "token_index": 3, "v_prefix": 0.5, "delta": 0.0},
        {"rollout_id": "b", "state_id": "b1", "token_index": 1, "v_prefix": 0.75, "delta": -0.75},
    ]
    return rollouts, labels


def source_namespace():
    root = os.environ.get("VALUE_MODEL_ROOT")
    if not root:
        pytest.skip("Optional source comparison requires VALUE_MODEL_ROOT")
    root = Path(root) / "delta_value_llm_exp"
    path = root / "05_train_token_scalar_model_accelerate_local.py"
    names = {
        "DenseTokenScalarDataset",
        "NormalizedTokenScalarDataset",
        "TerminalCompositionContinuationDataset",
        "_select_continuation_state_indices",
        "_target_stats",
        "_mask_for_row",
        "_masked_loss",
        "_terminal_target_stats",
        "_hybrid_terminal_loss",
        "_hybrid_terminal_composition_loss",
    }
    tree = ast.parse(path.read_text())
    selected = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names],
        type_ignores=[],
    )
    ns = {"math": math, "bisect_right": bisect_right, "VALUE_CENTERING_MODES": {"none"}}
    exec(compile(selected, str(path), "exec"), ns)
    spec = importlib.util.spec_from_file_location("source_scalar", root / "modeling_token_scalar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns["model"] = module
    return ns


@pytest.mark.parametrize("loss_type", ["mse", "nonzero_balanced_mse"])
@pytest.mark.parametrize("loss_mask", ["response", "selected_state", "selected_delta"])
@pytest.mark.parametrize("label_mode", ["selected_segment", "paired_next_state"])
@pytest.mark.parametrize("normalization", ["none", "standardize"])
def test_source_targets_collate_loss_gradients(records, loss_type, loss_mask, label_mode, normalization):
    src = source_namespace()
    rollouts, labels = records
    examples, _ = adapt_legacy_rows(rollouts, labels)
    config = ScalarConfig(
        "unused",
        loss_type=loss_type,
        loss_mask=loss_mask,
        delta_label_mode=label_mode,
        target_normalization=normalization,
        max_length=8192,
    )
    rows = ordinary_rows(examples, config)
    source = src["DenseTokenScalarDataset"](rollouts, labels, "delta", loss_mask, label_mode)
    for ours, theirs in zip(rows, source, strict=True):
        for key in ours:
            assert ours[key] == theirs[key]
    norm = normalization_metadata(rows, config)
    if norm["enabled"]:
        stats = src["_target_stats"](source)
        assert {k: norm[k] for k in stats} == stats
        source = src["NormalizedTokenScalarDataset"](source, norm["mean"], norm["std"])
    for window in (8192, 3):
        conf = replace(config, window_policy="legacy_tail", max_length=window)
        batch = collate(rows, conf, norm)
        expected = src["model"].collate_token_scalar_batch(list(source), 0, window)
        for key in ("input_ids", "attention_mask", "target", "target_mask", "signal_mask"):
            torch.testing.assert_close(batch[key], expected[key], atol=0, rtol=0)
        pred = torch.linspace(-0.5, 0.5, batch["target"].numel()).reshape_as(batch["target"]).requires_grad_()
        actual, valid, _, _ = per_sample_loss(pred, batch, conf, norm)
        expected_losses = torch.stack(
            [
                src["_masked_loss"](
                    pred[i : i + 1],
                    batch["target"][i : i + 1],
                    batch["target_mask"][i : i + 1],
                    batch["signal_mask"][i : i + 1],
                    loss_type,
                    1e-8,
                )
                for i in range(len(rows))
            ]
        )
        torch.testing.assert_close(actual, expected_losses)
        loss = actual.sum() / valid.sum().clamp_min(1)
        reference = expected_losses.sum() / valid.sum().clamp_min(1)
        torch.testing.assert_close(
            torch.autograd.grad(loss, pred, retain_graph=True)[0], torch.autograd.grad(reference, pred)[0]
        )


def test_source_accumulation_counts_real_zero_loss_rows_not_dp_padding(records):
    src = source_namespace()
    rollouts, labels = records
    examples, _ = adapt_legacy_rows(rollouts, labels)
    config = ScalarConfig(
        "unused",
        loss_mask="selected_state",
        target_normalization="none",
        max_length=8192,
    )
    rows = ordinary_rows(examples, config)
    source = src["DenseTokenScalarDataset"](rollouts, labels, "delta", "selected_state", "selected_segment")
    # This real row has no selected states, so source batch-size-one loss and
    # gradient are zero while its logical sample still occupies an accumulation
    # slot. The repeated fourth row is synthetic DP padding and occupies none.
    assert rows[2]["token_loss_mask"] == [0.0]
    batch = collate(rows + [dict(rows[0], id="synthetic", sample_valid=False)], config, normalization=None)
    assert batch["sample_valid_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]
    pred = torch.linspace(-0.4, 0.5, batch["target"].numel()).reshape_as(batch["target"]).requires_grad_()
    losses, sample_valid, _, _ = per_sample_loss(pred, batch, config, normalization=None)
    assert sample_valid.tolist() == [True, True, True, False]

    source_losses = []
    for index, row in enumerate(source):
        source_batch = src["model"].collate_token_scalar_batch([row], 0, config.max_length)
        width = source_batch["input_ids"].shape[1]
        source_losses.append(
            src["_masked_loss"](
                pred[index : index + 1, :width],
                source_batch["target"],
                source_batch["target_mask"],
                source_batch.get("signal_mask"),
                config.loss_type,
                1e-8,
            )
        )
    source_loss = torch.stack(source_losses).mean()
    actual_loss = (losses * sample_valid).sum() / sample_valid.sum()
    torch.testing.assert_close(actual_loss, source_loss)
    source_gradient = torch.autograd.grad(source_loss, pred, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual_loss, pred, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, source_gradient)

    # Sum microbatch contributions against the same all-real-sample denominator.
    accumulated_gradient = torch.zeros_like(pred)
    for start, end in ((0, 1), (1, 2), (2, 4)):
        micro = {key: value[start:end] for key, value in batch.items()}
        micro_loss, micro_valid, _, _ = per_sample_loss(pred[start:end], micro, config, normalization=None)
        accumulated_gradient += torch.autograd.grad(
            (micro_loss * micro_valid).sum() / sample_valid.sum(), pred, retain_graph=True
        )[0]
    torch.testing.assert_close(accumulated_gradient, source_gradient)


def test_hybrid_source(records):
    src = source_namespace()
    _, labels = records
    continuations = []
    for i, n in enumerate((5, 2100, 65)):
        continuations.append(
            {
                "rollout_id": "a",
                "state_id": "a0",
                "continuation_id": str(i),
                "continuation_index": i,
                "continuation_token_ids": [3] * n,
                "prompt_token_ids": [1, 2],
                "reward": float(i % 2),
            }
        )
    config = ScalarConfig("unused", objective="hybrid_terminal_composition", max_length=8192)
    rows, counts = continuation_rows(continuations, labels, config)
    assert counts["included"] == 3
    source = src["TerminalCompositionContinuationDataset"](
        continuations, labels, max_length=8192, continuation_state_count=64, continuation_state_min_gap=32
    )
    for row, theirs in zip(rows, source, strict=True):
        for key in row:
            assert row[key] == theirs[key]
    stats = fit_stats(rows, terminal=True)
    assert stats == src["_terminal_target_stats"](source)
    assert fit_stats(rows[:1], terminal=True)["std"] == 1.0
    norm = {"enabled": True, "mode": "standardize", "mean": 0.2, "std": 0.7}
    batch = collate(rows, config, norm)
    pred = torch.randn(batch["target"].shape, requires_grad=True)
    losses, valid, _, _ = per_sample_loss(pred, batch, config, norm, stats)
    expected = []
    for i, row in enumerate(rows):
        ref = src["model"].collate_token_scalar_batch([row], 0, 8192)
        expected.append(
            src["_hybrid_terminal_loss"](
                pred[i : i + 1, : ref["input_ids"].shape[1]], ref, norm, stats, 1, 1, "mse", 1e-8
            )
        )
    expected = torch.stack(expected)
    torch.testing.assert_close(losses, expected)
    torch.testing.assert_close(
        torch.autograd.grad(losses.mean(), pred, retain_graph=True)[0], torch.autograd.grad(expected.mean(), pred)[0]
    )
    assert valid.all()
    kept, counts = continuation_rows(continuations, labels, replace(config, continuation_budget=1))
    assert len(kept) == 1 and counts["over_budget"] == 2
    assert not continuation_rows(continuations, labels, config, {"a"})[0]


def test_continuation_fallback_coerces_serialized_token_index(records):
    src = source_namespace()
    _, labels = records
    continuation = {
        "rollout_id": "a",
        "token_index": "0",
        "continuation_id": "string-index",
        "continuation_index": 0,
        "continuation_token_ids": [3, 4],
        "prompt_token_ids": [1, 2],
        "reward": 1.0,
    }
    config = ScalarConfig("unused", objective="hybrid_terminal_composition", max_length=8192)
    rows, counts = continuation_rows([continuation], labels, config)
    source = src["TerminalCompositionContinuationDataset"](
        [continuation], labels, max_length=8192, continuation_state_count=64, continuation_state_min_gap=32
    )
    assert counts["input"] == counts["matched"] == counts["eligible"] == counts["included"] == 1
    assert counts.get("unmatched", 0) == counts.get("over_budget", 0) == 0
    assert len(rows) == len(source) == 1
    for key in rows[0]:
        assert rows[0][key] == source[0][key]


def test_full_lengths_zero_signal_and_microbatch(records):
    config = ScalarConfig("unused", loss_type="nonzero_balanced_mse", max_length=8192)
    examples, _ = adapt_legacy_rows(*records)
    rows = ordinary_rows(examples, config)
    norm = normalization_metadata(rows, config)
    batch = collate(rows, config, norm)
    assert batch["signal_mask"][0, 2] == 1  # selected delta zero remains signal
    assert rows[0]["token_targets"][0] == 0
    pred = torch.randn(batch["target"].shape, requires_grad=True)
    losses, valid, _, _ = per_sample_loss(pred, batch, config, norm)
    gradient = torch.autograd.grad(losses.sum() / valid.sum(), pred)[0]
    accumulated = torch.zeros_like(pred)
    for i in range(len(rows)):
        micro = {k: v[i : i + 1] for k, v in batch.items()}
        terms, _, _, _ = per_sample_loss(pred[i : i + 1], micro, config, norm)
        accumulated += torch.autograd.grad(terms.sum() / valid.sum(), pred)[0]
    torch.testing.assert_close(gradient, accumulated, rtol=0, atol=0)
    long = dict(rows[0], prompt_token_ids=[1] * 8188)
    assert scan_lengths([long], config)["rollout"] == 8192
    assert collate([long], config, norm)["target_mask"].sum() == 4
    with pytest.raises(ValueError, match="truncation forbidden"):
        collate([dict(long, prompt_token_ids=[1] * 8189)], config, norm)


@pytest.mark.parametrize("value_layer", [-1, 0, 1, -2])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_source_model(tiny_model, value_layer, dtype):
    src = source_namespace()
    config = ScalarConfig(str(tiny_model), value_layer=value_layer, dtype=dtype, gradient_checkpointing=False)
    model = DeltaScalarModel.from_config(config)
    original = src["model"].LLMTokenScalarModel(str(tiny_model), False, dtype, value_layer).module()
    original.load_state_dict(model.state_dict())
    assert model.scalar_head.weight.dtype == torch.float32
    tokens, mask = torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == "bfloat16"):
        pred, reference = model(tokens, mask), original(tokens, mask)
    torch.testing.assert_close(pred, reference, atol=0, rtol=0)
    pred.float().sum().backward()
    reference.float().sum().backward()
    for ours, theirs in zip(model.parameters(), original.parameters(), strict=True):
        if ours.grad is not None:
            torch.testing.assert_close(ours.grad, theirs.grad, atol=0, rtol=0)


def make_legacy(tiny_model, config, model, normalization):
    return {
        "model": model.state_dict(),
        "config": {
            "model": {"train_model": str(tiny_model), "dtype": config.dtype, "value_layer": config.value_layer},
            "train": {"max_length": config.max_length},
        },
        "target_type": "delta",
        "output_granularity": "token_sequence",
        "step": 17,
        "target_normalization": normalization,
        "training_method": {
            "delta_objective": config.objective,
            "loss_type": config.loss_type,
            "loss_mask": config.loss_mask,
            "delta_label_mode": config.delta_label_mode,
        },
    }


def test_checkpoint_and_frozen_prefixes(tiny_model, tmp_path, records):
    config = ScalarConfig(str(tiny_model), dtype="float32", max_length=3, target_normalization="none")
    model = DeltaScalarModel.from_config(config)
    norm = {"enabled": False, "mode": "none", "mean": None, "std": None}
    old = tmp_path / "old.pt"
    torch.save(make_legacy(tiny_model, config, model, norm), old)
    artifact = tmp_path / "artifact"
    imported = import_legacy(old, artifact, expected_model=str(tiny_model))
    assert imported["normalization"]["mean"] is None and not imported["resume_capable"]
    worker = FrozenDeltaWorker(artifact, device="cpu", microbatch=4)
    before = {k: v.clone() for k, v in worker.model.state_dict().items()}
    examples, _ = adapt_legacy_rows(*records)
    scored = worker.score(examples)
    for example, row in zip(examples, scored, strict=True):
        for state in example.states:
            tokens = (example.rollout.prompt_token_ids + example.rollout.response_token_ids[: state.token_index + 1])[
                -3:
            ]
            with torch.inference_mode():
                expected = model(torch.tensor([tokens]), torch.ones(1, len(tokens), dtype=torch.long))[0, -1].item()
            assert row["delta_pred_raw"][state.token_index] == pytest.approx(expected, abs=1e-7)
            assert row["critic_signal_mask"][state.token_index] == 1
    worker.microbatch = 1
    reversed_rows = worker.score(examples[::-1])[::-1]
    for first, second in zip(scored, reversed_rows, strict=True):
        assert first["delta_pred_raw"] == pytest.approx(second["delta_pred_raw"], abs=1e-7)
    for key, value in worker.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in worker.model.parameters())
    expanded = FrozenDeltaWorker(artifact, device="cpu", scoring_max_length=8192)
    assert expanded.score(examples)[0]["critic_window"]["semantic_changes"]
    with torch.no_grad():
        worker.model.scalar_head.weight.zero_()
        worker.model.scalar_head.bias.zero_()
    zero = worker.score(examples)[0]
    assert zero["delta_pred_raw"] == [0] * 4 and zero["critic_signal_mask"] == [1, 0, 0, 1]
    bad = make_legacy(tiny_model, config, model, norm)
    bad["training_method"]["delta_objective"] = "direct_mixed_advantage"
    torch.save(bad, old)
    with pytest.raises(ValueError, match="Unsupported objective"):
        import_legacy(old, tmp_path / "bad")


def test_standardized_score_and_full_window(tiny_model, tmp_path, records):
    config = ScalarConfig(str(tiny_model), dtype="float32", max_length=3)
    norm = {"enabled": True, "mode": "standardize", "mean": 0.2, "std": 0.7}
    token_hash, _ = tokenizer_fingerprint(str(tiny_model))
    write_artifact(
        tmp_path / "artifact",
        DeltaScalarModel.from_config(config).state_dict(),
        config,
        norm,
        tokenizer_sha256=token_hash,
    )
    worker = FrozenDeltaWorker(tmp_path / "artifact", device="cpu")
    examples, _ = adapt_legacy_rows(*records)
    with pytest.raises(ValueError, match="Selected prefix"):
        worker.score(examples)
    worker = FrozenDeltaWorker(tmp_path / "artifact", device="cpu", scoring_max_length=8192)
    for row in worker.score(examples):
        for raw, normalized, mask in zip(
            row["delta_pred_raw"], row["delta_pred_normalized"], row["critic_signal_mask"], strict=True
        ):
            assert raw == pytest.approx(normalized * 0.7 + 0.2 if mask else 0)
    assert read_artifact(tmp_path / "artifact")["normalization"] == norm
