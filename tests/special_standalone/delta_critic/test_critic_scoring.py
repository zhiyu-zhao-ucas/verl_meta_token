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

from types import SimpleNamespace

import pytest
import torch

from examples.delta_critic.checkpoint import (
    file_hash,
    import_legacy,
    tokenizer_fingerprint,
    write_artifact,
)
from examples.delta_critic.scalar_model import DeltaScalarModel
from examples.delta_critic.score import FrozenDeltaWorker, ScoringRow
from examples.delta_critic.training_config import ScalarConfig


@pytest.fixture
def tiny_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
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
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
    ).save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({str(index): index for index in range(32)}, unk_token="0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="0", pad_token="0", eos_token="1").save_pretrained(
        path
    )
    return path


def _source_checkpoint(model_path, config, model, normalization=None):
    return {
        "model": model.state_dict(),
        "config": {
            "model": {
                "train_model": str(model_path),
                "tokenizer_path": str(model_path),
                "dtype": config.dtype,
                "value_layer": config.value_layer,
            },
            "train": {"max_length": config.max_length, "bf16": False, "fp16": False},
        },
        "target_type": "delta",
        "output_granularity": "token_sequence",
        "step": 17,
        "target_normalization": normalization or {"enabled": False, "mode": "none", "mean": None, "std": None},
        "training_method": {
            "target_type": "delta",
            "output_dim": 1,
            "delta_objective": config.objective,
            "loss_type": config.loss_type,
            "loss_mask": config.loss_mask,
            "delta_label_mode": config.delta_label_mode,
            "target_normalization": config.target_normalization,
        },
    }


def _artifact(path, tiny_model, *, max_length=4, normalization=None):
    config = ScalarConfig(
        model_path=str(tiny_model),
        tokenizer_path=str(tiny_model),
        dtype="float32",
        value_layer=0,
        max_length=max_length,
        window_policy="legacy_tail",
        target_normalization="standardize" if normalization and normalization["enabled"] else "none",
    )
    model = DeltaScalarModel.from_config(config)
    token_hash, _ = tokenizer_fingerprint(str(tiny_model), expected_vocab_size=model.config.vocab_size)
    write_artifact(
        path,
        model.state_dict(),
        config,
        normalization or {"enabled": False, "mode": "none", "mean": None, "std": None},
        tokenizer_sha256=token_hash,
        source_sha256="source-checkpoint-hash",
        resume_capable=False,
    )
    return config, model


@pytest.fixture
def rows():
    return [
        ScoringRow("long", (1, 2, 3), (4, 5, 6, 7), (0, 3)),
        ScoringRow("short", (2,), (8, 9), (0, 1)),
    ]


def test_legacy_import_keeps_source_metadata_and_fp32_head(tiny_model, tmp_path):
    source_config = ScalarConfig(
        str(tiny_model),
        tokenizer_path=str(tiny_model),
        dtype="float32",
        value_layer=0,
        max_length=4,
        target_normalization="none",
    )
    source_model = DeltaScalarModel.from_config(source_config)
    normalization = {"enabled": False, "mode": "none", "mean": None, "std": None}
    checkpoint = tmp_path / "legacy.pt"
    torch.save(_source_checkpoint(tiny_model, source_config, source_model, normalization), checkpoint)

    artifact = tmp_path / "artifact"
    metadata = import_legacy(checkpoint, artifact, expected_model=str(tiny_model), expected_tokenizer=str(tiny_model))
    state = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
    assert metadata["source_path"] == str(checkpoint.resolve())
    assert metadata["source_sha256"] == file_hash(checkpoint)
    assert metadata["source_step"] == 17
    assert metadata["source_metadata"]["training_method"]["delta_label_mode"] == "selected_segment"
    assert metadata["config"]["value_layer"] == 0
    assert metadata["config"]["max_length"] == 4
    assert not metadata["resume_capable"]
    assert metadata["normalization"]["mean"] is None and metadata["normalization"]["std"] is None
    assert state["scalar_head.weight"].dtype == torch.float32
    for name, tensor in state.items():
        assert tensor.dtype == source_model.state_dict()[name].dtype


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(target_type="delta_distribution"), "token_sequence scalar delta"),
        (lambda c: c["training_method"].update(delta_objective="direct_mixed_advantage"), "Unsupported objective"),
        (lambda c: c["training_method"].update(loss_mask="prompt"), "Unsupported scalar supervision mask"),
        (lambda c: c["training_method"].update(delta_label_mode="unknown"), "Unsupported or missing delta label mode"),
        (lambda c: c["training_method"].update(value_centering="global"), "Value centering"),
        (lambda c: c["config"]["train"].update(max_length=0), "positive training max_length"),
        (
            lambda c: c["target_normalization"].update(enabled=True, mode="standardize", mean=0.0, std=float("nan")),
            "Nonfinite",
        ),
    ],
)
def test_legacy_import_rejects_unsupported_or_invalid_metadata(tiny_model, tmp_path, mutate, message):
    config = ScalarConfig(str(tiny_model), tokenizer_path=str(tiny_model), dtype="float32", target_normalization="none")
    model = DeltaScalarModel.from_config(config)
    old = _source_checkpoint(tiny_model, config, model)
    mutate(old)
    checkpoint = tmp_path / "bad.pt"
    torch.save(old, checkpoint)
    with pytest.raises(ValueError, match=message):
        import_legacy(checkpoint, tmp_path / "bad-artifact")


def test_score_reorders_microbatches_restores_rows_and_freezes_model(tiny_model, tmp_path, rows):
    normalization = {"enabled": True, "mode": "standardize", "mean": -0.25, "std": 0.5}
    config, reference = _artifact(tmp_path / "artifact", tiny_model, normalization=normalization)
    worker = FrozenDeltaWorker(tmp_path / "artifact", device="cpu", microbatch=2)
    before = {name: value.clone() for name, value in worker.model.state_dict().items()}

    scored = worker.score(rows)
    assert [row["rollout_id"] for row in scored] == ["long", "short"]
    for row, source in zip(scored, rows, strict=True):
        for index in source.selected_token_indices:
            full_ids = source.prompt_token_ids + source.response_token_ids[: index + 1]
            shift = max(0, len(full_ids) - config.max_length)
            window = full_ids[shift:]
            with torch.inference_mode():
                expected = reference(torch.tensor([window]), torch.ones((1, len(window)), dtype=torch.long))[
                    0, len(window) - 1
                ].item()
            assert row["delta_pred_normalized"][index] == pytest.approx(expected)
            expected_raw = expected * normalization["std"] + normalization["mean"]
            assert row["delta_pred_raw"][index] == pytest.approx(expected_raw)
            assert row["critic_signal_mask"][index] == 1.0
        assert row["critic_window"]["prefix_shifts"]
        assert all(
            row["critic_signal_mask"][index] == 0.0
            for index in set(range(len(source.response_token_ids))) - set(source.selected_token_indices)
        )
    for name, value in worker.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in worker.model.parameters())


def test_zero_prediction_remains_a_selected_signal(tiny_model, tmp_path, rows):
    config, model = _artifact(tmp_path / "artifact", tiny_model)
    with torch.no_grad():
        model.scalar_head.weight.zero_()
        model.scalar_head.bias.zero_()
    token_hash, _ = tokenizer_fingerprint(str(tiny_model), expected_vocab_size=model.config.vocab_size)
    write_artifact(
        tmp_path / "zero-artifact",
        model.state_dict(),
        config,
        {"enabled": False, "mode": "none", "mean": None, "std": None},
        tokenizer_sha256=token_hash,
    )
    worker = FrozenDeltaWorker(tmp_path / "zero-artifact", device="cpu", microbatch=3)
    scored = worker.score(rows)
    assert scored[0]["delta_pred_raw"] == [0.0, 0.0, 0.0, 0.0]
    assert scored[0]["critic_signal_mask"] == [1.0, 0.0, 0.0, 1.0]


class _FakeV1Worker:
    def __init__(self, artifact, config, model, pad_token_id):
        self.delta_initial_artifact = str(artifact)
        self.model_config = SimpleNamespace(hf_config=SimpleNamespace(delta_scalar_config=config.as_dict()))
        self.engine_config = SimpleNamespace(forward_only=True, infer_micro_batch_size_per_gpu=2)
        self.optimizer_config = None
        self.engine = SimpleNamespace(module=model, optimizer=None)
        self.pad_token_id = pad_token_id
        self.input_order = []

    def infer_batch(self, data):
        jagged = data["input_ids"]
        self.input_order.extend([tuple(row.tolist()) for row in jagged.unbind()])
        width = data["target"].shape[1]
        tokens = torch.nested.to_padded_tensor(jagged, self.pad_token_id, output_size=(len(data), width))
        output = self.engine.module(tokens, data["attention_mask"])
        predictions = torch.nested.as_nested_tensor([row for row in output], layout=torch.jagged)
        return __import__("tensordict").TensorDict({"delta_scalar": predictions}, batch_size=[len(data)])


def test_v1_infer_batch_path_uses_forward_only_worker(tiny_model, tmp_path, rows):
    artifact = tmp_path / "artifact"
    config, model = _artifact(artifact, tiny_model)
    token_hash, tokenizer = tokenizer_fingerprint(str(tiny_model), expected_vocab_size=model.config.vocab_size)
    worker = _FakeV1Worker(artifact, config, model, tokenizer.pad_token_id)
    scorer = FrozenDeltaWorker(artifact, device="cpu", microbatch=2, engine_worker=worker)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    scored = scorer.score(rows)
    assert worker.input_order == [
        (2, 8),
        (2, 8, 9),
        (1, 2, 3, 4),
        (4, 5, 6, 7),
    ]
    assert [row["rollout_id"] for row in scored] == ["long", "short"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


def test_transfer_queue_round_trip_preserves_row_keys_and_unpadded_lengths(tiny_model, tmp_path):
    tq = pytest.importorskip("transfer_queue")
    tensordict = pytest.importorskip("tensordict")
    artifact = tmp_path / "artifact"
    _artifact(artifact, tiny_model)
    worker = FrozenDeltaWorker(artifact, device="cpu", microbatch=2)
    started = False
    try:
        tq.init()
        started = True
        fields = tensordict.TensorDict(
            {
                name: torch.nested.as_nested_tensor([torch.tensor(row) for row in values], layout=torch.jagged)
                for name, values in {
                    "prompts": [[1, 2], [2]],
                    "responses": [[3, 4, 5], [3, 4]],
                    "selected_token_indices": [[0, 2], [1]],
                }.items()
            },
            batch_size=[2],
        )
        meta = tq.kv_batch_put(keys=["critic-row-a", "critic-row-b"], partition_id="critic-score-tests", fields=fields)
        scored_meta = worker.score_transfer_queue(meta)
        assert scored_meta.keys == meta.keys
        assert scored_meta.partition_id == meta.partition_id
        output = tq.kv_batch_get_by_meta(
            scored_meta,
            select_fields=[
                "critic_row_id",
                "critic_signal_mask",
                "delta_pred_normalized",
                "delta_pred_raw",
                "critic_checkpoint",
            ],
        )
        assert list(output["critic_row_id"]) == ["critic-row-a", "critic-row-b"]
        assert output["critic_signal_mask"][0].tolist() == [1.0, 0.0, 1.0]
        assert output["critic_signal_mask"][1].tolist() == [0.0, 1.0]
        assert output["delta_pred_raw"][0].shape == (3,)
        assert output["delta_pred_raw"][1].shape == (2,)
    finally:
        if started:
            tq.close()


def test_legacy_missing_label_requires_declared_source_and_preserves_original(tiny_model, tmp_path):
    config = ScalarConfig(str(tiny_model), dtype="float32", max_length=8, target_normalization="none")
    model = DeltaScalarModel.from_config(config)
    original = _source_checkpoint(tiny_model, config, model)
    original["training_method"].pop("delta_label_mode")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(original, checkpoint)
    with pytest.raises(ValueError, match="missing delta label"):
        import_legacy(checkpoint, tmp_path / "missing")
    with pytest.raises(ValueError, match="label_mode_source"):
        import_legacy(checkpoint, tmp_path / "unsourced", declared_label_mode="selected_segment")
    meta = import_legacy(
        checkpoint,
        tmp_path / "declared",
        declared_label_mode="selected_segment",
        label_mode_source="run/configs/selected_segment.yaml",
    )
    assert meta["config"]["delta_label_mode"] == "selected_segment"
    assert "delta_label_mode" not in meta["source_metadata"]["training_method"]
    assert meta["source_sha256"] == file_hash(checkpoint)
    assert meta["declared_metadata"] == {
        "delta_label_mode": "selected_segment",
        "source": "run/configs/selected_segment.yaml",
    }
    assert meta["inferred_metadata"] == {}


def test_legacy_label_declaration_cannot_override_recorded_mode(tiny_model, tmp_path):
    config = ScalarConfig(str(tiny_model), dtype="float32", max_length=8, target_normalization="none")
    model = DeltaScalarModel.from_config(config)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(_source_checkpoint(tiny_model, config, model), checkpoint)
    with pytest.raises(ValueError, match="conflicts"):
        import_legacy(
            checkpoint,
            tmp_path / "bad",
            declared_label_mode="paired_next_state",
            label_mode_source="other-experiment.yaml",
        )
