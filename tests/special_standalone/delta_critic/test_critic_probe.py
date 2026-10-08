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

"""Deterministic selection and truthful metric/gradient coverage for critic probes."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from examples.delta_critic.probe_critic import (
    _artifact_label,
    _compatible_artifacts,
    _reproject_normalized,
    _score_full_sequences,
    checkpoint_metrics,
    gradient_similarity,
    head_gradient_probe,
    local_td_metrics,
    paired_records,
    select_terminal_probe_rows,
    terminal_raw_metrics,
)
from examples.delta_critic.score import ScoringRow
from examples.delta_critic.training_config import ScalarConfig


def test_terminal_selector_is_seeded_group_round_robin_and_nested():
    rows = [
        {"id": f"{group}-{index}", "prefix_group_id": group}
        for group, count in (("a", 4), ("b", 3), ("c", 2), ("d", 1))
        for index in range(count)
    ]
    selected = select_terminal_probe_rows(rows, 7, seed=19)
    smaller = select_terminal_probe_rows(rows, 4, seed=19)
    repeat = select_terminal_probe_rows(rows, 7, seed=19)
    changed_seed = select_terminal_probe_rows(rows, 7, seed=23)

    assert [row["id"] for row in selected] == [row["id"] for row in repeat]
    assert [row["id"] for row in smaller] == [row["id"] for row in selected[:4]]
    assert len({row["id"] for row in selected}) == 7
    assert len({row["prefix_group_id"] for row in selected[:4]}) == 4
    assert [row["id"] for row in changed_seed] != [row["id"] for row in selected]


def test_local_td_metrics_use_raw_errors_and_exact_nonzero_sign_subset():
    rows = [
        {"prediction_raw": 0.8, "target_raw": 1.0},
        {"prediction_raw": -0.5, "target_raw": -1.0},
        {"prediction_raw": 0.4, "target_raw": 0.0},
    ]
    metrics = local_td_metrics(rows)
    assert metrics["raw"]["count"] == 3
    assert metrics["raw"]["mse"] == pytest.approx(0.15)
    assert metrics["nonzero_target_count"] == 2
    assert metrics["nonzero_raw"]["spearman"] == pytest.approx(1.0)
    assert metrics["nonzero_sign_accuracy"] == pytest.approx(1.0)


def test_nonzero_local_target_with_zero_prediction_is_a_sign_miss():
    metrics = local_td_metrics([{"prediction_raw": 0.0, "target_raw": -0.25}])
    assert metrics["nonzero_sign_accuracy"] == pytest.approx(0.0)


def test_terminal_metrics_report_zero_baseline_by_k_and_length():
    rows = [
        {
            "prediction_raw": 0.5,
            "target_raw": 1.0,
            "suffix_count": 2,
            "continuation_length": 20,
        },
        {
            "prediction_raw": 1.0,
            "target_raw": 1.0,
            "suffix_count": 2,
            "continuation_length": 60,
        },
        {
            "prediction_raw": -1.0,
            "target_raw": 0.0,
            "suffix_count": 1,
            "continuation_length": 300,
        },
    ]
    metrics = terminal_raw_metrics(rows)
    assert metrics["count"] == 3
    assert metrics["raw_mse"] == pytest.approx(5.0 / 12.0)
    assert metrics["zero_raw_rmse"] == pytest.approx((2.0 / 3.0) ** 0.5)
    assert metrics["by_suffix_count_k"]["2"]["raw_rmse"] == pytest.approx(0.5 / 2.0**0.5)
    assert metrics["by_continuation_length"]["1-32"]["count"] == 1
    assert metrics["by_continuation_length"]["257-512"]["count"] == 1


def test_paired_records_preserve_identical_positions_and_bootstrap_groups():
    local = [
        {
            "id": "rollout-1",
            "prompt_id": "query-1",
            "response_index": 0,
            "selected_response_indices": [1, 3],
            "selected_local_deltas": [0.2, -0.1],
        }
    ]
    terminal = [
        {
            "id": "branch-1",
            "rollout_id": "rollout-1",
            "prompt_id": "query-1",
            "response_index": 0,
            "prefix_group_id": "state-1",
            "anchor_state_id": "state-1",
            "anchor_token_index": 3,
            "terminal_suffix_response_indices": [0, 2],
            "response_token_ids": [10, 11, 12],
            "terminal_comp_target": 0.4,
        }
    ]
    scores = {
        "hybrid": {
            "local": {"rollout-1": {"delta_pred_raw": [0, 0.1, 0, -0.2]}},
            "terminal": {"branch-1": {"delta_pred_raw": [0.2, 0, 0.3]}},
        },
        "td": {
            "local": {"rollout-1": {"delta_pred_raw": [0, 0.3, 0, -0.1]}},
            "terminal": {"branch-1": {"delta_pred_raw": [0.1, 0, 0.6]}},
        },
    }
    records = paired_records(local, terminal, scores, ["hybrid", "td"])

    assert [row["token_index"] for row in records[:2]] == [1, 3]
    assert records[0]["prediction_raw"] == {"hybrid": 0.1, "td": 0.3}
    assert records[0]["anchor_state_id"] == "rollout-1:t1"
    assert records[0]["bootstrap_group"] == "rollout-1"
    assert records[2]["positions"] == [0, 2]
    assert records[2]["prediction_raw"] == {"hybrid": pytest.approx(0.5), "td": pytest.approx(0.7)}
    assert records[2]["position_prediction_raw"] == {"hybrid": [0.2, 0.3], "td": [0.1, 0.6]}
    assert records[2]["bootstrap_group"] == "state-1"
    assert checkpoint_metrics(records, "hybrid")["terminal"]["raw_rmse"] == pytest.approx(0.1)


def test_full_sequence_scorer_extracts_many_positions_and_denormalizes():
    class TokenEcho(nn.Module):
        def forward(self, input_ids, attention_mask):
            return input_ids.float() * attention_mask

    worker = SimpleNamespace(
        config=SimpleNamespace(window_policy="full", max_length=8, scoring_autocast="none"),
        engine_worker=None,
        metadata={
            "normalization": {"enabled": True, "mean": 0.5, "std": 2.0},
            "weights_sha256": "weights",
            "stats_version": "stats",
        },
        model=TokenEcho(),
        device=torch.device("cpu"),
        microbatch=2,
        pad_token_id=0,
    )
    rows = [
        ScoringRow("a", (3, 4), (5, 6, 7), (0, 2)),
        ScoringRow("b", (1,), (2,), ()),
    ]
    scored = _score_full_sequences(worker, rows)

    assert scored[0]["delta_pred_normalized"] == [5.0, 0.0, 7.0]
    assert scored[0]["delta_pred_raw"] == [10.5, 0.0, 14.5]
    assert scored[0]["critic_signal_mask"] == [1.0, 0.0, 1.0]
    assert scored[1]["delta_pred_raw"] == [0.0]


def test_gradient_similarity_is_an_exact_vector_cosine():
    metrics = gradient_similarity(torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0]))
    assert metrics["td_gradient_norm"] == pytest.approx(3.0)
    assert metrics["terminal_gradient_norm"] == pytest.approx(4.0)
    assert metrics["gradient_dot_product"] == pytest.approx(0.0)
    assert metrics["gradient_cosine"] == pytest.approx(0.0)
    assert metrics["gradient_dimension"] == 2
    assert gradient_similarity(torch.zeros(2), torch.ones(2))["gradient_cosine"] is None


def test_artifact_labels_disambiguate_checkpoints_at_the_same_step():
    config = {"objective": "local_td0"}
    first = {"config": config, "training_state": {"global_step": 80}, "weights_sha256": "a" * 64}
    second = {"config": config, "training_state": {"global_step": 80}, "weights_sha256": "b" * 64}
    assert _artifact_label(first) == "local_td0-step80-aaaaaaaa"
    assert _artifact_label(first) != _artifact_label(second)


def test_paired_raw_scoring_allows_different_normalization_metadata():
    config = {
        "model_path": "model",
        "tokenizer_path": None,
        "value_layer": -1,
        "max_length": 32,
        "window_policy": "full",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "scoring_autocast": "none",
    }
    left = {
        "tokenizer_sha256": "tokenizer",
        "config": config,
        "normalization": {"enabled": True, "mean": 0.0, "std": 0.2},
    }
    right = {
        "tokenizer_sha256": "tokenizer",
        "config": config,
        "normalization": {"enabled": True, "mean": 0.3, "std": 0.7},
    }
    _compatible_artifacts(left, right)


def test_head_gradient_probe_uses_separate_hybrid_losses_and_labels_scope():
    class TinyScalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 3)
            self.scalar_head = nn.Linear(3, 1)

        def forward(self, input_ids, attention_mask):
            return self.scalar_head(self.embedding(input_ids)).squeeze(-1)

    torch.manual_seed(4)
    model = TinyScalar()
    config = ScalarConfig(
        "unused",
        dtype="float32",
        objective="hybrid_terminal_composition",
        loss_type="mse",
        target_normalization="none",
        max_length=8,
        window_policy="full",
        terminal_normalization="td",
        terminal_length_weighting="inverse_count",
        hybrid_reduction="separate_samples",
        td_weight=1.0,
        terminal_weight=0.003,
        training_autocast="none",
    )
    td_rows = [
        {
            "id": "td",
            "prompt_token_ids": [1],
            "response_token_ids": [2, 3],
            "token_targets": [1.0, -0.5],
            "token_loss_mask": [1.0, 1.0],
            "token_signal_mask": [1.0, 0.0],
            "terminal_suffix_response_indices": [],
            "terminal_comp_target": 0.0,
            "terminal_comp_valid": False,
        }
    ]
    terminal_rows = [
        {
            "id": "terminal",
            "prompt_token_ids": [1],
            "response_token_ids": [4, 5],
            "token_targets": [0.0, 0.0],
            "token_loss_mask": [0.0, 0.0],
            "token_signal_mask": [0.0, 0.0],
            "terminal_suffix_response_indices": [0, 1],
            "terminal_comp_target": 0.2,
            "terminal_comp_valid": True,
        }
    ]
    none = {"enabled": False, "mode": "none", "mean": None, "std": None}
    result = head_gradient_probe(
        model,
        td_rows,
        terminal_rows,
        source_normalization=none,
        loss_config=config,
        loss_normalization=none,
        terminal_stats=None,
        pad_token_id=0,
        device="cpu",
        batch_size=1,
    )

    assert result["parameter_scope"] == "scalar_head"
    assert result["gradient_dimension"] == sum(parameter.numel() for parameter in model.scalar_head.parameters())
    assert result["td_rows"] == 1
    assert result["terminal_rows"] == 1
    assert result["terminal_weight_included"] == pytest.approx(0.003)
    assert result["td_gradient_norm"] > 0
    assert result["terminal_gradient_norm"] > 0
    assert -1.0 <= result["gradient_cosine"] <= 1.0
    assert all(parameter.requires_grad for parameter in model.embedding.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_reprojected_predictions_preserve_raw_values_across_head_normalizations():
    source = {"enabled": True, "mean": 0.25, "std": 2.0}
    target = {"enabled": True, "mean": -0.5, "std": 0.5}
    prediction = torch.tensor([0.5])
    projected = _reproject_normalized(prediction, source, target)
    reconstructed_raw = projected * target["std"] + target["mean"]
    torch.testing.assert_close(reconstructed_raw, torch.tensor([1.25]))
