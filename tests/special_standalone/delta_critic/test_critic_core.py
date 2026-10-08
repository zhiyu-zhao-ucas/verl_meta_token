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
from torch import nn

from examples.delta_critic.scalar_loss import delta_loss, per_sample_loss
from examples.delta_critic.scalar_model import DeltaScalarModel
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.training_data import collate, resolve_max_length


class TinyDecoder(nn.Module):
    def __init__(self, max_position_embeddings=8):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=1,
            max_position_embeddings=max_position_embeddings,
        )
        self._no_split_modules = []
        self.embedding = nn.Embedding(16, 4)
        self.transform = nn.Linear(4, 4)

    def get_decoder(self):
        return self

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False):
        initial = self.embedding(input_ids)
        final = torch.tanh(self.transform(initial))
        return SimpleNamespace(
            last_hidden_state=final,
            hidden_states=(initial, final) if output_hidden_states else None,
        )


def test_model_reads_current_token_hidden_state_and_enforces_native_context():
    torch.manual_seed(11)
    model = DeltaScalarModel(TinyDecoder(max_position_embeddings=5), value_layer=-1)
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    padded = model(ids, torch.tensor([[1, 1, 1], [1, 1, 0]]))
    unmasked = model(ids)
    torch.testing.assert_close(padded, unmasked)
    assert padded.shape == (2, 3)
    assert model.scalar_head.out_features == 1 and model.scalar_head.bias is not None
    assert model.scalar_head.weight.dtype == torch.float32
    with pytest.raises(ValueError, match="exceeds backbone max_position_embeddings"):
        model(torch.ones((1, 6), dtype=torch.long))


def test_native_context_resolver_rejects_oversized_override():
    native = SimpleNamespace(max_position_embeddings=40960)
    assert resolve_max_length(None, native) == 40960
    assert resolve_max_length(2048, native) == 2048
    with pytest.raises(ValueError, match="exceeds backbone max_position_embeddings"):
        resolve_max_length(40961, native)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        resolve_max_length(None, SimpleNamespace())


def test_local_td0_rejects_non_source_loss_weights():
    with pytest.raises(ValueError, match="local_td0 uses fixed unit weights"):
        ScalarConfig("unused", max_length=8, td_weight=0.5)


def test_collate_right_pads_and_places_response_target_at_prompt_plus_token_index():
    config = ScalarConfig("unused", max_length=5)
    rows = [
        {
            "id": "long",
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4, 5],
            "token_targets": [0.25, -0.5, 0.0],
            "token_loss_mask": [1, 1, 1],
            "token_signal_mask": [1, 0, 1],
        },
        {
            "id": "short",
            "prompt_token_ids": [6],
            "response_token_ids": [7],
            "token_targets": [1.5],
            "token_loss_mask": [1],
            "token_signal_mask": [0],
        },
    ]
    batch = collate(rows, config, normalization=None, pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    assert batch["target"].tolist() == [[0, 0, 0.25, -0.5, 0.0], [0, 1.5, 0, 0, 0]]
    assert batch["target_mask"].tolist() == [[0, 0, 1, 1, 1], [0, 1, 0, 0, 0]]
    assert batch["signal_mask"].tolist() == [[0, 0, 1, 0, 1], [0, 0, 0, 0, 0]]
    assert batch["terminal_suffix_positions"].shape == (2, 0)
    too_long = dict(rows[0], prompt_token_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="truncation forbidden"):
        collate([too_long], config, normalization=None)


def test_balanced_loss_uses_explicit_signal_even_for_zero_delta():
    config = ScalarConfig("unused", loss_type="nonzero_balanced_mse", max_length=5)
    row = {
        "id": "signal-zero",
        "prompt_token_ids": [1],
        "response_token_ids": [2, 3, 4],
        "token_targets": [0.0, 0.0, 0.0],
        "token_loss_mask": [1, 1, 1],
        "token_signal_mask": [1, 0, 0],
    }
    batch = collate([row], config, normalization=None)
    assert batch["signal_mask"].tolist() == [[0, 1, 0, 0]]
    pred = torch.tensor([[0.0, 2.0, 1.0, 1.0]], requires_grad=True)
    losses, valid, _, _ = per_sample_loss(pred, batch, config, normalization=None)
    assert valid.tolist() == [True]
    torch.testing.assert_close(losses, torch.tensor([1.25]))
    torch.autograd.grad(losses.sum(), pred)


def test_hybrid_collate_keeps_gather_positions_and_loss_uses_raw_suffix_sum():
    config = ScalarConfig("unused", objective="hybrid_terminal_composition", max_length=8)
    row = {
        "id": "hybrid",
        "prompt_token_ids": [1, 2],
        "response_token_ids": [3, 4, 5, 6, 7],
        "token_targets": [0.0] * 5,
        "token_loss_mask": [0] * 5,
        "token_signal_mask": [0] * 5,
        "terminal_suffix_response_indices": [0, 2, 4],
        "terminal_comp_target": 1.0,
        "terminal_comp_valid": True,
    }
    batch = collate(
        [row],
        config,
        normalization={"enabled": True, "mode": "standardize", "mean": 0.25, "std": 0.5},
    )
    assert batch["terminal_suffix_positions"].tolist() == [[2, 4, 6]]
    assert batch["terminal_suffix_mask"].tolist() == [[1.0, 1.0, 1.0]]
    pred = torch.full((1, 7), -0.5, requires_grad=True)
    losses, valid, td, terminal = per_sample_loss(
        pred,
        batch,
        config,
        normalization={"enabled": True, "mode": "standardize", "mean": 0.25, "std": 0.5},
        terminal_stats={"count": 2, "mean": 0.0, "std": 0.5},
    )
    # Three raw zero predictions sum to 0; the independently normalized terminal
    # error is (0 - 1) / 0.5, with the source 0.5 * squared-error coefficient.
    torch.testing.assert_close(losses, torch.tensor([2.0]))
    torch.testing.assert_close(td, torch.tensor([0.0]))
    torch.testing.assert_close(terminal, torch.tensor([2.0]))
    assert valid.tolist() == [True]
    grad = torch.autograd.grad(losses.sum(), pred)[0]
    assert grad[0, [2, 4, 6]].tolist() == pytest.approx([-2.0, -2.0, -2.0])
    assert grad[0, [0, 1, 3, 5]].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_v1_loss_callback_reports_sample_mean_and_coverage_metrics():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    config = ScalarConfig("unused", max_length=3)
    data = TensorDict(
        {
            "target": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "target_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            "signal_mask": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]]),
            "sample_valid_mask": torch.tensor([1.0, 1.0, 0.0]),
        },
        batch_size=[3],
    )
    tu.assign_non_tensor(
        data,
        delta_config=config.as_dict(),
        normalization=None,
        terminal_stats=None,
        dp_size=1,
        valid_sample_count=2,
    )
    pred = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [100.0, 100.0, 100.0]], requires_grad=True)
    loss, metrics = delta_loss({"delta_scalar": pred}, data)
    # Logical row losses are 0.5 and 0.5, regardless of their token counts;
    # the synthetic third row contributes neither numerator nor denominator.
    torch.testing.assert_close(loss, torch.tensor(0.5))
    assert metrics["critic/td_loss"].aggregate() == pytest.approx(0.5)
    assert metrics["critic/total_loss"].aggregate() == pytest.approx(0.5)
    assert metrics["critic/valid_rows"].aggregate() == pytest.approx(2.0)
    assert metrics["critic/target_tokens"].aggregate() == pytest.approx(3.0)
    assert metrics["critic/signal_tokens"].aggregate() == pytest.approx(2.0)
    assert metrics["critic/background_tokens"].aggregate() == pytest.approx(1.0)
    assert metrics["critic/input_tokens"].aggregate() == pytest.approx(5.0)


def test_terminal_defaults_keep_legacy_semantics_and_selection_defaults():
    config = ScalarConfig("unused")
    assert config.terminal_normalization == "independent"
    assert config.terminal_length_weighting == "none"
    assert config.hybrid_reduction == "all_samples"
    assert config.continuation_selection == "uniform"
    assert config.continuation_max_candidates == 5

    with pytest.raises(ValueError, match="terminal_length_weighting"):
        ScalarConfig("unused", terminal_length_weighting="per_token")
    with pytest.raises(ValueError, match="continuation_selection"):
        ScalarConfig("unused", continuation_selection="topk")
    with pytest.raises(ValueError, match="continuation_max_candidates"):
        ScalarConfig("unused", continuation_max_candidates=0)
    with pytest.raises(ValueError, match="continuation_max_candidates"):
        ScalarConfig("unused", continuation_max_candidates=1.5)


def test_inverse_count_uses_raw_sum_k_times_mean_and_masks_padding_and_invalid_rows():
    config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        terminal_normalization="td",
        terminal_length_weighting="inverse_count",
        terminal_weight=1.7,
        max_length=4,
    )
    batch = {
        "target": torch.zeros((3, 4)),
        "target_mask": torch.zeros((3, 4)),
        "signal_mask": torch.zeros((3, 4)),
        "sample_valid_mask": torch.tensor([1.0, 1.0, 0.0]),
        "terminal_comp_valid": torch.tensor([1.0, 1.0, 1.0]),
        "terminal_comp_target": torch.tensor([4.0, 1.0, float("nan")]),
        "terminal_suffix_positions": torch.tensor([[0, 2], [1, -1], [999, -1]]),
        "terminal_suffix_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
    }
    normalization = {"enabled": True, "mode": "standardize", "mean": 0.5, "std": 2.0}
    pred = torch.tensor(
        [[0.25, 8.0, 0.75, -8.0], [0.0, -0.25, 8.0, 8.0], [5.0, 5.0, 5.0, 5.0]],
        requires_grad=True,
    )

    losses, valid, _, terminal = per_sample_loss(pred, batch, config, normalization)
    # Each selected prediction is returned to raw units before summation: row 0
    # is (2 * .25 + .5) + (2 * .75 + .5) = 3, including K * mean.
    residual_0 = (2 * 0.25 + 0.5) + (2 * 0.75 + 0.5) - 4.0
    residual_1 = (2 * -0.25 + 0.5) - 1.0
    expected_terminal = torch.tensor([0.5 * (residual_0 / 2) ** 2, 0.5 * (residual_1 / 2) ** 2, 0.0])
    expected_losses = config.terminal_weight * expected_terminal * torch.tensor([0.5, 1.0, 0.0])
    torch.testing.assert_close(terminal, expected_terminal)
    torch.testing.assert_close(losses, expected_losses)
    assert valid.tolist() == [True, True, False]

    expected_pred = pred.detach().clone().requires_grad_()
    expected_raw_0 = (2 * expected_pred[0, 0] + 0.5) + (2 * expected_pred[0, 2] + 0.5)
    expected_raw_1 = 2 * expected_pred[1, 1] + 0.5
    literal = config.terminal_weight * (
        0.5 * ((expected_raw_0 - 4.0) / 2.0).square() / 2 + 0.5 * ((expected_raw_1 - 1.0) / 2.0).square()
    )
    actual_gradient = torch.autograd.grad(losses.sum(), pred)[0]
    expected_gradient = torch.autograd.grad(literal, expected_pred)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(losses).all()
    assert torch.equal(actual_gradient[2], torch.zeros_like(actual_gradient[2]))


def test_terminal_independent_std_matches_equivalent_td_scale_weight():
    normalization = {"enabled": True, "mode": "standardize", "mean": 0.25, "std": 2.0}
    batch = {
        "target": torch.zeros((1, 3)),
        "target_mask": torch.zeros((1, 3)),
        "signal_mask": torch.zeros((1, 3)),
        "terminal_comp_valid": torch.tensor([1.0]),
        "terminal_comp_target": torch.tensor([2.0]),
        "terminal_suffix_positions": torch.tensor([[0, 2]]),
        "terminal_suffix_mask": torch.tensor([[1.0, 1.0]]),
    }
    pred = torch.tensor([[0.0, -1.0, 0.5]], requires_grad=True)
    independent = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        terminal_normalization="independent",
        terminal_weight=0.25,
        max_length=3,
    )
    td_scaled = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        terminal_normalization="td",
        terminal_weight=4.0,
        max_length=3,
    )
    stats = {"count": 2, "mean": 0.0, "std": 0.5}
    independent_loss, _, _, _ = per_sample_loss(pred, batch, independent, normalization, stats)
    td_loss, _, _, _ = per_sample_loss(pred, batch, td_scaled, normalization, None)
    torch.testing.assert_close(independent_loss, td_loss)
    independent_grad = torch.autograd.grad(independent_loss.sum(), pred, retain_graph=True)[0]
    td_grad = torch.autograd.grad(td_loss.sum(), pred)[0]
    torch.testing.assert_close(independent_grad, td_grad)


@pytest.mark.parametrize("std", [0.0, -1.0, float("inf"), float("nan")])
def test_enabled_td_normalization_requires_finite_positive_scale(std):
    config = ScalarConfig("unused", max_length=2)
    batch = {
        "target": torch.zeros((1, 2)),
        "target_mask": torch.ones((1, 2)),
        "signal_mask": torch.zeros((1, 2)),
    }
    with pytest.raises(ValueError, match="finite positive std"):
        per_sample_loss(
            torch.zeros((1, 2)),
            batch,
            config,
            {"enabled": True, "mean": 0.0, "std": std},
        )


def test_separate_sample_inverse_count_reduction_preserves_dp_microbatch_gradients_and_raw_metrics():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    config = ScalarConfig(
        "unused",
        objective="hybrid_terminal_composition",
        terminal_normalization="td",
        terminal_length_weighting="inverse_count",
        hybrid_reduction="separate_samples",
        td_weight=0.3,
        terminal_weight=0.7,
        max_length=4,
    )
    normalization = {"enabled": True, "mode": "standardize", "mean": 0.5, "std": 2.0}
    batch = TensorDict(
        {
            "target": torch.zeros((4, 4)),
            "target_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0] * 4, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            "signal_mask": torch.zeros((4, 4)),
            "attention_mask": torch.ones((4, 4)),
            "sample_valid_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
            "terminal_comp_valid": torch.tensor([1.0, 1.0, 0.0, 1.0]),
            "terminal_comp_target": torch.tensor([3.5, -0.5, 0.0, float("nan")]),
            "terminal_suffix_positions": torch.tensor([[1, 3], [2, -1], [-1, -1], [999, -1]]),
            "terminal_suffix_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        },
        batch_size=[4],
    )
    coefficient = torch.tensor(0.4, requires_grad=True)
    features = torch.tensor(
        [[0.5, 1.0, -0.5, 1.5], [0.25, -0.25, 0.75, 1.0], [0.5, 1.5, -0.75, 0.25], [1.0, 1.0, 1.0, 1.0]]
    )
    pred = coefficient * features
    td_rows = 0.5 * (pred[[0, 2], 0]).square()
    raw_row_0 = (2 * pred[0, 1] + 0.5) + (2 * pred[0, 3] + 0.5)
    raw_row_1 = 2 * pred[1, 2] + 0.5
    residuals = torch.stack([raw_row_0 - 3.5, raw_row_1 - (-0.5)])
    unweighted_terminal = 0.5 * (residuals / 2.0).square()
    expected = (
        config.td_weight * td_rows.mean()
        + config.terminal_weight * (unweighted_terminal * torch.tensor([0.5, 1.0])).mean()
    )
    expected_gradient = torch.autograd.grad(expected, coefficient, retain_graph=True)[0]

    def run(dp_size, microbatch):
        rank_losses = []
        reference_metrics = None
        for rank in range(dp_size):
            indices = list(range(rank, len(batch), dp_size))
            rank_loss = coefficient.new_zeros(())
            for start in range(0, len(indices), microbatch):
                selected = indices[start : start + microbatch]
                micro = batch[selected].clone()
                tu.assign_non_tensor(
                    micro,
                    delta_config=config.as_dict(),
                    normalization=normalization,
                    terminal_stats=None,
                    dp_size=dp_size,
                    valid_sample_count=3,
                    td_sample_count=2,
                    terminal_sample_count=2,
                )
                loss, metrics = delta_loss({"delta_scalar": pred[selected]}, micro)
                rank_loss = rank_loss + loss
                if dp_size == 1 and microbatch == 4:
                    reference_metrics = metrics
            rank_losses.append(rank_loss)
        return sum(rank_losses) / dp_size, reference_metrics

    reference_loss, reference_metrics = run(dp_size=1, microbatch=4)
    torch.testing.assert_close(reference_loss, expected)
    for dp_size, microbatch in ((1, 1), (1, 2), (2, 1), (2, 2)):
        reduced, _ = run(dp_size=dp_size, microbatch=microbatch)
        actual_gradient = torch.autograd.grad(reduced, coefficient, retain_graph=True)[0]
        torch.testing.assert_close(reduced, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=1e-7)

    assert reference_metrics["critic/terminal_count"].aggregate() == pytest.approx(2.0)
    assert reference_metrics["critic/terminal_suffix_count"].aggregate() == pytest.approx(3.0)
    assert reference_metrics["critic/terminal_raw_squared_error"].aggregate() == pytest.approx(
        float(residuals.detach().square().sum())
    )
    assert reference_metrics["critic/terminal_loss"].aggregate() == pytest.approx(
        float(unweighted_terminal.detach().mean())
    )
    assert reference_metrics["critic/terminal_optimization_loss"].aggregate() == pytest.approx(
        float((config.terminal_weight * unweighted_terminal * torch.tensor([0.5, 1.0])).mean().detach())
    )
