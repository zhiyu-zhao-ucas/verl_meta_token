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
