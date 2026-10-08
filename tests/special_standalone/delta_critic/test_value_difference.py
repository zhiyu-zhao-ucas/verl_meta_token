# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Boundary alignment, shared controls, and inference-to-policy contract checks."""

import pytest
import torch

from examples.delta_critic.contracts import DeltaExample, Rollout, SelectedState
from examples.delta_critic.score import ScoringRow
from examples.delta_critic.target_ops import selected_labels, state_advantages
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.value_difference import boundary_batch, boundary_positions, boundary_rows, differences
from examples.delta_critic.value_difference_score import FrozenBoundaryWorker


def example():
    rollout = Rollout("r", (1, 2), (3, 4, 5, 6), 1.0, (1.0,) * 4)
    return DeltaExample(rollout, (SelectedState("r", 0, 0.25), SelectedState("r", 2, 0.75)))


def test_mc_boundary_differences_are_the_existing_selected_segment_labels():
    item = example()
    value = boundary_rows([item], "value")[0]
    delta = boundary_rows([item], "direct_delta")[0]
    assert value["boundary_positions"] == (1, 3, 5)  # prompt end, before y2, full response
    assert value["boundary_targets"] == (0.25, 0.75, 1.0)
    assert differences(value["boundary_targets"]) == selected_labels(item, "selected_segment")
    assert delta["boundary_targets"] == (0.5, 0.25, 0.0)
    assert {key: val for key, val in value.items() if key != "boundary_targets"} == {
        key: val for key, val in delta.items() if key != "boundary_targets"
    }


def test_supervision_shift_includes_prompt_end_and_predicted_terminal_anchor():
    config = ScalarConfig(model_path="unused", max_length=8, loss_type="mse", target_normalization="none")
    rows = boundary_rows([example()], "value")
    batch = boundary_batch(rows, config, 0)
    torch.testing.assert_close(batch["target"][0], torch.tensor([0, 0.25, 0, 0.75, 0, 1.0]))
    torch.testing.assert_close(batch["target_mask"][0], torch.tensor([0, 1, 0, 1, 0, 1.0]))
    # Never train a guessed value at an unobserved prefix, nor at p+i instead of p+i-1.
    assert batch["target_mask"][0, 2] == 0


@pytest.mark.parametrize("selected", [(2, 0), (1, 1), (-1,), (4,), (True,)])
def test_ambiguous_or_invalid_boundary_indices_fail(selected):
    with pytest.raises(ValueError):
        boundary_positions(2, 4, selected)


class CausalToy(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        assert not torch.is_grad_enabled()
        return (input_ids * attention_mask).float().cumsum(-1) / 100


def toy_scorer(target):
    scorer = FrozenBoundaryWorker.__new__(FrozenBoundaryWorker)
    scorer.target = target
    scorer.model = CausalToy()
    scorer.device = torch.device("cpu")
    scorer.microbatch = 2
    scorer.pad_token_id = 0
    scorer.config = ScalarConfig(model_path="unused", max_length=32, target_normalization="none")
    scorer.metadata = {"weights_sha256": "weights", "stats_version": "stats"}
    return scorer


def test_value_scoring_uses_next_selected_boundary_and_predicted_endpoint():
    scorer = toy_scorer("value")
    # Metadata reward is deliberately different: the endpoint must be predicted.
    row = ScoringRow("r", (1, 2), (3, 4, 5, 6), (0, 2), metadata={"terminal_reward": 100})
    result = scorer.score([row])[0]
    assert result["delta_pred_raw"] == pytest.approx([0.07, 0, 0.11, 0])
    assert result["critic_window"]["boundary_predictions_raw"] == pytest.approx([0.03, 0.10, 0.21])
    assert result["critic_window"]["terminal_source"] == "model_prediction"
    # Verify unchanged policy segment broadcasting [i,j), including the tail.
    vector = state_advantages(example(), scorer.config.contract(), raw_predictions={0: 0.07, 2: 0.11})
    assert vector.values == (0.07, 0.07, 0.11, 0.11)


def test_full_row_batch_padding_does_not_change_prefix_values_or_row_order():
    rows = [ScoringRow("long", (1, 2), (3, 4, 5, 6), (0, 2)), ScoringRow("short", (2,), (3,), (0,))]
    batched = toy_scorer("value").score(rows)
    assert [row["rollout_id"] for row in batched] == ["long", "short"]
    individual = [toy_scorer("value").score([row])[0] for row in rows]
    assert batched == individual
    assert batched[1]["delta_pred_raw"] == pytest.approx([0.03])


def test_direct_delta_control_reads_same_boundary_without_value_subtraction():
    row = ScoringRow("r", (1, 2), (3, 4, 5, 6), (0, 2))
    result = toy_scorer("direct_delta").score([row])[0]
    assert result["delta_pred_raw"] == pytest.approx([0.03, 0, 0.10, 0])


def test_empty_selection_has_no_policy_signal_and_oversized_context_fails():
    row = ScoringRow("r", (1,), (2,), ())
    assert toy_scorer("value").score([row])[0]["critic_signal_mask"] == [0]
    with pytest.raises(ValueError, match="truncation forbidden"):
        toy_scorer("value").score([ScoringRow("wide", (1,), (2,) * 32, (0,))])


def test_dummy_eval_rows_do_not_enter_loss_or_point_denominators():
    from tensordict import TensorDict

    from examples.delta_critic.value_difference import boundary_mse_loss
    from verl.utils import tensordict_utils as tu

    data = TensorDict(
        {
            "target": torch.tensor([[0.25, 0.75], [100.0, 100.0]]),
            "target_mask": torch.ones(2, 2),
            "sample_valid_mask": torch.tensor([1, 0]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, valid_sample_count=1)
    prediction = torch.tensor([[0.5, 0.5], [200.0, 200.0]], requires_grad=True)
    loss, _ = boundary_mse_loss({"delta_scalar": prediction}, data)
    assert loss.item() == pytest.approx(0.0625)
    loss.backward()
    torch.testing.assert_close(prediction.grad[1], torch.zeros(2))


def test_new_best_replaces_previous_weights_and_keeps_only_one_artifact(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from examples.delta_critic.train_boundary_scalar import save_best

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        lambda module, options: {"scalar_head.weight": torch.ones(1, 2), "scalar_head.bias": torch.zeros(1)},
    )
    worker = SimpleNamespace(engine=SimpleNamespace(module=None))
    config = ScalarConfig(model_path="unused", max_length=8, target_normalization="none")
    save_best(worker, tmp_path, config, 5, 0.2, {})
    first = (tmp_path / "best").resolve()
    assert (first / "COMPLETE").exists()
    save_best(worker, tmp_path, config, 10, 0.1, {})
    assert not first.exists()
    assert (tmp_path / "best").resolve().name == "step_00000010"
    assert len(list(tmp_path.glob("step_*/model.pt"))) == 1
    assert json.loads((tmp_path / "selection.json").read_text())["mse"] == 0.1
