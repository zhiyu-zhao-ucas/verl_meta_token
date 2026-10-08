# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from examples.delta_critic.online_critic import OnlineDeltaWorker
from examples.delta_critic.online_optimizer import FP32MasterAdamW
from examples.delta_critic.online_update_data import paired_rows, paired_training_config
from examples.delta_critic.scalar_loss import per_sample_loss
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.training_data import collate


def record(step=1, role="train", rewards=(0.0, 1.0), second_rewards=None):
    rid = f"{step}:{role}"
    response = [2, 3, 4, 5]
    result = {
        "rollout": {
            "id": rid,
            "prompt_token_ids": [1],
            "response_token_ids": response,
            "terminal_reward": 1.0,
            "actor_version": str(step - 1),
        },
        "states": [{"token_index": 0}, {"token_index": 2}],
        "full_selected_indices": [0, 2, 3],
        "critic_role": role,
        "prompt_stable_id": role,
        "labels": [],
    }
    for index, values in ((0, rewards), (2, rewards if second_rewards is None else second_rewards)):
        label = {
            "state_id": f"{rid}:t{index}",
            "rollout_id": rid,
            "token_index": index,
            "v_prefix": sum(values) / len(values),
            "mc_continuations": [],
        }
        for branch, reward in enumerate(values):
            label["mc_continuations"].append(
                {
                    "continuation_index": branch,
                    "reward": reward,
                    "token_ids": [6, 7, 8],
                    "delta_top_logprobs": [[{"prob": 0.7}, {"prob": 0.3}]] * 3,
                }
            )
        result["labels"].append(label)
    return result


def config():
    return paired_training_config(
        ScalarConfig("tiny", dtype="float32", max_length=64, gradient_checkpointing=False), state_count=2, min_gap=1
    )


class TinyScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Embedding(20, 3)
        self.scalar_head = torch.nn.Linear(3, 1)

    def forward(self, input_ids, attention_mask):
        return self.scalar_head(self.backbone(input_ids)).squeeze(-1)


def worker(mode="td", interval=4):
    def initialize(self, artifact, **kwargs):
        self.config = ScalarConfig("tiny", dtype="float32", max_length=64, gradient_checkpointing=False)
        self.checkpoint_config = self.config
        self.metadata = {
            "normalization": {"enabled": True, "mean": 0.1, "std": 0.2},
            "weights_sha256": "initial",
            "stats_version": "stats",
        }
        self.model = TinyScalar().eval().requires_grad_(False)
        self.device = torch.device("cpu")
        self.pad_token_id = 0

    with patch("examples.delta_critic.online_critic.FrozenDeltaWorker.__init__", initialize):
        return OnlineDeltaWorker(
            "tiny",
            update_config={
                "protocol": "paired_fresh_v1",
                "mode": mode,
                "interval": interval,
                "train_pairs_per_step": 1,
                "diagnostic_pairs_per_step": 1,
                "branches_per_state": 2,
                "learning_rate": 2e-6,
                "terminal_state_count": 2,
                "terminal_min_gap": 1,
            },
        )


def test_true_zero_is_supervised_unknown_second_endpoint_is_not():
    rows = paired_rows([record()], config(), branches_per_state=2)
    td = rows["train"]["td"][0]
    assert td["token_targets"] == [0.0] * 4
    assert td["token_loss_mask"] == [1.0, 0.0, 0.0, 0.0]
    batch = collate([td], config(), {"enabled": False})
    pred = torch.ones_like(batch["target"], requires_grad=True)
    losses, _, _, _ = per_sample_loss(pred, batch, config(), {"enabled": False})
    losses.sum().backward()
    assert pred.grad.tolist() == [[0.0, 1.0, 0.0, 0.0, 0.0]]


def test_nonadjacent_pair_rejected_even_if_only_two_mc_states():
    value = record()
    value["full_selected_indices"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="adjacent"):
        paired_rows([value], config(), branches_per_state=2)


def test_loo_terminal_targets_use_other_siblings_and_anchor_start():
    rows = paired_rows([record(rewards=(0.0, 1.0))], config(), branches_per_state=2)
    assert [row["terminal_comp_target"] for row in rows["train"]["terminal"]] == [-1.0, 1.0, -1.0, 1.0]
    assert all(row["terminal_suffix_response_indices"][0] == 0 for row in rows["train"]["terminal"])


def test_saved_values_and_sibling_budget_are_checked():
    value = record()
    value["labels"][0]["v_prefix"] = 0.75
    with pytest.raises(ValueError, match="reward mean"):
        paired_rows([value], config(), branches_per_state=2)
    value = record()
    value["labels"][0]["mc_continuations"].pop()
    with pytest.raises(ValueError, match="exactly 2"):
        paired_rows([value], config(), branches_per_state=2)


def test_four_steps_make_one_update_and_fresh_data_is_cleared():
    value = worker()
    before = deepcopy(value.model.state_dict())
    for step in range(1, 4):
        metrics = value.buffer_fit_mc([record(step), record(step, "diagnostic")], step)
        assert metrics["delta_critic/optimizer_steps"] == 0
        assert all(torch.equal(before[name], tensor) for name, tensor in value.model.state_dict().items())
    metrics = value.buffer_fit_mc([record(4), record(4, "diagnostic")], 4)
    assert metrics["delta_critic/optimizer_steps"] == 1
    assert metrics["delta_critic/update_step"] == 1
    assert metrics["delta_critic/td_rows"] == 4
    assert value.pending_windows == []
    assert metrics["delta_critic/backbone_change_norm"] > 0
    assert metrics["delta_critic/head_change_norm"] > 0
    assert not any(param.requires_grad for param in value.model.parameters())
    with pytest.raises(ValueError, match="contiguous"):
        value.buffer_fit_mc([record(4), record(4, "diagnostic")], 4)


def test_buffer_and_optimizer_resume_at_nonrefresh_checkpoint(tmp_path):
    value = worker("hybrid")
    for step in range(1, 6):
        value.buffer_fit_mc([record(step), record(step, "diagnostic")], step)
    path = tmp_path / "online.pt"
    value.save_online(path, 5)
    restored = worker("hybrid")
    restored.load_online(path, 5)
    assert len(restored.pending_windows) == 1
    for step in range(6, 9):
        value.buffer_fit_mc([record(step), record(step, "diagnostic")], step)
        restored.buffer_fit_mc([record(step), record(step, "diagnostic")], step)
    for name, tensor in value.model.state_dict().items():
        torch.testing.assert_close(tensor, restored.model.state_dict()[name], rtol=0, atol=0)
    assert value.update_step == restored.update_step == 2


def test_hybrid_uses_separate_td_and_terminal_means():
    value = worker("hybrid", interval=1)
    records = [record(), record(role="diagnostic")]
    rows = paired_rows(records, value.fit_config, branches_per_state=2)["train"]
    expected = {}
    with torch.no_grad():
        for kind in ("td", "terminal"):
            total = 0.0
            for row in rows[kind]:
                batch = value._paired_batch([row])
                prediction = value.model(batch["input_ids"], batch["attention_mask"])
                loss, _, td, _ = per_sample_loss(prediction, batch, value.fit_config, value.metadata["normalization"])
                total += float(td.sum() if kind == "td" else loss.sum())
            expected[kind] = total / len(rows[kind])
    metrics = value.buffer_fit_mc(records, 1)
    assert metrics["delta_critic/td_loss"] == pytest.approx(expected["td"])
    assert metrics["delta_critic/terminal_optimization_loss"] == pytest.approx(expected["terminal"])
    assert metrics["delta_critic/optimizer_steps"] == 1


def test_missing_quota_and_prompt_role_change_fail_before_buffer_commit():
    value = worker()
    with pytest.raises(ValueError, match="quota"):
        value.buffer_fit_mc([record()], 1)
    assert value.last_policy_step == 0 and not value.pending_windows
    value.buffer_fit_mc([record(), record(role="diagnostic")], 1)
    swapped = record(2)
    swapped["prompt_stable_id"] = "diagnostic"
    with pytest.raises(ValueError, match="partition"):
        value.buffer_fit_mc([swapped, record(2, "diagnostic")], 2)
    assert value.last_policy_step == 1


def test_stale_actor_version_fails_before_buffer_commit():
    value = worker()
    stale = record()
    stale["rollout"]["actor_version"] = "99"
    with pytest.raises(ValueError, match="current pre-update actor"):
        value.buffer_fit_mc([stale, record(role="diagnostic")], 1)
    assert value.last_policy_step == 0 and not value.pending_windows
    wrong_branch = record()
    wrong_branch["labels"][0]["mc_actor_version"] = "99"
    with pytest.raises(ValueError, match="branch actor version"):
        value.buffer_fit_mc([wrong_branch, record(role="diagnostic")], 1)


def test_fp32_master_preserves_sub_bfloat16_updates_and_fp32_accumulation():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    model.weight.data.fill_(1.0)
    optimizer = FP32MasterAdamW(model, learning_rate=0.001)
    for _ in range(4):
        optimizer.zero_grad()
        for _ in range(3):
            model(torch.ones(1, 1, dtype=torch.bfloat16)).float().sum().backward()
        assert optimizer.masters[0].grad.dtype == torch.float32
        assert optimizer.masters[0].grad.item() == 3.0
        assert model.weight.grad is None
        optimizer.step()
    assert optimizer.masters[0].item() == pytest.approx(0.996, abs=1e-6)
    assert model.weight.item() < 1.0
    assert all(
        item.dtype == torch.float32
        for key, item in optimizer.optimizer.state[optimizer.masters[0]].items()
        if isinstance(item, torch.Tensor)
    )


def test_diagnostics_preserve_exact_zero_labels_and_raw_terminal_targets():
    value = worker(interval=1)
    rows = paired_rows([record(role="diagnostic")], value.fit_config, branches_per_state=2)["diagnostic"]
    metrics, predictions = value._paired_diagnostics(rows, return_predictions=True)
    assert metrics["td_count"] == 1
    assert metrics["td_target_mean"] == 0.0
    assert metrics["td_nonzero_target_count"] == 0
    assert metrics["td_nonzero_sign_accuracy_valid"] == 0
    assert metrics["td_zero_baseline_skill_valid"] == 0
    assert metrics["td_pearson_valid"] == 0
    assert metrics["terminal_count"] == 4
    assert metrics["terminal_target_mean"] == 0.0
    assert metrics["terminal_target_std"] == metrics["terminal_zero_baseline_rmse"] == 1.0
    assert metrics["terminal_zero_baseline_skill_valid"] == 1
    # Verify terminal prediction is the denormalized suffix sum, not the
    # normalized terminal loss or its length-weighted optimizer contribution.
    batch = value._paired_batch([rows["terminal"][0]])
    with torch.no_grad():
        normalized = value.model(batch["input_ids"], batch["attention_mask"])
    positions = batch["terminal_suffix_positions"][0]
    active = batch["terminal_suffix_mask"][0].bool()
    expected_sum = (normalized[0, positions[active]] * 0.2 + 0.1).sum()
    assert predictions["terminal"][0] == pytest.approx(float(expected_sum), abs=1e-6)


def test_diagnostic_update_compares_identical_rows_without_extra_forwards():
    value = worker(interval=1)
    with patch.object(value.model, "forward", wraps=value.model.forward) as forward:
        metrics = value.buffer_fit_mc(
            [record(second_rewards=(1.0, 1.0)), record(role="diagnostic", second_rewards=(1.0, 1.0))], 1
        )
    # One train TD forward, plus one TD and four terminal rows pre/post update.
    assert forward.call_count == 11
    for kind in ("td", "terminal"):
        prefix = f"delta_critic/diagnostic_update/{kind}"
        expected_change = (
            metrics[f"delta_critic/diagnostic_postfit/{kind}_rmse"]
            - metrics[f"delta_critic/diagnostic_prefit/{kind}_rmse"]
        )
        assert metrics[f"{prefix}_rmse_change"] == pytest.approx(expected_change)
        assert metrics[f"{prefix}_rmse_change_valid"] == 1
        assert metrics[f"{prefix}_prediction_drift_rmse"] > 0.0
        assert metrics[f"{prefix}_count"] == (1 if kind == "td" else 4)
    assert not any(parameter.requires_grad for parameter in value.model.parameters())
    assert all(parameter.grad is None for parameter in value.model.parameters())
