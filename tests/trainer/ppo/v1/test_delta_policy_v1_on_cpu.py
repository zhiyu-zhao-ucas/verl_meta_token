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

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import BatchMeta, KVBatchMeta

from examples.delta_critic.checkpoint import tokenizer_fingerprint, tokenizer_fingerprint_from_tokenizer
from examples.delta_critic.policy_config import DeltaPolicyConfig
from examples.delta_critic.policy_online import validate_online_v1_settings
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import transferqueue_utils as tq_utils
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.padding import response_from_nested


@pytest.mark.parametrize("kl_mask_scope", ["policy", "response"])
def test_online_v1_settings_preserve_requested_kl_scope(kl_mask_scope):
    config = {
        "algorithm": {
            "delta_policy": {
                "enabled": True,
                "critic_artifact": "frozen-critic",
                "reference_policy_id": "initial-actor",
                "kl_mask_scope": kl_mask_scope,
            }
        },
        "actor_rollout_ref": {"actor": {"strategy": "fsdp2", "use_kl_loss": True}, "rollout": {"name": "vllm"}},
        "trainer": {"v1": {"trainer_mode": "sync"}},
        "critic": {"enable": False},
    }
    assert validate_online_v1_settings(config).kl_mask_scope == kl_mask_scope


@pytest.mark.parametrize("collect_mc,expansion", [(False, False), (True, False), (True, True)])
def test_delta_agent_loop_postprocess_writes_selected_fields_to_tq(monkeypatch, collect_mc, expansion):
    candidates = [[{"prob": 0.6}], [{"prob": 0.4}], [{"prob": 0.7}]]
    output = SimpleNamespace(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22],
        response_mask=[1, 1, 1],
        reward_score=1.0,
        extra_fields={
            "delta_top_logprobs": candidates,
            "global_steps": 0,
            "min_global_steps": 0,
            "max_global_steps": 0,
        },
    )

    def as_dict():
        return {
            "prompts": torch.tensor(output.prompt_ids),
            "responses": torch.tensor(output.response_ids),
            "response_mask": torch.tensor(output.response_mask),
            "rm_scores": torch.tensor([0.0, 0.0, 1.0]),
            "extra_fields": output.extra_fields.copy(),
        }

    output.as_dict = as_dict

    async def noop(*args, **kwargs):
        pass

    worker = SimpleNamespace(
        config=OmegaConf.create(
            {
                "algorithm": {
                    "delta_policy": {
                        "enabled": True,
                        "short_outcome_baseline": 0.25,
                        "selection": {"states_per_response": 2, "min_token_gap": 1},
                    }
                }
            }
        ),
        _compute_score=noop,
        _compute_teacher_logprobs=noop,
        _compute_multi_modal_inputs=lambda output, input_ids: None,
        _compute_position_ids=lambda input_ids, attention_mask, multi_modal_inputs: torch.arange(
            input_ids.shape[-1]
        ).unsqueeze(0),
    )
    if collect_mc:
        worker.config.algorithm.delta_policy.rollout_mode = "selected_prefix_mc"
        worker.config.algorithm.delta_policy.mc = {"concurrency": 2, "continuations_per_state": 2}
        worker.config.algorithm.delta_policy.expansion = {"enabled": expansion, "short_response_tokens": 100}
        worker.config.actor_rollout_ref = {"rollout": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}
        worker.llm_client = object()
        worker.tokenizer = object()

        async def collect(output_arg, selected, **kwargs):
            assert output_arg is output and selected == ([1] if expansion else [0, 1])
            assert kwargs["uid"] == "row"
            labels = []
            if expansion:
                labels = [
                    {
                        "prefix_response_token_ids": [20],
                        "mc_continuations": [
                            {
                                "continuation_index": 0,
                                "token_ids": [30, 31, 32],
                                "reward": 1.0,
                                "delta_top_logprobs": [[{"prob": 0.8}], [{"prob": 0.2}], [{"prob": 0.7}]],
                            },
                            {
                                "continuation_index": 1,
                                "token_ids": [40],
                                "reward": 0.0,
                                "delta_top_logprobs": [[{"prob": 0.5}]],
                            },
                        ],
                    }
                ]
            return {"rollout": {"id": "row", "terminal_reward": 0.0}, "states": [], "labels": labels}

        monkeypatch.setattr("examples.delta_critic.policy_mc.collect_online_mc", collect)
    writes = []

    async def capture_put(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put", capture_put)
    asyncio.run(
        AgentLoopWorkerTQ.__ray_actor_class__._agent_loop_postprocess(
            worker, output, False, uid="row", session_id=0, global_steps=0, delta_expand=expansion
        )
    )

    assert len(writes) == 1
    assert writes[0]["keys"] == (["row_0_0", "row_0_1", "row_0_2"] if expansion else ["row_0_0"])
    fields = writes[0]["fields"]
    assert fields["rollout_actor_version"][0].item() == 0
    assert fields["selected_token_indices"][0].tolist() == [0, 1]
    assert fields["policy_token_mask"][0].tolist() == [1.0, 1.0, 1.0]
    assert fields["delta_top_logprobs"][0] == candidates
    if collect_mc:
        assert fields["delta_mc_record"][0]["rollout"]["id"] == "row"
        assert fields["rm_scores"][0].tolist() == [0.0, 0.0, 0.0]
    else:
        assert fields["rm_scores"][0].tolist() == [0.0, 0.0, 1.0]
    if expansion:
        assert fields["prompts"][1].tolist() == [10, 11, 20]
        assert fields["responses"][1].tolist() == [30, 31, 32]
        assert fields["rm_scores"][1].tolist() == [0.0, 0.0, 1.0]
        assert fields["rm_scores"][2].tolist() == [0.0]
        assert fields["selected_token_indices"][1].tolist() == [1, 2]
        assert fields["selected_token_indices"][2].tolist() == [0]
        assert fields["rollout_actor_version"][1].item() == 0
        assert "delta_group_rewards" not in fields.keys()
        # A continuation measures itself together with its prefix.
        assert fields["delta_true_reward"][0].item() == 0.0
        assert fields["delta_reward_text_length"][0].item() == 3
        assert fields["delta_true_reward"][1].item() == 1.0
        assert fields["delta_reward_text_length"][1].item() == 1 + 3
        assert fields["delta_true_reward"][2].item() == 0.0
        assert fields["delta_reward_text_length"][2].item() == 1 + 1
        assert fields["delta_reward_baseline"].tolist() == pytest.approx([0.25, 0.0, 1.0])
        assert [row.sum().item() for row in fields["rm_scores"].unbind()] == fields["delta_true_reward"].tolist()


@pytest.mark.parametrize("stats_mode", ["per_rollout", "initial_rollout"])
@pytest.mark.parametrize("collect_mc", [False, True])
@pytest.mark.parametrize("update_protocol", ["legacy", "paired_fresh_v1"])
def test_delta_trainer_prepares_full_batch_and_zeros_padding(
    monkeypatch, tmp_path, stats_mode, collect_mc, update_protocol
):
    rows = [
        {
            "prompts": torch.tensor([10]),
            "responses": torch.tensor([20, 21]),
            "response_mask": torch.tensor([1, 1]),
            "policy_token_mask": torch.tensor([1.0, 1.0]),
            "selected_token_indices": torch.tensor([0]),
            "delta_top_logprobs": [[{"prob": 0.6}], [{"prob": 0.8}]],
            "old_log_probs": torch.tensor([-1.0, -1.0]),
            "ref_log_prob": torch.tensor([-1.1, -1.1]),
            "rm_scores": torch.tensor([0.0, 1.0]),
        },
        {
            "prompts": torch.tensor([11]),
            "responses": torch.tensor([30, 31]),
            "response_mask": torch.tensor([1, 1]),
            "policy_token_mask": torch.tensor([1.0, 1.0]),
            "selected_token_indices": torch.tensor([0]),
            "delta_top_logprobs": [[{"prob": 0.7}], [{"prob": 0.9}]],
            "old_log_probs": torch.tensor([-1.2, -1.2]),
            "ref_log_prob": torch.tensor([-1.3, -1.3]),
            "rm_scores": torch.tensor([0.0, 0.0]),
        },
    ]
    if collect_mc:
        for row in rows:
            row["delta_mc_record"] = {
                "rollout": {
                    "prompt_token_ids": row["prompts"].tolist(),
                    "response_token_ids": row["responses"].tolist(),
                    "actor_version": "0",
                },
                "states": [{"token_index": 0}],
                "labels": [],
            }
    data = list_of_dict_to_tensordict([*rows, rows[0]])
    batch = KVBatchMeta(
        keys=["a", "b", "padding"],
        tags=[
            {"min_global_steps": 0, "max_global_steps": 0},
            {"min_global_steps": 0, "max_global_steps": 0},
            {"is_padding": True},
        ],
        partition_id="train",
    )

    class FakeScorer:
        metadata = {"weights_sha256": "frozen-critic"}
        shift = 0.0
        update_step = 0

        def buffer_fit_mc(self, records, policy_step):
            assert policy_step == self.update_step + 1
            return self.fit_mc(records)

        def fit_mc(self, records):
            assert len(written) == self.update_step + 1, "advantages must be written before fitting"
            assert len(records) == 2, "padding must not train the critic"
            self.shift += 100.0
            self.update_step += 1
            return {"delta_critic/update_step": self.update_step}

        def score(self, examples):
            assert [example.rollout.rollout_id for example in examples] == ["a", "b"]
            return [
                {"rollout_id": "a", "delta_pred_raw": [1.0 + self.shift, 0.0], "critic_signal_mask": [1.0, 0.0]},
                {"rollout_id": "b", "delta_pred_raw": [3.0 + self.shift, 0.0], "critic_signal_mask": [1.0, 0.0]},
            ]

    trainer = SimpleNamespace(
        _delta_policy_scorer=FakeScorer(),
        _validate_delta_rollout_versions=lambda batch: None,
        _tq_field_rows=PPOTrainer._tq_field_rows,
        _short_response_reward_deltas=PPOTrainer._short_response_reward_deltas,
        _short_response_reward_baselines=PPOTrainer._short_response_reward_baselines,
        global_steps=1,
        delta_policy_config=DeltaPolicyConfig.online(),
        delta_policy_settings={"reference_policy_id": "initial-actor", "advantage_stats_mode": stats_mode},
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path))),
    )
    if collect_mc:
        trainer.delta_policy_settings["rollout_mode"] = "selected_prefix_mc"
        if update_protocol == "paired_fresh_v1":
            trainer.delta_policy_settings["critic_update"] = {"enabled": True, "protocol": update_protocol}
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", lambda **kwargs: data)
    written = []

    def capture_put(**kwargs):
        written.append(kwargs["fields"])
        return batch

    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put", capture_put)
    metrics = {}
    PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)

    assert len(written) == 1
    fields = written[0]
    assert fields["row_weight"].tolist() == [1.0, 1.0, 0.0]
    assert fields["sample_valid_mask"].tolist() == [1.0, 1.0, 0.0]
    assert fields["advantages"][2].tolist() == [0.0, 0.0]
    assert metrics["delta_policy/advantage_stats_count"] == 4
    assert batch.extra_info["delta_policy_config"]["behavior_logprob_source"] == "actor_snapshot"
    assert metrics["delta_policy/advantage_stats_mean"] == 2.0
    assert metrics["delta_policy/advantage_stats_std"] == 1.0
    assert metrics["delta_policy/health/all/rows"] == 2
    assert metrics["delta_policy/health/all/raw_delta_count"] == 2
    assert metrics["delta_policy/health/all/raw_delta_mean"] == 2.0
    assert metrics["delta_policy/health/all/advantage_count"] == 4
    assert metrics["delta_policy/health/positive_reward/row_advantage_mean"] == -1.0
    assert metrics["delta_policy/health/zero_reward/row_advantage_mean"] == 1.0
    assert fields["advantages"][0].tolist() == [-1.0, -1.0]
    if collect_mc:
        assert trainer._delta_policy_scorer.shift == 100.0
        assert metrics["delta_critic/prediction_update_step"] == 0
        assert metrics["delta_critic/update_step"] == 1
        for record in data["delta_mc_record"]:
            record["rollout"]["actor_version"] = "1"
    trainer._delta_policy_scorer.shift = 4.0
    trainer.global_steps = 2
    # Simulate a new trainer restoring the fixed statistics from disk.
    trainer._delta_policy_stats = None
    PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)
    assert metrics["delta_policy/advantage_stats_mean"] == (2.0 if stats_mode == "initial_rollout" else 6.0)
    assert written[-1]["advantages"][0].tolist() == ([3.0, 3.0] if stats_mode == "initial_rollout" else [-1.0, -1.0])
    assert written[-1]["advantages"][2].tolist() == [0.0, 0.0]
    if stats_mode == "initial_rollout":
        trainer.delta_policy_settings["reference_policy_id"] = "changed-actor"
        with pytest.raises(ValueError, match="different policy/critic"):
            PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)
        trainer.delta_policy_settings["reference_policy_id"] = "initial-actor"
        (tmp_path / "delta_advantage_stats.json").unlink()
        trainer._delta_policy_stats = None
        with pytest.raises(ValueError, match="refusing to recalibrate"):
            PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)


def test_shared_original_calibration_preserves_continuation_actor_loss(monkeypatch, tmp_path):
    keys = ["o0", "o1", "c0", "c1"]
    values = [1.0, 3.0, 100.0, 200.0]
    rows = [
        {
            "prompts": torch.tensor([10]),
            "responses": torch.tensor([20, 21]),
            "response_mask": torch.ones(2),
            "policy_token_mask": torch.ones(2),
            "selected_token_indices": torch.tensor([0]),
            "delta_top_logprobs": [[{"prob": 0.6}], [{"prob": 0.8}]],
            "old_log_probs": torch.full((2,), -1.0),
            "ref_log_prob": torch.full((2,), -1.1),
            "rm_scores": torch.zeros(2),
            "delta_is_continuation": index >= 2,
        }
        for index in range(4)
    ]
    data = list_of_dict_to_tensordict(rows)
    batch = KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id="train")

    class Scorer:
        metadata = {"weights_sha256": "same-initial-critic"}

        def score(self, examples):
            return [
                {"rollout_id": key, "delta_pred_raw": [value, 0.0], "critic_signal_mask": [1.0, 0.0]}
                for key, value in zip(keys, values, strict=True)
            ]

    shared = tmp_path / "common" / "calibration.json"
    trainer = SimpleNamespace(
        _delta_policy_scorer=Scorer(),
        _validate_delta_rollout_versions=lambda batch: None,
        _tq_field_rows=PPOTrainer._tq_field_rows,
        _short_response_reward_deltas=PPOTrainer._short_response_reward_deltas,
        _short_response_reward_baselines=PPOTrainer._short_response_reward_baselines,
        global_steps=1,
        delta_policy_config=DeltaPolicyConfig.online(advantage_clip=5.0),
        delta_policy_settings={
            "reference_policy_id": "same-base",
            "advantage_stats_mode": "initial_rollout",
            "advantage_stats_fit_population": "originals",
            "advantage_stats_path": str(shared),
            "mc": {"continuations_per_state": 1},
            "expansion": {"enabled": True, "prompts_per_step": 2, "states_per_step": 2},
        },
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path / "writer"))),
    )
    written = []
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", lambda **kwargs: data)
    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put",
        lambda **kwargs: written.append(kwargs["fields"]) or batch,
    )
    metrics = {}
    PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)
    assert metrics["delta_policy/advantage_stats_count"] == 4
    assert metrics["delta_policy/advantage_stats_mean"] == 2.0
    assert metrics["delta_policy/advantage_stats_std"] == 1.0
    assert written[-1]["row_weight"].tolist() == [1.0] * 4
    assert written[-1]["advantages"][2].tolist() == [5.0, 5.0]
    assert written[-1]["policy_loss_mask"][2].tolist() == [1.0, 1.0]
    assert shared.exists()
    trainer._delta_policy_stats = None
    trainer.delta_policy_settings["advantage_stats_read_only"] = True
    trainer.config.trainer.default_local_dir = str(tmp_path / "reader")
    values[:] = [4.0, 6.0, -100.0, -200.0]
    PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)
    assert metrics["delta_policy/advantage_stats_mean"] == 2.0
    assert written[-1]["advantages"][0].tolist() == [2.0, 2.0]
    assert (tmp_path / "reader" / "delta_advantage_stats.json").exists()
    shared.unlink()
    trainer._delta_policy_stats = None
    with pytest.raises(ValueError, match="Shared original calibration is missing"):
        PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)


def test_short_response_baseline_rows_must_be_binary_calibration_values():
    batch = KVBatchMeta(keys=["short"], tags=[{}], partition_id="train")
    data = {
        "delta_reward_text_length": torch.tensor([1]),
        "delta_reward_baseline": torch.tensor([-0.01]),
    }
    with pytest.raises(ValueError, match=r"baseline must be finite and in \[0, 1\]"):
        PPOTrainer._short_response_reward_baselines(batch, data, short_response_budget=100, fallback_baseline=0.5)


@pytest.mark.parametrize("stats_mode", ["per_rollout", "initial_rollout"])
@pytest.mark.parametrize("continuation_actor_loss", [False, True])
@pytest.mark.parametrize("kl_mask_scope", ["policy", "response"])
def test_expansion_uses_delta_for_every_row_independent_of_true_rewards(
    monkeypatch, tmp_path, stats_mode, continuation_actor_loss, kl_mask_scope
):
    from examples.delta_critic.policy_loss import per_sample_policy_losses

    keys = ["a", "b", "c", "d"]
    responses = [[20, 21], [30], [40, 41, 42], [50]]
    selected = [[0], [0], [1], [0]]
    rows = []
    for index, tokens in enumerate(responses):
        length = len(tokens)
        rows.append(
            {
                "prompts": torch.tensor([10, 11] if index < 2 else [10, 11, 20]),
                "responses": torch.tensor(tokens),
                "response_mask": torch.ones(length),
                "policy_token_mask": torch.ones(length),
                "selected_token_indices": torch.tensor(selected[index]),
                "delta_top_logprobs": [[{"prob": 0.6}]] * length,
                "old_log_probs": torch.full((length,), -1.0),
                "ref_log_prob": torch.full((length,), -1.1),
                "rm_scores": torch.zeros(length),
                "delta_is_continuation": index >= 2,
                "delta_mc_record": {},
            }
        )
    # Divergence before the first selected token must be visible in full-response
    # diagnostics and regularized only when the response KL scope is requested.
    rows[2]["ref_log_prob"][0] = -3.0
    for index in range(2):
        rows[index]["delta_mc_record"] = {
            "rollout": {
                "prompt_token_ids": rows[index]["prompts"].tolist(),
                "response_token_ids": responses[index],
                "actor_version": "0",
            },
            "states": [{"token_index": 0}] if index == 0 else [],
            "labels": (
                [{"token_index": 0, "v_prefix": 0.0, "mc_continuations": [{"reward": 0.0}, {"reward": 0.0}]}]
                if index == 0
                else []
            ),
        }
    batch = KVBatchMeta(
        keys=[*keys, "padding"],
        tags=[{} for _ in keys] + [{"is_padding": True}],
        partition_id="train",
    )

    class Scorer:
        metadata = {"weights_sha256": "frozen"}

        def score(self, examples):
            assert [e.rollout.rollout_id for e in examples] == keys
            assert examples[2].rollout.prompt_token_ids == (10, 11, 20)
            return [
                {"rollout_id": "a", "delta_pred_raw": [1.0, 0.0], "critic_signal_mask": [1.0, 0.0]},
                {"rollout_id": "b", "delta_pred_raw": [3.0], "critic_signal_mask": [1.0]},
                {"rollout_id": "c", "delta_pred_raw": [0.0, 5.0, 0.0], "critic_signal_mask": [0.0, 1.0, 0.0]},
                {"rollout_id": "d", "delta_pred_raw": [7.0], "critic_signal_mask": [1.0]},
            ]

        def fit_mc(self, records):
            raise AssertionError("Expansion must keep its critic frozen")

    trainer = SimpleNamespace(
        _delta_policy_scorer=Scorer(),
        _validate_delta_rollout_versions=lambda batch: None,
        _tq_field_rows=PPOTrainer._tq_field_rows,
        _short_response_reward_deltas=PPOTrainer._short_response_reward_deltas,
        _short_response_reward_baselines=PPOTrainer._short_response_reward_baselines,
        global_steps=1,
        delta_policy_config=DeltaPolicyConfig.online(kl_mask_scope=kl_mask_scope),
        delta_policy_settings={
            "reference_policy_id": "initial",
            "advantage_stats_mode": stats_mode,
            "rollout_mode": "selected_prefix_mc",
            "mc": {"continuations_per_state": 2},
            "expansion": {
                "enabled": True,
                "prompts_per_step": 2,
                "states_per_step": 1,
                "continuation_actor_loss": continuation_actor_loss,
            },
        },
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path))),
    )
    written, exported = [], []
    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get",
        lambda **kwargs: list_of_dict_to_tensordict([*rows, rows[2]]),
    )

    def put(**kwargs):
        written.append(kwargs["fields"])
        return batch

    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put", put)
    monkeypatch.setattr(
        "examples.delta_critic.policy_mc.export_mc_attempt", lambda records, directory: exported.append(records)
    )
    gradients = []
    for rewards in ([0.0, 0.0], [0.0, 1.0], [1.0, 1.0]):
        for index, reward in enumerate(rewards):
            rows[index]["rm_scores"][-1] = reward
            rows[index + 2]["rm_scores"][-1] = reward
            rows[0]["delta_mc_record"]["labels"][0]["mc_continuations"][index]["reward"] = reward
        rows[0]["delta_mc_record"]["labels"][0]["v_prefix"] = sum(rewards) / len(rewards)
        metrics = {}
        PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)
        fields = written[-1]
        assert len(exported[-1]) == 2
        assert fields["row_weight"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
        assert fields["critic_signal_mask"][2].tolist() == [0.0, 1.0, 0.0]
        assert fields["policy_loss_mask"][2].tolist() == ([0.0, 1.0, 1.0] if continuation_actor_loss else [0.0] * 3)
        expected_kl_mask = [1.0, 1.0, 1.0] if kl_mask_scope == "response" else [0.0, 1.0, 1.0]
        assert fields["kl_mask"][2].tolist() == (expected_kl_mask if continuation_actor_loss else [0.0] * 3)
        assert fields["kl_mask"][4].tolist() == [0.0] * 3
        assert fields["advantages"][4].tolist() == [0.0] * 3
        assert metrics["delta_policy/policy_token_coverage"] == pytest.approx(6 / 7 if continuation_actor_loss else 1.0)
        outside_kl = metrics["delta_policy/preupdate_ref_kl_outside_policy_row_mean"]
        assert outside_kl == pytest.approx(
            torch.exp(torch.tensor(-2.0)).item() + 1.0 if continuation_actor_loss else 0.0
        )
        if continuation_actor_loss:
            assert (
                metrics["delta_policy/preupdate_ref_kl_response_row_mean"]
                > metrics["delta_policy/preupdate_ref_kl_policy_row_mean"]
            )
        expected_zero_rows = rewards.count(0.0) * (2 if continuation_actor_loss else 1)
        assert metrics["delta_policy/zero_reward_rows"] == expected_zero_rows
        if rewards == [0.0, 0.0]:
            assert metrics["delta_policy/zero_reward_positive_row_advantage_fraction"] == 0.5
        assert metrics["delta_policy/continuation_rows"] == 2
        assert metrics["delta_policy/continuation_advantage_from_critic"] == 1.0
        population = torch.tensor([1.0, 1.0, 3.0, 5.0, 5.0, 7.0] if continuation_actor_loss else [1.0, 1.0, 3.0])
        assert metrics["delta_policy/advantage_stats_count"] == len(population)
        expected = (5.0 - population.mean().item()) / population.std(correction=0).item()
        assert fields["advantages"][2].tolist() == pytest.approx([0.0, expected, expected])
        dense = {name: fields[name].to_padded_tensor(0.0) for name in ("advantages", "policy_loss_mask", "kl_mask")}
        current = torch.full_like(dense["advantages"], -1.0, requires_grad=True)
        dense.update(
            old_log_probs=current.detach(),
            ref_log_prob=current.detach() - 0.1,
            row_weight=fields["row_weight"],
            sample_valid_mask=fields["sample_valid_mask"],
        )
        losses = per_sample_policy_losses(current, dense, trainer.delta_policy_config)
        (losses["loss"] * losses["row_weight"]).sum().backward()
        gradients.append(current.grad.clone())
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[2], rtol=0, atol=0)
    if continuation_actor_loss:
        assert gradients[0][2, 1:].abs().sum() > 0
        assert bool(gradients[0][2, 0].abs() > 0) == (kl_mask_scope == "response")
    else:
        assert gradients[0][2:].abs().sum() == 0
    if stats_mode == "initial_rollout":
        stats_path = tmp_path / "delta_advantage_stats.json"
        legacy_stats = json.loads(stats_path.read_text())
        legacy_stats.pop("advantage_contract")
        stats_path.write_text(json.dumps(legacy_stats))
        with pytest.raises(ValueError, match="different policy/critic contract"):
            PPOTrainer._compute_delta_policy_advantage(trainer, batch, {})


def test_expansion_calibrates_short_outcomes_separately_then_clips_every_row(monkeypatch, tmp_path):
    """Short outcomes use their sibling baseline and stay outside critic statistics."""
    keys = ["a", "b", "c", "d"]
    responses = [[20, 21], [30], [40, 41, 42], [50]]
    selected = [[0], [0], [1], [0]]
    # (reward, length the budget compares against). Only a and c sit inside it.
    short = [(1.0, 2), (1.0, 150), (0.0, 40), (1.0, 200)]
    rows = []
    for index, tokens in enumerate(responses):
        length = len(tokens)
        rows.append(
            {
                "prompts": torch.tensor([10, 11] if index < 2 else [10, 11, 20]),
                "responses": torch.tensor(tokens),
                "response_mask": torch.ones(length),
                "policy_token_mask": torch.ones(length),
                "selected_token_indices": torch.tensor(selected[index]),
                "delta_top_logprobs": [[{"prob": 0.6}]] * length,
                "old_log_probs": torch.full((length,), -1.0),
                "ref_log_prob": torch.full((length,), -1.1),
                "rm_scores": torch.zeros(length),
                "delta_is_continuation": index >= 2,
                "delta_mc_record": {},
                "delta_true_reward": torch.tensor(short[index][0]),
                "delta_reward_text_length": torch.tensor(short[index][1]),
                # c and d are leave-one-out siblings: c's baseline comes only
                # from d, even though c's own observed reward is zero.
                "delta_reward_baseline": torch.tensor([0.25, 0.5, 1.0, 0.0][index]),
            }
        )
    for index in range(2):
        rows[index]["delta_mc_record"] = {
            "rollout": {
                "prompt_token_ids": rows[index]["prompts"].tolist(),
                "response_token_ids": responses[index],
                "actor_version": "0",
            },
            "states": [{"token_index": 0}] if index == 0 else [],
            "labels": (
                [{"token_index": 0, "v_prefix": 0.0, "mc_continuations": [{"reward": 0.0}, {"reward": 0.0}]}]
                if index == 0
                else []
            ),
        }
    batch = KVBatchMeta(
        keys=[*keys, "padding"],
        tags=[{} for _ in keys] + [{"is_padding": True}],
        partition_id="train",
    )
    # Predictions only reach the two rows outside the budget.
    predictions = {"a": [1.0, 0.0], "b": [3.0], "c": [0.0, 5.0, 0.0], "d": [5.0]}

    class Scorer:
        metadata = {"weights_sha256": "frozen"}

        def score(self, examples):
            assert [example.rollout.rollout_id for example in examples] == keys
            return [
                {
                    "rollout_id": key,
                    "delta_pred_raw": predictions[key],
                    "critic_signal_mask": [float(bool(value)) for value in predictions[key]],
                }
                for key in keys
            ]

        def fit_mc(self, records):
            raise AssertionError("Expansion must keep its critic frozen")

    trainer = SimpleNamespace(
        _delta_policy_scorer=Scorer(),
        _validate_delta_rollout_versions=lambda batch: None,
        _tq_field_rows=PPOTrainer._tq_field_rows,
        _short_response_reward_deltas=PPOTrainer._short_response_reward_deltas,
        _short_response_reward_baselines=PPOTrainer._short_response_reward_baselines,
        global_steps=1,
        delta_policy_config=DeltaPolicyConfig.online(
            advantage_clip=1.0, short_outcome_baseline=0.25, short_outcome_scale=0.25
        ),
        delta_policy_settings={
            "reference_policy_id": "initial",
            "advantage_stats_mode": "initial_rollout",
            "rollout_mode": "selected_prefix_mc",
            "mc": {"continuations_per_state": 2},
            "expansion": {
                "enabled": True,
                "prompts_per_step": 2,
                "states_per_step": 1,
                "short_response_tokens": 100,
            },
        },
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path))),
    )
    written = []
    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get",
        lambda **kwargs: list_of_dict_to_tensordict([*rows, rows[2]]),
    )
    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put",
        lambda **kwargs: written.append(kwargs["fields"]) or batch,
    )
    monkeypatch.setattr("examples.delta_critic.policy_mc.export_mc_attempt", lambda records, directory: None)
    metrics = {}
    PPOTrainer._compute_delta_policy_advantage(trainer, batch, metrics)
    fields = written[-1]
    # Row a carries the reward over its whole response, row c the other reward.
    assert metrics["delta_policy/short_response_rows"] == 2
    assert metrics["delta_policy/short_response_positive_rows"] == 1
    assert metrics["delta_policy/short_response_mean_reward"] == pytest.approx(0.5)
    assert metrics["delta_policy/short_response_mean_baseline"] == pytest.approx(0.625)
    assert metrics["delta_policy/short_response_mean_advantage"] == pytest.approx(-0.5)
    # Historical continuation rm_scores could be zero placeholders. Diagnose
    # rewards from delta_true_reward whenever that authoritative field exists.
    assert metrics["delta_policy/positive_reward_rows"] == 3
    assert metrics["delta_policy/zero_reward_rows"] == 1
    assert metrics["delta_policy/positive_reward_row_advantage_mean"] == pytest.approx(1 / 3)
    assert metrics["delta_policy/zero_reward_row_advantage_mean"] == -1.0
    assert metrics["delta_policy/advantage_clip"] == pytest.approx(1.0)
    # Both complete short responses exceed the clip bound before clipping.
    assert metrics["delta_policy/clipped_advantage_tokens"] == 5
    # Critic population contains only ordinary rows b and d. Short outcomes use
    # calibrated advantages in their own units and cannot move its mean or std.
    population = torch.tensor([3.0, 5.0])
    mean, std = population.mean().item(), population.std(correction=0).item()
    assert metrics["delta_policy/advantage_stats_count"] == 2
    assert metrics["delta_policy/advantage_stats_mean"] == pytest.approx(mean)
    assert metrics["delta_policy/advantage_stats_std"] == pytest.approx(std)
    assert fields["advantages"][0].tolist() == [1.0, 1.0]
    assert fields["advantages"][2].tolist() == [-1.0, -1.0, -1.0]
    # Rows b and d retain their critic deltas and use critic-only statistics.
    assert fields["advantages"][1].tolist() == pytest.approx([(3.0 - mean) / std])
    assert fields["advantages"][3].tolist() == [1.0]
    assert fields["state_mask"][0].tolist() == [1.0, 1.0]
    assert fields["policy_loss_mask"][2].tolist() == [1.0, 1.0, 1.0]
    assert fields["row_weight"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert fields["advantages"][4].tolist() == [0.0] * 3
    stats_path = tmp_path / "delta_advantage_stats.json"
    assert "short_outcome_advantage_v2_le100_baseline0.25_scale0.25_loo_siblings_or_fixed" in stats_path.read_text()


def test_delta_policy_config_survives_tq_and_reaches_actor_mini_batches(monkeypatch):
    class FakeTQClient:
        async def async_kv_retrieve_meta(self, keys, partition_id, create):
            assert create is False
            return BatchMeta(global_indexes=list(range(len(keys))), partition_ids=[partition_id] * len(keys))

        async def async_get_data(self, meta):
            return TensorDict(
                {
                    "row_weight": torch.tensor([1.0, 0.0, 1.0, 1.0]),
                    "sample_valid_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                },
                batch_size=[4],
            )

    config = {"mode": "online_ppo", "behavior_logprob_source": "actor_snapshot"}
    meta = KVBatchMeta(
        keys=["a", "b", "c", "d"],
        tags=[{}] * 4,
        partition_id="train",
        extra_info={
            "delta_policy_config": config,
            "mini_batch_size": 2,
            "epochs": 1,
            "seed": 0,
            "dataloader_kwargs": {"shuffle": False},
        },
    )
    fake_client = FakeTQClient()
    monkeypatch.setattr(tq_utils, "TQ_INITIALIZED", True)
    monkeypatch.setattr(tq_utils.tq, "get_client", lambda: fake_client)

    data = tq_utils._meta_to_realdata(meta)
    assert "delta_policy_config" in data.keys()
    assert data["delta_policy_config"] == config

    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        is_mp_src_rank_with_outputs=lambda: False,
    )
    worker = SimpleNamespace(engine=engine, profiler=MagicMock())
    received = []

    def capture_mini_batch(mini_batch):
        received.append(mini_batch)
        return None

    worker.train_batch = capture_mini_batch
    assert TrainingWorker.train_mini_batch(worker, data) is None

    assert len(received) == 2
    assert all("delta_policy_config" in mini_batch.keys() for mini_batch in received)
    assert all(mini_batch["delta_policy_config"] == config for mini_batch in received)
    # The engine worker attaches a denominator for each logical minibatch before dispatch.
    assert [mini_batch["global_valid_sample_weight"] for mini_batch in received] == [1.0, 2.0]


def test_actor_logprob_response_slice_matches_transferqueue_response_mask():
    full_sequence_logprobs = [torch.arange(6, dtype=torch.float32), torch.arange(5, dtype=torch.float32) + 10]
    response_masks = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0])]
    full_nested = torch.nested.as_nested_tensor(full_sequence_logprobs, layout=torch.jagged)
    mask_nested = torch.nested.as_nested_tensor(response_masks, layout=torch.jagged)

    response_logprobs = response_from_nested(full_nested, mask_nested)

    assert response_logprobs.offsets().diff().tolist() == mask_nested.offsets().diff().tolist() == [3, 2]
    torch.testing.assert_close(response_logprobs[0], full_sequence_logprobs[0][-4:-1])
    torch.testing.assert_close(response_logprobs[1], full_sequence_logprobs[1][-3:-1])


def test_v1_tokenizer_fingerprint_uses_checkpoint_artifact_contract():
    class FakeTokenizer:
        special_tokens_map = {"pad_token": "<pad>", "eos_token": "<eos>"}

        def get_vocab(self):
            return {"<pad>": 0, "<eos>": 1, "hello": 2}

    tokenizer = FakeTokenizer()
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        artifact_fingerprint, loaded_tokenizer = tokenizer_fingerprint("unused-local-tokenizer")

    assert loaded_tokenizer is tokenizer
    assert tokenizer_fingerprint_from_tokenizer(tokenizer) == artifact_fingerprint
