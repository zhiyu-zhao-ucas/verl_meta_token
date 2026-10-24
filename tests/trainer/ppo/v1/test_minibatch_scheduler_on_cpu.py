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
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def _trainer(estimator, per_minibatch):
    actor = {
        "ppo_mini_batch_size": 32,
        "ppo_epochs": 1,
        "data_loader_seed": 42,
        "shuffle": False,
        "calculate_entropy": False,
        "entropy_coeff": 0,
        "optim": {"step_lr_scheduler_per_minibatch": per_minibatch},
    }
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": actor, "rollout": {"n": 4, "temperature": 1}},
            "critic": {**actor, "ppo_mini_batch_size": 64, "ppo_epochs": 2},
            "algorithm": {"adv_estimator": estimator},
            "data": {
                "train_files": [],
                "val_files": [],
                "train_batch_size": 256,
                "val_batch_size": 64,
                "dataloader_num_workers": 0,
            },
            "trainer": {"total_epochs": 1, "total_training_steps": 20, "v1": {"sampler": {}}},
        }
    )
    trainer = SimpleNamespace(
        config=config,
        trainer_mode="sync",
        parameter_sync_step=1,
        tokenizer=None,
        processor=None,
        delta_policy_config=None,
        actor_rollout_wg=MagicMock(),
        critic_wg=MagicMock(),
    )
    trainer.actor_rollout_wg.update_actor.return_value = {"metrics": {"mfu": 0.0}}
    trainer.critic_wg.train_mini_batch.return_value.get.return_value = {"metrics": {"mfu": 0.0}}
    return trainer


@pytest.mark.parametrize(
    "estimator,per_minibatch,actor_expected",
    [
        ("gae", False, False),
        ("gae", True, True),
        ("grpo", True, True),
        ("vineppo", False, True),
    ],
)
def test_scheduler_flag_reaches_workers(estimator, per_minibatch, actor_expected):
    trainer = _trainer(estimator, per_minibatch)
    actor_batch = SimpleNamespace(extra_info={})
    critic_batch = SimpleNamespace(extra_info={})
    PPOTrainer._update_actor(trainer, actor_batch, {})
    PPOTrainer._update_critic(trainer, critic_batch, {})
    assert actor_batch.extra_info["scheduler_per_minibatch"] is actor_expected
    assert critic_batch.extra_info["scheduler_per_minibatch"] is per_minibatch
    assert actor_batch.extra_info["mini_batch_size"] == 128


@pytest.mark.parametrize(
    "estimator,per_minibatch,actor_steps,critic_steps",
    [
        ("gae", False, 20, 20),
        ("gae", True, 160, 160),
        ("grpo", True, 160, 160),
        ("vineppo", False, 160, 20),
    ],
)
def test_schedule_horizon_counts_optimizer_updates(estimator, per_minibatch, actor_steps, critic_steps):
    trainer = _trainer(estimator, per_minibatch)
    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}] * 512, [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader"),
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        PPOTrainer._init_dataloader(trainer)
    warning.assert_not_called()
    assert trainer.config.actor_rollout_ref.actor.optim.total_training_steps == actor_steps
    assert trainer.config.critic.optim.total_training_steps == critic_steps
