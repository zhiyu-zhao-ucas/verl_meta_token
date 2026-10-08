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

import math

import pytest
import torch
from tensordict import TensorDict

from examples.delta_critic.policy_config import DeltaPolicyConfig
from examples.delta_critic.policy_diagnostics import policy_batch_diagnostics
from examples.delta_critic.policy_loss import delta_policy_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric.utils import Metric, reduce_metrics


def _row(**updates):
    row = {
        "response_mask": [1, 1, 1],
        "policy_loss_mask": [1, 1, 0],
        "advantages": [-1.0, 2.0, 99.0],
        "state_mask": [1, 1, 1],
        "raw_selected_deltas": [-0.2, 0.4],
        "is_continuation": False,
        "is_outcome": False,
        "actor_enabled": True,
        "reward": 1.0,
        "advantage_clip_count": 1,
    }
    return dict(row, **updates)


def test_batch_diagnostics_distinguish_states_tokens_sources_and_disabled_rows():
    metrics = policy_batch_diagnostics(
        [
            _row(),
            _row(is_continuation=True, is_outcome=True, reward=0.0, advantages=[0.0, 0.0, 99.0]),
            _row(is_continuation=True, actor_enabled=False, policy_loss_mask=[0, 0, 0]),
        ]
    )
    all_prefix = "delta_policy/health/all"
    assert metrics[f"{all_prefix}/rows"] == 3
    assert metrics[f"{all_prefix}/active_pg_rows"] == 2
    assert metrics[f"{all_prefix}/zero_advantage_rows"] == 1
    assert metrics[f"{all_prefix}/response_tokens"] == 9
    assert metrics[f"{all_prefix}/policy_tokens"] == 4
    assert metrics[f"{all_prefix}/advantage_count"] == 4
    assert metrics[f"{all_prefix}/advantage_mean"] == 0.25
    assert metrics[f"{all_prefix}/advantage_zero_fraction"] == 0.5
    assert metrics[f"{all_prefix}/advantage_negative_fraction"] == 0.25
    assert metrics[f"{all_prefix}/raw_delta_count"] == 2
    assert metrics[f"{all_prefix}/raw_delta_mean"] == pytest.approx(0.1)
    assert metrics[f"{all_prefix}/state_advantage_clip_fraction"] == pytest.approx(2 / 6)
    assert metrics["delta_policy/health/short_outcome/raw_delta_count"] == 0
    assert metrics["delta_policy/health/zero_reward/advantage_zero_fraction"] == 1
    assert metrics["delta_policy/health/continuation/actor_enabled_rows"] == 1


def test_empty_batch_scopes_are_finite_and_have_zero_denominators():
    metrics = policy_batch_diagnostics([])
    assert metrics
    assert all(value == 0.0 for value in metrics.values())


def _actor_data(config, start=0, stop=3, *, dp_size=1):
    data = TensorDict(
        {
            "old_log_probs": torch.zeros((3, 3)),
            "advantages": torch.tensor([[1.0, -1.0, 99.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0]]),
            "policy_loss_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
            "row_weight": torch.tensor([2.0, 1.0, 1.0]),
            "sample_valid_mask": torch.tensor([1.0, 0.0, 1.0]),
        },
        batch_size=[3],
    )[start:stop]
    tu.assign_non_tensor(data, delta_policy_config=config.as_dict(), dp_size=dp_size, global_valid_sample_weight=3.0)
    return data


def test_actor_diagnostics_use_pg_mask_padding_and_global_row_denominator():
    config = DeltaPolicyConfig.online(advantage_normalization="none", kl_coef=0.0)
    current = torch.tensor(
        [[math.log(1.5), math.log(0.5), 15.0], [15.0, -15.0, 15.0], [5.0, 5.0, 5.0]], requires_grad=True
    )
    loss, metrics = delta_policy_loss({"log_probs": current}, _actor_data(config))
    prefix = "actor/delta_forward_"
    # Only the first row supplies policy tokens; its row weight is 2 out of 3.
    assert metrics[prefix + "active_pg_row_fraction"].aggregate() == pytest.approx(2 / 3)
    assert metrics[prefix + "ratio_row_mean"].aggregate() == pytest.approx(2 / 3)
    assert metrics[prefix + "ratio_abs_deviation_row_mean"].aggregate() == pytest.approx(1 / 3)
    assert metrics[prefix + "clip_fraction_row_mean"].aggregate() == pytest.approx(2 / 3)
    assert metrics[prefix + "ratio_outside_clip_fraction_row_mean"].aggregate() == pytest.approx(2 / 3)
    expected_kl = ((1 / 1.5 - 1 + math.log(1.5)) + (2 - 1 + math.log(0.5))) / 3
    assert metrics[prefix + "old_kl_row_mean"].aggregate() == pytest.approx(expected_kl)
    loss.backward()
    assert torch.equal(current.grad[1:], torch.zeros_like(current.grad[1:]))
    assert current.grad[0, 2].item() == 0.0


def test_actor_metrics_are_invariant_to_microbatch_and_dp_partition():
    config = DeltaPolicyConfig.online(advantage_normalization="none", kl_coef=0.0)
    current = torch.tensor([[0.3, -0.4, 15.0], [15.0, -15.0, 15.0], [5.0, 5.0, 5.0]])
    _, full = delta_policy_loss({"log_probs": current}, _actor_data(config))
    split = [delta_policy_loss({"log_probs": current[i : i + 1]}, _actor_data(config, i, i + 1))[1] for i in range(3)]
    ranks = [
        delta_policy_loss({"log_probs": current[:1]}, _actor_data(config, 0, 1, dp_size=2))[1],
        delta_policy_loss({"log_probs": current[1:]}, _actor_data(config, 1, 3, dp_size=2))[1],
    ]
    for key in full:
        assert sum(part[key].aggregate() for part in split) == pytest.approx(full[key].aggregate())
        assert Metric.aggregate_dp([part[key] for part in ranks]) == pytest.approx(full[key].aggregate())


def test_first_minibatch_ratio_and_sign_sensitive_clipping_are_observations():
    config = DeltaPolicyConfig.online(advantage_normalization="none", kl_coef=0.0)
    _, initial = delta_policy_loss({"log_probs": torch.zeros((3, 3))}, _actor_data(config))
    prefix = "actor/delta_forward_"
    assert initial[prefix + "old_kl_row_mean"].aggregate() == 0.0
    assert initial[prefix + "ratio_abs_deviation_row_mean"].aggregate() == 0.0
    current = torch.tensor([[math.log(0.5), math.log(1.5), 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    _, opposite = delta_policy_loss({"log_probs": current}, _actor_data(config))
    assert opposite[prefix + "clip_fraction_row_mean"].aggregate() == 0.0
    assert opposite[prefix + "ratio_outside_clip_fraction_row_mean"].aggregate() == pytest.approx(2 / 3)


def test_forward_metrics_average_multiple_minibatches_after_worker_reduction():
    key = "actor/delta_forward_clip_fraction_row_mean"
    assert reduce_metrics({key: [0.0, 0.8]})[key] == pytest.approx(0.4)
