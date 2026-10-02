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

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from examples.delta_critic.contracts import DeltaConfig, DeltaExample, Rollout, SelectedState
from examples.delta_critic.policy_config import DeltaPolicyConfig, delta_policy_config_from_mapping
from examples.delta_critic.policy_data import (
    policy_response_batch,
    policy_training_batch,
    prepare_policy_rows,
)
from examples.delta_critic.policy_loss import delta_policy_loss, per_sample_policy_losses
from examples.delta_critic.target_ops import state_advantages


def _source_module():
    root = Path(os.environ.get("VALUE_MODEL_ROOT", "/scratch2/zhiyu/code/value_model"))
    source = root / "delta_value_llm_exp" / "offline_grpo.py"
    if not source.is_file():
        pytest.fail(f"value_model parity source not found: {source}; set VALUE_MODEL_ROOT")
    spec = importlib.util.spec_from_file_location("value_model_offline_grpo_source", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _example(rollout_id, prompt, response, states, policy_mask=None, stored_logprobs=None):
    rollout = Rollout(
        rollout_id=rollout_id,
        prompt_token_ids=tuple(prompt),
        response_token_ids=tuple(response),
        terminal_reward=0.0,
        policy_token_mask=tuple(policy_mask or [1.0] * len(response)),
        prompt_id=f"prompt-{rollout_id}",
        metadata={"behavior_logprobs": stored_logprobs or [-0.1] * len(response)},
    )
    selected = tuple(SelectedState(rollout_id, token_index, v_prefix=0.0, delta=delta) for token_index, delta in states)
    return DeltaExample(rollout, selected)


def test_config_defaults_keep_historical_and_online_kl_contracts_separate():
    offline = DeltaPolicyConfig.historical_offline()
    online = DeltaPolicyConfig.online()
    assert (offline.clip_range, offline.kl_coef, offline.kl_reference) == (0.2, 0.01, "behavior")
    assert offline.kl_estimator == "source_sampled_reverse"
    assert offline.kl_mask_scope == "policy"
    assert (online.kl_coef, online.kl_reference, online.kl_estimator) == (0.001, "reference", "low_var_kl")
    assert online.kl_mask_scope == "policy"
    assert online.behavior_logprob_source == "actor_snapshot"
    assert delta_policy_config_from_mapping({"delta_policy": {"clip_range": 0.1}}, mode="online_ppo").clip_range == 0.1
    with pytest.raises(ValueError, match="fixed reference policy"):
        DeltaPolicyConfig.online(kl_reference="behavior")
    with pytest.raises(ValueError, match="unclamped source"):
        DeltaPolicyConfig.historical_offline(kl_estimator="low_var_kl")
    with pytest.raises(ValueError, match="max_length"):
        DeltaPolicyConfig.historical_offline(max_length=2.5)


def test_precomputed_behavior_logprob_requires_every_rollout_id():
    example = _example("r1", [1], [2, 3], [(0, 0.5)])
    config = DeltaPolicyConfig.historical_offline(behavior_logprob_source="precomputed", advantage_normalization="none")
    with pytest.raises(ValueError, match="Missing precomputed behavior logprobs.*r1"):
        prepare_policy_rows([example], {"r1": {0: 0.5}}, {}, config)


def test_online_rejects_rollout_logprobs_with_sampling_distortion():
    example = _example("r1", [1], [2, 3], [(0, 0.5)], stored_logprobs=[-0.2, -0.4])
    example.rollout.metadata["rollout_log_probs"] = None
    with pytest.raises(ValueError, match="actor_snapshot"):
        DeltaPolicyConfig.online(behavior_logprob_source="rollout", kl_coef=0.0)
    config = DeltaPolicyConfig.online(kl_coef=0.0, advantage_normalization="none")
    with pytest.raises(ValueError, match="Missing actor_snapshot"):
        prepare_policy_rows([example], {"r1": {0: 0.5}}, None, config)


def test_constant_training_advantages_fail_clearly_during_standardization():
    examples = [
        _example("r1", [1], [2], [(0, 0.5)]),
        _example("r2", [3], [4], [(0, 0.5)]),
    ]
    config = DeltaPolicyConfig.historical_offline()
    predictions = {"r1": {0: 1.0}, "r2": {0: 1.0}}
    with pytest.raises(ValueError, match="std"):
        prepare_policy_rows(examples, predictions, None, config, fit_stats=True)


def test_short_response_reward_joins_the_same_standardization_as_critic_deltas():
    """The realized reward is a raw delta, so it is normalized like every other one."""
    examples = [
        _example("short", [1], [2, 3], [(0, 0.5)]),
        _example("long", [1], [2, 3], [(0, 0.5)]),
    ]
    behavior = {"short": [-0.1, -0.1], "long": [-0.1, -0.1]}
    config = DeltaPolicyConfig.online(kl_coef=0.0)
    rows, stats = prepare_policy_rows(
        examples,
        {"short": {0: 9.0}, "long": {0: 0.5}},
        behavior,
        config,
        reward_deltas_by_id={"short": 1.0},
        fit_stats=True,
    )
    by_id = {row["id"]: row for row in rows}
    # Population is the two reward tokens plus the two critic tokens: 1, 1, 0.5, 0.5.
    assert (stats.count, stats.mean, stats.std) == pytest.approx((4, 0.75, 0.25))
    # The whole response is a state segment for the substituted row.
    assert by_id["short"]["state_mask"] == (1.0, 1.0)
    assert by_id["short"]["advantages"] == pytest.approx((1.0, 1.0))
    # The critic's own value for that row never reaches the advantage.
    assert by_id["long"]["advantages"] == pytest.approx((-1.0, -1.0))
    assert by_id["long"]["state_mask"] == (1.0, 1.0)


def test_advantage_clip_bounds_normalized_rows_and_reports_them():
    examples = [
        _example("short", [1], [2, 3], [(0, 0.5)]),
        _example("long", [1], [2, 3], [(0, 0.5)]),
    ]
    behavior = {"short": [-0.1, -0.1], "long": [-0.1, -0.1]}
    config = DeltaPolicyConfig.online(kl_coef=0.0, advantage_clip=0.25)
    rows, _ = prepare_policy_rows(
        examples,
        {"short": {0: 9.0}, "long": {0: 0.5}},
        behavior,
        config,
        reward_deltas_by_id={"short": 1.0},
        fit_stats=True,
    )
    by_id = {row["id"]: row for row in rows}
    # Unclipped z-scores would be +1 and -1; the bound covers substituted and critic rows alike.
    assert by_id["short"]["advantages"] == pytest.approx((0.25, 0.25))
    assert by_id["long"]["advantages"] == pytest.approx((-0.25, -0.25))
    assert by_id["short"]["advantage_clip_count"] == 2
    assert by_id["long"]["advantage_clip_count"] == 2
    with pytest.raises(ValueError, match="advantage_clip"):
        DeltaPolicyConfig.online(kl_coef=0.0, advantage_clip=0.0)


def test_offline_advantages_collation_and_loss_match_value_model_source():
    source = _source_module()
    examples = [
        _example(
            "r1",
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [(0, 0.5), (3, -0.25)],
            policy_mask=[1, 1, 0, 1, 1],
            stored_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
        ),
        _example("r2", [9], [10, 11, 12], [(1, 1.25)], policy_mask=[1, 1, 1], stored_logprobs=[-0.6, -0.7, -0.8]),
    ]
    predictions = {"r1": {0: 0.5, 3: -0.25}, "r2": {1: 1.25}}
    config = DeltaPolicyConfig.historical_offline(max_length=5)
    rows, stats = prepare_policy_rows(
        examples,
        predictions,
        None,
        config,
        fit_stats=True,
    )
    assert stats is not None

    source_rows = []
    for example in examples:
        raw = state_advantages(
            example,
            # Values are supplied directly from the same critic predictions used above.
            DeltaConfig("selected_segment", "selected_state", "critic", "segment", "none", "standardize"),
            raw_predictions=predictions[example.rollout.rollout_id],
        )
        source_rows.append(
            {
                "id": example.rollout.rollout_id,
                "prompt_id": example.rollout.prompt_id,
                "prompt_token_ids": list(example.rollout.prompt_token_ids),
                "response_token_ids": list(example.rollout.response_token_ids),
                "policy_token_mask": list(example.rollout.policy_token_mask),
                "old_logprobs": list(rows[len(source_rows)]["old_logprobs"]),
                "response_advantage": 0.0,
                "state_advantages": list(raw.values),
                "state_mask": list(raw.mask),
            }
        )
    source_stats = source.state_advantage_stats(source_rows)
    assert stats.count == source_stats["count"]
    assert stats.mean == pytest.approx(source_stats["mean"], abs=1e-12)
    assert stats.std == pytest.approx(source_stats["std"], abs=1e-12)
    source.normalize_state_advantages(source_rows, source_stats)

    batch = policy_training_batch(rows, pad_token_id=0, config=config)
    source_batch = source.collate_offline_grpo_batch(source_rows, pad_token_id=0, max_length=5)
    assert torch.equal(batch["input_ids"], source_batch["input_ids"])
    assert torch.equal(batch["attention_mask"], source_batch["attention_mask"])
    torch.testing.assert_close(batch["old_log_probs"], source_batch["old_logprobs"])
    torch.testing.assert_close(batch["advantages"], source_batch["state_advantages"])
    torch.testing.assert_close(batch["policy_loss_mask"], source_batch["state_mask"])

    current = torch.tensor([[0.15, -0.45, 0.2, -0.35], [-0.1, 0.3, 0.4, 0.0]], requires_grad=True)
    actual = per_sample_policy_losses(current, batch, config)
    source_row_losses = []
    for index in range(current.shape[0]):
        mask = source_batch["state_mask"][index : index + 1]
        pg = source.clipped_policy_loss(
            current[index : index + 1],
            source_batch["old_logprobs"][index : index + 1],
            source_batch["state_advantages"][index : index + 1],
            mask,
            config.clip_range,
        )
        kl = source.sampled_reverse_kl(
            current[index : index + 1],
            source_batch["old_logprobs"][index : index + 1],
            mask,
        )
        torch.testing.assert_close(actual["pg_loss"][index], pg)
        torch.testing.assert_close(actual["kl_loss"][index], kl)
        source_row_losses.append(pg + config.kl_coef * kl)
    expected = torch.stack(source_row_losses).mean()
    observed = actual["loss"].mean()
    torch.testing.assert_close(observed, expected)
    expected_grad = torch.autograd.grad(expected, current, retain_graph=True)[0]
    observed_grad = torch.autograd.grad(observed, current)[0]
    torch.testing.assert_close(observed_grad, expected_grad, rtol=1e-6, atol=1e-7)


def test_policy_masks_keep_zero_advantage_selected_tokens_and_ignore_padding():
    examples = [
        _example("r1", [1], [2, 3, 4], [(0, 0.0), (2, 1.0)], policy_mask=[1, 0, 1]),
        _example("r2", [5], [6, 7], [(0, 2.0)], policy_mask=[1, 1]),
    ]
    config = DeltaPolicyConfig.historical_offline(advantage_normalization="none", max_length=None)
    rows, _ = prepare_policy_rows(
        examples,
        {"r1": {0: 0.0, 2: 1.0}, "r2": {0: 2.0}},
        None,
        config,
        row_weights_by_id={"r1": 1.0, "r2": 0.0},
    )
    assert rows[0]["state_mask"] == (1.0, 1.0, 1.0)
    assert rows[0]["policy_loss_mask"] == (1.0, 0.0, 1.0)
    assert rows[1]["sample_valid"] is False
    batch = policy_training_batch(rows, 0, config)
    assert batch["global_valid_sample_weight"] == 1.0
    assert batch["policy_loss_mask"][0].tolist() == [1.0, 0.0, 1.0]
    assert batch["policy_loss_mask"][1].tolist() == [1.0, 1.0, 0.0]
    assert batch["sample_valid_mask"].tolist() == [1.0, 0.0]
    assert batch["advantages"][0, 0] == 0.0


def test_policy_response_adapter_keeps_response_alignment_and_validates_masks():
    examples = [
        _example("r1", [1], [2, 3], [(0, 0.5)], policy_mask=[1, 0]),
        _example("r2", [4], [5], [(0, 1.0)]),
    ]
    config = DeltaPolicyConfig.online(kl_coef=0.0, advantage_normalization="none", kl_mask_scope="response")
    rows, _ = prepare_policy_rows(
        examples,
        {"r1": {0: 0.5}, "r2": {0: 1.0}},
        {"r1": [-0.1, -0.1], "r2": [-0.1]},
        config,
        row_weights_by_id={"r1": 1.0, "r2": 0.0},
    )
    batch = policy_response_batch(rows, config)
    assert batch["old_log_probs"].shape == (2, 2)
    assert batch["kl_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert batch["sample_valid_mask"].tolist() == [1.0, 0.0]
    assert batch["global_valid_sample_weight"] == 1.0

    malformed = [dict(rows[0], response_mask=(1.0, 0.5))]
    with pytest.raises(ValueError, match="binary"):
        policy_response_batch(malformed, config)


def test_online_loss_uses_behavior_for_clip_fixed_reference_for_masked_kl():
    config = DeltaPolicyConfig.online(advantage_normalization="none")
    current = torch.tensor([[1.0, -10.0, 0.5]], requires_grad=True)
    batch = {
        "old_log_probs": torch.tensor([[0.0, -9.0, 0.0]]),
        "advantages": torch.tensor([[1.0, 99.0, -1.0]]),
        "policy_loss_mask": torch.tensor([[1.0, 0.0, 1.0]]),
        "response_mask": torch.ones((1, 3)),
        "ref_log_prob": torch.zeros((1, 3)),
    }
    losses = per_sample_policy_losses(current, batch, config)
    ratio = torch.exp(torch.tensor([1.0, 0.5]))
    expected_pg = -torch.minimum(
        ratio * torch.tensor([1.0, -1.0]), ratio.clamp(0.8, 1.2) * torch.tensor([1.0, -1.0])
    ).mean()
    expected_kl_values = torch.exp(torch.tensor([-1.0, -0.5])) - torch.tensor([-1.0, -0.5]) - 1.0
    expected_kl = expected_kl_values.mean()
    torch.testing.assert_close(losses["pg_loss"], expected_pg.reshape(1))
    torch.testing.assert_close(losses["kl_loss"], expected_kl.reshape(1))
    assert losses["kl_loss"].item() < 10.0  # excluded middle response token has very large reference KL
    assert losses["kl_token_count"].item() == 2
    losses["loss"].sum().backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()


@pytest.mark.parametrize(
    "mask,scope",
    [
        (torch.tensor([[1.0, float("nan")]]), "response"),
        (torch.tensor([[1.0, 0.5]]), "response"),
        (torch.ones((1, 3)), "response"),
    ],
)
def test_malformed_kl_mask_is_rejected_including_response_fallback(mask, scope):
    config = DeltaPolicyConfig.online(advantage_normalization="none", kl_mask_scope=scope)
    batch = {
        "old_log_probs": torch.zeros((1, 2)),
        "advantages": torch.ones((1, 2)),
        "policy_loss_mask": torch.ones((1, 2)),
        # Deliberately omit kl_mask so response_mask is the configured fallback.
        "response_mask": mask,
        "ref_log_prob": torch.zeros((1, 2)),
    }
    with pytest.raises(ValueError, match="kl_mask"):
        per_sample_policy_losses(torch.zeros((1, 2)), batch, config)


def test_malformed_explicit_kl_mask_is_rejected_even_when_kl_is_disabled():
    config = DeltaPolicyConfig.online(advantage_normalization="none", kl_coef=0.0)
    batch = {
        "old_log_probs": torch.zeros((1, 2)),
        "advantages": torch.ones((1, 2)),
        "policy_loss_mask": torch.ones((1, 2)),
        "kl_mask": torch.tensor([[1.0, float("nan")]]),
    }
    with pytest.raises(ValueError, match="kl_mask"):
        per_sample_policy_losses(torch.zeros((1, 2)), batch, config)


def test_zero_active_mask_is_finite_zero_loss():
    config = DeltaPolicyConfig.historical_offline()
    current = torch.tensor([[1.0, -1.0]], requires_grad=True)
    batch = {
        "old_log_probs": torch.zeros_like(current),
        "advantages": torch.zeros_like(current),
        "policy_loss_mask": torch.zeros_like(current),
    }
    losses = per_sample_policy_losses(current, batch, config)
    assert losses["loss"].item() == 0.0
    assert losses["policy_token_count"].item() == 0.0
    losses["loss"].sum().backward()
    assert torch.equal(current.grad, torch.zeros_like(current))


def _nested(sequences, *, dtype=torch.float32):
    return torch.nested.as_nested_tensor([torch.tensor(row, dtype=dtype) for row in sequences], layout=torch.jagged)


def test_v1_callback_extracts_response_before_padding_nested_fields():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    config = DeltaPolicyConfig.online(advantage_normalization="none")
    data = TensorDict(
        {
            "prompts": _nested([[1, 2], [3]], dtype=torch.long),
            "responses": _nested([[4, 5], [6, 7]], dtype=torch.long),
            "old_log_probs": _nested([[0.0, 0.0], [0.0, 0.0]]),
            "advantages": _nested([[1.0, 1.0], [-1.0, -1.0]]),
            "policy_loss_mask": _nested([[1.0, 0.0], [1.0, 1.0]]),
            "kl_mask": _nested([[1.0, 0.0], [1.0, 1.0]]),
            "ref_log_prob": _nested([[0.0, 0.0], [0.0, 0.0]]),
            "row_weight": torch.ones(2),
            "sample_valid_mask": torch.ones(2),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        delta_policy_config=config.as_dict(),
        dp_size=1,
        global_valid_sample_weight=2.0,
    )
    # Full-sequence outputs have lengths 4 and 3. Response logprobs are taken
    # from [prompt_len - 1 : prompt_len + response_len - 1].
    current_full = _nested([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    loss, metrics = delta_policy_loss({"log_probs": current_full}, data)
    expected_batch = {
        "old_log_probs": torch.zeros((2, 2)),
        "advantages": torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
        "policy_loss_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "kl_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "ref_log_prob": torch.zeros((2, 2)),
        "row_weight": torch.ones(2),
        "sample_valid_mask": torch.ones(2),
    }
    expected_current = torch.tensor([[0.2, 0.3], [0.5, 0.6]])
    expected = per_sample_policy_losses(expected_current, expected_batch, config)["loss"].mean()
    torch.testing.assert_close(loss, expected)
    assert metrics["actor/delta_rows"].aggregate() == pytest.approx(2.0)


def test_v1_callback_rejects_misaligned_nested_full_sequence_boundaries():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    config = DeltaPolicyConfig.online(advantage_normalization="none")
    data = TensorDict(
        {
            "prompts": _nested([[1, 2], [3]], dtype=torch.long),
            "responses": _nested([[4, 5], [6, 7]], dtype=torch.long),
            "old_log_probs": _nested([[0.0, 0.0], [0.0, 0.0]]),
            "advantages": _nested([[1.0, 1.0], [-1.0, -1.0]]),
            "policy_loss_mask": _nested([[1.0, 0.0], [1.0, 1.0]]),
            "ref_log_prob": _nested([[0.0, 0.0], [0.0, 0.0]]),
            "row_weight": torch.ones(2),
            "sample_valid_mask": torch.ones(2),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        delta_policy_config=config.as_dict(),
        dp_size=1,
        global_valid_sample_weight=2.0,
    )
    # The total packed token count is still seven, but the per-row boundaries
    # disagree with prompt+response lengths [4, 3].
    current_full = _nested([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7]])
    with pytest.raises(ValueError, match="boundaries"):
        delta_policy_loss({"log_probs": current_full}, data)


def _dense_v1_batch(config, start, stop, *, dp_size, global_weight):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    batch = TensorDict(
        {
            "old_log_probs": torch.zeros((stop - start, 3)),
            "advantages": torch.tensor([[1.0, 0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, -0.5]])[
                start:stop
            ],
            "policy_loss_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])[
                start:stop
            ],
            "kl_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])[start:stop],
            "ref_log_prob": torch.zeros((stop - start, 3)),
            "row_weight": torch.ones(stop - start),
            "sample_valid_mask": torch.ones(stop - start),
        },
        batch_size=[stop - start],
    )
    tu.assign_non_tensor(
        batch, delta_policy_config=config.as_dict(), dp_size=dp_size, global_valid_sample_weight=global_weight
    )
    return batch


def test_v1_global_row_denominator_is_microbatch_and_dp_partition_invariant():
    config = DeltaPolicyConfig.online(advantage_normalization="none")
    current = torch.tensor(
        [[0.2, 0.1, -0.3], [0.0, -0.5, 0.4], [0.3, 0.2, -0.1], [0.1, 0.4, 0.2]],
        requires_grad=True,
    )
    full_batch = _dense_v1_batch(config, 0, 4, dp_size=1, global_weight=4.0)
    full_loss, _ = delta_policy_loss({"log_probs": current}, full_batch)
    full_grad = torch.autograd.grad(full_loss, current, retain_graph=True)[0]

    split_losses = []
    for start, stop in ((0, 1), (1, 3), (3, 4)):
        loss, _ = delta_policy_loss(
            {"log_probs": current[start:stop]},
            _dense_v1_batch(config, start, stop, dp_size=1, global_weight=4.0),
        )
        split_losses.append(loss)
    split_loss = torch.stack(split_losses).sum()
    split_grad = torch.autograd.grad(split_loss, current, retain_graph=True)[0]
    torch.testing.assert_close(split_loss, full_loss)
    torch.testing.assert_close(split_grad, full_grad)

    rank0, _ = delta_policy_loss(
        {"log_probs": current[:2]}, _dense_v1_batch(config, 0, 2, dp_size=2, global_weight=4.0)
    )
    rank1, _ = delta_policy_loss(
        {"log_probs": current[2:]}, _dense_v1_batch(config, 2, 4, dp_size=2, global_weight=4.0)
    )
    dp_equivalent = (rank0 + rank1) / 2.0
    dp_grad = torch.autograd.grad(dp_equivalent, current)[0]
    torch.testing.assert_close(dp_equivalent, full_loss)
    torch.testing.assert_close(dp_grad, full_grad)


def test_v1_callback_requires_full_logical_update_denominator():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    config = DeltaPolicyConfig.online(advantage_normalization="none")
    data = TensorDict(
        {
            "old_log_probs": torch.zeros((1, 2)),
            "advantages": torch.ones((1, 2)),
            "policy_loss_mask": torch.ones((1, 2)),
            "ref_log_prob": torch.zeros((1, 2)),
            "row_weight": torch.ones(1),
            "sample_valid_mask": torch.ones(1),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, delta_policy_config=config.as_dict(), dp_size=1)
    with pytest.raises(ValueError, match="global_valid_sample_weight"):
        delta_policy_loss({"log_probs": torch.zeros((1, 2))}, data)
