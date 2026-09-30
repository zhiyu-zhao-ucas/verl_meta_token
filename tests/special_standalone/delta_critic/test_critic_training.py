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

"""Focused regression tests for V1 batch scaling and update-order restore."""

import pytest
import torch
import torch.multiprocessing as mp
from tensordict import TensorDict

from examples.delta_critic.engine import (
    microbatch_padding_width,
    resolve_autocast_dtype,
    trim_dense_columns,
)
from examples.delta_critic.scalar_loss import delta_loss, per_sample_loss
from examples.delta_critic.train import (
    RowIterator,
    coverage_metrics,
    expected_continuation_rows,
    reduce_engine_metrics,
    reserve_step_directory,
    resolve_config,
)
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.training_data import collate, continuation_rows


def _rows():
    return [
        {
            "id": "three_tokens",
            "prompt_token_ids": [1],
            "response_token_ids": [2, 3, 4],
            "token_targets": [0.0, 0.2, -0.1],
            "token_loss_mask": [1.0, 1.0, 1.0],
            "token_signal_mask": [1.0, 0.0, 0.0],
        },
        {
            "id": "one_token",
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4, 5],
            "token_targets": [0.4, 0.0, 0.0],
            "token_loss_mask": [1.0, 0.0, 0.0],
            "token_signal_mask": [1.0, 0.0, 0.0],
        },
        {
            "id": "two_tokens",
            "prompt_token_ids": [2],
            "response_token_ids": [3, 4, 5],
            "token_targets": [-0.3, 0.1, 0.0],
            "token_loss_mask": [1.0, 1.0, 0.0],
            "token_signal_mask": [1.0, 0.0, 0.0],
        },
        {
            "id": "real_zero_supervision",
            "prompt_token_ids": [1],
            "response_token_ids": [2, 3, 4],
            "token_targets": [0.0, 0.0, 0.0],
            "token_loss_mask": [0.0, 0.0, 0.0],
            "token_signal_mask": [0.0, 0.0, 0.0],
        },
    ]


def test_row_mean_gradient_is_invariant_to_dp_and_microbatch():
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric.utils import AggregationType, Metric

    config = ScalarConfig("unused", max_length=8, target_normalization="none")
    normalization = {"enabled": False, "mode": "none", "mean": None, "std": None}
    real_rows = _rows()
    # Simulate two synthetic rows used only to make per-rank microbatch counts
    # even. Their duplicate payload must not affect loss or token coverage.
    padding_rows = [
        {
            **real_rows[0],
            "id": f"padding_{i}",
            "token_loss_mask": [0.0, 0.0, 0.0],
            "sample_valid": False,
        }
        for i in range(2)
    ]
    rows = real_rows + padding_rows
    batch = TensorDict(collate(rows, config, normalization), batch_size=[len(rows)])
    prediction_shape = batch["target"].shape
    feature = torch.arange(1, prediction_shape[1] + 1, dtype=torch.float32).expand(len(rows), -1)
    coefficient = torch.tensor(0.6, requires_grad=True)
    predictions = coefficient * feature
    per_row, valid, _, _ = per_sample_loss(predictions, batch, config, normalization)
    reference = (per_row * valid).sum() / valid.sum()
    expected_grad = torch.autograd.grad(reference, coefficient, retain_graph=True)[0]

    reference_data = batch.clone()
    tu.assign_non_tensor(
        reference_data,
        delta_config=config.as_dict(),
        normalization=normalization,
        terminal_stats=None,
        dp_size=1,
        valid_sample_count=len(real_rows),
    )
    _, reference_metrics = delta_loss({"delta_scalar": predictions}, reference_data, dp_group=None)

    assert int(valid.sum()) == len(real_rows)  # the zero-target real row still counts
    assert float(per_row[3].detach()) == 0.0
    assert torch.all(per_row[4:] == 0)

    for dp_size, microbatch in ((1, 1), (1, 2), (2, 1), (2, 3)):
        rank_losses = []
        rank_metric_totals = []
        for rank in range(dp_size):
            row_indices = list(range(rank, len(rows), dp_size))
            local_loss = coefficient.new_zeros(())
            local_metrics = {}
            for start in range(0, len(row_indices), microbatch):
                indices = row_indices[start : start + microbatch]
                data = batch[indices]
                tu.assign_non_tensor(
                    data,
                    delta_config=config.as_dict(),
                    normalization=normalization,
                    terminal_stats=None,
                    dp_size=dp_size,
                    valid_sample_count=len(real_rows),
                )
                loss, metrics = delta_loss({"delta_scalar": predictions[indices]}, data, dp_group=None)
                local_loss = local_loss + loss
                for name, metric in metrics.items():
                    local_metrics.setdefault(name, Metric(aggregation=AggregationType.SUM)).extend(metric)
            rank_losses.append(local_loss)
            rank_metric_totals.append(local_metrics)

        reduced_loss = sum(rank_losses) / dp_size
        actual_grad = torch.autograd.grad(reduced_loss, coefficient, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=1e-7)
        for name in rank_metric_totals[0]:
            actual_metric = reduce_engine_metrics({name: [rank[name] for rank in rank_metric_totals]})[name]
            assert actual_metric == pytest.approx(reference_metrics[name].aggregate(), abs=1e-7), name


def test_row_iterator_restores_next_update_exactly():
    uninterrupted = RowIterator(size=5, seed=23)
    prefix = uninterrupted.take(7)
    state = uninterrupted.state()
    expected_next = uninterrupted.take(11)

    resumed = RowIterator(size=5, seed=23, **state)
    assert resumed.take(11) == expected_next
    assert prefix != expected_next[: len(prefix)]


def test_coverage_excludes_only_synthetic_rows():
    rows = _rows()
    summary = coverage_metrics(rows)
    assert summary["real_rows"] == 4
    assert summary["valid_rows"] == 4
    assert summary["supervised_rows"] == 3
    assert summary["target_tokens"] == 6
    assert summary["signal_tokens"] == 3
    assert summary["terminal_rows"] == 0


def _reserve_step_directory_worker(rank, world_size, rendezvous_file, step_dir, expect_existing):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        try:
            reserve_step_directory(step_dir, device=torch.device("cpu"))
        except FileExistsError:
            assert expect_existing, "a fresh step directory must not be reported as an overwrite"
        else:
            assert not expect_existing, "an existing step directory must be refused by every rank"
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("expect_existing", [False, True])
def test_step_directory_reservation_agrees_across_ranks(tmp_path, expect_existing):
    """Rank zero creates the step directory, so no other rank may test it itself.

    A per-rank existence check races rank zero's ``mkdir`` and aborts a
    legitimate save with a bogus overwrite error, which is what broke the
    two-GPU run.
    """
    step_dir = tmp_path / "step_00000001"
    if expect_existing:
        step_dir.mkdir()
    mp.spawn(
        _reserve_step_directory_worker,
        args=(2, str(tmp_path / "rdzv"), str(step_dir), expect_existing),
        nprocs=2,
        join=True,
    )
    assert step_dir.is_dir()
    assert list(step_dir.iterdir()) == []


def test_trim_dense_columns_skips_keys_absent_from_scoring_batches():
    """The frozen scoring batch carries only `target`/`loss_mask`.

    Trimming it must not require `target_mask`/`signal_mask`, which training
    batches do carry.
    """
    scoring = {"target": torch.zeros(2, 5), "loss_mask": torch.zeros(2, 5)}
    trim_dense_columns(scoring, 3)
    assert scoring["target"].shape == (2, 3)
    assert scoring["loss_mask"].shape == (2, 5)  # not a padded column

    training = {key: torch.zeros(2, 5) for key in ("target", "target_mask", "signal_mask")}
    trim_dense_columns(training, 4)
    assert all(value.shape == (2, 4) for value in training.values())

    with pytest.raises(ValueError, match="positive"):
        trim_dense_columns({"target": torch.zeros(1, 2)}, 0)


def test_resolve_autocast_dtype_maps_config_names():
    assert resolve_autocast_dtype("none", "bfloat16") is torch.float32
    assert resolve_autocast_dtype("bfloat16", "bfloat16") is torch.bfloat16
    assert resolve_autocast_dtype("model_dtype", "bfloat16") is torch.bfloat16
    assert resolve_autocast_dtype("model_dtype", "float32") is torch.float32
    with pytest.raises(ValueError, match="Unsupported autocast"):
        resolve_autocast_dtype("bogus", "bfloat16")


def test_budgeted_hybrid_split_is_not_reported_incomplete():
    """A configured continuation budget drops rows on purpose; the trainer's
    completeness guard must not read that as missing data."""
    labels = [
        {
            "rollout_id": "r",
            "state_id": "r:t0",
            "token_index": 0,
            "v_prefix": 0.1,
            "prompt_token_ids": [1, 2],
        }
    ]
    continuations = [
        {
            "state_id": "r:t0",
            "rollout_id": "r",
            "continuation_index": index,
            "continuation_id": f"c{index}",
            "continuation_token_ids": [3, 4],
            "reward": 1.0,
        }
        for index in (0, 1)
    ]
    budgeted = ScalarConfig("unused", objective="hybrid_terminal_composition", continuation_budget=1)
    _, meta = continuation_rows(continuations, labels, budgeted)
    assert (meta["included"], meta["eligible"], meta["over_budget"]) == (1, 2, 1)
    assert meta["included"] == expected_continuation_rows(meta)

    # Unlimited budget still requires every eligible row to be kept.
    unbudgeted = ScalarConfig("unused", objective="hybrid_terminal_composition")
    _, meta = continuation_rows(continuations, labels, unbudgeted)
    assert meta["included"] == expected_continuation_rows(meta) == 2


def test_attention_backend_override_reaches_the_run_config():
    config_path = "examples/delta_critic/config_train_qwen3_8b.yaml"
    assert resolve_config(config_path).attention_implementation == "sdpa"
    assert resolve_config(config_path, "flash_attention_2").attention_implementation == "flash_attention_2"
    assert resolve_config(config_path, "eager").attention_implementation == "eager"
    with pytest.raises(ValueError, match="attention_implementation"):
        resolve_config(config_path, "bogus")


def test_microbatch_padding_width_uses_the_longest_real_row():
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    assert microbatch_padding_width(mask) == 4
    assert microbatch_padding_width(mask[:1]) == 3
    with pytest.raises(ValueError, match="empty"):
        microbatch_padding_width(torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="nonempty"):
        microbatch_padding_width(torch.ones(3, dtype=torch.long))


def _trim_to_width(batch, width):
    """Mirror what DeltaFSDPEngine.forward_step does to a micro-batch."""
    trimmed = batch.clone()
    for key in ("target", "target_mask", "signal_mask"):
        trimmed[key] = batch[key][:, :width]
    return trimmed


@pytest.mark.parametrize("loss_type", ["mse", "nonzero_balanced_mse"])
def test_trimming_padding_does_not_change_the_per_row_loss(loss_type):
    """The engine runs each micro-batch at its own longest row instead of the
    global batch width. Padding columns carry a zero target and a zero mask, so
    dropping them must not move the loss."""
    config = ScalarConfig("unused", max_length=64, target_normalization="none", loss_type=loss_type)
    normalization = {"enabled": False, "mode": "none", "mean": None, "std": None}
    rows = _rows()
    batch = TensorDict(collate(rows, config, normalization), batch_size=[len(rows)])
    global_width = batch["target"].shape[1]

    torch.manual_seed(7)
    predictions = torch.randn(len(rows), global_width, requires_grad=True)
    for index in range(len(rows)):
        micro = batch[index : index + 1]
        width = microbatch_padding_width(micro["attention_mask"])
        assert width <= global_width
        full_pred = predictions[index : index + 1]
        reference, _, _, _ = per_sample_loss(full_pred, micro, config, normalization)
        trimmed_pred = full_pred[:, :width]
        trimmed = _trim_to_width(micro, width)
        assert trimmed_pred.shape == trimmed["target"].shape == trimmed["target_mask"].shape
        actual, _, _, _ = per_sample_loss(trimmed_pred, trimmed, config, normalization)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_trimming_keeps_the_row_with_no_padding_untouched():
    config = ScalarConfig("unused", max_length=64, target_normalization="none")
    normalization = {"enabled": False, "mode": "none", "mean": None, "std": None}
    rows = _rows()
    for row in rows:  # one row per micro-batch: trim must be a no-op there
        batch = TensorDict(collate([row], config, normalization), batch_size=[1])
        width = microbatch_padding_width(batch["attention_mask"])
        assert width == len(row["prompt_token_ids"]) + len(row["response_token_ids"])
        assert width == batch["target"].shape[1]
