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
import json

import pytest
import torch

from examples.delta_critic.policy_config import DeltaPolicyConfig
from examples.delta_critic.policy_train import (
    RowIterator,
    actor_tensor_batch,
    export_final_hf,
    read_precomputed_behavior,
)
from examples.delta_critic.policy_training_config import PolicyTrainingConfig


def _row():
    return {
        "id": "long-response",
        "prompt_token_ids": (1,),
        "response_token_ids": (2, 3, 4, 5),
        "old_logprobs": (-0.1, -0.2, -0.3, -0.4),
        "advantages": (10.0, 20.0, 30.0, 40.0),
        "state_mask": (1.0, 1.0, 1.0, 1.0),
        "policy_token_mask": (1.0, 1.0, 1.0, 1.0),
        "policy_loss_mask": (1.0, 1.0, 1.0, 1.0),
        "response_mask": (1.0, 1.0, 1.0, 1.0),
        "row_weight": 1.0,
        "sample_valid": True,
    }


def test_actor_batch_preserves_tail_window_action_alignment():
    config = DeltaPolicyConfig.historical_offline(advantage_normalization="none", max_length=3, window="legacy_tail")
    runtime = PolicyTrainingConfig(model_path="unused", dtype="float32", gradient_checkpointing=False)
    data = actor_tensor_batch([_row()], 0, config, runtime, dp_size=1, global_valid_sample_weight=1.0)

    # The legacy tail window retains [3, 4, 5]. Token 3 has no predecessor,
    # so it becomes the synthetic one-token prompt and its action is omitted.
    assert data["input_ids"].unbind()[0].tolist() == [3, 4, 5]
    assert data["prompts"].unbind()[0].tolist() == [3]
    assert data["responses"].unbind()[0].tolist() == [4, 5]
    torch.testing.assert_close(data["old_log_probs"].values(), torch.tensor([-0.3, -0.4]))
    torch.testing.assert_close(data["advantages"].values(), torch.tensor([30.0, 40.0]))
    torch.testing.assert_close(data["policy_loss_mask"].values(), torch.ones(2))


def test_row_iterator_keeps_partial_epoch_tail_in_its_epoch():
    iterator = RowIterator(size=5, seed=42)
    first = iterator.take(4)
    tail = iterator.take(4)
    assert len(first) == 4
    assert len(tail) == 1
    assert len(set(first + tail)) == 5
    assert iterator.state() == {"epoch": 1, "cursor": 0}
    next_epoch = iterator.take(4)
    assert len(next_epoch) == 4
    assert iterator.state() == {"epoch": 1, "cursor": 4}


def test_precomputed_old_logprobs_require_matching_scoring_provenance(tmp_path):
    config = DeltaPolicyConfig.online(behavior_logprob_source="precomputed")
    source = tmp_path / "old.jsonl"
    row = {
        "rollout_id": "r1",
        "logprobs": [-0.1, -0.2],
        "provenance": {
            "actor_fingerprint": "actor-a",
            "temperature": config.train_logprob_temperature,
            "window": config.window,
            "max_length": config.max_length,
        },
    }
    source.write_text(json.dumps(row) + "\n")
    values, identity = read_precomputed_behavior(source, config, actor_fingerprint="actor-a")
    assert values == {"r1": [-0.1, -0.2]}
    assert identity == "actor-a"
    with pytest.raises(ValueError, match="current actor snapshot"):
        read_precomputed_behavior(source, config, actor_fingerprint="actor-b")
    row["provenance"]["temperature"] = 0.7
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="temperature mismatch"):
        read_precomputed_behavior(source, config, actor_fingerprint="actor-a")


def test_complete_checkpoint_can_recover_final_export_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda values, src: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    checkpoint = tmp_path / "step_00000003"
    source = checkpoint / "huggingface"
    source.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("complete\n")
    (source / "model.safetensors").write_bytes(b"initial")
    export_final_hf(checkpoint, tmp_path)
    destination = tmp_path / "final_hf/model.safetensors"
    assert destination.read_bytes() == b"initial"
    export_final_hf(checkpoint, tmp_path)
    destination.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        export_final_hf(checkpoint, tmp_path)
    assert destination.read_bytes() == b"changed"
