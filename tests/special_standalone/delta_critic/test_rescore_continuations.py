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
from types import SimpleNamespace

import pytest
import torch

from examples.delta_critic.rescore_continuations import continuation_top_logprobs, write_rescored_rows


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", max_position_embeddings=20)
        self.weight = torch.nn.Parameter(torch.eye(8))
        self.seen = None

    @property
    def base_model(self):
        return self

    def forward(self, input_ids, **kwargs):
        self.seen = input_ids.tolist()
        return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 8).float())

    def get_output_embeddings(self):
        return lambda hidden: hidden @ self.weight


def test_teacher_forcing_uses_before_token_prefix_and_chunking_preserves_probabilities():
    actor = TinyActor()
    rows = continuation_top_logprobs(actor, [1, 2], [3, 4, 5], top_k=1, logits_chunk_size=1)
    assert actor.seen == [[1, 2, 3, 4]]
    assert [row[0]["token_id"] for row in rows] == [2, 3, 4]
    expected = torch.tensor([1.0, *([0.0] * 7)]).log_softmax(0)[0].item()
    assert [row[0]["logprob"] for row in rows] == pytest.approx([expected] * 3)
    chunked = continuation_top_logprobs(actor, [1, 2], [3, 4, 5], top_k=1, logits_chunk_size=3)
    assert rows == chunked


def test_chunked_scoring_matches_real_tiny_qwen3_causal_lm_logits():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    actor = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
        )
    ).eval()
    prefix, response = [1, 2], [3, 4, 5]
    scored = continuation_top_logprobs(actor, prefix, response, top_k=3, logits_chunk_size=2)
    with torch.inference_mode():
        full = actor(torch.tensor([[*prefix, *response[:-1]]]), use_cache=False).logits[0, len(prefix) - 1 :]
        values, indices = full.float().log_softmax(-1).topk(3, -1)
    assert [[candidate["token_id"] for candidate in row] for row in scored] == indices.tolist()
    torch.testing.assert_close(
        torch.tensor([[candidate["logprob"] for candidate in row] for row in scored]),
        values,
        rtol=1e-6,
        atol=1e-6,
    )


def test_rescoring_preserves_tokens_rewards_and_refuses_wrong_actor_or_overwrite(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "state_id": "s",
                "rollout_id": "r",
                "token_index": 1,
                "prompt_token_ids": [1],
                "prefix_response_token_ids": [2],
            }
        )
        + "\n"
    )
    continuation = {
        "state_id": "s",
        "rollout_id": "r",
        "token_index": 1,
        "actor_version": "actor",
        "continuation_token_ids": [3, 4],
        "reward": 1.0,
        "sampling_config": {"temperature": 0.7, "top_p": 0.9},
    }
    source = tmp_path / "source.jsonl"
    original = json.dumps(continuation) + "\n"
    source.write_text(original)
    output = tmp_path / "scored.jsonl"
    assert write_rescored_rows(source, labels, output, TinyActor(), model_path="unused", actor_version="actor") == 1
    result = json.loads(output.read_text())
    for key, value in continuation.items():
        assert result[key] == value
    assert len(result["delta_top_logprobs"]) == 2
    assert result["delta_top_logprobs"] == continuation_top_logprobs(TinyActor(), [1, 2], [3, 4])
    assert result["delta_top_logprobs_provenance"]["logprobs_mode"] == "raw_logprobs"
    assert source.read_text() == original
    with pytest.raises(FileExistsError, match="overwrite"):
        write_rescored_rows(source, labels, output, TinyActor(), model_path="unused", actor_version="actor")
    refused = tmp_path / "refused.jsonl"
    with pytest.raises(ValueError, match="actor_version"):
        write_rescored_rows(source, labels, refused, TinyActor(), model_path="unused", actor_version="other")
    assert not refused.exists()
    assert not list(tmp_path.glob(".refused.jsonl.*"))
