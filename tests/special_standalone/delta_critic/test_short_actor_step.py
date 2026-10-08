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

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from examples.delta_critic.short_actor_step import (
    ShortRow,
    chunked_response_logprobs,
    group_metrics,
    load_short_rows,
    measure_rows,
    perform_one_step,
)


def _saved_row(row_id="r1", *, is_short=True):
    return {
        "row_id": row_id,
        "is_short": is_short,
        "prompt_token_ids": [1, 2],
        "response_token_ids": [3, 4, 5],
        "policy_token_mask": [1.0, 0.0, 1.0],
        "reward": 0.0,
        "baseline": 0.0,
        "baseline_source": "leave_one_out_siblings",
        "advantage_replay": {"old_advantage_mean": 0.5, "new_advantage_mean": -1.0},
    }


class _CumulativeDecoder(nn.Module):
    def __init__(self, hidden_size=4, vocab_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        hidden = self.embedding(input_ids).cumsum(dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


class _CountingHead(nn.Linear):
    def __init__(self, hidden_size=4, vocab_size=8):
        super().__init__(hidden_size, vocab_size, bias=False)
        self.max_rows_seen = 0

    def forward(self, hidden):
        self.max_rows_seen = max(self.max_rows_seen, hidden.shape[0])
        return super().forward(hidden)


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _CumulativeDecoder()
        self.lm_head = _CountingHead()


def _actor_row():
    return ShortRow(
        row_id="short",
        prompt=(1, 2),
        response=(3, 4, 5),
        mask=(1.0, 0.0, 1.0),
        reward=0.0,
        baseline=0.0,
        baseline_source="leave_one_out_siblings",
        old_advantage=0.5,
        new_advantage=-1.0,
    )


def test_loader_keeps_exact_short_tokens_mask_and_both_advantages(tmp_path):
    batch = tmp_path / "batch.jsonl"
    batch.write_text(json.dumps(_saved_row()) + "\n" + json.dumps(_saved_row("long", is_short=False)) + "\n")

    rows = load_short_rows(batch)

    assert len(rows) == 1
    assert rows[0].prompt == (1, 2)
    assert rows[0].response == (3, 4, 5)
    assert rows[0].mask == (1.0, 0.0, 1.0)
    assert rows[0].old_advantage == 0.5
    assert rows[0].new_advantage == -1.0


def test_loader_fails_closed_when_saved_mask_does_not_align(tmp_path):
    row = _saved_row()
    row["policy_token_mask"] = [1.0]
    batch = tmp_path / "bad.jsonl"
    batch.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="policy_token_mask"):
        load_short_rows(batch)


def test_chunked_head_matches_full_reference_and_bounds_logits_rows():
    torch.manual_seed(11)
    actor = _TinyActor()
    response = (3, 4, 5)
    prompt = (1, 2)

    logprobs, eos_logprob = chunked_response_logprobs(
        actor,
        prompt,
        response,
        chunk_size=2,
        eos_token_id=7,
    )
    chunk_rows_seen = actor.lm_head.max_rows_seen

    tokens = torch.tensor([[*prompt, *response[:-1]]])
    hidden = actor.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, return_dict=True)
    selected_hidden = hidden.last_hidden_state[0, len(prompt) - 1 :]
    full_logits = actor.lm_head(selected_hidden).float()
    full_logprobs = full_logits.log_softmax(-1)
    expected = full_logprobs.gather(-1, torch.tensor(response).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(logprobs, expected)
    torch.testing.assert_close(eos_logprob, full_logprobs[-1, 7])
    assert chunk_rows_seen <= 2


def test_paired_cpu_steps_use_old_and_new_advantages_and_report_group_metrics():
    torch.manual_seed(23)
    initial = _TinyActor()
    row = _actor_row()
    device = torch.device("cpu")
    before = measure_rows(
        initial,
        [row],
        device=device,
        chunk_size=2,
        eos_token_id=5,
        autocast_dtype="none",
    )
    behavior = [before[0].logprobs]
    old_actor = copy.deepcopy(initial)
    new_actor = copy.deepcopy(initial)

    old_step = perform_one_step(
        old_actor,
        [row],
        behavior,
        variant="old",
        device=device,
        chunk_size=2,
        autocast_dtype="none",
    )
    new_step = perform_one_step(
        new_actor,
        [row],
        behavior,
        variant="new",
        device=device,
        chunk_size=2,
        autocast_dtype="none",
    )

    assert old_step["grad_norm_before_clip"] > 0
    assert new_step["grad_norm_before_clip"] > 0
    assert old_step["grad_norm_after_clip"] <= 1.0
    assert new_step["grad_norm_after_clip"] <= 1.0
    assert any(not torch.equal(a, b) for a, b in zip(old_actor.parameters(), new_actor.parameters(), strict=True))

    after = measure_rows(
        new_actor,
        [row],
        device=device,
        chunk_size=2,
        eos_token_id=5,
        autocast_dtype="none",
    )
    grouped_before = next(iter(group_metrics([row], before, eos_token_id=5).values()))
    grouped_after = next(iter(group_metrics([row], after, eos_token_id=5).values()))
    assert grouped_before["mean_response_logprob_row_mean"] != grouped_after["mean_response_logprob_row_mean"]
    assert grouped_before["mean_eos_probability_at_last_response_decision"] is not None
