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
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from examples.delta_critic.mc_labeling import MCConfig, MCRequest
from examples.delta_critic.online_critic import OnlineDeltaWorker
from examples.delta_critic.policy_mc import collect_online_mc, export_mc_attempt, export_mc_records, online_mc_config
from examples.delta_critic.training_config import ScalarConfig
from examples.delta_critic.v1_sampler import sample_v1_continuations


@pytest.mark.parametrize("expansion", [False, True])
def test_mc_uses_exact_selected_prefixes_and_checks_actor_version(tmp_path, expansion):
    calls = []

    class Client:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                token_ids=[7], extra_fields={"global_steps": 3, "delta_top_logprobs": [[{"prob": 0.6}]]}
            )

    class Tokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return " ".join(map(str, ids))

    output = SimpleNamespace(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22],
        response_mask=[1, 1, 1],
        reward_score=-1.0,
        extra_fields={"global_steps": 3},
    )
    policy = {
        "rollout_mode": "selected_prefix_mc",
        "expansion": {"enabled": expansion},
        "mc": {
            "continuations_per_state": 2,
            "max_continuation_tokens": 12,
            "max_total_tokens": 5,
            "reward": {"method": "exact_or_numeric"},
        },
    }
    kwargs = dict(
        policy=policy,
        rollout_config={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        server_client=Client(),
        tokenizer=Tokenizer(),
        uid="x",
        session_id=0,
        sample_kwargs={"reward_model": {"ground_truth": "7"}},
        request_semaphore=asyncio.Semaphore(2),
    )
    record = asyncio.run(collect_online_mc(output, [0, 2], **kwargs))
    assert sorted(call["prompt_ids"] for call in calls) == [[10, 11], [10, 11], [10, 11, 20, 21], [10, 11, 20, 21]]
    assert [label["v_prefix"] for label in record["labels"]] == [1.0, 1.0]
    assert all(label["mc_num_samples"] == 2 for label in record["labels"])
    if expansion:
        assert all(call["sampling_params"]["delta_top_logprobs"] == 20 for call in calls)
        assert all(
            continuation["delta_top_logprobs"] == [[{"prob": 0.6}]]
            for label in record["labels"]
            for continuation in label["mc_continuations"]
        )
    assert sorted(call["sampling_params"]["max_tokens"] for call in calls) == [1, 1, 3, 3]
    assert {
        label["token_index"]: label["mc_sampling_config"]["effective_max_tokens"] for label in record["labels"]
    } == {0: 3, 2: 1}
    assert record["rollout"]["terminal_reward"] == 0.0
    assert record["rollout"]["trainer_reward"] == -1.0
    assert record["rollout"]["reward_config"] == {"method": "exact_or_numeric"}
    export_mc_records([record], tmp_path)

    from examples.delta_critic.legacy_adapter import adapt_legacy_rows

    rollouts = [json.loads(line) for line in (tmp_path / "rollouts_train_regen.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (tmp_path / "mc_labels_train.jsonl").read_text().splitlines()]
    examples, diagnostics = adapt_legacy_rows(rollouts, labels)
    assert not diagnostics and len(examples[0].states) == 2
    assert len((tmp_path / "continuations_train.jsonl").read_text().splitlines()) == 4
    assert "mc_continuations" in record["labels"][0], "export must not mutate queued records"
    export_mc_records([record], tmp_path)
    altered = dict(record, rollout=dict(record["rollout"], terminal_reward=1.0))
    with pytest.raises(ValueError, match="overwrite different MC data"):
        export_mc_records([altered], tmp_path)
    output.extra_fields["global_steps"] = 4
    with pytest.raises(ValueError, match="actor version"):
        asyncio.run(collect_online_mc(output, [0], **kwargs))


@pytest.mark.parametrize("answer,gold", [(r"\boxed{\frac{1}{2}}", "0.5"), (r"\boxed{\sqrt{4}}", "2")])
@pytest.mark.parametrize("collector", ["online", "batch"])
def test_collectors_score_real_math_in_processes(answer, gold, collector):
    pytest.importorskip("math_verify")
    from examples.delta_critic.batch_sample import sample_labeled_rollout, sampling_settings
    from examples.delta_critic.sampling_io import score_text

    class Tokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return answer if ids else ""

    class Client:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=[7], extra_fields={"global_steps": 3})

    async def collect():
        if collector == "online":
            return await collect_online_mc(
                SimpleNamespace(
                    prompt_ids=[10],
                    response_ids=[7],
                    response_mask=[1],
                    reward_score=-1.0,
                    extra_fields={"global_steps": 3, "finish_reason": "stop"},
                ),
                [0],
                policy={
                    "rollout_mode": "selected_prefix_mc",
                    "mc": {
                        "continuations_per_state": 2,
                        "max_continuation_tokens": 12,
                    },
                },
                rollout_config={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                server_client=Client(),
                tokenizer=Tokenizer(),
                uid="math",
                session_id=0,
                sample_kwargs={"reward_model": {"ground_truth": gold}},
                request_semaphore=asyncio.Semaphore(2),
            )
        config = {
            "rollout": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 12, "top_k_logprobs": 1},
            "mc": {"temperature": 1.0, "top_p": 0.95, "max_continuation_tokens": 12, "continuations_per_state": 2},
            "smoke": {"states_per_response": 1, "top_mass": 0.8},
            "state_selection": {
                "max_candidates": 1,
                "entropy_weight": 0.0,
                "low_top1_weight": 1.0,
                "final_window_tokens": 64,
                "final_window_weight": 0.0,
            },
            "model": {"actor_model": "3"},
        }
        generation, mc_config = sampling_settings(config, "prefix_only")
        rollout, states, labels = await sample_labeled_rollout(
            {"prompt_id": "math", "gold_answer": gold, "prompt": "question"},
            [10],
            SimpleNamespace(
                token_ids=[7],
                log_probs=[-0.3],
                stop_reason="stop",
                extra_fields={
                    "finish_reason": "stop",
                    "delta_top_logprobs": [
                        [
                            {
                                "token_id": 7,
                                "prob": 0.7,
                                "logprob": -0.3,
                                "token": answer,
                            }
                        ]
                    ],
                },
            ),
            tokenizer=Tokenizer(),
            client=Client(),
            config=config,
            split="train",
            response_index=0,
            actor_version="3",
            generation=generation,
            mc_config=mc_config,
            mc_concurrency=2,
            score_semaphore=asyncio.Semaphore(2),
        )
        return {"rollout": rollout, "states": states, "labels": labels}

    record = asyncio.run(collect())
    assert score_text(answer, gold, {}) == 1.0
    assert record["rollout"]["terminal_reward"] == 1.0
    assert record["labels"][0]["v_prefix"] == 1.0


def test_export_attempt_survives_resampling_and_legacy_partial_export(tmp_path, monkeypatch):
    record = mc_record()
    # Simulate an interrupted pre-fix export before the policy checkpoint.
    legacy = tmp_path / "rollouts_train_regen.jsonl"
    legacy.write_text("old incomplete export\n")
    first = export_mc_attempt([record], tmp_path)
    first_contents = (first / "rollouts_train_regen.jsonl").read_text()
    assert export_mc_attempt([record], tmp_path) == first
    retried = copy.deepcopy(record)
    retried["rollout"]["id"] = "new-uuid"
    retried["rollout"]["terminal_reward"] = 0.0
    for label in retried["labels"]:
        label["rollout_id"] = "new-uuid"
    second = export_mc_attempt([retried], tmp_path)
    assert first != second
    assert (first / "rollouts_train_regen.jsonl").read_text() == first_contents
    assert json.loads((second / "rollouts_train_regen.jsonl").read_text())["terminal_reward"] == 0.0
    assert legacy.read_text() == "old incomplete export\n"

    def interrupt(records, directory):
        directory.mkdir()
        (directory / "rollouts_train_regen.jsonl").write_text("partial")
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patcher:
        patcher.setattr("examples.delta_critic.policy_mc.export_mc_records", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            export_mc_attempt([record], tmp_path / "interrupted")
    assert list((tmp_path / "interrupted").iterdir()) == []
    assert export_mc_attempt([record], tmp_path / "interrupted").is_dir()


def test_mc_mode_requires_explicit_continuation_cap():
    with pytest.raises(ValueError, match="max_continuation_tokens"):
        online_mc_config({"rollout_mode": "selected_prefix_mc"}, {}, 0)
    assert online_mc_config({}, {}, 0) is None


@pytest.mark.parametrize("candidates", [None, [], [[{"prob": 0.5}]], [[], []]])
def test_delta_continuation_rejects_missing_or_misaligned_token_probabilities(candidates):
    class Client:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=[7, 8], extra_fields={"delta_top_logprobs": candidates})

    config = MCConfig(
        mode="prefix_only",
        continuations_per_state=1,
        sampling_config={"max_tokens": 12, "delta_top_logprobs": 20},
        actor_version="0",
        continuation_skip_special_tokens=True,
    )
    with pytest.raises(ValueError, match="token-aligned delta_top_logprobs"):
        asyncio.run(
            sample_v1_continuations(MCRequest("r", (3,), (1, 2)), config, server_client=Client(), tokenizer=None)
        )


def test_mc_total_cap_exhausted_prefix_never_calls_server():
    class Client:
        async def generate(self, **kwargs):
            raise AssertionError("No tokens remain under the total cap")

    config = MCConfig(
        mode="prefix_only",
        continuations_per_state=2,
        sampling_config={"max_tokens": 12},
        actor_version="0",
        continuation_skip_special_tokens=True,
        max_total_tokens=5,
    )
    request = MCRequest("r", (3, 4, 5), (1, 2))
    result = asyncio.run(sample_v1_continuations(request, config, server_client=Client(), tokenizer=None))
    assert [row.token_ids for row in result] == [(), ()]


def test_mc_total_cap_excludes_selected_states_without_generation_room():
    from verl.trainer.ppo.v1.agent_loop_tq import _online_delta_rollout_fields

    output = SimpleNamespace(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22, 23],
        response_mask=[1, 1, 1, 1],
        extra_fields={"delta_top_logprobs": [[{"prob": 0.5}]] * 4, "global_steps": 0},
    )
    settings = {
        "rollout_mode": "selected_prefix_mc",
        "mc": {"max_total_tokens": 4},
        "selection": {
            "states_per_response": 4,
            "min_token_gap": 0,
            "entropy_weight": 0.0,
            "low_top1_weight": 1.0,
            "final_window_tokens": 0,
            "final_window_weight": 0.0,
            "max_candidates": 1,
        },
    }
    fields = _online_delta_rollout_fields(output, settings)
    assert fields["selected_token_indices"].tolist() == [0, 1]
    assert fields["policy_token_mask"].tolist() == [1.0] * 4


class TinyScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(30, 1)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids).squeeze(-1)


def make_worker(update_config=None):
    def init(self, artifact, **kwargs):
        self.config = ScalarConfig(model_path="tiny", dtype="float32", max_length=64, gradient_checkpointing=False)
        self.checkpoint_config = self.config
        self.metadata = {
            "normalization": {"enabled": True, "mean": 0.1, "std": 0.2},
            "weights_sha256": "initial",
            "stats_version": "stats",
        }
        self.model = TinyScalar().eval().requires_grad_(False)
        self.device = torch.device("cpu")
        self.pad_token_id = 0

    with patch("examples.delta_critic.online_critic.FrozenDeltaWorker.__init__", init):
        return OnlineDeltaWorker("tiny", update_config={"learning_rate": 0.01, **(update_config or {})})


def mc_record():
    return {
        "rollout": {"id": "r", "prompt_token_ids": [1], "response_token_ids": [2, 3], "terminal_reward": 1.0},
        "states": [{"token_index": 0}, {"token_index": 1}],
        "labels": [
            {"rollout_id": "r", "token_index": 0, "v_prefix": 0.25},
            {"rollout_id": "r", "token_index": 1, "v_prefix": 0.75},
        ],
    }


def test_online_critic_updates_and_checkpoint_restores_next_update(tmp_path):
    torch.manual_seed(17)
    worker = make_worker()
    initial = worker.model.embedding.weight.detach().clone()
    metrics = worker.fit_mc([mc_record()])
    assert metrics["delta_critic/update_step"] == 1
    assert not torch.equal(initial, worker.model.embedding.weight)
    assert not any(p.requires_grad for p in worker.model.parameters())
    checkpoint = tmp_path / "critic.pt"
    worker.save_online(checkpoint, 1)
    worker.fit_mc([mc_record()])
    expected = worker.model.embedding.weight.detach().clone()
    restored = make_worker()
    restored.load_online(checkpoint, 1)
    restored.fit_mc([mc_record()])
    torch.testing.assert_close(restored.model.embedding.weight, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="mismatch"):
        restored.load_online(checkpoint, 2)


def test_online_critic_mini_batches_make_separate_optimizer_updates():
    records = []
    for index in range(4):
        record = mc_record()
        record["rollout"]["id"] = f"r{index}"
        for label in record["labels"]:
            label["rollout_id"] = f"r{index}"
        records.append(record)
    worker = make_worker({"mini_batch_size": 2, "microbatch": 1})
    metrics = worker.fit_mc(records)
    assert metrics["delta_critic/optimizer_steps"] == 2
    assert metrics["delta_critic/update_step"] == 2
    assert metrics["delta_critic/rows"] == 4


def test_online_critic_rejects_incomplete_mc_labels():
    record = mc_record()
    record["labels"].pop()
    with pytest.raises(ValueError, match="every selected state"):
        make_worker().fit_mc([record])
