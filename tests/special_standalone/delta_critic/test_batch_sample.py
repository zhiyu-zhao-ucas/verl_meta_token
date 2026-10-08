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
"""Exercise V1 prompt loading and delta sampling through the V1 client boundary."""

import ast
import asyncio
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.delta_critic.batch_sample import (
    _recover_partial_outputs,
    collect_labeled_rollouts,
    sample_labeled_rollout,
    sampling_settings,
)
from examples.delta_critic.mc_labeling import MCRequest
from examples.delta_critic.sampling_io import normalize_prompt_row, score_text
from examples.delta_critic.v1_sampler import sample_v1_continuations

CONFIG_PATH = Path("examples/delta_critic/config_sampling_qwen3_8b.yaml")

# Output-recording switches this migration adds on top of the source sampling
# config. They decide what the collector persists, not how it samples, so the
# source parity check tolerates exactly these keys and rejects any other drift.
LOCAL_ONLY_CONFIG_KEYS = {"mc": {"save_continuations", "save_individual_rewards"}}


def _config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["smoke"]["states_per_response"] = 2
    config["state_selection"]["min_token_gap"] = 0
    config["mc"]["continuations_per_state"] = 2
    config["reward"]["method"] = "exact_or_numeric"
    return config


def test_batch_sample_v1_uncertainty_to_paired_mc():
    config = _config()
    generation, mc_config = sampling_settings(config, "paired_next_state")
    assert generation["repetition_penalty"] == 1.05
    assert generation["stop"] == ["\n\n---\n\n"]
    assert generation["include_stop_str_in_output"] is False
    mc_config = replace(mc_config, actor_version="actor-1")
    prompt = {"prompt_id": "p1", "prompt": "Problem", "gold_answer": "2"}
    output = SimpleNamespace(
        token_ids=[20, 21],
        log_probs=[-0.3, -0.8],
        stop_reason="completed",
        extra_fields={
            "finish_reason": "stop",
            "delta_top_logprobs": [
                [
                    {"token_id": 20, "prob": 0.3, "logprob": -0.3, "token": "1"},
                    {"token_id": 99, "prob": 0.0, "logprob": float("-inf"), "token": None},
                ],
                [{"token_id": 21, "prob": 0.7, "logprob": -0.8, "token": "2"}],
            ],
        },
    )

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return "".join({20: "1", 21: "2", 30: "2"}[item] for item in ids)

    class Client:
        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                token_ids=[30], extra_fields={"delta_top_logprobs": [[{"token_id": 30, "prob": 0.7}]]}
            )

    client = Client()
    rollout, states, labels = asyncio.run(
        sample_labeled_rollout(
            prompt,
            [10],
            output,
            tokenizer=Tokenizer(),
            client=client,
            config=config,
            split="train",
            response_index=0,
            actor_version="actor-1",
            generation=generation,
            mc_config=mc_config,
            mc_concurrency=2,
        )
    )
    assert rollout["id"] == "train-p1-0"
    assert rollout["response"] == "12"
    assert rollout["behavior_logprobs"] == [-0.3, -0.8]
    assert len(rollout["tokens"][0]["top_candidates"]) == 1
    assert [state["token_index"] for state in states] == [0, 1]
    assert [label["mc_num_samples"] for label in labels] == [2, 2]
    assert labels[1]["v_next"] == rollout["terminal_reward"]
    assert len(client.calls) == 4  # two unique prefixes, two samples each
    assert all(call["sampling_params"] == mc_config.sampling_config for call in client.calls)


def test_v1_mc_continuations_keep_sample_order_under_global_limit():
    config = _config()
    config["mc"]["continuations_per_state"] = 4
    _, mc_config = sampling_settings(config, "prefix_only")
    request = MCRequest("rollout-1", (20,), (10,))

    class Client:
        def __init__(self):
            self.active = 0
            self.peak = 0

        async def generate(self, **kwargs):
            index = int(kwargs["request_id"].rsplit(":", 1)[1])
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep((4 - index) * 0.001)
            self.active -= 1
            return SimpleNamespace(
                token_ids=[index], extra_fields={"delta_top_logprobs": [[{"token_id": index, "prob": 0.7}]]}
            )

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return str(ids[0])

    client = Client()
    rows = asyncio.run(
        sample_v1_continuations(
            request,
            mc_config,
            server_client=client,
            tokenizer=Tokenizer(),
            request_semaphore=asyncio.Semaphore(2),
        )
    )
    assert [row.token_ids for row in rows] == [(0,), (1,), (2,), (3,)]
    assert client.peak == 2


def test_v1_mc_continuations_cancel_other_requests_on_error():
    config = _config()
    config["mc"]["continuations_per_state"] = 3
    _, mc_config = sampling_settings(config, "prefix_only")
    request = MCRequest("rollout-1", (), (10,))

    class Client:
        def __init__(self):
            self.cancelled = 0

        async def generate(self, **kwargs):
            index = int(kwargs["request_id"].rsplit(":", 1)[1])
            if index == 0:
                await asyncio.sleep(0)
                raise RuntimeError("server failed")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

    client = Client()
    with pytest.raises(RuntimeError, match="server failed"):
        asyncio.run(
            sample_v1_continuations(
                request,
                mc_config,
                server_client=client,
                tokenizer=None,
            )
        )
    assert client.cancelled == 2


def test_v1_native_mc_outputs_follow_completion_indices():
    path = Path("verl/workers/rollout/vllm_rollout/vllm_async_server.py")
    module = ast.parse(path.read_text())
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_collect_delta_mc_token_ids"
    )
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    collect = namespace["_collect_delta_mc_token_ids"]

    outputs = [SimpleNamespace(index=1, token_ids=[21]), SimpleNamespace(index=0, token_ids=[20])]
    assert collect(outputs, 2) == [[20], [21]]
    with pytest.raises(ValueError, match="incomplete"):
        collect(outputs[:1], 2)


@pytest.mark.parametrize(
    ("mc_sampling_mode", "batch_size", "expected_peak"),
    [
        ("requests", None, 2),
        ("native", None, 2),
        ("native", 1, 2),
    ],
)
def test_collect_labeled_rollouts_overlaps_rollout_and_mc_with_global_limit(
    mc_sampling_mode, batch_size, expected_peak
):
    config = _config()
    config["smoke"]["states_per_response"] = 1
    config["mc"]["continuations_per_state"] = 3
    generation, mc_config = sampling_settings(config, "prefix_only")
    mc_config = replace(mc_config, actor_version="actor-1")
    prompts = [
        {"prompt_id": str(i), "prompt": f"Prompt {i}", "prompt_token_ids": [10 + i], "gold_answer": "2"}
        for i in range(3)
    ]

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return "".join({20: "1", 30: "2"}[item] for item in ids)

    class Client:
        def __init__(self):
            self.mc_started = asyncio.Event()
            self.active_mc = 0
            self.peak_mc = 0
            self.mc_counts = []

        async def generate(self, **kwargs):
            if kwargs["request_id"].startswith("delta-rollout:"):
                if kwargs["request_id"].endswith(":1:0"):
                    await self.mc_started.wait()
                return SimpleNamespace(
                    token_ids=[20],
                    log_probs=[-0.3],
                    stop_reason="completed",
                    extra_fields={
                        "finish_reason": "stop",
                        "delta_top_logprobs": [[{"token_id": 20, "prob": 0.7, "logprob": -0.3, "token": "1"}]],
                    },
                )
            self.mc_started.set()
            self.active_mc += 1
            self.peak_mc = max(self.peak_mc, self.active_mc)
            count = kwargs["sampling_params"].get("delta_mc_n", 1)
            self.mc_counts.append(count)
            await asyncio.sleep(0.002)
            self.active_mc -= 1
            return SimpleNamespace(
                token_ids=[30],
                extra_fields={"delta_top_logprobs": [[{"token_id": 30, "prob": 0.7}]]},
            )

    client = Client()
    collected = []
    streamed_rollouts = []
    streamed_labels = []
    asyncio.run(
        asyncio.wait_for(
            collect_labeled_rollouts(
                prompts,
                tokenizer=Tokenizer(),
                client=client,
                config=config,
                split="train",
                actor_version="actor-1",
                generation=generation,
                mc_config=mc_config,
                rollout_concurrency=2,
                mc_concurrency=2,
                mc_sampling_mode=mc_sampling_mode,
                mc_native_batch_size=batch_size,
                emit=lambda rollout, states, labels: collected.append((rollout, states, labels)),
                emit_rollout=lambda rollout, states: streamed_rollouts.append((rollout, states)),
                emit_label=lambda rollout, label: streamed_labels.append((rollout, label)),
            ),
            timeout=2,
        )
    )
    assert {result[0]["prompt_id"] for result in collected} == {"0", "1", "2"}
    assert all(len(result[2]) == 1 for result in collected)
    assert len(streamed_rollouts) == 3
    assert len(streamed_labels) == 3
    assert client.peak_mc == expected_peak
    expected_counts = [1] * 9
    assert sorted(client.mc_counts) == expected_counts


def test_partial_output_recovery_keeps_only_committed_states(tmp_path):
    temporary = {
        name: tmp_path / f".{name}.jsonl.incomplete" for name in ("rollouts", "states", "mc_labels", "continuations")
    }
    temporary["rollouts"].write_text(json.dumps({"id": "train-p1-0"}) + "\n")
    temporary["states"].write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"rollout_id": "train-p1-0", "state_id": "s1"},
                {"rollout_id": "train-p1-0", "state_id": "s2"},
            )
        )
    )
    temporary["mc_labels"].write_text(json.dumps({"state_id": "s1", "mc_num_samples": 2}) + "\n")
    temporary["continuations"].write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"state_id": "s1", "continuation_index": 0},
                {"state_id": "s1", "continuation_index": 1},
                {"state_id": "s2", "continuation_index": 0},
            )
        )
        + '{"truncated"'
    )

    resume, counts = _recover_partial_outputs(
        temporary,
        continuations_per_state=2,
        save_continuations=True,
        mc_mode="prefix_only",
    )

    assert counts == {"rollouts": 1, "states": 2, "mc_labels": 1, "continuations": 2}
    assert [row["state_id"] for row in resume["train-p1-0"]["labels"]] == ["s1"]
    continuation_rows = [json.loads(line) for line in temporary["continuations"].read_text().splitlines()]
    assert {row["state_id"] for row in continuation_rows} == {"s1"}

    temporary["continuations"].write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"state_id": "s1", "continuation_index": 0},
                {"state_id": "s1", "continuation_index": 2},
            )
        )
    )
    resume, counts = _recover_partial_outputs(
        temporary,
        continuations_per_state=2,
        save_continuations=True,
        mc_mode="prefix_only",
    )
    assert counts["mc_labels"] == 0
    assert resume["train-p1-0"]["labels"] == []


def test_v1_prompt_loader_uses_trainer_batches_and_chat_template(tmp_path):
    pytest.importorskip("torchdata")
    pytest.importorskip("datasets")
    from omegaconf import OmegaConf

    from examples.delta_critic.v1_prompts import load_v1_prompts

    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": f"q{i}"}],
                    "reward_model": {"ground_truth": str(i)},
                    "data_source": "fixture",
                }
            )
            + "\n"
            for i in range(3)
        )
    )

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            assert tokenize and add_generation_prompt
            assert kwargs["enable_thinking"] is False
            return [101] + [ord(char) for char in messages[0]["content"]] + [102]

        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return ",".join(map(str, ids))

    config = OmegaConf.create(
        {
            "data": {
                "train_files": str(path),
                "val_files": str(path),
                "train_batch_size": 2,
                "val_batch_size": 2,
                "dataloader_num_workers": 0,
                "shuffle": False,
                "validation_shuffle": False,
                "seed": 42,
                "prompt_key": "prompt",
                "filter_overlong_prompts": False,
                "apply_chat_template_kwargs": {"enable_thinking": False},
            },
            "actor_rollout_ref": {"rollout": {"prompt_length": 3}},
            "algorithm": {"filter_groups": {"enable": False}},
            "trainer": {"v1": {"trainer_mode": "sync", "sampler": {"sync_refill_failed_groups": False}}},
        }
    )
    tokenizer = Tokenizer()
    train = load_v1_prompts(config, tokenizer=tokenizer, processor=None, hf_model_type=None, split="train")
    validation = load_v1_prompts(config, tokenizer=tokenizer, processor=None, hf_model_type=None, split="test")
    assert [row["gold_answer"] for row in train] == ["0", "1"]  # V1 train drop_last
    assert [row["gold_answer"] for row in validation] == ["0", "1", "2"]
    assert train[0]["raw_prompt"] == [{"role": "user", "content": "q0"}]
    assert train[0]["prompt_token_ids"] == [ord("q"), ord("0"), 102]  # V1 left cap
    assert train[0]["prompt"] == "113,48,102"

    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    async def agent_loop_ids(messages):
        loop = object.__new__(SingleTurnAgentLoop)
        loop.loop = asyncio.get_running_loop()
        loop.rollout_config = config.actor_rollout_ref.rollout
        loop.continuous_token_builder = create_continuous_token_builder(
            tokenizer,
            hf_model_type=None,
            chat_template_kwargs={"enable_thinking": False},
        )
        return await loop.ct_build_initial_tokens(messages)

    assert train[0]["prompt_token_ids"] == asyncio.run(agent_loop_ids(train[0]["raw_prompt"]))


def test_source_prompt_and_math_reward_parity():
    root = os.environ.get("VALUE_MODEL_ROOT")
    if not root:
        pytest.skip("Set VALUE_MODEL_ROOT for source differential test")
    source_dir = Path(root) / "delta_value_llm_exp"
    source_config = yaml.safe_load((source_dir / "config_qwen3_8b_densecritic_aligned_sweeps.yaml").read_text())
    local_config = yaml.safe_load(CONFIG_PATH.read_text())
    for section in ("model", "data", "prompt", "reward", "smoke", "state_selection", "mc"):
        # Local configs cover the sampling-relevant subset of the source
        # config, so only the direction "every local key is a source key" is
        # checked; omitted source keys belong to the critic training wrapper.
        local_only = set(local_config[section]) - set(source_config[section])
        assert local_only == LOCAL_ONLY_CONFIG_KEYS.get(section, set()), (
            f"Unexpected local-only keys in {section}: {sorted(local_only)}"
        )
        for key, value in local_config[section].items():
            if key in LOCAL_ONLY_CONFIG_KEYS.get(section, set()):
                continue
            assert value == source_config[section][key]
    for key, value in local_config["rollout"].items():
        assert value == source_config["rollout"][key]
    common_spec = importlib.util.spec_from_file_location("source_delta_common", source_dir / "common.py")
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    reward_spec = importlib.util.spec_from_file_location("source_delta_reward", source_dir / "reward.py")
    reward = importlib.util.module_from_spec(reward_spec)
    reward_spec.loader.exec_module(reward)
    for row in (
        {"id": "a", "problem": "1 + 1", "answer": "2"},
        {"unique_id": "b", "prompt": "Compute 1 + 1", "gold_answer": "2"},
    ):
        assert normalize_prompt_row(row, 3) == common.normalize_prompt_row(row, 3)
    for response, gold in (
        (r"The answer is \boxed{2}.", "2"),
        (r"1 + 1 = \boxed{3}.", "2"),
        ("There is no answer.", "2"),
    ):
        config = {"method": "math_verify", "fallback_on_import_error": False}
        assert score_text(response, gold, config) == reward.score_text(response, gold, {"reward": config})


def test_normalize_official_dapo_math_row():
    row = {
        "prompt": [{"role": "user", "content": "Solve this problem.\n\n1 + 1 = ?"}],
        "reward_model": {"ground_truth": "2", "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": "dapo-example"},
        "data_source": "math_dapo",
    }

    normalized = normalize_prompt_row(row, 0)

    assert normalized == {
        "prompt_id": "dapo-example",
        "prompt": "Solve this problem.\n\n1 + 1 = ?",
        "gold_answer": "2",
    }
