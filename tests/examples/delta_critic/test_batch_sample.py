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

import asyncio
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.delta_critic.batch_sample import sample_labeled_rollout, sampling_settings
from examples.delta_critic.sampling_io import normalize_prompt_row, score_text


CONFIG_PATH = Path("examples/delta_critic/config_sampling_qwen3_8b.yaml")


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
    mc_config = replace(mc_config, actor_version="actor-1")
    prompt = {"prompt_id": "p1", "prompt": "Problem", "gold_answer": "2"}
    output = SimpleNamespace(
        token_ids=[20, 21], log_probs=[-0.3, -0.8], stop_reason="completed",
        extra_fields={
            "finish_reason": "stop",
            "delta_top_logprobs": [
                [{"token_id": 20, "prob": 0.3, "logprob": -0.3, "token": "1"}],
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
            return SimpleNamespace(token_ids=[30])

    client = Client()
    rollout, states, labels = asyncio.run(sample_labeled_rollout(
        prompt, [10], output, tokenizer=Tokenizer(), client=client,
        config=config, split="train", response_index=0, actor_version="actor-1",
        generation=generation, mc_config=mc_config, mc_concurrency=2,
    ))
    assert rollout["id"] == "train-p1-0"
    assert rollout["response"] == "12"
    assert rollout["behavior_logprobs"] == [-0.3, -0.8]
    assert [state["token_index"] for state in states] == [0, 1]
    assert [label["mc_num_samples"] for label in labels] == [2, 2]
    assert labels[1]["v_next"] == rollout["terminal_reward"]
    assert len(client.calls) == 4  # two unique prefixes, two samples each
    assert all(call["sampling_params"] == mc_config.sampling_config for call in client.calls)


def test_v1_prompt_loader_uses_trainer_batches_and_chat_template(tmp_path):
    pytest.importorskip("torchdata")
    pytest.importorskip("datasets")
    from omegaconf import OmegaConf

    from examples.delta_critic.v1_prompts import load_v1_prompts

    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(
        json.dumps({
            "prompt": [{"role": "user", "content": f"q{i}"}],
            "reward_model": {"ground_truth": str(i)},
            "data_source": "fixture",
        }) + "\n" for i in range(3)
    ))

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            assert tokenize and add_generation_prompt
            assert kwargs["enable_thinking"] is False
            return [101] + [ord(char) for char in messages[0]["content"]] + [102]

        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return ",".join(map(str, ids))

    config = OmegaConf.create({
        "data": {
            "train_files": str(path), "val_files": str(path), "train_batch_size": 2,
            "val_batch_size": 2, "dataloader_num_workers": 0,
            "shuffle": False, "validation_shuffle": False, "seed": 42,
            "prompt_key": "prompt", "filter_overlong_prompts": False,
            "apply_chat_template_kwargs": {"enable_thinking": False},
        },
        "actor_rollout_ref": {"rollout": {"prompt_length": 3}},
        "algorithm": {"filter_groups": {"enable": False}},
        "trainer": {"v1": {"trainer_mode": "sync", "sampler": {"sync_refill_failed_groups": False}}},
    })
    tokenizer = Tokenizer()
    train = load_v1_prompts(config, tokenizer=tokenizer, processor=None,
                            hf_model_type=None, split="train")
    validation = load_v1_prompts(config, tokenizer=tokenizer, processor=None,
                                 hf_model_type=None, split="test")
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
            tokenizer, hf_model_type=None, chat_template_kwargs={"enable_thinking": False},
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
        for key, value in local_config[section].items():
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
