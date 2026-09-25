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
"""Exercise source-style batch sampling through the V1 client boundary."""

import asyncio
import importlib.util
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
