# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop import AgentLoopOutput
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _DeltaExpansionCoordinator, _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_mc_quota_coordinator_releases_all_128_originals():
    async def run():
        coordinator = _DeltaExpansionCoordinator()
        protocol = {"quota_timeout_seconds": 2}
        tasks = [
            coordinator.register(
                "batch",
                index,
                {"role": "train" if index < 112 else "diagnostic", "eligible": index != 0},
                protocol,
                128,
                1,
            )
            for index in range(128)
        ]
        selected = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
        assert sum(selected) == 12
        assert not selected[0]
        assert sum(selected[112:]) == 2
        assert coordinator.batches == {}

    asyncio.run(run())


def test_mc_quota_coordinator_fails_every_waiter_when_partition_has_too_few_candidates():
    async def run():
        coordinator = _DeltaExpansionCoordinator()
        outcomes = await asyncio.gather(
            *[
                coordinator.register(
                    "batch",
                    index,
                    {"role": "train", "eligible": True},
                    {"quota_timeout_seconds": 2},
                    128,
                    1,
                )
                for index in range(128)
            ],
            return_exceptions=True,
        )
        assert all(isinstance(outcome, ValueError) for outcome in outcomes)
        assert coordinator.batches == {}

    asyncio.run(run())


@pytest.mark.parametrize("selected", [True, False])
def test_paired_expansion_preserves_actor_grid_and_emits_two_distinct_sibling_groups(monkeypatch, selected):
    output = AgentLoopOutput(
        prompt_ids=[10],
        response_ids=[20, 21, 22],
        response_mask=[1, 1, 1],
        reward_score=0.0,
        metrics={},
        extra_fields={
            "delta_top_logprobs": [[{"prob": 0.1}], [{"prob": 0.2}], [{"prob": 0.3}]],
            "global_steps": 0,
            "min_global_steps": 0,
            "max_global_steps": 0,
        },
    )

    async def noop(*args, **kwargs):
        pass

    async def register(*args):
        return selected

    worker = SimpleNamespace(
        config=OmegaConf.create(
            {
                "algorithm": {
                    "delta_policy": {
                        "enabled": True,
                        "selection": {"states_per_response": 3, "min_token_gap": 1},
                        "rollout_mode": "selected_prefix_mc",
                        "mc": {"continuations_per_state": 8, "max_response_total_tokens": 2048},
                        "expansion": {
                            "enabled": True,
                            "prompts_per_step": 128,
                            "short_response_tokens": 100,
                            "protocol": {"enabled": True, "sampling_layout": "pair"},
                        },
                    }
                },
                "actor_rollout_ref": {"rollout": {}},
            }
        ),
        _compute_score=noop,
        _compute_teacher_logprobs=noop,
        _compute_multi_modal_inputs=lambda output, input_ids: None,
        _compute_position_ids=lambda input_ids, attention_mask, multi_modal_inputs: torch.arange(
            input_ids.shape[-1]
        ).unsqueeze(0),
        _delta_expansion_coordinator=SimpleNamespace(register=SimpleNamespace(remote=register)),
        llm_client=object(),
        tokenizer=object(),
    )

    async def collect(output_arg, selected_indices, **kwargs):
        assert selected_indices == ([0, 1] if selected else [])
        return {
            "rollout": {"terminal_reward": 0.0},
            "states": [],
            "labels": [
                {
                    "prefix_response_token_ids": output.response_ids[:index],
                    "mc_continuations": [
                        {
                            "continuation_index": sibling,
                            "token_ids": [30 + sibling],
                            "reward": float(sibling % 2),
                            "delta_top_logprobs": [[{"prob": 0.3}]],
                        }
                        for sibling in range(8)
                    ],
                }
                for index in selected_indices
            ],
        }

    writes = []

    async def capture(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr("examples.delta_critic.policy_mc.collect_online_mc", collect)
    monkeypatch.setattr("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put", capture)
    asyncio.run(
        AgentLoopWorkerTQ.__ray_actor_class__._agent_loop_postprocess(
            worker,
            output,
            False,
            uid="new-uid",
            session_id=0,
            global_steps=1,
            delta_expand=False,
            delta_protocol_batch_id="batch",
            delta_protocol_index=0,
            delta_critic_role="train",
            delta_prompt_id="stable-prompt",
        )
    )
    saved = writes[0]
    assert saved["keys"] == [f"new-uid_0_{index}" for index in range(17 if selected else 1)]
    fields = saved["fields"]
    assert fields["selected_token_indices"][0].tolist() == [0, 1, 2]
    record = fields["delta_mc_record"][0]
    assert record["full_selected_indices"] == [0, 1, 2]
    assert record["valid_td_indices"] == ([0] if selected else [])
    assert record["critic_role"] == "train"
    if selected:
        assert fields["prompts"][1].tolist() == [10]
        assert fields["prompts"][9].tolist() == [10, 20]
        assert all(fields["delta_is_continuation"][index] for index in range(1, 17))


def test_original_cache_replay_uses_new_runtime_metadata(tmp_path):
    from examples.delta_critic.online_protocol import atomic_json, original_cache_contract

    protocol = {"enabled": True, "original_cache_mode": "read", "original_cache_dir": str(tmp_path)}
    config = OmegaConf.create(
        {
            "algorithm": {"delta_policy": {"expansion": {"protocol": protocol}}},
            "actor_rollout_ref": {
                "model": {"path": "same-base"},
                "rollout": {"prompt_length": 1024, "response_length": 2048},
            },
        }
    )
    sampling = {"temperature": 1.0}
    output = AgentLoopOutput(
        prompt_ids=[10],
        response_ids=[20],
        response_mask=[1],
        metrics={},
        extra_fields={"global_steps": 0, "delta_top_logprobs": [[{"prob": 0.4}]]},
    )
    atomic_json(
        tmp_path / "stable_session_0.json",
        {
            "contract": original_cache_contract(config, sampling, "stable"),
            "output": output.model_dump(mode="json"),
        },
    )
    observed = []

    async def capture(cached, validate, **kwargs):
        observed.append((cached, kwargs))

    worker = SimpleNamespace(config=config, _agent_loop_postprocess=capture)
    asyncio.run(
        AgentLoopWorkerTQ.__ray_actor_class__._run_agent_loop(
            worker,
            sampling,
            {"validate": False},
            agent_name="single_turn_agent",
            global_steps=1,
            uid="reader-uid",
            session_id=0,
            delta_prompt_id="stable",
            delta_critic_role="diagnostic",
        )
    )
    assert observed[0][0].response_ids == [20]
    assert observed[0][1]["uid"] == "reader-uid"
    assert observed[0][1]["delta_critic_role"] == "diagnostic"
