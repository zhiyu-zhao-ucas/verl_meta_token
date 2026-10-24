# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.vineppo.config import configure_vineppo
from verl.trainer.ppo.vineppo.core import microbatch_loss
from verl.utils import tensordict_utils as tu


def config(overrides=()):
    path = Path(__file__).resolve().parents[4] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(path), version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=vineppo",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                *overrides,
            ],
        )


def test_standard_config_enables_reference_and_disables_critic():
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass

    cfg = config()
    configure_vineppo(cfg)
    assert not need_critic(cfg) and need_reference_policy(cfg)
    assert cfg.algorithm.rollout_correction is None
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    assert actor.vineppo and actor.kl_loss_coef == 0.0001
    assert omega_conf_to_dataclass(cfg.algorithm).vineppo.mc_samples == 9


@pytest.mark.parametrize(
    "override",
    [
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "algorithm.gamma=0.9",
        "algorithm.use_kl_in_reward=true",
        "trainer.v1.trainer_mode=fully_async",
        "actor_rollout_ref.rollout.name=sglang",
        "reward.reward_manager.name=limited",
        "actor_rollout_ref.actor.entropy_coeff=0.01",
        "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
    ],
)
def test_unsupported_semantics_rejected_before_launch(override):
    with pytest.raises(ValueError, match="VinePPO requires"):
        configure_vineppo(config([override]))


def test_other_algorithms_keep_their_config():
    cfg = config(["algorithm.adv_estimator=grpo"])
    before = OmegaConf.to_container(cfg)
    configure_vineppo(cfg)
    assert before == OmegaConf.to_container(cfg)


def test_registry_dispatch_does_not_whiten_at_rollout():
    from verl.trainer.ppo.ray_trainer import compute_advantage

    cfg = config(["algorithm.vineppo.mc_samples=2"])
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.tensor([[0.0, 1.0]]),
                "response_mask": torch.ones(1, 2),
                "vineppo_step_ids": torch.tensor([[0, 1]]),
                "vineppo_values": torch.tensor([[0.25, 0.5]]),
                "vineppo_mc_counts": torch.tensor([[2, 2]]),
                "vineppo_num_steps": torch.tensor([2]),
            },
            batch_size=[1],
        )
    )
    result = compute_advantage(data, "vineppo", config=cfg.algorithm)
    torch.testing.assert_close(result.batch["advantages"], torch.tensor([[0.25, 0.5]]))


def test_actor_loss_adapter_nested_inputs_and_real_accumulation_metadata():
    from verl.workers.config import ActorConfig
    from verl.workers.utils.losses import ppo_loss

    nested = lambda rows: torch.nested.as_nested_tensor(
        [torch.tensor(row, dtype=torch.float32) for row in rows], layout=torch.jagged
    )
    old = [[0.0, 0.0], [0.0]]
    adv = [[1.0, 3.0], [4.0]]
    data = TensorDict(
        {
            "prompts": nested([[1], [1]]),
            "responses": nested([[2, 3], [2]]),
            "response_mask": nested([[1, 1], [1]]),
            "old_log_probs": nested(old),
            "ref_log_prob": nested(old),
            "advantages": nested(adv),
        },
        batch_size=[2],
    )
    # Full-sequence log probabilities are shifted by the real padding adapter.
    logp = torch.tensor([0.1, 0.2, 0.0, 0.3, 0.0], requires_grad=True)
    tu.assign_non_tensor(data, num_micro_batches=3)
    cfg = ActorConfig(
        strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1, vineppo=True, use_kl_loss=True, kl_loss_coef=0.1
    )
    loss, _ = ppo_loss(cfg, {"log_probs": logp}, data)
    expected, _ = microbatch_loss(
        torch.zeros(2, 2),
        torch.stack([logp[:2], torch.stack([logp[3], logp[4]])]),
        torch.tensor([[1.0, 3.0], [4.0, 0.0]]),
        torch.tensor([[1, 1], [1, 0]]),
        ref_log_prob=torch.zeros(2, 2),
        kl_coef=0.1,
        num_micro_batches=3,
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logp.grad).all()
    del data["num_micro_batches"]
    with pytest.raises(ValueError, match="accumulation count"):
        ppo_loss(cfg, {"log_probs": logp}, data)


def test_failed_session_aborts_sampler_and_default_behavior_is_unchanged(monkeypatch):
    from verl.trainer.ppo.v1 import replay_buffer as module

    monkeypatch.setattr(
        module.tq,
        "kv_list",
        lambda: {
            "train": {
                "p": {
                    "is_prompt": True,
                    "global_steps": 1,
                    "status": "failure",
                }
            }
        },
    )
    args = dict(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=1,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
    )
    with pytest.raises(RuntimeError, match="aborting the training round"):
        module.ReplayBuffer(**args, fail_on_generation_error=True)._sync_metadata_from_transfer_queue()
    normal = module.ReplayBuffer(**args)
    normal._sync_metadata_from_transfer_queue()
    assert normal.failure_keys["train"] == {"p"}


def test_session_publication_waits_for_mc_and_controller_reads_all_metadata(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.trainer.ppo.vineppo.rollout import VinePPOCollector

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)

        def __call__(self, text, **kwargs):
            return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    cfg = config(["algorithm.vineppo.mc_samples=2", "algorithm.vineppo.splitter=gsm8k"])
    configure_vineppo(cfg)
    writes = []

    async def run():
        release = asyncio.Event()
        calls = 0

        async def generate(**kwargs):
            nonlocal calls
            calls += 1
            await release.wait()
            return SimpleNamespace(token_ids=[ord("x")], extra_fields={"global_steps": 0, "finish_reason": "stop"})

        async def score(*args):
            return 0.5

        async def noop(*args, **kwargs):
            pass

        async def capture(**kwargs):
            writes.append(kwargs)

        monkeypatch.setattr("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put", capture)
        from verl.utils.config import omega_conf_to_dataclass

        worker = SimpleNamespace(
            config=cfg,
            _vineppo_collector=VinePPOCollector(
                omega_conf_to_dataclass(cfg.algorithm.vineppo),
                Tokenizer(),
                generate,
                score,
                context_length=100,
                sampling_params={},
            ),
            _compute_score=noop,
            _compute_teacher_logprobs=noop,
            _compute_multi_modal_inputs=lambda *args: None,
            _compute_position_ids=lambda ids, *args: torch.arange(ids.shape[-1]).unsqueeze(0),
        )
        output = AgentLoopOutput(
            prompt_ids=[ord("Q")],
            response_ids=[ord("a"), ord("\n"), ord("b"), 0],
            response_mask=[1] * 4,
            reward_score=1.0,
            num_turns=2,
            metrics={},
            extra_fields={"global_steps": 0, "finish_reason": "stop"},
        )
        task = asyncio.create_task(
            AgentLoopWorkerTQ.__ray_actor_class__._agent_loop_postprocess(
                worker,
                output,
                False,
                uid="row",
                session_id=0,
                global_steps=1,
                index=0,
            )
        )
        while calls < 1:
            await asyncio.sleep(0)
        assert writes == []
        release.set()
        await task
        assert calls == 4 and len(writes) == 1
        assert writes[0]["keys"] == ["row_0_0"]  # MC never produces actor rows.
        assert writes[0]["fields"]["responses"][0].tolist() == output.response_ids

    asyncio.run(run())
    saved = writes[0]

    class Batch:
        keys = saved["keys"]
        tags = saved["tags"]
        partition_id = "train"

        def __len__(self):
            return 1

    selected = []

    def get(**kwargs):
        selected.extend(kwargs["select_fields"])
        result = saved["fields"].select(*kwargs["select_fields"], strict=False)
        result["response_mask"] = torch.nested.as_nested_tensor(
            list(result["response_mask"].unbind()), layout=torch.jagged
        )
        return result

    results = []

    def put(**kwargs):
        results.append(kwargs["fields"])
        return Batch()

    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", get)
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put", put)
    trainer = SimpleNamespace(config=cfg, delta_policy_config=None, global_steps=1)
    metrics = {}
    PPOTrainer._compute_advantage(trainer, Batch(), metrics)
    assert "vineppo_values" in selected
    assert metrics["vineppo/mc_samples"] == 4
    assert metrics["vineppo/mc_completion_rate"] == 1
    torch.testing.assert_close(results[0]["advantages"][0], torch.tensor([0.0, 0.5, 0.5, 0.5]))
    trainer.global_steps = 2
    with pytest.raises(ValueError, match="different actor version"):
        PPOTrainer._compute_advantage(trainer, Batch(), {})


@pytest.mark.parametrize("fallback", [0.0, None])
def test_main_verifier_error_is_graded_incorrect_or_aborts(monkeypatch, fallback):
    import asyncio
    from types import SimpleNamespace

    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
    from verl.trainer.ppo.vineppo.rollout import VinePPOCollector
    from verl.utils.config import omega_conf_to_dataclass

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)

        def __call__(self, text, **kwargs):
            return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    cfg = config(
        [
            "algorithm.vineppo.mc_samples=2",
            f"algorithm.vineppo.score_error_reward={'null' if fallback is None else fallback}",
        ]
    )
    configure_vineppo(cfg)
    writes = []

    async def generate(**kwargs):
        return SimpleNamespace(token_ids=[ord("x"), 0], extra_fields={"global_steps": 0, "finish_reason": "stop"})

    async def score(*args):
        return 1.0

    async def failing_score(*args, **kwargs):
        raise RuntimeError("math-verify could not parse the nonempty gold answer")

    async def noop(*args, **kwargs):
        pass

    async def capture(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put", capture)
    worker = SimpleNamespace(
        config=cfg,
        _vineppo_collector=VinePPOCollector(
            omega_conf_to_dataclass(cfg.algorithm.vineppo),
            Tokenizer(),
            generate,
            score,
            context_length=100,
            sampling_params={},
        ),
        _compute_score=failing_score,
        _compute_teacher_logprobs=noop,
        _compute_multi_modal_inputs=lambda *args: None,
        _compute_position_ids=lambda ids, *args: torch.arange(ids.shape[-1]).unsqueeze(0),
    )
    output = AgentLoopOutput(
        prompt_ids=[ord("Q")],
        response_ids=[ord("a"), 0],
        response_mask=[1] * 2,
        num_turns=2,
        metrics={},
        extra_fields={"global_steps": 0, "finish_reason": "stop"},
    )
    postprocess = AgentLoopWorkerTQ.__ray_actor_class__._agent_loop_postprocess(
        worker, output, False, uid="row", session_id=0, global_steps=1, index=0
    )
    if fallback is None:
        with pytest.raises(RuntimeError, match="gold answer"):
            asyncio.run(postprocess)
        assert writes == []
        return
    asyncio.run(postprocess)
    fields = writes[0]["fields"]
    assert fields["rm_scores"][0].tolist() == [0.0, 0.0]
    assert bool(fields["vineppo_main_score_error"][0]) and fields["vineppo_values"][0].tolist() == [1.0]


@pytest.mark.parametrize("validate,limit,expected", [(False, 4096, 4096), (True, 4096, None), (False, None, None)])
def test_training_response_limit_preserves_evaluation(monkeypatch, validate, limit, expected):
    import asyncio
    from types import SimpleNamespace

    from tensordict import NonTensorData

    from verl.trainer.ppo.v1 import agent_loop_tq

    async def run():
        cfg = config(
            [
                "actor_rollout_ref.rollout.response_length=10240",
                f"actor_rollout_ref.rollout.train_response_length={'null' if limit is None else limit}",
            ]
        )
        captured = []

        async def trajectory_info(*args):
            return [{}]

        async def capture(prompt, sampling_params, **kwargs):
            captured.append(dict(sampling_params))

        monkeypatch.setattr(agent_loop_tq, "get_trajectory_info", trajectory_info)
        worker = SimpleNamespace(config=cfg, background_tasks=set(), _run_prompt=capture)
        batch = TensorDict(
            {"global_steps": torch.tensor([1]), "index": torch.tensor([0]), "validate": NonTensorData(validate)},
            batch_size=[1],
        )
        await agent_loop_tq.AgentLoopWorkerTQ.__ray_actor_class__.generate_sequences(worker, batch)
        await asyncio.gather(*list(worker.background_tasks))
        assert len(captured) == 1
        assert captured[0].get("max_tokens") == expected

    asyncio.run(run())
