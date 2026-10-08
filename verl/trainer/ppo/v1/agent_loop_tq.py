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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import json
import logging
import os
from types import SimpleNamespace
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _online_delta_policy_settings(config) -> dict[str, Any] | None:
    policy = config.get("algorithm", {}).get("delta_policy", {})
    if not policy or not policy.get("enabled", False):
        return None
    selection = policy.get("selection", {})
    return {
        "selection": {
            "states_per_response": int(selection.get("states_per_response", 64)),
            "min_token_gap": int(selection.get("min_token_gap", 32)),
            "entropy_weight": float(selection.get("entropy_weight", 0.0)),
            "low_top1_weight": float(selection.get("low_top1_weight", 1.0)),
            "final_window_tokens": int(selection.get("final_window_tokens", 64)),
            "final_window_weight": float(selection.get("final_window_weight", 0.0)),
            "max_candidates": int(selection.get("max_candidates", 5)),
            "top_k_logprobs": int(selection.get("top_k_logprobs", 20)),
        }
    }


def _online_delta_rollout_fields(output, settings: dict[str, Any]) -> dict[str, Any]:
    """Validate merged-token alignment and expose the rollout contract as TQ fields."""
    from examples.delta_critic.policy_online import select_uncertainty_indices

    extra_fields = output.extra_fields
    candidates = extra_fields.get("delta_top_logprobs")
    if candidates is None:
        raise ValueError("Online delta rollout is missing vLLM delta_top_logprobs")
    if len(candidates) != len(output.response_ids):
        raise ValueError(
            "vLLM delta_top_logprobs rows do not align with response tokens after continuous-token merging"
        )
    selection = settings["selection"]
    selection_mask = list(output.response_mask)
    if settings.get("rollout_mode", "critic_prediction") == "selected_prefix_mc":
        total_cap = settings.get("mc", {}).get("max_total_tokens")
        response_cap = settings.get("mc", {}).get("max_response_total_tokens")
        if total_cap is not None:
            # A state at response index i uses prompt + response[:i] as its
            # generation prefix. Keep only states with room for a token.
            selection_mask = [
                allowed if len(output.prompt_ids) + index < total_cap else 0
                for index, allowed in enumerate(selection_mask)
            ]
        if response_cap is not None:
            reserve = settings.get("mc", {}).get("min_continuation_room", 1)
            selection_mask = [
                allowed if index <= response_cap - reserve else 0 for index, allowed in enumerate(selection_mask)
            ]
    selected = select_uncertainty_indices(
        candidates,
        selection_mask,
        states_per_response=selection["states_per_response"],
        min_token_gap=selection["min_token_gap"],
        entropy_weight=selection["entropy_weight"],
        low_top1_weight=selection["low_top1_weight"],
        final_window_tokens=selection["final_window_tokens"],
        final_window_weight=selection["final_window_weight"],
        max_candidates=selection["max_candidates"],
    )
    version = extra_fields.get("global_steps")
    min_version = extra_fields.get("min_global_steps", version)
    max_version = extra_fields.get("max_global_steps", version)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Online delta rollout is missing an integer actor global_steps version")
    if min_version != version or max_version != version:
        raise ValueError("Online delta policy requires one actor version across every generated response")
    return {
        "delta_top_logprobs": candidates,
        "selected_token_indices": torch.tensor(selected, dtype=torch.int64),
        "policy_token_mask": torch.tensor(output.response_mask, dtype=torch.float32),
        "rollout_actor_version": torch.tensor(version, dtype=torch.int64),
    }


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


class _DeltaExpansionCoordinator:
    """Wait for all original eligibility results before assigning fixed MC quotas.

    Worker generate_sequences launches every original as a background asyncio
    task before returning. Waiting here therefore cannot occupy the worker RPC
    slots needed to start the remaining original generations.
    """

    def __init__(self):
        self.batches = {}

    async def register(self, batch_id, index, entry, protocol, prompt_count, policy_step):
        from examples.delta_critic.online_protocol import choose_expansion_prompts

        state = self.batches.setdefault(
            batch_id, {"entries": {}, "event": asyncio.Event(), "error": None, "released": 0}
        )
        index = int(index)
        if index in state["entries"]:
            state["error"] = "Duplicate original registered with MC quota coordinator"
            state["event"].set()
        state["entries"][index] = entry
        if len(state["entries"]) == prompt_count and not state["event"].is_set():
            try:
                state["selected"] = choose_expansion_prompts(
                    state["entries"], protocol, prompt_count=prompt_count, policy_step=policy_step
                )
            except Exception as error:
                state["error"] = str(error)
            state["event"].set()
        try:
            await asyncio.wait_for(state["event"].wait(), timeout=float(protocol.get("quota_timeout_seconds", 1800)))
            if state["error"]:
                raise ValueError(state["error"])
            return index in state["selected"]
        finally:
            state["released"] += 1
            if state["released"] == prompt_count:
                self.batches.pop(batch_id, None)


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    def set_delta_expansion_coordinator(self, coordinator):
        self._delta_expansion_coordinator = coordinator

    async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        protocol = self.config.algorithm.get("delta_policy", {}).get("expansion", {}).get("protocol", {})
        if protocol.get("enabled", False) and not trajectory["validate"] and int(kwargs["global_steps"]) == 1:
            from examples.delta_critic.online_protocol import first_original_cache_path, original_cache_contract

            path = first_original_cache_path(protocol, kwargs["delta_prompt_id"], kwargs["session_id"])
            if path is not None:
                contract = original_cache_contract(self.config, sampling_params, kwargs["delta_prompt_id"])
                mode = protocol.get("original_cache_mode", "read")
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved["contract"] != contract:
                        raise ValueError("Shared first-batch original sampling/model contract differs")
                    output = AgentLoopOutput.model_validate(saved["output"])
                    if output.extra_fields.get("global_steps") != 0:
                        raise ValueError("Shared first-batch originals must come from actor version zero")
                    output.metrics.generate_sequences = 0.0
                    return await self._agent_loop_postprocess(output, False, **kwargs)
                if mode != "write":
                    raise ValueError(f"Shared first-batch original is missing: {path}")
                kwargs["_delta_original_cache_path"] = str(path)
                kwargs["_delta_original_cache_contract"] = contract
        return await super()._run_agent_loop(sampling_params, trajectory, agent_name=agent_name, trace=trace, **kwargs)

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        delta_policy_settings = _online_delta_policy_settings(self.config)
        if delta_policy_settings is not None and not validate:
            sampling_params["delta_top_logprobs"] = delta_policy_settings["selection"]["top_k_logprobs"]

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        delta_policy_settings = _online_delta_policy_settings(self.config)
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        cache_path = kwargs.pop("_delta_original_cache_path", None)
        cache_contract = kwargs.pop("_delta_original_cache_contract", None)
        if cache_path is not None:
            from examples.delta_critic.online_protocol import atomic_json

            if len(outputs) != 1 or outputs[0].extra_fields.get("global_steps") != 0:
                raise ValueError("Shared originals must be single-turn outputs from actor version zero")
            atomic_json(cache_path, {"contract": cache_contract, "output": outputs[0].model_dump(mode="json")})

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            expansion = self.config.algorithm.get("delta_policy", {}).get("expansion", {}) if not validate else {}
            expand_this_prompt = bool(expansion.get("enabled", False) and kwargs.get("delta_expand", False))
            # Short-response delta substitution: publish the realized outcome
            # reward and the token count it describes. A continuation counts its
            # prefix, so it is measured like a fresh response.
            short_response_budget = expansion.get("short_response_tokens")
            record = None
            if delta_policy_settings is not None and not validate:
                field.update(_online_delta_rollout_fields(output, delta_policy_settings))
                policy = self.config.algorithm.delta_policy
                if policy.get("rollout_mode", "critic_prediction") == "selected_prefix_mc":
                    from examples.delta_critic.policy_mc import collect_online_mc

                    if not hasattr(self, "_delta_mc_semaphore"):
                        self._delta_mc_semaphore = asyncio.Semaphore(policy.mc.get("concurrency", 16))
                    mc_indices = field["selected_token_indices"].tolist()
                    full_selected_indices = list(mc_indices)
                    protocol = expansion.get("protocol", {})
                    if protocol.get("enabled", False):
                        from examples.delta_critic.online_protocol import select_mc_state_indices

                        if not hasattr(self, "_delta_expansion_coordinator"):
                            raise RuntimeError("Matched sampling requires an initialized expansion coordinator")
                        eligible_indices = select_mc_state_indices(
                            full_selected_indices,
                            output.extra_fields["delta_top_logprobs"],
                            protocol=protocol,
                            response_cap=int(policy.mc.max_response_total_tokens),
                            prompt_length=len(output.prompt_ids),
                            total_cap=policy.mc.get("max_total_tokens"),
                            max_candidates=delta_policy_settings["selection"]["max_candidates"],
                        )
                        expand_this_prompt = await self._delta_expansion_coordinator.register.remote(
                            kwargs["delta_protocol_batch_id"],
                            int(kwargs["delta_protocol_index"]),
                            {"role": kwargs["delta_critic_role"], "eligible": bool(eligible_indices)},
                            dict(protocol),
                            int(expansion["prompts_per_step"]),
                            int(kwargs["global_steps"]),
                        )
                        field["delta_expand"] = expand_this_prompt
                        mc_indices = eligible_indices if expand_this_prompt else []
                    elif expansion.get("enabled", False):
                        if expand_this_prompt:
                            if not mc_indices:
                                raise ValueError("Budgeted expansion prompt has no eligible selected state")
                            # The first uncertainty-ranked state is recovered by
                            # scoring one position with the same selector.
                            from examples.delta_critic.policy_online import select_uncertainty_indices

                            selection = delta_policy_settings["selection"]
                            best = select_uncertainty_indices(
                                output.extra_fields["delta_top_logprobs"],
                                [int(j in mc_indices) for j in range(len(output.response_ids))],
                                states_per_response=1,
                                min_token_gap=selection["min_token_gap"],
                                entropy_weight=selection["entropy_weight"],
                                low_top1_weight=selection["low_top1_weight"],
                                final_window_tokens=selection["final_window_tokens"],
                                final_window_weight=selection["final_window_weight"],
                                max_candidates=selection["max_candidates"],
                            )
                            mc_indices = best
                        else:
                            mc_indices = []
                    record = await collect_online_mc(
                        output,
                        mc_indices,
                        policy=policy,
                        rollout_config=self.config.actor_rollout_ref.rollout,
                        server_client=self.llm_client,
                        tokenizer=self.tokenizer,
                        uid=uid,
                        session_id=session_id,
                        sample_kwargs=kwargs,
                        request_semaphore=self._delta_mc_semaphore,
                    )
                    if protocol.get("enabled", False):
                        role = kwargs["delta_critic_role"]
                        prompt_id = kwargs["delta_prompt_id"]
                        record.update(
                            full_selected_indices=full_selected_indices,
                            critic_role=role,
                            prompt_stable_id=prompt_id,
                            valid_td_indices=mc_indices[:1] if len(mc_indices) == 2 else [],
                        )
                        record["rollout"].update(
                            critic_role=role,
                            prompt_stable_id=prompt_id,
                            full_selected_indices=full_selected_indices,
                            valid_td_indices=record["valid_td_indices"],
                        )
                    field["delta_mc_record"] = record
                    # Use the collector's reward for both terminal TD targets
                    # and the trainer's reward tensors/metrics. The original
                    # reward-manager score remains in record.trainer_reward.
                    field["rm_scores"] = torch.zeros_like(responses, dtype=torch.float32)
                    field["rm_scores"][-1] = record["rollout"]["terminal_reward"]
                    field["delta_is_continuation"] = False
                    if short_response_budget is not None:
                        field["delta_true_reward"] = torch.tensor(
                            float(record["rollout"]["terminal_reward"]), dtype=torch.float32
                        )
                        field["delta_reward_text_length"] = torch.tensor(len(output.response_ids), dtype=torch.int64)
                        field["delta_reward_baseline"] = torch.tensor(
                            float(policy.get("short_outcome_baseline", 0.5)), dtype=torch.float32
                        )
                field["extra_fields"] = {
                    key: value for key, value in field["extra_fields"].items() if key != "delta_top_logprobs"
                }
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )
            if expand_this_prompt:
                protocol = expansion.get("protocol", {})
                state_count = (
                    2 if protocol.get("enabled", False) and protocol.get("sampling_layout", "pair") == "pair" else 1
                )
                if record is None or len(record["labels"]) != state_count:
                    raise ValueError("Budgeted expansion did not fill its selected-state quota")
                for label_index, label in enumerate(record["labels"]):
                    completions = label.get("mc_continuations", [])
                    expected = int(policy.mc.continuations_per_state)
                    if len(completions) != expected or any(not item["token_ids"] for item in completions):
                        raise ValueError("Budgeted expansion did not fill its continuation quota")
                    for continuation in completions:
                        token_ids = continuation["token_ids"]
                        prefix_ids = label["prefix_response_token_ids"]
                        child_prompt = torch.tensor(output.prompt_ids + prefix_ids, dtype=torch.int64)
                        child_response = torch.tensor(token_ids, dtype=torch.int64)
                        child_input = torch.cat([child_prompt, child_response])
                        child = dict(field)
                        child.update(
                            prompts=child_prompt,
                            responses=child_response,
                            response_mask=torch.ones_like(child_response),
                            loss_mask=torch.ones_like(child_response),
                            policy_token_mask=torch.ones_like(child_response, dtype=torch.float32),
                            rm_scores=torch.zeros(len(token_ids), dtype=torch.float32),
                            input_ids=child_input,
                            attention_mask=torch.ones_like(child_input),
                            position_ids=torch.arange(len(child_input), dtype=torch.int64),
                            delta_mc_record={},
                            delta_is_continuation=True,
                            extra_fields={key: value for key, value in field["extra_fields"].items()},
                        )
                        # Keep training/exported reward metrics consistent with the
                        # same completion score used by the short-outcome actor path.
                        child["rm_scores"][-1] = float(continuation["reward"])
                        if short_response_budget is not None:
                            default_baseline = float(policy.get("short_outcome_baseline", 0.5))
                            sibling_rewards = [
                                float(item["reward"])
                                for item in completions
                                if int(item["continuation_index"]) != int(continuation["continuation_index"])
                            ]
                            baseline = (
                                sum(sibling_rewards) / len(sibling_rewards) if sibling_rewards else default_baseline
                            )
                            child.update(
                                delta_true_reward=torch.tensor(float(continuation["reward"]), dtype=torch.float32),
                                delta_reward_text_length=torch.tensor(
                                    len(prefix_ids) + len(token_ids), dtype=torch.int64
                                ),
                                delta_reward_baseline=torch.tensor(baseline, dtype=torch.float32),
                            )
                        # The prefix is conditioning context. Select and train
                        # newly generated tokens exactly like an ordinary response.
                        child.update(
                            _online_delta_rollout_fields(
                                SimpleNamespace(
                                    response_ids=token_ids,
                                    response_mask=[1] * len(token_ids),
                                    extra_fields={
                                        **output.extra_fields,
                                        "delta_top_logprobs": continuation.get("delta_top_logprobs"),
                                    },
                                ),
                                delta_policy_settings,
                            )
                        )
                        if not child["selected_token_indices"].numel():
                            raise ValueError("Delta-scored continuation has no eligible selected state")
                        if "rollout_log_probs" in child:
                            child["rollout_log_probs"] = torch.zeros_like(child_response, dtype=torch.float32)
                        if "routed_experts" in child:
                            child["routed_experts"] = torch.zeros(
                                (len(child_input), *child["routed_experts"].shape[1:]),
                                dtype=child["routed_experts"].dtype,
                            )
                        child["multi_modal_inputs"] = None
                        child_index = int(continuation["continuation_index"])
                        keys.append(
                            f"{uid}_{session_id}_{(i * state_count + label_index) * expected + child_index + 1}"
                        )
                        fields.append(child)
                        tags.append(
                            {
                                **tags[-1],
                                "prompt_len": len(child_prompt),
                                "response_len": len(child_response),
                                "seq_len": len(child_input),
                            }
                        )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        protocol = instance.config.algorithm.get("delta_policy", {}).get("expansion", {}).get("protocol", {})
        if protocol.get("enabled", False):
            instance._delta_expansion_coordinator = ray.remote(num_cpus=0, max_concurrency=1024)(
                _DeltaExpansionCoordinator
            ).remote()
            await asyncio.gather(
                *[
                    worker.set_delta_expansion_coordinator.remote(instance._delta_expansion_coordinator)
                    for worker in instance.agent_loop_workers
                ]
            )
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
