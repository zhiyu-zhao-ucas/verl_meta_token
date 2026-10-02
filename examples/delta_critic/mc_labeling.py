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
"""Prefix/next continuation MC requests and labels independent of GPU backend."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from .legacy_adapter import adapt_legacy_rows, terminal_after_last_token


@dataclass(frozen=True)
class MCRequest:
    rollout_id: str
    prefix_response_token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.prefix_response_token_ids


@dataclass(frozen=True)
class MCContinuation:
    token_ids: tuple[int, ...]
    text: str
    delta_top_logprobs: list[list[dict[str, Any]]] | None = None


@dataclass(frozen=True)
class MCConfig:
    mode: Literal["prefix_only", "paired_next_state"]
    continuations_per_state: int
    sampling_config: Mapping[str, Any]
    actor_version: str
    continuation_skip_special_tokens: bool
    treat_length_truncation_as_terminal: bool = True
    assume_legacy_last_token_terminal: bool = False
    save_individual_rewards: bool = False
    save_continuations: bool = False
    max_total_tokens: int | None = None
    max_response_total_tokens: int | None = None
    require_nonempty: bool = False

    def __post_init__(self):
        if self.mode not in {"prefix_only", "paired_next_state"}:
            raise ValueError(f"Unsupported MC mode={self.mode!r}")
        if (
            isinstance(self.continuations_per_state, bool)
            or not isinstance(self.continuations_per_state, int)
            or self.continuations_per_state <= 0
            or not self.sampling_config
            or not self.actor_version
        ):
            raise ValueError("MC budget, sampling_config and actor_version are required")
        if self.sampling_config.get("n", 1) != 1:
            raise ValueError("MC sampling_config.n must be 1; budget is continuations_per_state")
        if not isinstance(self.continuation_skip_special_tokens, bool):
            raise ValueError("continuation_skip_special_tokens must be explicitly set")
        if self.max_total_tokens is not None and (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
            or self.max_total_tokens < 1
        ):
            raise ValueError("max_total_tokens must be a positive integer")
        if self.max_response_total_tokens is not None and (
            isinstance(self.max_response_total_tokens, bool)
            or not isinstance(self.max_response_total_tokens, int)
            or self.max_response_total_tokens < 1
        ):
            raise ValueError("max_response_total_tokens must be a positive integer")


def build_mc_requests(
    rollouts: Iterable[dict[str, Any]],
    states: Iterable[dict[str, Any]],
    config: MCConfig,
) -> tuple[list[MCRequest], dict[str, tuple[MCRequest, MCRequest | None]]]:
    """Deduplicate prefixes within each rollout, matching value_model's request keys."""
    rollouts = list(rollouts)
    rollout_map = {str(row["id"]): row for row in rollouts}
    if len(rollout_map) != len(rollouts):
        raise ValueError("Duplicate rollout_id in MC rollouts")
    requests: dict[tuple[str, tuple[int, ...]], MCRequest] = {}
    links = {}
    seen_positions = set()
    for state in states:
        rid = str(state["rollout_id"])
        if rid not in rollout_map:
            raise ValueError(f"State references missing rollout_id={rid}")
        rollout = rollout_map[rid]
        if rollout.get("actor_version") is not None and rollout["actor_version"] != config.actor_version:
            raise ValueError(f"MC actor_version mismatch for rollout_id={rid}")
        response = tuple(rollout["response_token_ids"])
        index = state["token_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(response):
            raise ValueError(f"token_index={index} outside rollout_id={rid}")
        if (rid, index) in seen_positions:
            raise ValueError(f"Duplicate selected token_index={index} for rollout_id={rid}")
        seen_positions.add((rid, index))
        before_ids = response[:index]
        if tuple(state["prefix_response_token_ids"]) != before_ids:
            raise ValueError(f"prefix_response_token_ids mismatch for state={state.get('state_id')}")
        if tuple(state["prompt_token_ids"]) != tuple(rollout["prompt_token_ids"]):
            raise ValueError("prompt_token_ids mismatch")
        if state.get("prompt_id") != rollout.get("prompt_id") or state.get("split") != rollout.get("split"):
            raise ValueError("state prompt_id/split mismatch")
        if state.get("actor_version") is not None and state["actor_version"] != config.actor_version:
            raise ValueError(f"MC actor_version mismatch for state={state.get('state_id')}")
        if any(value != 1 for value in rollout.get("policy_token_mask", [1] * len(response))):
            raise ValueError("MC continuation requires single-turn generated response tokens")
        before = requests.setdefault((rid, before_ids), MCRequest(rid, before_ids, tuple(rollout["prompt_token_ids"])))
        after = None
        if config.mode == "paired_next_state":
            shortcut = index == len(response) - 1 and terminal_after_last_token(
                rollout.get("finish_reason"),
                treat_length_truncation_as_terminal=config.treat_length_truncation_as_terminal,
                assume_legacy_last_token_terminal=config.assume_legacy_last_token_terminal,
            )
            if not shortcut:
                after_ids = response[: index + 1]
                after = requests.setdefault(
                    (rid, after_ids), MCRequest(rid, after_ids, tuple(rollout["prompt_token_ids"]))
                )
        if state.get("state_id") is None or str(state["state_id"]) == "":
            raise ValueError("state_id is required")
        sid = str(state["state_id"])
        if sid in links:
            raise ValueError(f"Duplicate state_id={sid}")
        links[sid] = (before, after)
    return list(requests.values()), links


def label_mc_states(
    rollouts: Iterable[dict[str, Any]],
    states: Iterable[dict[str, Any]],
    config: MCConfig,
    *,
    sample: Callable[[MCRequest, MCConfig], Sequence[MCContinuation]],
    decode_prefix: Callable[[tuple[int, ...]], str],
    score: Callable[[str, dict[str, Any]], float],
) -> list[dict[str, Any]]:
    """Sample each unique prefix and score decoded prefix + completion text.

    The scorer sees the same string composition as value_model. It receives the
    rollout so callers can use its gold answer and task-specific reward config.
    """
    rollouts = list(rollouts)
    states = list(states)
    requests, links = build_mc_requests(rollouts, states, config)
    rollout_map = {str(row["id"]): row for row in rollouts}
    values = {}
    for request in requests:
        continuations = list(sample(request, config))
        if len(continuations) != config.continuations_per_state:
            raise ValueError("MC sampler returned a count different from continuations_per_state")
        prefix_text = decode_prefix(request.prefix_response_token_ids)
        rewards = []
        empty = 0
        records = []
        for continuation_index, continuation in enumerate(continuations):
            reward = float(score(prefix_text + continuation.text, rollout_map[request.rollout_id]))
            if not isfinite(reward):
                raise ValueError("MC reward must be finite")
            rewards.append(reward)
            empty += not continuation.token_ids
            if config.save_continuations:
                records.append(
                    {
                        "continuation_index": continuation_index,
                        "token_ids": list(continuation.token_ids),
                        "text": continuation.text,
                        "reward": reward,
                        **(
                            {"delta_top_logprobs": continuation.delta_top_logprobs}
                            if continuation.delta_top_logprobs is not None
                            else {}
                        ),
                    }
                )
        values[request] = (sum(rewards) / len(rewards), rewards, empty, records)
    return _assemble_labels(rollouts, states, config, links, values)


async def label_mc_states_async(
    rollouts: Iterable[dict[str, Any]],
    states: Iterable[dict[str, Any]],
    config: MCConfig,
    *,
    sample: Callable[[MCRequest, MCConfig], Awaitable[Sequence[MCContinuation]]],
    decode_prefix: Callable[[tuple[int, ...]], str],
    score: Callable[[str, dict[str, Any]], float],
    request_concurrency: int = 1,
    score_in_thread: bool = False,
    score_semaphore: asyncio.Semaphore | None = None,
    score_batch_async: Callable[[list[str], dict[str, Any]], Awaitable[Sequence[float]]] | None = None,
    on_label: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Async variant for the V1 LLM server client.

    Signal-based scorers must use a process-backed ``score_batch_async``;
    ``score_in_thread`` is only suitable for thread-safe scoring functions.
    """
    if score_batch_async is not None and score_in_thread:
        raise ValueError("score_batch_async and score_in_thread are mutually exclusive")
    rollouts = list(rollouts)
    states = list(states)
    requests, links = build_mc_requests(rollouts, states, config)
    rollout_map = {str(row["id"]): row for row in rollouts}
    if isinstance(request_concurrency, bool) or not isinstance(request_concurrency, int) or request_concurrency < 1:
        raise ValueError("request_concurrency must be positive")

    async def evaluate(request):
        continuations = list(await sample(request, config))
        if len(continuations) != config.continuations_per_state:
            raise ValueError("MC sampler returned a count different from continuations_per_state")

        async_rewards = None
        if score_batch_async is not None:
            prefix_text = decode_prefix(request.prefix_response_token_ids)
            texts = [prefix_text + continuation.text for continuation in continuations]
            if score_semaphore is None:
                async_rewards = list(await score_batch_async(texts, rollout_map[request.rollout_id]))
            else:
                async with score_semaphore:
                    async_rewards = list(await score_batch_async(texts, rollout_map[request.rollout_id]))
            if len(async_rewards) != len(continuations):
                raise ValueError("MC scorer returned a count different from continuations")

        def score_continuations():
            prefix_text = decode_prefix(request.prefix_response_token_ids)
            rewards = []
            empty = 0
            records = []
            for continuation_index, continuation in enumerate(continuations):
                reward = float(
                    async_rewards[continuation_index]
                    if async_rewards is not None
                    else score(prefix_text + continuation.text, rollout_map[request.rollout_id])
                )
                if not isfinite(reward):
                    raise ValueError("MC reward must be finite")
                rewards.append(reward)
                empty += not continuation.token_ids
                if config.save_continuations:
                    records.append(
                        {
                            "continuation_index": continuation_index,
                            "token_ids": list(continuation.token_ids),
                            "text": continuation.text,
                            "reward": reward,
                            **(
                                {"delta_top_logprobs": continuation.delta_top_logprobs}
                                if continuation.delta_top_logprobs is not None
                                else {}
                            ),
                        }
                    )
            return sum(rewards) / len(rewards), rewards, empty, records

        if score_in_thread:
            if score_semaphore is None:
                result = await asyncio.to_thread(score_continuations)
            else:
                async with score_semaphore:
                    result = await asyncio.to_thread(score_continuations)
        else:
            result = score_continuations()
        return request, result

    semaphore = asyncio.Semaphore(request_concurrency)

    async def bounded_evaluate(request):
        async with semaphore:
            return await evaluate(request)

    tasks = [asyncio.create_task(bounded_evaluate(request)) for request in requests]
    values = {}
    emitted = set()
    try:
        for task in asyncio.as_completed(tasks):
            request, value = await task
            values[request] = value
            if on_label is not None:
                for state in states:
                    state_id = str(state["state_id"])
                    before, after = links[state_id]
                    if state_id not in emitted and before in values and (after is None or after in values):
                        on_label(_assemble_labels(rollouts, [state], config, links, values)[0])
                        emitted.add(state_id)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return _assemble_labels(rollouts, states, config, links, values)


def _assemble_labels(rollouts, states, config, links, values):
    rollout_map = {str(row["id"]): row for row in rollouts}
    labels = []
    for state in states:
        before, after = links[str(state["state_id"])]
        value, rewards, empty, records = values[before]
        row = {
            "state_id": state["state_id"],
            "rollout_id": state["rollout_id"],
            "token_index": state["token_index"],
            "prompt_id": state.get("prompt_id"),
            "split": state.get("split"),
            "prompt_token_ids": list(before.prompt_token_ids),
            "prefix_response_token_ids": list(before.prefix_response_token_ids),
            "v_prefix": value,
            "mc_num_samples": len(rewards),
            "mc_empty_continuations": empty,
            "mc_sampling_config": {
                **config.sampling_config,
                **(
                    {
                        "max_response_total_tokens": config.max_response_total_tokens,
                        "effective_max_response_tokens": max(
                            0,
                            min(
                                config.sampling_config["max_tokens"],
                                config.max_response_total_tokens - len(before.prefix_response_token_ids),
                            ),
                        ),
                    }
                    if config.max_response_total_tokens is not None
                    else {}
                ),
                **(
                    {
                        "max_total_tokens": config.max_total_tokens,
                        "effective_max_tokens": max(
                            0,
                            min(
                                config.sampling_config["max_tokens"],
                                config.max_total_tokens - len(before.input_ids),
                            ),
                        ),
                    }
                    if config.max_total_tokens is not None
                    else {}
                ),
            },
            "mc_actor_version": config.actor_version,
            "mc_continuation_skip_special_tokens": config.continuation_skip_special_tokens,
        }
        if config.save_individual_rewards:
            row["prefix_continuation_rewards"] = rewards
        if config.save_continuations:
            row["mc_continuations"] = records
        if config.mode == "paired_next_state":
            if after is None:
                next_value = float(rollout_map[before.rollout_id]["terminal_reward"])
                next_rewards, next_empty = [], 0
            else:
                next_value, next_rewards, next_empty, _ = values[after]
                if config.save_individual_rewards:
                    row["next_continuation_rewards"] = next_rewards
            row.update(
                v_next=next_value,
                delta=next_value - value,
                mc_next_num_samples=len(next_rewards),
                mc_next_empty_continuations=next_empty,
            )
        labels.append(row)
    # Step 1 validates state linkage, prefix alignment and all numeric outputs.
    adapt_legacy_rows(rollouts, labels)
    return labels
