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
"""Thin adapter to V1 LLMServerClient.generate for MC continuations."""

import asyncio
from typing import Any

from .mc_labeling import MCConfig, MCContinuation, MCRequest


async def sample_v1_continuations(
    request: MCRequest,
    config: MCConfig,
    *,
    server_client: Any,
    tokenizer: Any,
    request_semaphore: asyncio.Semaphore | None = None,
    native_n_batch_size: int | None = None,
) -> list[MCContinuation]:
    """Use the active actor server; generate MC continuations concurrently.

    V1 TokenOutput contains token IDs but no completion text. Decode those IDs
    with the actor tokenizer before the source-style prefix+continuation score.
    The caller owns the server lifecycle and must ensure its weights match
    ``config.actor_version``; this adapter cannot verify remote weights.
    """
    if native_n_batch_size is not None and native_n_batch_size < 1:
        raise ValueError("native_n_batch_size must be positive")
    if native_n_batch_size is not None and config.sampling_config.get("delta_top_logprobs") is not None:
        raise ValueError("Delta-scored continuations require individual requests with token-aligned top logprobs")
    semaphore = request_semaphore or asyncio.Semaphore(config.continuations_per_state)
    max_tokens = config.sampling_config["max_tokens"]
    if config.max_total_tokens is not None:
        max_tokens = min(max_tokens, config.max_total_tokens - len(request.input_ids))
    if config.max_response_total_tokens is not None:
        max_tokens = min(max_tokens, config.max_response_total_tokens - len(request.prefix_response_token_ids))
    if max_tokens <= 0:
        return [MCContinuation((), "") for _ in range(config.continuations_per_state)]
    sampling_params = {**config.sampling_config, "max_tokens": max_tokens}

    def checked_tokens(token_ids):
        if len(token_ids) > max_tokens:
            raise ValueError("MC continuation exceeded the prefix-inclusive token cap")
        return tuple(token_ids)

    if native_n_batch_size is not None:

        async def generate_batch(start: int) -> list[MCContinuation]:
            count = min(native_n_batch_size, config.continuations_per_state - start)
            async with semaphore:
                output = await server_client.generate(
                    request_id=f"delta-mc:{config.actor_version}:{request.rollout_id}:"
                    f"{len(request.prefix_response_token_ids)}:{start}",
                    prompt_ids=list(request.input_ids),
                    sampling_params={**sampling_params, "delta_mc_n": count},
                )
            rows = (output.extra_fields or {}).get("delta_mc_token_ids")
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError("V1 backend must return every delta_mc_n continuation")
            return [
                MCContinuation(
                    checked_tokens(token_ids),
                    tokenizer.decode(token_ids, skip_special_tokens=config.continuation_skip_special_tokens),
                )
                for token_ids in rows
            ]

        # Keep one request per state in flight. The shared semaphore can then
        # admit requests from other states instead of filling every slot with
        # batches of the first state.
        continuations = []
        for start in range(0, config.continuations_per_state, native_n_batch_size):
            continuations.extend(await generate_batch(start))
        return continuations

    async def generate_one(sample_index: int) -> MCContinuation:
        for attempt in range(4):
            request_id = (
                f"delta-mc:{config.actor_version}:{request.rollout_id}:"
                f"{len(request.prefix_response_token_ids)}:{sample_index}"
            )
            if config.require_nonempty:
                request_id += f":attempt{attempt}"
            async with semaphore:
                output = await server_client.generate(
                    request_id=request_id,
                    prompt_ids=list(request.input_ids),
                    sampling_params=dict(sampling_params),
                )
            token_ids = checked_tokens(output.token_ids)
            if token_ids or not config.require_nonempty:
                candidates = None
                if config.sampling_config.get("delta_top_logprobs") is not None:
                    candidates = (output.extra_fields or {}).get("delta_top_logprobs")
                    if candidates is None or len(candidates) != len(token_ids) or any(not row for row in candidates):
                        raise ValueError("MC continuation is missing token-aligned delta_top_logprobs")
                return MCContinuation(
                    token_ids,
                    tokenizer.decode(token_ids, skip_special_tokens=config.continuation_skip_special_tokens),
                    delta_top_logprobs=candidates,
                )
        raise ValueError("MC continuation stayed empty after four attempts")

    tasks = [asyncio.create_task(generate_one(i)) for i in range(config.continuations_per_state)]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
