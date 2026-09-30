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
    semaphore = request_semaphore or asyncio.Semaphore(config.continuations_per_state)

    if native_n_batch_size is not None:

        async def generate_batch(start: int) -> list[MCContinuation]:
            count = min(native_n_batch_size, config.continuations_per_state - start)
            async with semaphore:
                output = await server_client.generate(
                    request_id=f"delta-mc:{config.actor_version}:{request.rollout_id}:"
                    f"{len(request.prefix_response_token_ids)}:{start}",
                    prompt_ids=list(request.input_ids),
                    sampling_params={**config.sampling_config, "delta_mc_n": count},
                )
            rows = (output.extra_fields or {}).get("delta_mc_token_ids")
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError("V1 backend must return every delta_mc_n continuation")
            return [
                MCContinuation(
                    tuple(token_ids),
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
        async with semaphore:
            output = await server_client.generate(
                request_id=f"delta-mc:{config.actor_version}:{request.rollout_id}:"
                f"{len(request.prefix_response_token_ids)}:{sample_index}",
                prompt_ids=list(request.input_ids),
                sampling_params=dict(config.sampling_config),
            )
        token_ids = tuple(output.token_ids)
        return MCContinuation(
            token_ids, tokenizer.decode(token_ids, skip_special_tokens=config.continuation_skip_special_tokens)
        )

    tasks = [asyncio.create_task(generate_one(i)) for i in range(config.continuations_per_state)]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
