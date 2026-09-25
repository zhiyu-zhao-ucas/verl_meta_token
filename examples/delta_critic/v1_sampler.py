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

from typing import Any

from .mc_labeling import MCConfig, MCContinuation, MCRequest


async def sample_v1_continuations(
    request: MCRequest, config: MCConfig, *, server_client: Any, tokenizer: Any,
) -> list[MCContinuation]:
    """Use the active actor server; one V1 request per MC continuation.

    V1 TokenOutput contains token IDs but no completion text. Decode those IDs
    with the actor tokenizer before the source-style prefix+continuation score.
    The caller owns the server lifecycle and must ensure its weights match
    ``config.actor_version``; this adapter cannot verify remote weights.
    """
    results = []
    for sample_index in range(config.continuations_per_state):
        output = await server_client.generate(
            request_id=f"delta-mc:{config.actor_version}:{request.rollout_id}:"
                       f"{len(request.prefix_response_token_ids)}:{sample_index}",
            prompt_ids=list(request.input_ids),
            sampling_params=dict(config.sampling_config),
        )
        token_ids = tuple(output.token_ids)
        results.append(MCContinuation(
            token_ids, tokenizer.decode(token_ids, skip_special_tokens=config.continuation_skip_special_tokens)
        ))
    return results
