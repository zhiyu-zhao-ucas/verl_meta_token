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
"""Manual single-GPU V1 rollout -> MC -> contract smoke test."""

import argparse
import asyncio
import json
import os
from functools import partial

import ray
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from examples.delta_critic.collection import export_v1_rollouts
from examples.delta_critic.legacy_adapter import adapt_legacy_rows
from examples.delta_critic.mc_labeling import MCConfig, label_mc_states_async
from examples.delta_critic.state_selection import select_states
from examples.delta_critic.v1_sampler import sample_v1_continuations
from verl.workers.rollout.llm_server import LLMServerManager


async def main(model_path: str, actor_version: str, gpu_memory_utilization: float):
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")
    config.trainer.n_gpus_per_node = 1
    config.trainer.nnodes = 1
    config.actor_rollout_ref.model.path = model_path
    rollout_config = config.actor_rollout_ref.rollout
    rollout_config.name = "vllm"
    rollout_config.mode = "async"
    rollout_config.nnodes = 1
    rollout_config.tensor_model_parallel_size = 1
    rollout_config.data_parallel_size = 1
    rollout_config.pipeline_model_parallel_size = 1
    rollout_config.load_format = "auto"
    rollout_config.skip_tokenizer_init = False
    rollout_config.prompt_length = 128
    rollout_config.response_length = 16
    rollout_config.gpu_memory_utilization = gpu_memory_utilization
    rollout_config.standalone_gpu_memory_utilization = gpu_memory_utilization
    ray.init(num_gpus=1, include_dashboard=False, log_to_driver=False)
    try:
        manager = await LLMServerManager.create(config=config)
        client = manager.get_client()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        prompt_ids = tokenizer.encode("Compute 1 + 1. Answer with a number: ", add_special_tokens=False)
        rollout_sampling = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 16, "logprobs": True}
        output = await client.generate(
            request_id="delta-v1-smoke-rollout", prompt_ids=prompt_ids, sampling_params=rollout_sampling
        )
        response_ids = list(output.token_ids)
        if not response_ids:
            raise RuntimeError("Actor generated no response tokens")
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        finish_reason = output.extra_fields.get("finish_reason")
        if finish_reason is None:
            raise RuntimeError("V1 output did not preserve raw finish_reason")
        # This is an interface check; the simple reward is not a math benchmark.
        row = {
            "uid": "delta-smoke-prompt-1", "prompt_id": "delta-smoke-prompt-1", "global_steps": 0,
            "prompts": prompt_ids, "responses": response_ids, "response_mask": [1] * len(response_ids),
            "rollout_log_probs": output.log_probs, "reward_score": float("2" in response_text),
            "stop_reason": output.stop_reason, "extra_fields": output.extra_fields, "gold_answer": "2",
        }
        rollouts = export_v1_rollouts(
            [row], actor_version=actor_version, sampling_config=rollout_sampling,
            eval_fraction=0.0, split_seed="delta-v1-smoke",
        )
        indices = sorted({0, len(response_ids) - 1})
        states = select_states(
            rollouts, strategy="indices", states_per_response=len(indices),
            indices_by_rollout={rollouts[0]["id"]: indices},
        )
        mc_config = MCConfig(
            mode="paired_next_state", continuations_per_state=2,
            sampling_config={"temperature": 0.7, "top_p": 0.9, "max_tokens": 8},
            actor_version=actor_version, continuation_skip_special_tokens=True,
        )
        labels = await label_mc_states_async(
            rollouts, states, mc_config,
            sample=partial(sample_v1_continuations, server_client=client, tokenizer=tokenizer),
            decode_prefix=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
            score=lambda text, rollout: float("2" in text),
        )
        examples, diagnostics = adapt_legacy_rows(rollouts, labels)
        print("DELTA_V1_SMOKE_RESULT=" + json.dumps({
            "model": model_path, "response_tokens": len(response_ids),
            "finish_reason": finish_reason, "stop_reason": output.stop_reason,
            "behavior_logprobs": len(output.log_probs or []), "selected_indices": indices,
            "mc_rows": len(labels), "mc_budgets": [r["mc_num_samples"] for r in labels],
            "validated_states": len(examples[0].states), "diagnostics": diagnostics,
        }, sort_keys=True), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--actor-version", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()
    asyncio.run(main(args.model_path, args.actor_version, args.gpu_memory_utilization))
