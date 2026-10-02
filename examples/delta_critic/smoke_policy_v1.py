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
"""Run a tiny synchronous V1 delta-policy rollout, score, and actor update."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.distributed.tensor import Replicate, Shard

from examples.delta_critic.policy_online import validate_online_v1_settings
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import validate_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-path", type=Path, required=True, help="Local tiny HF actor model")
parser.add_argument("--critic-artifact", type=Path, required=True, help="Frozen scalar critic checkpoint")
parser.add_argument("--output", type=Path, required=True, help="New output directory")
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--mc", action="store_true", help="Collect prefix MC and update the delta critic after prediction")
parser.add_argument("--expansion", action="store_true", help="Train on a fixed budget of prefix continuations")
parser.add_argument("--gpus", type=int, choices=(1, 2), default=1)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=False)
tiny_model = root / "tiny-chat"
shutil.copytree(args.model_path.resolve(), tiny_model)
tokenizer_config_path = tiny_model / "tokenizer_config.json"
tokenizer_config = json.loads(tokenizer_config_path.read_text())
if not tokenizer_config.get("chat_template"):
    tokenizer_config["chat_template"] = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
tokenizer_config_path.write_text(json.dumps(tokenizer_config))
initial_weights = load_file(str(tiny_model / "model.safetensors"))
reference_policy_id = hashlib.sha256((tiny_model / "model.safetensors").read_bytes()).hexdigest()
rows = [
    {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": f"What is {number} + 1? Answer with a number."}],
        "reward_model": {"ground_truth": str(number + 1)},
        "extra_info": {"index": number},
    }
    for number in range(1, 4 * args.steps + 1)
]
pd.DataFrame(rows).to_parquet(root / "train.parquet")

with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "verl/trainer/config")):
    config = compose(config_name="ppo_trainer")

settings = {
    "data.train_files": str(root / "train.parquet"),
    "data.val_files": str(root / "train.parquet"),
    "data.train_batch_size": 4,
    "data.val_batch_size": 4,
    "data.max_prompt_length": 64,
    "data.max_response_length": 12,
    "data.dataloader_num_workers": 0,
    "trainer.n_gpus_per_node": args.gpus,
    "trainer.total_epochs": 1,
    "trainer.total_training_steps": args.steps,
    "trainer.val_before_train": False,
    "trainer.test_freq": -1,
    "trainer.save_freq": 1,
    "trainer.logger": ["console"],
    "trainer.default_local_dir": str(root / "checkpoints"),
    "trainer.resume_mode": "disable",
    "trainer.project_name": "delta_policy_smoke",
    "trainer.experiment_name": "v1_tiny",
    "reward.num_workers": 1,
    "critic.enable": False,
    "actor_rollout_ref.model.path": str(tiny_model),
    "actor_rollout_ref.model.use_remove_padding": False,
    "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
    "actor_rollout_ref.actor.strategy": "fsdp2",
    "actor_rollout_ref.actor.use_kl_loss": True,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 4,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.actor.ppo_epochs": 1,
    "actor_rollout_ref.actor.optim.lr": 0.001,
    "actor_rollout_ref.actor.fsdp_config.param_offload": False,
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.data_parallel_size": args.gpus,
    "actor_rollout_ref.rollout.gpu_memory_utilization": args.gpu_memory_utilization,
    "actor_rollout_ref.rollout.standalone_gpu_memory_utilization": args.gpu_memory_utilization,
    "actor_rollout_ref.rollout.max_model_len": 128,
    "actor_rollout_ref.rollout.temperature": 0.7,
    "actor_rollout_ref.rollout.top_p": 0.95,
    "actor_rollout_ref.rollout.n": 1,
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
    "algorithm.delta_policy": {
        "enabled": True,
        "critic_artifact": str(args.critic_artifact.resolve()),
        "reference_policy_id": reference_policy_id,
        "label_mode": "selected_segment",
        "behavior_logprob_source": "actor_snapshot",
        "selection": {
            "strategy": "uncertainty",
            "states_per_response": 2,
            "min_token_gap": 1,
            "top_k_logprobs": 5,
            "max_candidates": 5,
        },
    },
}
if args.mc:
    settings["algorithm.delta_policy"].update(
        rollout_mode="selected_prefix_mc",
        mc={
            "continuations_per_state": 2,
            "concurrency": 2,
            "max_continuation_tokens": 12,
            "reward": {"method": "exact_or_numeric"},
        },
        critic_update={"epochs": 1, "microbatch": 1, "learning_rate": 1e-3},
    )
if args.expansion:
    settings["data.train_batch_size"] = 6
    settings["data.gen_batch_size"] = 4
    settings["actor_rollout_ref.actor.ppo_mini_batch_size"] = 3
    settings["algorithm.delta_policy"].update(
        rollout_mode="selected_prefix_mc",
        expansion={"enabled": True, "prompts_per_step": 4, "states_per_step": 1},
        mc={
            "continuations_per_state": 2,
            "concurrency": 2,
            "max_continuation_tokens": 12,
            "max_response_total_tokens": 12,
            "min_continuation_room": 1,
            "save_continuations": True,
            "reward": {"method": "exact_or_numeric"},
        },
    )
for key, value in settings.items():
    OmegaConf.update(config, key, value, force_add=True)
OmegaConf.resolve(config)
validate_online_v1_settings(config)
validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=False)
run_ppo(config, TaskRunnerV1)

previous = initial_weights
changes = []
for step in range(1, args.steps + 1):
    checkpoint_dir = root / "checkpoints" / f"global_step_{step}" / "actor"
    rank_states = [
        torch.load(
            checkpoint_dir / f"model_world_size_{args.gpus}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(args.gpus)
    ]
    if any(set(state) != set(initial_weights) for state in rank_states):
        raise RuntimeError(f"V1 actor checkpoint tensor keys changed at step {step}")
    current = {}
    for key, before in initial_weights.items():
        values = [state[key] for state in rank_states]
        if args.gpus == 1:
            tensor = values[0].to_local() if hasattr(values[0], "to_local") else values[0]
        else:
            placements = values[0].placements
            if any(value.placements != placements for value in values) or len(placements) != 1:
                raise RuntimeError(f"Inconsistent V1 actor shard placements for {key}")
            if isinstance(placements[0], Shard):
                dimension = placements[0].dim
                tensor = torch.cat([value.to_local() for value in values], dim=dimension)
                tensor = tensor.narrow(dimension, 0, before.shape[dimension])
            elif isinstance(placements[0], Replicate):
                tensor = values[0].to_local()
                for value in values[1:]:
                    torch.testing.assert_close(tensor, value.to_local(), atol=0, rtol=0)
            else:
                raise RuntimeError(f"Unsupported V1 actor shard placement for {key}: {placements}")
        if tensor.shape != before.shape:
            raise RuntimeError(f"V1 actor checkpoint shape changed for {key} at step {step}")
        current[key] = tensor
    change = max(float((current[key].float() - previous[key].float()).abs().max()) for key in current)
    if change <= 0.0:
        raise RuntimeError(f"V1 actor weights did not change at step {step}")
    changes.append(change)
    previous = current
critic_changes = []
if args.mc and not args.expansion:
    previous_critic = torch.load(args.critic_artifact / "model.pt", map_location="cpu", weights_only=True)
    for step in range(1, args.steps + 1):
        state = torch.load(
            root / "checkpoints" / f"global_step_{step}" / "delta_critic.pt", map_location="cpu", weights_only=True
        )
        if state["update_step"] != step:
            raise RuntimeError("Online critic update count does not match actor steps")
        change = max(
            (state["model"][key].float() - value.float()).abs().max().item() for key, value in previous_critic.items()
        )
        if change <= 0:
            raise RuntimeError(f"Online critic weights did not change at step {step}")
        critic_changes.append(change)
        previous_critic = state["model"]
report = {
    "steps": args.steps,
    "gpus": args.gpus,
    "mc": args.mc,
    "critic_max_abs_changes": critic_changes,
    "actor_max_abs_changes": changes,
    "reference_policy_id": reference_policy_id,
    "critic_artifact": str(args.critic_artifact.resolve()),
}
(root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print("DELTA_POLICY_V1_SMOKE_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
