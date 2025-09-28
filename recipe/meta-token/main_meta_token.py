"""Entry point for the Meta Token PPO recipe.

This module mirrors the high-level structure of other recipes (for example,
``recipe/entropy``) while delegating the training loop to
``MetaTokenPPOTrainer``. It wires together Ray initialization, dataset
construction, reward loading, and worker registration so users can launch
Meta Token experiments with a single command.
"""

from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager

from .meta_token_trainer import MetaTokenPPOTrainer


@hydra.main(config_path="config", config_name="meta_token_trainer", version_base=None)
def main(config):
	"""Hydra entry point."""

	run_meta_token(config)


def run_meta_token(config) -> None:
	"""Initialize Ray (if needed) and kick off the remote controller."""

	if not ray.is_initialized():
		default_runtime_env = {
			"env_vars": {
				"TOKENIZERS_PARALLELISM": "true",
				"NCCL_DEBUG": "WARN",
				"VLLM_LOGGING_LEVEL": "WARN",
			}
		}
		ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
		runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
		runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
		ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
		print(f"ray init kwargs: {ray_init_kwargs}")
		ray.init(**OmegaConf.to_container(ray_init_kwargs))

	runner = TaskRunner.remote()
	ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on the Ray head node
class TaskRunner:
	"""Driver process that orchestrates dataset loading and trainer execution."""

	def run(self, config):
		from pprint import pprint

		from omegaconf import OmegaConf

		from verl.single_controller.ray import RayWorkerGroup
		from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
		from verl.utils import hf_processor, hf_tokenizer
		from verl.utils.fs import copy_to_local

		pprint(OmegaConf.to_container(config, resolve=True))
		OmegaConf.resolve(config)

		local_path = copy_to_local(config.actor_rollout_ref.model.path)

		trust_remote_code = config.data.get("trust_remote_code", False)
		tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
		processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

		if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
			assert config.critic.strategy in {"fsdp", "fsdp2"}
			from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

			actor_rollout_cls = (
				AsyncActorRolloutRefWorker
				if config.actor_rollout_ref.rollout.mode == "async"
				else ActorRolloutRefWorker
			)
			critic_cls = CriticWorker
			ref_policy_cls = ActorRolloutRefWorker
		elif config.actor_rollout_ref.actor.strategy == "megatron":
			assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
			from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

			actor_rollout_cls = ActorRolloutRefWorker
			critic_cls = CriticWorker
			ref_policy_cls = ActorRolloutRefWorker
		else:
			raise NotImplementedError(f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}")

		role_worker_mapping = {
			Role.ActorRollout: ray.remote(actor_rollout_cls),
			Role.Critic: ray.remote(critic_cls),
		}

		global_pool_id = "global_pool"
		resource_pool_spec = {
			global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
		}
		mapping = {
			Role.ActorRollout: global_pool_id,
			Role.Critic: global_pool_id,
		}

		# Optional reward model workers.
		if config.reward_model.enable:
			if config.reward_model.strategy in {"fsdp", "fsdp2"}:
				from verl.workers.fsdp_workers import RewardModelWorker
			elif config.reward_model.strategy == "megatron":
				from verl.workers.megatron_workers import RewardModelWorker
			else:
				raise NotImplementedError(f"Unsupported reward model strategy: {config.reward_model.strategy}")

			role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
			mapping[Role.RewardModel] = global_pool_id

		if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
			# Reuse the actor worker for the reference policy when KL control is enabled.
			role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
			mapping[Role.RefPolicy] = global_pool_id

		reward_kwargs_cfg = config.reward_model.get("reward_kwargs", {})
		if OmegaConf.is_config(reward_kwargs_cfg):
			reward_kwargs = OmegaConf.to_container(reward_kwargs_cfg, resolve=True)
		elif isinstance(reward_kwargs_cfg, dict):
			reward_kwargs = reward_kwargs_cfg
		else:
			reward_kwargs = {}
		reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)
		val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs)

		resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

		from verl.utils.dataset.rl_dataset import collate_fn

		train_dataset = create_rl_dataset(
			config.data.train_files,
			config.data,
			tokenizer,
			processor,
			is_train=True,
		)
		val_dataset = create_rl_dataset(
			config.data.val_files,
			config.data,
			tokenizer,
			processor,
			is_train=False,
		)
		train_sampler = create_rl_sampler(config.data, train_dataset)

		trainer = MetaTokenPPOTrainer(
			config=config,
			tokenizer=tokenizer,
			processor=processor,
			role_worker_mapping=role_worker_mapping,
			resource_pool_manager=resource_pool_manager,
			ray_worker_group_cls=RayWorkerGroup,
			reward_fn=reward_fn,
			val_reward_fn=val_reward_fn,
			train_dataset=train_dataset,
			val_dataset=val_dataset,
			collate_fn=collate_fn,
			train_sampler=train_sampler,
		)
		trainer.init_workers()
		trainer.fit()


if __name__ == "__main__":
	main()
