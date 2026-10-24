#!/usr/bin/env bash
# Opt-in GPU smoke. Run from the repository root in an activated verl environment:
# python tests/trainer/ppo/vineppo/prepare_smoke.py /tmp/vineppo-smoke
# CUDA_VISIBLE_DEVICES=0,1 bash tests/trainer/ppo/vineppo/run_smoke.sh /tmp/vineppo-smoke train
# Repeat with `resume` (continues checkpoint 1) and `fault` (must exit nonzero before updating).
# The toy verifier rewards any nonempty completed text; this is not a reasoning benchmark.
set -euo pipefail
root=$(realpath "${1:?Pass the prepared fixture directory}")
phase=${2:?Select train, resume, or fault}
export RAY_ADDRESS=local
export MODEL="$root/model" TRAIN_FILES="$root/data.parquet" VAL_FILES="$root/data.parquet"
export MC_CONTEXT_LENGTH=96
args=(trainer.resume_mode=disable trainer.default_local_dir="$root/$phase")
case "$phase" in
  train) ;;
  resume) args+=(trainer.resume_mode=resume_path trainer.resume_from_path="$root/train/global_step_1") ;;
  fault) args+=(data.max_response_length=8 algorithm.vineppo.mc_samples=2
    algorithm.vineppo.mc_max_new_tokens=95 trainer.total_training_steps=1
    reward.custom_reward_function.path="$root/failing_reward.py") ;;
  *) echo 'Expected train, resume, or fault' >&2; exit 2 ;;
esac
bash examples/vineppo/run_qwen3_4b_vineppo_fsdp2.sh \
  trainer.n_gpus_per_node=2 trainer.nnodes=1 \
  trainer.total_training_steps=2 trainer.total_epochs=1 \
  trainer.logger='[console]' trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=1 \
  data.train_batch_size=2 data.max_prompt_length=16 data.max_response_length=32 \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=128 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=128 actor_rollout_ref.rollout.max_num_seqs=16 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=false \
  algorithm.vineppo.mc_samples=9 algorithm.vineppo.mc_max_new_tokens=32 \
  algorithm.vineppo.splitter=gsm8k algorithm.vineppo.concurrency=2 \
  reward.num_workers=1 reward.custom_reward_function.path="$root/reward.py" \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
  +ray_kwargs.ray_init.num_cpus=16 "${args[@]}"
