#!/usr/bin/env bash
# Native VinePPO; use an activated verl environment (managed with uv).
# MODEL=/models/model TRAIN_FILES=/data/train.parquet VAL_FILES=/data/test.parquet \
# MC_CONTEXT_LENGTH=2499 bash examples/vineppo/run_qwen3_4b_vineppo_fsdp2.sh
# GSM8K: append algorithm.vineppo.splitter=gsm8k (root + at most 25 intermediate states).
# MC_CONTEXT_LENGTH is an experiment setting, distinct from the model architecture limit.
set -euo pipefail
: "${MODEL:?Set MODEL to a Hugging Face model directory or ID}"
: "${TRAIN_FILES:?Set TRAIN_FILES to the training parquet}"
: "${VAL_FILES:?Set VAL_FILES to the validation parquet}"
: "${MC_CONTEXT_LENGTH:?Set the MC context budget explicitly}"

"${PYTHON_BIN:-python}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=vineppo \
  algorithm.vineppo.mc_context_length="$MC_CONTEXT_LENGTH" \
  trainer.use_v1=true \
  trainer.v1.trainer_mode=sync \
  actor_rollout_ref.model.path="$MODEL" \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size=128 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  "$@"
