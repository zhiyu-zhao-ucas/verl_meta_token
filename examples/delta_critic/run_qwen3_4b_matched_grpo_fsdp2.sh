#!/usr/bin/env bash
set -euo pipefail

# Match the local PPO run in outputs/matched_ppo_local_20261018/run_retry2.
# Reuse its recipe for actor, data, sampling, and evaluation settings.
# The requested 512 prompts x 8 responses preserve the original GRPO batch;
# this is 4x the responses/round of historical PPO (256 x 4), not budget matched.
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${CODE_DIR}"

export RUN_ROOT=${RUN_ROOT:-${CODE_DIR}/outputs/grpo_b512_n8_local}
export DATA_DIR=${DATA_DIR:-${CODE_DIR}/outputs/matched_ppo_local_20261018/data}
export OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/run_seed42}
export MODEL_PATH=${MODEL_PATH:-/scratch2/zhiyu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539}
export PYTHON_BIN=${PYTHON_BIN:-/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3,7}

# Keep token-mean loss and zero-coefficient KL settings from PPO.
# Group-standardized advantages are intrinsic to this GRPO comparison;
# gamma/lambda and the inherited critic optimizer settings are inactive.
# For a strict paired comparison, run PPO and GRPO from the same source
# snapshot: the historical PPO run predates current reward-parser changes.
# Inspect without training: bash examples/delta_critic/run_qwen3_4b_matched_grpo_fsdp2.sh --cfg job
exec bash "${CODE_DIR}/examples/delta_critic/run_matched_ppo_brezel.sh" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    critic.enable=false \
    data.train_batch_size=512 \
    data.gen_batch_size=512 \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=2 \
    trainer.project_name=matched_grpo \
    trainer.experiment_name=b512_n8_seed42 \
    "$@"
