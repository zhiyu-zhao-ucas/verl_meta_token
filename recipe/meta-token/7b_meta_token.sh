#!/usr/bin/env bash
set -xeuo pipefail

export WANDB_API_KEY=${WANDB_API_KEY:-YOUR_WANDB_API_KEY}

project_name=${PROJECT_NAME:-"MetaToken-Qwen2.5-7B"}
exp_name=${EXP_NAME:-"meta-token-1gpu"}

adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

max_prompt_length=${MAX_PROMPT_LEN:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LEN:-$((1024 * 8))}
train_prompt_bsz=${TRAIN_BSZ:-4}          # Conservative defaults for 1 GPU
train_prompt_mini_bsz=${TRAIN_MINI_BSZ:-1}
gen_prompt_bsz=${GEN_BSZ:-$((train_prompt_bsz * 1))}
n_resp_per_prompt=${NUM_RESP_PER_PROMPT:-1}
max_token=${MAX_TOKEN:-4096}

NNODES=1   # Single node / single GPU

MODEL_PATH=${MODEL_PATH:-"/PATH/TO/YOUR/MODEL"}
CKPTS_DIR=${CKPTS_DIR:-"/PATH/TO/CHECKPOINTS"}
TRAIN_FILE=${TRAIN_FILE:-"/PATH/TO/TRAIN.jsonl"}
TEST_FILE=${TEST_FILE:-"/PATH/TO/VAL.jsonl"}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}

use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
infer_micro_batch_size=${INFER_MICRO_BSZ:-null}
train_micro_batch_size=${TRAIN_MICRO_BSZ:-null}
offload=${OFFLOAD:-False}

HYDRA_FULL_ERROR=1 python -m recipe.meta-token.main_meta_token \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_token} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.test_freq=4 \
    trainer.save_freq=8 \
    trainer.total_epochs=1000 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${CKPTS_DIR}" \
    global_profiler.save_path="${CKPTS_DIR}/profiling"
