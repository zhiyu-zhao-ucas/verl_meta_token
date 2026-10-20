#!/usr/bin/env bash
# Sample Monte-Carlo (MC) labelled data for critic training.
#
# Pipeline per question: 1 rollout -> pick uncertain token states -> N MC
# continuations per state -> math_verify reward -> per-state MC value labels.
# Runs two jobs: a TRAIN split and a disjoint EVAL split (held-out slice of the
# same dataset, starting right after the train questions).
#
# Usage (from anywhere):
#   bash examples/delta_critic/run_mc_sampling.sh
# Every knob below is an env var, e.g. the three you will most often change:
#   MODEL=Qwen/Qwen3-8B N_TRAIN=8000 N_EVAL=500 MC_PER_STATE=32 \
#   GPUS=0,1,2,3 OUT_DIR=/data/mc_run1 bash examples/delta_critic/run_mc_sampling.sh
#
# Requirements: repo checked out, python env with verl + vllm + ray + datasets +
# math-verify installed, and internet (or a pre-populated HF cache) for the model
# and dataset.
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

# ============================== knobs =====================================
MODEL=${MODEL:-Qwen/Qwen3-4B-Base}      # HF id or local path of the model that generates
N_TRAIN=${N_TRAIN:-512}                 # number of TRAIN questions
N_EVAL=${N_EVAL:-100}                   # number of EVAL questions (disjoint from train)
MC_PER_STATE=${MC_PER_STATE:-16}        # MC continuations per selected state
STATES_PER_RESPONSE=${STATES_PER_RESPONSE:-64}  # max selected states per rollout
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}   # context window (4B-Base default 32768)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-30720} # rollout + MC continuation cap (<= MAX_MODEL_LEN-2048)
BASE_CONFIG=${BASE_CONFIG:-examples/delta_critic/config_sampling_qwen3_4b_base_dapo.yaml}
OUT_DIR=${OUT_DIR:-$REPO_ROOT/outputs/mc_sampling_$(date +%Y%m%d_%H%M%S)}
GPUS=${GPUS:-0}                         # one GPU = one TP1 replica; see "Scaling" below
ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY:-64}  # concurrent rollouts
MC_CONCURRENCY=${MC_CONCURRENCY:-128}   # concurrent MC sequences in flight
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
# ==========================================================================
# Scaling: each batch_sample process drives ONE vLLM replica on one GPU
# (tensor_parallel_size=1 in the config). To use K GPUs, this script launches K
# shards in parallel (one per GPU in $GPUS), each with a disjoint slice of the
# questions, writing to $OUT_DIR/{train,eval}/shard_<i>/. Concatenate shards with
# `cat $OUT_DIR/train/shard_*/mc_labels_train.jsonl` etc.

export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$OUT_DIR"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
K=${#GPU_LIST[@]}

# Build a per-shard YAML (question range + MC count + model) from BASE_CONFIG.
# args: split(train|rank_eval) offset count outfile
make_config() {
    python - "$BASE_CONFIG" "$1" "$2" "$3" "$4" <<'PY'
import sys, yaml
base, split, offset, count, out = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["model"]["actor_model"] = __import__("os").environ["MODEL"]
cfg["model"]["max_model_len"] = int(__import__("os").environ["MAX_MODEL_LEN"])
cfg["rollout"]["max_tokens"] = int(__import__("os").environ["MAX_NEW_TOKENS"])
cfg["mc"]["max_continuation_tokens"] = int(__import__("os").environ["MAX_NEW_TOKENS"])
cfg["mc"]["continuations_per_state"] = int(__import__("os").environ["MC_PER_STATE"])
cfg["smoke"]["states_per_response"] = int(__import__("os").environ["STATES_PER_RESPONSE"])
key = f"{split}_prompts"
cfg["smoke"][key] = int(count)
cfg["smoke"][f"{split}_prompt_offset"] = int(offset)
if split == "rank_eval":  # batch_sample checks eval/train overlap against the configured train slice
    cfg["smoke"]["train_prompts"] = int(__import__("os").environ["N_TRAIN"])
    cfg["smoke"]["train_prompt_offset"] = 0
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
}
export N_TRAIN MODEL MAX_MODEL_LEN MAX_NEW_TOKENS MC_PER_STATE STATES_PER_RESPONSE

# run_split <name: train|eval> <batch_sample split: train|rank_eval> <start> <total>
run_split() {
    local name=$1 split=$2 start=$3 total=$4
    local per=$(( (total + K - 1) / K )) pids=() i
    for ((i = 0; i < K; i++)); do
        local off=$(( start + i * per ))
        local cnt=$(( per ))
        (( i * per + cnt > total )) && cnt=$(( total - i * per ))
        (( cnt <= 0 )) && continue
        local dir=$OUT_DIR/$name/shard_$i
        mkdir -p "$dir"
        local expected_config=$dir/config.expected.yaml
        make_config "$split" "$off" "$cnt" "$expected_config"
        if [[ -f $dir/config.yaml ]]; then
            if ! cmp -s "$expected_config" "$dir/config.yaml"; then
                rm -f "$expected_config"
                echo "[$name] shard $i config differs from existing run" >&2
                exit 1
            fi
            rm -f "$expected_config"
        else
            mv "$expected_config" "$dir/config.yaml"
        fi
        if [[ -f $dir/data/rollouts_${split}_regen.jsonl &&
              -f $dir/data/states_${split}.jsonl &&
              -f $dir/data/mc_labels_${split}.jsonl &&
              ( -f $dir/data/continuations_${split}.jsonl ||
                -f $dir/data/continuations_${split}.parquet ) ]]; then
            echo "[$name] shard $i already complete; skipping"
            continue
        fi
        (
            CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python -m examples.delta_critic.batch_sample \
                --config "$dir/config.yaml" \
                --split "$split" \
                --prompt-source value_model \
                --model-path "$MODEL" \
                --actor-version "$MODEL" \
                --mc-mode prefix_only \
                --rollout-concurrency "$ROLLOUT_CONCURRENCY" \
                --mc-concurrency "$MC_CONCURRENCY" \
                --gpu-memory-utilization "$GPU_MEM_UTIL" \
                --output-dir "$dir/data" > "$dir/log.txt" 2>&1
        ) &
        pids+=($!)
    done
    local rc=0 p
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    (( rc == 0 )) || { echo "[$name] a shard failed; see $OUT_DIR/$name/shard_*/log.txt" >&2; exit 1; }
}

# Train questions are [0, N_TRAIN); eval questions are [N_TRAIN, N_TRAIN+N_EVAL)
# of the same dataset (no overlap). Both read the dataset's train split.
run_split train train     0        "$N_TRAIN"
run_split eval  rank_eval "$N_TRAIN" "$N_EVAL"

echo "Done. Outputs under $OUT_DIR"
echo "  train: $OUT_DIR/train/shard_*/data/{rollouts_train_regen,states_train,mc_labels_train,continuations_train}.jsonl"
echo "  eval : $OUT_DIR/eval/shard_*/data/{rollouts_rank_eval_regen,states_rank_eval,mc_labels_rank_eval,continuations_rank_eval}.jsonl"
