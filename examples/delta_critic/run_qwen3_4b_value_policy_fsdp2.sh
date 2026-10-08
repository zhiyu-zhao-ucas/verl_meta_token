#!/usr/bin/env bash
# Train Qwen3-4B value, select heldout-MSE best, then train policy with FSDP2.
# Default starts a detached pipeline including the matched direct-delta control.
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
PYTHON_BIN=${PYTHON_BIN:-/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin/python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-codex-value-difference}
UV_BIN=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$UV_BIN" && -x /tmp/verl-uv-bin/uv ]]; then
  UV_BIN=/tmp/verl-uv-bin/uv
fi
if [[ -z "$UV_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Set UV_BIN and PYTHON_BIN to the existing compatible uv/Python environment." >&2
  exit 1
fi
ACTION=start
if [[ ${1:-} == prepare || ${1:-} == start || ${1:-} == run ]]; then
  ACTION=$1
  shift
fi
exec "$UV_BIN" run --offline --no-project --python "$PYTHON_BIN" \
  python -m examples.delta_critic.value_policy_pipeline "$ACTION" "$@"
