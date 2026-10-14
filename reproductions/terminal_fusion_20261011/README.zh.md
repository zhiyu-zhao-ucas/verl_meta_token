# Terminal fusion C/D 从零复现

本目录保存 2026-10-11 至 12 日完成的 C/D 两组、各两次 policy 运行的冻结配置、奖励函数、启动器与源码差异。接收者只需 GitHub 仓库、公开 Qwen 模型和本机 GPU；**不需要**从 toast 传模型、critic 或 `outputs`。重新训练的 critic 和 actor 会有新的权重哈希，因此复现的是代码、数据、预算和实验协议，不声称逐 bit 重现历史权重或指标。

## 0. 资源和输入

- Linux x86_64、Python 3.12、CUDA 13 兼容驱动、`uv`；先取得 8 张空闲 GPU 用于初始 TD80 critic，随后 4 张用于 Hybrid80，policy 同时使用 2 张。采样阶段可使用 1 到 8 张。实际显存和磁盘要能容纳 Qwen3-4B、长达 32768 token 的 critic 样本、Ray/vLLM 和多份 checkpoint。
- 公开 Base 模型：`Qwen/Qwen3-4B-Base`，固定 revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`。handoff 的 `bootstrap` 会下载它。
- 固定 policy 训练 parquet 已在 GitHub 的 `examples/delta_critic/handoff_assets.tgz`：train SHA256 `6e020de627a86126f79e344a07e318d75c61a5797e25cf99d5b9deeb73f91942`，validation SHA256 `753e608c85527689dfac6dd27895c1cb859884b5fd6c95413521510f35dc2db7`。它们与历史 C/D 配置引用的文件逐字节一致。包还含 512 个初始 critic prompt、初始 critic 源码和 688 个评测 prompt；不含权重。
- 如果用户以后把原 critic 权重上传 Hugging Face，可把下载目录传给 `--critic` 并校验权重哈希。下面的默认流程全部从 Base 新建权重。

## 1. 克隆并安装

```bash
git clone --branch codex/terminal-fusion-repro https://github.com/zhiyu-zhao-ucas/verl_meta_token.git
cd verl_meta_token
uv python install 3.12.13
uv sync --frozen --python 3.12.13 --extra fsdp --extra vllm --extra math
export UV_BIN="$(command -v uv)"
export PYTHON_BIN="$PWD/.venv/bin/python"
export ASSET_DIR="$PWD/outputs/td_hybrid_handoff_assets"
export BOOTSTRAP_DIR="$PWD/outputs/my_initial_critics"
```

`uv.lock` 与 `pyproject.toml` 随仓库提交。原实验核心版本为 Python 3.12.13、torch 2.13.0+cu130、vLLM 0.29.0、transformers 5.12.1、Ray 2.55.1、math-verify 0.9。`bootstrap` 的环境检查会指出不匹配项。安装 wheel 和下载公开 Base 需要网络；后续训练进入离线模式。受 CUDA 驱动和 wheel 可用性影响，其他硬件平台可能需要先建立兼容环境。

## 2. 从公开 Base 重建 Hybrid80 critic

在已分配的 GPU 上按顺序运行：

```bash
export SAMPLE_GPUS=0,1,2,3,4,5,6,7
export TD_INIT_GPUS=0,1,2,3,4,5,6,7
export HYBRID_INIT_GPUS=0,1,2,3
bash examples/delta_critic/run_td_hybrid_handoff.sh bootstrap
```

该入口自动校验并解压 GitHub 中的 handoff 包，下载固定 revision Base，从 512 题重新生成原回答与 MC 续写，做 P95 长度筛选，训练 TD80，再从 Base 独立训练 Hybrid80。仅 C/D policy 需要最终 Hybrid80，但历史初始流程先生成 TD80；`bootstrap` 会同时记录采样、配置和日志。完成后确认：

```bash
test -s "$BOOTSTRAP_DIR/hybrid_checkpoints/step_00000080/model.pt"
test -s "$BOOTSTRAP_DIR/hybrid_checkpoints/step_00000080/metadata.json"
test -s "$ASSET_DIR/inputs/train.parquet"
test -s "$ASSET_DIR/inputs/validation.parquet"
sha256sum "$BOOTSTRAP_DIR/hybrid_checkpoints/step_00000080/model.pt"
```

本机产出的 critic SHA256 需单独记录。它的 `delta_label_mode` 必须为 `selected_segment`，tokenizer 必须与 Base 相符。

## 3. 准备历史 C/D policy 运行

四份原始配置位于 `original/{C,D}_run_{1,2}/config.yaml`，原始奖励函数和 `train_once.py` 也在本目录。`prepare.py` 从公开 Git 提交 `5b6b0db8a6114341c9264cae7df3bded372785b9` 恢复 541 个共同源码文件，覆盖本目录的 10/12 个历史差异文件，并按 `source_manifest.json`/`source_run_2_manifest.json` 逐文件校验 SHA256。它只把 Base、critic、parquet、日志和 Ray 临时目录改为本机路径；不会启动训练。接收者无需旧机器的源码快照。

```bash
export TF_RUN_DIR="$PWD/outputs/terminal_fusion_rebuilt_001"
"$PYTHON_BIN" reproductions/terminal_fusion_20261011/prepare.py \
  --repo "$PWD" \
  --output "$TF_RUN_DIR" \
  --base-model "$PWD/outputs/td_hybrid_base_model" \
  --critic "$BOOTSTRAP_DIR/hybrid_checkpoints/step_00000080" \
  --train "$ASSET_DIR/inputs/train.parquet" \
  --validation "$ASSET_DIR/inputs/validation.parquet" \
  --gpus 0,1

# CPU 配置和冻结源码预检查；不启动训练。
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" D_run_1 --preflight
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" C_run_1 --preflight
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" D_run_2 --preflight
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" C_run_2 --preflight
```

`prepare.py` 拒绝覆盖已有目录，并把所有输入身份写到 `reproduction_inputs.json`。需要新试次时换一个全新的 `TF_RUN_DIR`。冻结启动器只接受来自 `{0,1,5,6,7}` 的两张不同物理卡；`--gpus 0,1` 适合新机器有卡 0 和 1 的情况。四次运行应顺序执行，以免复用 Ray 端口。

### 历史 policy 设置

| 设置 | C 组 | D 组 |
| --- | --- | --- |
| `terminal_fusion` | `full_trajectory` | `full_trajectory` |
| `terminal_advantage_mode` | `mc_grpo` | `mc_grpo` |
| `terminal_include_delta` | `false` | `true` |
| 额外回答生成 | `independent_full_response` | 同左 |
| 原始优势尺度 | `raw_scale_50`、`advantage_center=false` | 同左 |
| 损失聚合 | `token_mean` | 同左 |
| 每步样本 | 128 个问题 + 24×8 个额外回答 = 320 行 | 同左 |
| 训练 | fresh Base；2 GPU；10 step；保存 3/5/6/7/10 | 同左 |

每次训练前需由实验负责人确认优势方案。本仓库当前本地偏好是 `advantage_center=false`、`advantage_stats_mode=initial_rollout`（只按标准差缩放）；**历史 C/D** 的配置则是 `raw_scale_50`。若选择本地偏好，须另存修改后的配置和新实验名称，不能把结果称为完全相同的历史设置。

## 4. 运行 policy

确认使用历史 `raw_scale_50` 方案后，再逐一执行：

```bash
set -o pipefail
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" C_run_1 2>&1 | tee "$TF_RUN_DIR/C_run_1/train.log"
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" D_run_1 2>&1 | tee "$TF_RUN_DIR/D_run_1/train.log"
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" C_run_2 2>&1 | tee "$TF_RUN_DIR/C_run_2/train.log"
"$PYTHON_BIN" "$TF_RUN_DIR/train_once.py" D_run_2 2>&1 | tee "$TF_RUN_DIR/D_run_2/train.log"
```

每次启动前再次核对该 run 的 `--preflight` 和 `reproduction_inputs.json`。运行产物位于各自的 `checkpoints/global_step_{3,5,6,7,10}/actor`、`metrics.jsonl`、`rollouts/`、`validation/`、`started.json`、`finished.json`。如果训练失败，换新运行目录重做该试次；冻结启动器拒绝在旧目录上覆盖或续训。

## 5. 导出并评测

`evaluate.py` 复用 handoff 包中与历史 C/D 评测 SHA256 相同的脚本和 688 道固定题（MATH500、AIME 2024/2025、DAPO holdout）。它将历史评测脚本中仅限原机器的 Python 路径改为当前 `sys.executable`，导出 FSDP checkpoint 为 HF safetensors，然后按每题 16 个独立样本、temperature 1、8192 token 上限生成，最后执行逐条 math-verify 和仓库 `main_eval` 双重核对。`2048` 分数只是同批 8192 输出的前缀诊断。

```bash
"$PYTHON_BIN" reproductions/terminal_fusion_20261011/evaluate.py \
  --run-dir "$TF_RUN_DIR" --assets-dir "$ASSET_DIR" \
  --run D_run_1 --step 10 --gpu 0
cat "$TF_RUN_DIR/evaluation/D_run_1_step_10/results.csv"
```

要覆盖历史 C/D 全部 checkpoint，可在确认资源足够后依次对四个 run 和 `3 5 6 7 10` 五个 step 调用同一命令。每次调用写独立目录；评测中断时检查该目录日志并用新运行目录重试。主要指标在 `results.csv` 的 `full_8192`、`repository_math_verify`、`pass_at_1` 与 `pass_at_16` 列。比较 D−C 时先按两个 repeat 对齐，分别计算每个 step/数据集的均值。

## 必须同步的内容

| 内容 | 位置或来源 | 处理 |
| --- | --- | --- |
| 历史 policy 源码差异、四份配置、reward、launcher、重建/评测脚本 | 本目录，GitHub | 已包含 |
| 共同源码、`uv.lock`、环境定义 | 此 Git 提交及 `pyproject.toml` | 已包含 |
| 固定 train/validation、critic prompt、评测题目/脚本 | `examples/delta_critic/handoff_assets.tgz` | 已包含，解压时校验 |
| Qwen3-4B-Base | 公开 Hugging Face 固定 revision | 接收者下载 |
| 初始 TD80/Hybrid80 critic | 从 Base 在接收者机器训练 | 不传原权重 |
| C/D actor checkpoint、rollouts、优势统计、评测输出 | 接收者机器新生成 | 不传原产物 |

历史数值比较应同时记录 Base revision、训练数据 SHA、critic 新 SHA、四份生成配置 SHA、源码 manifest、GPU/包版本、随机种子、训练日志与评测结果。新 critic 的采样和 GPU 运算可造成数值差异，因此在报告中标明 `rebuilt critic`。

## 可选：用 Qwen3-4B 重新采集初始 critic 数据

此项是单独的数据采集实验。模型固定为公开的 `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`，**不是** `Qwen/Qwen3-4B-Base`。采集使用上文同一份 512 题封存输入。`collect_qwen3_4b_critic.py prepare` 会逐条检查这 512 题在 Qwen3-4B tokenizer 下的 token ID，并验证正式配置与初始 critic 采集的有效配置相比只有 `model.actor_model` 变化。temperature 1.0、top_p 0.95、top_k 20、原回答及 MC 最多 30720 token、每回答最多 64 个 state、每 state 16 个 MC 分支、seed 42 均保持一致。每卡 rollout/MC 并发分别为 8/128；GPU 张数可以不同。

```bash
# 先通过 Hugging Face 下载固定 revision，并把 MODEL 指向下载好的 snapshot。
export MODEL=/path/to/Qwen3-4B/snapshot
export COLLECT_DIR="$PWD/outputs/qwen3_4b_initial_critic_512_001"
"$PYTHON_BIN" reproductions/terminal_fusion_20261011/collect_qwen3_4b_critic.py prepare \
  --output "$COLLECT_DIR" --model "$MODEL" \
  --archive "$PWD/examples/delta_critic/handoff_assets.tgz" --gpus 0,1,2,5,6,7
"$PYTHON_BIN" reproductions/terminal_fusion_20261011/collect_qwen3_4b_critic.py run \
  --output "$COLLECT_DIR"
```

输出在 `$COLLECT_DIR/collection/`，配置和模型、tokenizer、题目、代码包身份在 `collection_plan.json`。只使用当前机器空闲的物理 GPU；运行入口会在启动前拒绝显存已占用超过 512 MiB 的卡。采集中断时保持同一目录并重新执行 `run`，原采样器会校验配置并恢复未完成的临时输出。该 Qwen3-4B tokenizer 的指纹与 Base 不同；后续如果用这些数据训练 critic，critic 与 policy 的模型和 tokenizer 身份也必须相应改为 Qwen3-4B，不能直接放入上面基于 Base 的 C/D policy 配置。
