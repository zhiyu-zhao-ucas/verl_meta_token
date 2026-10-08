# 从数据采集复现 warmup Hybrid

Last updated: 10/08/2026

本文针对 `outputs/delta_policy_warmup3_20261007/hybrid`：**Qwen3-4B-Base actor 从头训练 10 个 policy step，前 3 次 actor optimizer 更新做零起点线性学习率预热；Hybrid80 critic 全程冻结。** 它不是 Hybrid step5 之后更新 critic 的 online 实验，也不是 `trainer.critic_warmup`。

## 先区分两种“复现”

- **固定历史权重**：提供原 Base、TD80、Hybrid80 权重；复用原训练/验证 parquet 和封存源码，能对齐权重与输入身份。warmup actor 的采样与 GPU 数值轨迹仍不能保证逐条一致。原 Hybrid80 的 `model.pt` SHA256 是 `fe95a8bbb5ff54c24d5dec483491b3571378bc049c97969f7d4778c55c2fb48e`。
- **从采集数据开始**：重新采 512 题的初始 critic 数据、训练新 TD80/Hybrid80，然后从 Base 训练 warmup Hybrid actor。它复用采样预算和算法设置，**新 MC 标签、新 critic 权重以及 actor 轨迹都不等于 2026-10-07 原运行**。不能把它的成绩称为原 checkpoint 的数值复现。

下面给出第二种从头路径；若要第一种，跳过阶段 1，把 `CRITIC_MODE=historical`、`TD_CRITIC`、`HYBRID_CRITIC` 指向原完整 artifact，再从阶段 2 开始。即使只跑 Hybrid，现有 `prepare` 仍会同时校验 TD80 和 Hybrid80，所以历史模式需提供两者。

## 流程总图

```text
固定 512 题 + Qwen3-4B-Base
  → 原回答：每题 1 条，最多 30720 新 token
  → uncertainty 选 state：每回答最多 64 个候选，gap 32
  → 每 state 16 条 MC 续写、数学 reward → v_prefix / delta label
  → P95 长度过滤 → 438 train / 48 heldout（原运行计数；新采样可能不同）
  → TD80 与 Hybrid80 分别从 Base + 新 scalar head 训练 80 update
  → Base actor + 固定 Hybrid80 + 4096 题 policy parquet
  → 每 policy step：128 原回答 + 24 state × 8 续写 = 320 actor 行
  → 冻结 critic 打优势，前 3 个 actor update 使用 warmup LR
  → 共 10 policy step / 20 actor update → step5、step10 保存与评测
```

初始 Hybrid80 的训练有 TD128 + terminal128/global batch、4 GPU、microbatch1、80 update；原历史 artifact 的 terminal 配置是 `terminal_weight=1`、`terminal_normalization=td`、`continuation_budget=8`。不要把当前示例 YAML 中的 `terminal_weight=0.003` 当成原 warmup run 的 critic 配方。actor 阶段的 24×8 扩展 MC 是 **policy 每步的新数据**：critic 不更新，actor 的长答优势来自冻结 critic 的预测。原 run 的 `algorithm.delta_policy.critic_update=null`。

## 阶段 0：环境与输入

在仓库根目录准备 Linux、Bash、`uv`、Python 3.12 兼容环境、Qwen3-4B-Base 完整 snapshot，以及足够磁盘和 GPU。交接包 `examples/delta_critic/handoff_assets.tgz` 含固定题目、parquet、采样配置和封存源码；`reproductions/td_hybrid_20261006/assets.tgz` 含 warmup 所依赖的 10 月 6 日两处评分修复。两个包均在仓库内，均不含大模型权重。原环境的关键版本是 Python3.12.13、torch2.13.0+cu130、vLLM0.29.0、transformers5.12.1、Ray2.55.1、Hydra1.3.3、math-verify0.9；脚本会核对身份，不应绕过检查。

```bash
export UV_BIN=/path/to/uv
export PYTHON_BIN=/path/to/compatible_env/bin/python
export BASE_MODEL=/data/models/Qwen3-4B-Base
export BOOTSTRAP_DIR="$PWD/outputs/my_warmup_initial_critics"
export WORK_DIR="$PWD/outputs/my_warmup_hybrid"
export SAMPLE_GPUS=0,1,2,3
export TD_INIT_GPUS=0,1,2,3,4,5,6,7
export HYBRID_INIT_GPUS=0,1,2,3
export HYBRID_GPUS=0,1
export EVAL_GPU=0
```

`BASE_MODEL` 可以省略，`bootstrap` 会下载固定 Qwen3-4B-Base revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`；下载需要网络，后续训练使用离线模式。`BOOTSTRAP_DIR`、`WORK_DIR` 均应是**尚不存在**的新目录。GPU 编号改成本机空闲卡；初始 TD80 用 8 卡、初始 Hybrid80 用 4 卡、policy Hybrid 用 2 卡，阶段顺序运行，可以复用物理卡。

## 阶段 1：重新采样并训练初始 critic

```bash
bash examples/delta_critic/run_td_hybrid_handoff.sh bootstrap
```

`bootstrap` 内部顺序执行 `prepare-bootstrap → initial_collect → initial_p95 → initial_td → initial_hybrid`：先从包内 512 道固定题目采样、评分 MC，接着做 P95 过滤，再训练 TD80 与 Hybrid80。中间文件位于 `$BOOTSTRAP_DIR/train/data_p95/`，包括 `rollouts_train_regen.jsonl`、`mc_labels_train.jsonl`、`continuations_train.jsonl` 和 `split.json`；最终权重位于 `$BOOTSTRAP_DIR/td_checkpoints/step_00000080` 与 `hybrid_checkpoints/step_00000080`。日志在 `$BOOTSTRAP_DIR/logs/`。检查 `initial_collect.done`、`initial_p95.done`、`initial_td.done`、`initial_hybrid.done`，以及两个 step80 的 `metadata.json`/`model.pt`。这些是**新训练权重**。

## 阶段 2：生成独立 Hybrid actor 运行目录

```bash
export CRITIC_MODE=rebuilt
export TD_CRITIC="$BOOTSTRAP_DIR/td_checkpoints/step_00000080"
export HYBRID_CRITIC="$BOOTSTRAP_DIR/hybrid_checkpoints/step_00000080"
bash examples/delta_critic/run_td_hybrid_handoff.sh check
bash examples/delta_critic/run_td_hybrid_handoff.sh prepare
```

`check` 只核对 CPU 环境、输入路径/哈希，不训练；`prepare` 写入独立 `$WORK_DIR`，绑定 Base、parquet、新 critic 和封存源码。**此时不要执行 `all`、`fresh-all` 或普通 `hybrid`**：普通 Hybrid 配方没有三次 warmup。

下面的准备器把交接源码中的两处旧评分函数替换为 10 月 6 日封存版本，应用与原 `warmup3` 相同的 `engine_workers.py` scheduler patch，再在 resolved Hybrid 配置中设置 `actor_rollout_ref.actor.optim.lr_warmup_steps=3`。它逐文件核验新旧 SHA256，更新 `code_manifest.json`，写 `warmup_hybrid_preparation.json`；原 TD、online 分支保持不变。

```bash
"$UV_BIN" run --offline --no-project --python "$PYTHON_BIN" python \
  -m examples.delta_critic.prepare_warmup_hybrid_handoff prepare \
  --work-dir "$WORK_DIR"
```

准备器只接受尚未启动 Hybrid 的工作目录。得到的 Hybrid 源码关键文件 SHA256 应为：`sampling_io.py=b9ad51be…`、`agent_loop_tq.py=6f6d3a4c…`、`engine_workers.py=ab0aff4d…`；完整值在 `warmup_hybrid_preparation.json`。**单改 YAML 的 `lr_warmup_steps=3` 不够**：原 worker 每个 policy step 有两个 optimizer minibatch，默认 scheduler 只在该 step 最后前进；原 warmup run 额外修改 worker，使 scheduler 每次 optimizer 更新都前进。

## 阶段 3：从 Base 训练 warmup Hybrid actor

```bash
bash examples/delta_critic/run_td_hybrid_handoff.sh hybrid
"$UV_BIN" run --offline --no-project --python "$PYTHON_BIN" python \
  -m examples.delta_critic.prepare_warmup_hybrid_handoff verify \
  --work-dir "$WORK_DIR"
```

这一步不会加载旧 actor step5：actor 和固定 reference 均从 Base 起跑；加载的是**冻结** Hybrid80 critic。训练使用 4096 题 parquet，验证为 DAPO128 + MATH 前100题；DP2/TP1，seed42，原回答最大 2048 token，续写“前缀+新回答”总长最多 2048。每 step 128 原回答、24 个选中 state 各 8 续写，合计 320 行，分两个 160 行 minibatch、每卡 microbatch1；actor 只做 1 epoch。短答 `≤100 token` 使用历史 Hybrid outcome 规则，长答从冻结 Hybrid critic 取 selected-segment 优势。actor 原始 optimizer horizon 保留 20 update，driver 只在 10 个 policy step 后停止。

LR **按 optimizer update，而非 policy step**：第 1–5 次实际 LR 为 `0, 3.333333e-6, 6.666667e-6, 1e-5, 1e-5`，之后保持 `1e-5`。每个 DP rank 的 `hybrid/lr_trace/rank_*.jsonl` 应有 20 条；`verify` 逐条检查 `lr_used`、`lr_after_update`、`scheduler_advanced` 和顺序。还应检查 `logs/hybrid.done`、`hybrid/metrics.jsonl` 的 10 个 step、`hybrid/checkpoints/global_step_5` 与 `global_step_10`、`checkpoints/delta_advantage_stats.json`。优势均值/方差来自**本次首次 rollout**，新采样不会等于原 run 保存的 `mean=0.001423975117185974, std=0.017577480929294457`。

## 阶段 4：独立评测

```bash
bash examples/delta_critic/run_td_hybrid_handoff.sh eval-hybrid
```

评测导出 step5/10，使用 DAPO128、MATH500、AIME2024/2025 各30题，共688题，每题16次、最多8192新 token。查看 `$WORK_DIR/evaluation_tools/hybrid_step_5/summary.json`、`hybrid_step_10/summary.json`、`results.csv`，确保两项 `repository_main_eval_crosschecks` 通过，并固定同一 `full_8192` 评分列比较。训练内名为 `math500` 的 greedy 指标只有 MATH 前100题。不要用它替代完整独立评测。

原 2026-10-07 warmup run 已完成 Hybrid 的 10 policy step、20 optimizer update，两个 DP rank 的 LR trace 验证通过。该 run 的其它因果解释需谨慎：其无 warmup 对照有一次明显崩溃，而另一条无 warmup Hybrid 运行未同样崩溃；不能仅据两次 run 断言 warmup 必然提升 Hybrid。复现时先确认源码、critic 身份、每步 128+192 布局、LR trace 和评分口径，再讨论准确率。

## 运行中断与身份边界

`logs/<stage>.done` 表示完成；存在 `.log` 但无 `.done` 时，入口拒绝从该阶段直接重跑，避免重复 optimizer 更新。不要删除日志或伪造成功标记；失败训练用新的 `WORK_DIR`，初始 critic 采样/训练失败用新的 `BOOTSTRAP_DIR`。拷贝旧权重时用 `rsync -aL` 解引用符号链接。warmup 准备器及评测都不更新 critic。

若需要**原权重身份**的复现：在阶段 0 提供原 TD80、Hybrid80 与 Base 完整 snapshot，阶段 1 跳过，阶段 2 设置 `CRITIC_MODE=historical` 和原 artifact。若坚持“从 MC 数据重新采集”同时还要求复现原 `warmup1007` 的具体曲线，两者不能同时保证：新 collector 的 MC 样本和 critic 权重会改变 actor 的训练信号。必须把原标签、原 critic 和新采样运行分成不同实验身份。
