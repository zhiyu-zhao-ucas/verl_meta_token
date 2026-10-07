# 2026-10-06 TD / Hybrid 修复版封存

这份提交保存 `outputs/delta_historical_bugfix_retry_20261006_02` 实际运行的代码。
`assets.tgz` 包含 3,841 个文件：完整 TD / Hybrid 源码、独立评测源码、原 driver、
配置、reward、prompt 审计、训练和验证 parquet，以及启动和训练核验记录。
恢复不读取当前工作树中的 `verl` / `examples`，也不需要旧 `outputs` 中的源码。

`launch_integrity_audit.json` 记录了封存时的核查：TD 1,304 个源文件、Hybrid
1,308 个源文件，以及实际 Ray 包的 727 / 730 个 Python 文件全部匹配。
原 driver、配置、reward、runner、prompt 审计和环境版本记录均匹配启动封存。
并行训练和后续评测调度是后来增加的脚本，分别保存在 `orchestration/`，没有
混称为启动前已有的训练代码。它们是历史运行记录，新入口不再依赖这些接管脚本。

## 校验并恢复完整代码

使用训练时的兼容环境；入口不安装或更新依赖。

```bash
export PYTHON_BIN=/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin/python
export UV_CACHE_DIR=/tmp/uv-codex-audit
export REPRO=reproductions/td_hybrid_20261006/run.py
export RUN_DIR="$PWD/outputs/oct6_sealed_rerun"

uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" verify-bundle
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" restore --work-dir "$RUN_DIR"
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" check --work-dir "$RUN_DIR" --weights
```

恢复入口拒绝覆盖已有目录。归档 SHA256、每个文件 SHA256、模型和 critic 权重身份
在 `bundle.json` 中；归档只接受清单内的普通文件，拒绝路径穿越和重复成员。
恢复后的训练源码和原 driver 为只读。每次启动前重新检查源码、配置、数据、
模型、critic、tokenizer、评测文件和记录的 Python 包版本，发现变化即停止。

Base 三个权重分片、两个完整 critic80 和 actor checkpoint 太大，不放入 Git。
它们的原路径和 SHA256 已记录；在原机器使用既有文件，迁移机器时须另行保存这些
模型素材及兼容运行环境。训练数据已在归档中。环境记录保证版本可核对，历史没有
记录所有第三方库文件的启动时字节哈希；不能将它说成完整二进制环境认证。

## 从 Base 重新运行

确认指定 GPU 空闲后，在两个终端分别运行；这里的卡号仅对应原机器示例。

```bash
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" launch \
  --work-dir "$RUN_DIR" --arm td --gpus 3,4,5,6
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" launch \
  --work-dir "$RUN_DIR" --arm hybrid --gpus 2,7
```

入口用与旧队列相同的 GPU 文件锁，并检查显存和计算进程，不抢占已有任务。
同一 arm 已启动过则拒绝重复启动。训练输出保存在各自 `train.log`、`metrics.jsonl`
和 `checkpoints/` 中；成功后验证十步指标、实际 MC 布局、固定统计和 checkpoint。

保持 TD DP4 / Hybrid DP2、原初始化 horizon 100 / 20、循环停止点 10、固定 critic80、
每步 128 原回答 + 192 续写、两次 160 行 optimizer minibatch、LR 1e-5、seed42。
TD / Hybrid 分别保留自己的历史短答配方和自定义 loss 分母。

```bash
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" evaluate \
  --work-dir "$RUN_DIR" --arm td --gpus 2,3,4,5,6,7
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" evaluate \
  --work-dir "$RUN_DIR" --arm hybrid --gpus 2,3,4,5,6,7
```

评测保留原 step5 / step10、688 题、每题 16 次、8192 token 协议和评分脚本。

## 在这次 step10 基础上继续训练

必须保留原完整 checkpoint：全部 model / optim / extra_state 分片、`data.pt`、
`fsdp_config.json` 和 `checkpoints/delta_advantage_stats.json`。不能只拿导出的模型续训。

```bash
export CONT_DIR="$PWD/outputs/oct6_td_continue20"
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" prepare-resume \
  --from-run "$PWD/outputs/delta_historical_bugfix_retry_20261006_02" \
  --arm td --step 10 --stop-step 20 --work-dir "$CONT_DIR" --port-base 35220
uv run --offline --no-project --python "$PYTHON_BIN" python "$REPRO" launch \
  --work-dir "$CONT_DIR" --arm td --gpus 3,4,5,6
```

Hybrid 对应 `--arm hybrid`、两张卡；停止点不得超过原 horizon 20。
TD 停止点不得超过原 horizon 100。多个新 run 并行时指定不同 `--port-base`。

续训使用独立输出目录，恢复 actor、optimizer、scheduler、RNG、dataloader，复制原
固定优势统计，保持 critic 和训练源码不变，并禁止加载后删除原 checkpoint。
原 fresh driver 保留；续训入口生成独立 `continue_driver.py`，只适配恢复步数、
停止点和 prompt 审计范围，具体替换及哈希记入 `continuation_driver.json`。
step10 之后保存新 prompt hash，但不会声称与不存在的历史前十步审计之外的轨迹一致。
以上 `evaluate` 命令仅用于原 step5 / step10 协议；新增 step20 不是原十步复现实验。

## 版本与效果边界

从固定 Git commit / tag 恢复本目录，可以重新获得归档和入口；即使之后工作树
继续修改，也不会改变这份已提交的素材。不要用后来版本的根目录代码替换包内源码。
包内原 `driver.py` 和源码可逐字节恢复；恢复新路径仅派生配置和独立续训 driver。

代码与输入身份可校验，不承诺重新采样后的轨迹和准确率逐数值一致。当前运行已经
改善了全文评分，但最终答案评分仍有差距；原始结果及诊断记录随封存保留。

## CPU 验证

```bash
uv run --offline --no-project --python "$PYTHON_BIN" python -m pytest -q \
  tests/special_standalone/delta_critic/test_frozen_reproduction.py
```

封存工作已在实际 TD / Hybrid step10 checkpoint 上通过续训准备检查；没有在封存时
额外启动 GPU 续训，续训更新的数值行为仍以实际运行日志为准。
