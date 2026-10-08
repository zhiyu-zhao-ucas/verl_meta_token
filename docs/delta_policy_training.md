# Delta policy 训练：实现状态与续作交接

Last updated: 2026-10-08（补充校准后的短回答 outcome advantage 与在线诊断说明）。

## 2026-10-02 短回答优势修正

短回答现在使用独立校准的 outcome advantage：
`A_outcome = (reward - baseline) / short_outcome_scale`。原回答使用
`algorithm.delta_policy.short_outcome_baseline`（默认 0.5）；同一 prefix 的
continuation 使用其他 sibling rewards 的 leave-one-out 均值，缺少 sibling
时使用固定 baseline。`short_outcome_scale` 默认 0.5，因此固定 baseline
下二元 reward 0/1 对应 -1/+1。

这些行广播到全部 response token，并经过统一 advantage clip，但不再减去
critic 的固定 mean、不参与 critic advantage 统计拟合。这样负的 critic
mean 不会把错误短答的 reward=0 转成正优势。长回答继续使用原来的 local
critic 预测及标准化。长度口径仍是原回答 response 长度、continuation
prefix 加续写长度；短答规则本身没有改变。

新的 outcome 合约及 baseline/scale 绑定到保存的统计 identity，拒绝直接
恢复旧 raw-reward 混合标准化统计。若首批全部为 outcome 行，保存等待
critic 统计的状态，直到首个含有效 critic 行的批次再拟合；不制造虚假
mean/std。旧实验日志和 frozen source snapshot 仍反映下面记录的历史规则，
正在运行的训练进程不会自动切换。

## 历史规则：短回答用真实 reward 作 delta，标准化输出加 clip

两个新开关都只作用于 `algorithm.delta_policy`，缺省不生效，因此不影响其它
已运行实验和离线路径。

`expansion.short_response_tokens`（本 run 取 100）：response 长度不超过该值
的行，其整条 response 的 delta 用真实 reward 代替 critic 预测。
`target_ops.state_advantages(..., reward_delta=...)` 返回
`[reward] * len(response)` 并置全 1 mask，取代原来的 selected-segment 广播。
长度口径：原回答按 response token 数（不含 prompt），continuation 按
`len(prefix) + len(continuation)`（把续写当成一条新回答）。reward 口径：
原回答取 `math_verify` 对该回答的分数（即 MC collector 的
`terminal_reward`），continuation 取 prefix 加 continuation 文本的分数
（MC labeller 的 `completion["reward"]`）。原回答和 continuation 一视同仁。

该 reward 与 critic 预测同处 reward 量纲，因此直接进入同一套标准化，
`(reward - mean) / std`，不另设统计。本 run 自己拟合出的统计为
`mean=0.0033, std=0.0268`（step 1 拟合后冻结），故 `reward=1` 会得到约
`+37`，`reward=0` 约 `-0.12`；对照 20261001 基线的 `mean=0.0037,
std=0.0191` 则是 `+52`。无论哪一套，都必须配合 clip 才有界。

`advantage_clip`（本 run 取 5.0）：在标准化之后对 advantage 向量对称截断到
`[-clip, +clip]`，对 critic 行和 reward 行一视同仁。实现在纯函数
`target_ops.clip_advantages`，由 `policy_data.prepare_policy_rows` 调用，
属于共享 advantage 合约而不是 trainer。截断发生在标准化统计拟合之后，
不改变拟合出的 mean/std。逐行截断的 token 数记入 `advantage_clip_count`，
由 trainer 汇总为 `delta_policy/clipped_advantage_tokens`。

两条规则都写进 `delta_advantage_stats.json` 的 identity
（`...:short_reward_le100+advantage_clip_5.0`），防止恢复时静默混用别的统计。

新增 TQ 字段 `delta_true_reward`（float）与 `delta_reward_text_length`（int），
只在开关开启时由 `agent_loop_tq.py` 写入；trainer 通过
`PPOTrainer._short_response_reward_deltas` 读取，用长度字段与预算比较。

一个容易误读的点，写下来避免以后搞错：expansion 模式下 `states_per_step` 只决定
哪些原回答拿到 MC label（因而获得 continuation 扩展），**不决定哪些行参与训练**。
每条原回答都会在 `_online_delta_rollout_fields` 里独立跑一次 uncertainty 选点
（`states_per_response=64`、`min_token_gap=32`），因此 128 条原回答本来就有
advantage 和 policy mask，segment 基本覆盖整条 response。
`states_train.jsonl` 里只有 24 条，是因为它记录的是 MC record 的 state，不是
TQ 的 `selected_token_indices`。算术上也对得上：20261001 基线
`advantage_stats_count=84728`，而它的原回答 token 总数为 85823（98.7%），
只有 128 条原回答几乎全 mask 才成立。本次改动没有改变这一点。

另需注意本轮 run 与 20261001 基线的可比性：基线 metrics 里没有
`delta_policy/continuation_advantage_from_critic`、`continuation_actor_loss_enabled`
和 `actor_loss_continuation_rows` 三个字段（当前 expansion 分支无条件写入），
而 `mask_only_20261002` 有后两个、没有第一个，说明代码是分几次改的，
基线停留在上面 2026-10-02 那次"continuation 改用 delta critic"之前。
数值上也印证：基线的 `advantage_stats_count=84728` 只等于原回答的 token 量，
说明其标准化 population 里没有 continuation。所以在 20261002 这轮里，
continuation 换了目标函数、原回答换了短回答规则，两者不能互相归因。

## 2026-10-02 expansion 中的 continuation 使用 delta critic

启用 `algorithm.delta_policy.expansion.enabled` 时，原回答和 continuation
统一使用冻结 delta critic 的预测作为 actor advantage。移除 continuation
按 8 条真实奖励计算组内 advantage 的分支。`math_verify` 奖励仍可用于
评估和 MC 数据导出；continuation 的 TQ `rm_scores` 置零，不参与该模式的
actor advantage 或 critic 在线更新。

continuation 请求返回逐 token 的 `delta_top_logprobs`，复用原回答的
uncertainty 选点规则。原 prompt 加 response prefix 作为 conditioning
context，仅新生成的 continuation tokens 进入选点和 policy loss。
原回答与续写一起经过 critic 打分、segment broadcast、同一套 advantage
标准化、PPO clip 和 reference KL；MC 展开仍只发生在原回答选出的 state。
缺少或错位的 continuation token 概率会报错，不回退到真实奖励。

第一轮固定标准化统计现在覆盖所有参与 actor 训练的 delta samples；统计
文件保存 `advantage_contract`，拒绝直接复用旧混合目标的标准化统计。
已有 `continuation_actor_loss=false` 对照开关仍只清零 continuation 的
PG/KL mask，保留原 batch 的行数分母。该开关下标准化统计只使用原回答。
此处是训练代码变更，不代表已运行的训练进程或历史 checkpoint 已切换目标。

## 2026-10-01 MC 一致性修复与旧 critic 采集对照

本次修复引用审查中的三个问题：

- `sampling_io.score_texts_async` 使用可复用的 spawn 子进程执行评分；
  math-verify 在子进程主线程运行，其 signal 超时机制可正常工作。
  `policy_mc.py` 和旧 `batch_sample.py` 都改用此路径，避免在线程中将
  数学等价答案错误记为 0。真实 `\frac{1}{2}`/`0.5`、`\sqrt{4}`/`2`
  在两条 collector 路径中均已回归验证。每个 collector/AgentLoop 进程
  复用一个 CPU 评分进程，MC 生成请求仍异步并发。
- online 原回答与 MC continuation 统一使用 `mc.reward` 重评分，得到
  同尺度的 0/1 terminal reward 和 MC value；TQ 的 `rm_scores` 同步使用
  新 terminal reward。原 reward manager 分数保留在 `trainer_reward`，
  reward 配方保留在 `reward_config`，避免 DAPO 的 -1 错答与 MC 的 0 混用。
- trainer 导出改为 `mc_data/step_XXXXXXXX/attempt_<内容哈希>/`。
  四份 JSONL 写完后才发布批次目录；相同内容复用目录，中断后同一步
  重新采样得到不同 UUID/token 时保留独立目录。旧 step 根目录文件保留。
  这些目录是采样记录，不是 checkpoint；下游选择数据时须识别重试批次，
  不能假定一个 step 只有一次采样。

对照依据是实际 4B 历史采集配置
`outputs/delta_qwen3_4b_base_dapo_20260927/config_base512.yaml`、
采集入口 `collect.py`、online 两份保存配置，以及
`examples/delta_critic/config_online_mc_overlay.yaml`。不能用仓库的旧 8B
示例配置代替实际 4B 实验参数。

| 项目 | 之前的 4B critic 数据采集 | online selected-prefix MC |
| --- | --- | --- |
| actor | 固定 Qwen3-4B-Base | 每轮使用更新前的当前 actor；原回答和所有 continuation 校验同版本 |
| 题目 | DAPO 前 512 条 train，另取 offset 512 的 100 条 rank_eval | 现有 policy 数据准备排除旧 critic prompt ID，4096 条去重训练题，128 条 DAPO 验证题和 100 条 MATH-500 |
| prompt | 模型 chat template，`enable_thinking=true` | 同设置；MC 直接拼接已有 token IDs，不重新套模板或编码 |
| 原回答上限 | 30720 | 原 online 配置为 30720；单独的 `_10k/config.yaml` 为 10000，启动入口直接读取各自配置 |
| MC 上限/次数 | 30720；每状态 16 次 | overlay 同为 30720/16，与原回答上限独立；实际仍受剩余模型上下文限制 |
| 采样分布参数 | temperature=1、top_p=0.95、top_k=20、min_p=0 | overlay MC 相同，online 原回答 temperature/top_p/top_k 相同 |
| 选点 | 最多 64 点、gap=32、top-logprobs=20，按 `1-top1_prob` 排序 | 当前配置相同，复用 gap 和 tie-break 实现 |
| 标签 | prefix-only MC，selected-segment 差分 | 同定义：相邻选点 `V(next)-V(current)`，最后一段 `terminal_reward-V(last)` |
| 并发 | 历史采集每 shard MC concurrency=128，native batch size=1 | overlay 每 AgentLoop worker 共享 16 个请求槽；调度与随机采样结果不保证逐条相同 |
| critic 更新 | 固定数据收集完成后独立训练 | 本轮先固定 policy advantage，再拟合 critic；更新后的 critic 用于下一轮，无历史 replay |
| 数据标识/导出 | 数据集 prompt UUID，train/rank_eval 分开，含完整 top-k 与 behavior logprobs | prompt token 哈希、全为 train；保留训练 critic 必需字段，但没有旧 collector 的完整 top-k/behavior logprob 导出 |

上述选点等价只针对当前 entropy_weight=0 的配置。开启 entropy 项时，旧
collector 对全部返回候选计算 entropy，online 当前只取 max_candidates
个；不能据此宣称所有可选配方都完全相同。online 的 state 导出记录传入
的 indices，selection score/candidates 不复现旧采集器的详细诊断字段。
合并两种数据时，要通过题目或精确 prompt tokens 对齐 ID、保留 holdout
隔离；不能直接用不同体系的 prompt ID 去重。

现有两份 frozen-policy 配置均未启用 `selected_prefix_mc`：它们只生成
原回答、读取冻结 critic，不收集 MC、不在线更新 critic。需要合并 overlay
启动新实验才能使用这里的新流程。本次仅修代码和验证，未改动运行配置。
历史数据文件未重评分；旧 collector 存在同类线程评分路径，但旧标签的
实际受影响范围尚未逐条审计，不能把代码修复当成已有数据已修复。

验证：在现有 `verl-delta-vllm-20260925` Python 环境运行以下命令，
**179 passed**：

```bash
CUDA_VISIBLE_DEVICES='' VALUE_MODEL_ROOT=/scratch2/zhiyu/code/value_model python -m pytest -q \
  tests/special_standalone/delta_critic \
  tests/trainer/ppo/v1/test_delta_policy_v1_on_cpu.py \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py
```

日志 `/tmp/delta-online-mc-fixes-tests.log`；Ruff check 与 `git diff --check`
通过。覆盖真实数学评分、TQ reward 契约、旧格式 round-trip、不同 UUID
重试、导出中断、critic 恢复，以及 advantage 固定后再拟合的顺序。
另在真实 CPU Ray async actor 内验证了子进程评分，两道数学等价题均为 1；
脚本 `/tmp/delta_mc_ray_verify.py`，日志 `/tmp/delta-mc-ray-verify.log`。
本次未运行新模式 GPU 端到端训练。

## 2026-10-01 新增 selected-prefix MC 在线采样模式

用户明确目标：online rollout 应复用原 critic 数据采集方式，产出的 MC
数据可继续训练 critic；本轮 policy 使用 critic 拟合本轮数据之前的预测。
新模式 `algorithm.delta_policy.rollout_mode=selected_prefix_mc` 实现：

1. 当前 actor 生成回答并按原 uncertainty 配方选点。
2. 同版本 actor 在每个选点之前的精确 token 前缀做 MC continuation。
3. `prefix_only` MC 标签沿用原 `selected_segment` 语义：相邻选点 MC
   value 差，最后一个选点使用回答终局 reward。不是 paired-next-state。
4. 当前 critic 预测 delta，完成 policy segment advantage 准备并写入 TQ。
5. 使用本轮 MC 标签拟合 critic；不重算本轮 advantage。下一轮用更新后的 critic。

配置覆盖片段见 `examples/delta_critic/config_online_mc_overlay.yaml`；应合并
到现有同步 V1 完整配置，而非当作独立 trainer 配置启动。默认旧模式仍为
`critic_prediction`，已有 4B 正式运行未修改/重启。新模式要求
`reward_model.ground_truth`，使用与旧采集器相同的 math-verify；continuation
上限单独显式配置，不能从当前 10000-token answer cap 猜测。服务仍按模型
剩余上下文限制实际 continuation 长度。MC 请求逐个检查 actor version。

每轮在 `trainer.default_local_dir/mc_data/step_XXXXXXXX/attempt_<内容哈希>/` 保存旧采集器兼容的
`rollouts_train_regen.jsonl`、`states_train.jsonl`、`mc_labels_train.jsonl`、
`continuations_train.jsonl`。原 token IDs、MC individual rewards、actor version
和 sampling config 保留，prompt ID 由 prompt token IDs 稳定计算。可复用
`adapt_legacy_rows` 和现有 scalar critic 数据入口；与旧数据合并时仍需按题目
保持原 holdout 划分。在线更新当前只训练本轮数据，尚无自动混合历史数据的
replay sampler。产出格式兼容不等于固定 critic 下的预测分布始终一致。

在线 critic 从原 artifact 初始化，复用其 objective、balanced/MSE loss、
loss mask、target normalization（不重新拟合统计），每轮按整批真实轨迹均值
累积 microbatch，常数 LR AdamW；训练后回到 eval 模式。此后端为 driver 上
单设备 critic，actor 仍走原 FSDP2。训练 critic 的梯度和 optimizer 额外占用
显存，应为新实验配置容量，不能把冻结 scorer 的显存预算直接当作训练预算。

actor checkpoint 同目录增加 `delta_critic.pt`，保存 critic 模型、optimizer、
update count 和 RNG，恢复绑定原 artifact、统计、update config、policy step；
缺少 critic 文件的恢复直接失败。本地 checkpoint 是当前支持路径。
`smoke_policy_v1 --mc` 可验证新模式 actor 和 critic 连续两轮都发生参数更新。
验证：delta 全套及受影响同步 V1 CPU 回归，133 passed、40 skipped、10 warnings
（31.60秒）；定向新增 MC/TQ 接线测试另行复验。Ruff 和 `git diff --check`
通过。覆盖真实 scalar 参数更新、保存恢复后下一步逐参数一致、collector 格式
导出后重新导入、旧数据文件拒绝覆盖、policy advantage 先于 critic 拟合固定。
尚未进行新模式真实 GPU smoke；当前 GPU 有已有任务，没有启动/替换正式实验。

## 2026-10-01 正式 4B 在线实验（当前运行）

用户要求从 Qwen3-4B-Base 在线更新 policy，冻结此前在 brezel 上用
4B-base DAPO 数据训练的 balanced delta critic step 80；随后明确要求
policy standardization 的 mean/std 只在开始校准一次，之后固定，以及
在本机 GPU 6、7 上两卡共同训练 actor。已新增
`algorithm.delta_policy.advantage_stats_mode=initial_rollout`：首次完整
rollout batch 的 raw delta 段广播 token 按总体标准差校准，保存到
`trainer.default_local_dir/delta_advantage_stats.json`。后续更新/恢复重用，
恢复缺少该文件则拒绝重新校准。critic target normalization 的统计始终
来自 checkpoint；不要与 policy advantage 的初始校准统计混淆。

运行目录：`outputs/delta_policy_4b_frozen_20261001/`。详细配置和复现说明
见该目录 README.md/config.yaml/manifest.json；tmux session 为
`delta_policy_4b_frozen_20261001`。GPU 6、7 都用于 actor FSDP2 DP=2、
rollout TP=1/DP=2，冻结 scorer 与 GPU 6 共享显存。全局 batch 32、每卡
microbatch 1、LR 1e-5、clip 0.2、固定初始 reference KL=0.001、初始预算
100 更新；完整上下文 32768（prompt 2048 + response 30720），每 5 更新
固定验证 DAPO 128 + MATH-500 100 并保存 checkpoint。

本次固定统计模式及相关回归：166 passed, 10 warnings（33.63秒），
日志 `fixed_stats_tests.log`；Ruff 和 `git diff --check` 通过。两卡大模型
actor/reference/rollout/scorer 均成功初始化。记录时仍在 step 0 初始验证，
尚未得到首次校准 mean/std 或 policy 更新指标，不能把启动成功写成训练
效果已验收。实时进展以 train.log、metrics.jsonl、status.json 为准。
输出目录内另有 baseline_comparison_zh.md 对照仓库 PPO/GRPO 配方。

启动期间因 Ray socket 路径过长修正过一次，并应用户的固定统计和两卡
actor 指令在任何 policy 更新前重启；旧启动日志已保留。不要重复运行
launch.sh 覆盖现有日志；恢复训练应显式设置 resume 并保留固定统计文件。

本页是新 session 的交接入口，记录 [迁移计划](delta_migration_plan.md) 中 policy 部分的实现、验收证据和下一步。**先读本页的交接快照，再看后面的历史配方；文件存在不等于链路已经验证。**

## 双卡续作快照（2026-09-30 22:12 SGT，优先于下方旧快照）

- 用户确认有两张空卡后，在 GPU 6、7 完成独立 policy trainer 的真实两卡 DP 更新。复用单卡 baseline 的相同输入/配置，三步后最终模型逐参数对照最大差 `2.4782494e-6`，小于烟测阈值；报告 `/tmp/delta-policy-2gpu-20260930/report.json`，两卡日志 `/tmp/delta-policy-2gpu-20260930/two_gpu.log`，产物 `/tmp/delta-policy-2gpu-20260930/two_gpu/final_hf`。
- 同步 V1 也完成两卡真实 `AgentLoop → TQ → 冻结 critic → FSDP2 actor` 单步更新。`smoke_policy_v1.py` 新增 `--gpus 2`，自动载入并重组两份 FSDP2 shard，核对全部 25 个 actor 张量；本次 25/25 均变化，最大变化 `0.0010100007`。日志显示 `actor/delta_rows=4.0`、非零 grad norm，证明逻辑 minibatch 的 4 条真实行已跨卡归约。可复现报告 `/tmp/delta-policy-v1-two-gpu-repro/report.json`，日志 `/tmp/delta-policy-v1-two-gpu-repro.log`。
- 两卡 DP 执行链路已验收。同步 V1 中断恢复的精确对照、正式 8B policy 效果仍未验收；本次仅使用随机小模型和玩具 reward。

复现同步 V1 两卡验收（新输出目录；先确认两卡空闲）：

```bash
export PATH=/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin:$PATH
CUDA_VISIBLE_DEVICES=6,7 HF_HUB_OFFLINE=1 python -m examples.delta_critic.smoke_policy_v1 \
  --model-path /tmp/delta-policy-rounds-actor-snapshot/tiny \
  --critic-artifact /tmp/delta-policy-rounds-actor-snapshot/run/round_000/critic/step_00000001 \
  --output /tmp/delta-policy-v1-two-gpu-next --steps 1 --gpus 2
```

## 续作快照（2026-09-30 21:52 SGT，优先于下方旧快照）

- **同步 V1 现已完成单卡随机 tiny Qwen 两步真实训练**：`AgentLoop → vLLM top-k → TransferQueue → 冻结 critic → 整轮 advantage 统计 → old/ref logprob → FSDP2 actor`。两步分别使用 4 条真实样本，actor checkpoint 相邻最大参数变化为 `0.0010100007`、`0.0010113716`；第 2 步 KL 为 `0.0237395`，rollout staleness 为 `0`。报告 `/tmp/delta-policy-v1-two-step/report.json`，原始日志 `/tmp/delta-policy-v1-two-step.log`。这是执行链路验收，不代表 8B 训练效果。
- 修复 `agent_loop_tq.py` postprocess 对 `delta_policy_settings` 的未定义局部变量；加入真实 postprocess/TQ 字段与整轮准备（包括 padding 行）的 CPU 回归。可复现入口为 `examples/delta_critic/smoke_policy_v1.py`，传入本地 tiny actor、冻结 critic 和新输出目录。脚本会核对两步 actor checkpoint 确实变化。
- 加上源仓差分的 delta 与相关 V1 CPU 测试：**168 passed, 10 warnings**；所涉 policy/V1 文件 Ruff check 和 `git diff --check` 通过。单卡 V1 save 后的精确恢复、两卡 DP、正式 8B 仍未验收；独立阶段编排的两轮/恢复证据仍见下一节。

复现同步 V1 两步烟测（先确认所选 GPU 空闲；路径为本次本机产物）：

```bash
export PATH=/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin:$PATH
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 python -m examples.delta_critic.smoke_policy_v1 \
  --model-path /tmp/delta-policy-rounds-actor-snapshot/tiny \
  --critic-artifact /tmp/delta-policy-rounds-actor-snapshot/run/round_000/critic/step_00000001 \
  --output /tmp/delta-policy-v1-two-step-recheck --steps 2
```

## 续作快照（2026-09-30 21:15 SGT，优先于下方 17:03 记录）

- **独立文件阶段链路已在 GPU 7 用随机 tiny Qwen 完成真实两轮**：采样/MC → critic → policy，各轮 actor 参数最大变化均为 `0.0010000467`；固定 reference 哈希未变化。结果为 `/tmp/delta-policy-rounds-actor-snapshot/report.json`，日志 `/tmp/delta-policy-rounds-actor-snapshot.log`。同目录 `--resume` 约 0.9 秒结束，未重新执行阶段，日志 `/tmp/delta-policy-rounds-actor-snapshot-resume.log`。这是执行链路验收，不代表 8B 效果。
- 本轮纠正在线 PPO 的 old logprob 语义：rollout 采样实际用了 `temperature=0.7, top_p=0.95, repetition_penalty=1.05`，不能直接与 actor 在训练温度 1.0 的 logprob 求比值。`DeltaPolicyConfig.online()` 现在默认 `behavior_logprob_source=actor_snapshot`，更新前通过冻结的当前 actor 用完整上下文和 `train_logprob_temperature` 计算 old；显式 `rollout`/`stored` 在线配置拒绝。外部 `precomputed` 文件须带 actor 指纹、温度、窗口和长度 provenance；在线时指纹必须匹配当前 actor。采样 logprob 保留原始记录。此修正之后才重新跑了上面的两轮烟测。
- 独立训练器新增初始 actor 文件指纹、结构化 normalization/provenance、独立的 `--scoring-microbatch`（默认 1），使 scorer/reference/old 推断不随 actor 物理 microbatch 改变。完成 checkpoint 若中断于 `final_hf` 导出前，`--resume` 可重建导出；已有不同内容的导出拒绝覆盖。`--initialize` 与 `--resume` 可同时提供以复现初始身份。
- 最新 GPU trainer 报告 `/tmp/delta-policy-training-final/report.json`：完成 checkpoint 重导出与三步中断恢复均逐参数完全一致（最大差 `0`）；microbatch=1/2 三步后最大参数差 `4.0833e-5`（loss 最大约 `1e-6`，Adam 放大了归约顺序差异）；离线历史模式参数变化 `0.00200209`。两卡 DP 尚未验收，因为 GPU 0–5 被占用、6 显示 100% 利用率，仅 GPU 7 空闲。日志 `/tmp/delta-policy-training-final.log` 及该目录各子日志。
- 独立路径改动后的 delta CPU 全套：**153 passed, 10 warnings**，日志 `/tmp/delta-policy-final-cpu.log`。同步 V1 接线正在本轮实现及验收，**不能把独立两轮结果当作 AgentLoop → TransferQueue → actor 已通过**。下方 17:03 快照保留为历史进度，凡状态冲突以本节和后续更新为准。

## 交接快照（2026-09-30 17:03 SGT，本轮收尾）

### 1. 用户目标与最近澄清

- 原任务范围：离线 policy 训练、在线 V1 接入、rollout → MC → critic → policy 交替迭代，以及保存/恢复和小模型验收。
- 本轮用户要求继续，并允许 Luna max 子代理；最后要求把当前进展写成足够清晰的文档，供新 session 继续。因此本轮收拢代码与测试状态，**没有启动 8B 正式训练，也不把未完成部分标为完成**。
- 原引用任务 ID：`01a0e7e9-ca55-7943-aa1b-f73625c6dd66`。本轮已经调用 `read_thread`；返回的历史消息有截断。新 session 若要依赖该任务，应再次调用 `read_thread`，不要只依赖标题。
- 沿用的算法选择：在线 clipping 对本轮 old policy，KL 对固定 reference（默认初始 actor）；离线历史公式 clipping/KL 共用 behavior anchor。在线每轮刷新 old，reference 不刷新。
- **2048-token 窗口不是用户指令。** 本轮用户追问“什么时候说过：2048-token 窗口？”，已明确答复：没有找到用户要求；数字来自此前留下的源实验历史对照。它只能作为显式历史复现选项，不能当成用户同意对当前训练截断。当前代码 `DeltaPolicyConfig.historical_offline()` 仍有 `max_length=2048/window=legacy_tail` 默认，**这是现存实现状态，不是用户确认**；新 session 应特别核对离线当前训练与历史复现的入口/默认值区分。在线默认完整上下文，超配置或模型原生上限报错，不自动截断。

### 2. 仓库与工作区

- 根目录：`/scratch2/zhiyu/code/new_verl/verl_meta_token`
- 分支：`delta`；本轮开始和交接时 HEAD 为 `97a817f8`（`Document delta critic migration and validation evidence`）。
- 当前 policy 工作主要是**未跟踪的新文件**，另有上一 session 留下的已跟踪文件修改；本轮没有 commit/push/PR。新 session 必须先看 `git status --short`，不要 reset/clean 丢掉这些文件。
- 已跟踪修改：`docs/delta_migration_plan.md`、`examples/delta_critic/checkpoint.py`、`tests/special_standalone/delta_critic/test_critic_scoring.py`。后两项是此前旧 checkpoint label-mode 显式声明兼容工作，不能当成本轮新写的 policy 代码回退。
- 遵守根 `AGENTS.md`。若以后提 PR，需要 duplicate checks、人类逐行审查/相关测试及 AI assistance 声明；本次只是本地继续实现与交接。

### 3. 代码实际状态

| 文件（相对根目录） | 已有内容 | 当前边界 |
| --- | --- | --- |
| `examples/delta_critic/policy_config.py` | 离线历史/在线两种 loss 配置，anchor、mask、温度、窗口约束 | 非整数 `max_length` 已拒绝；2048 来源见上方澄清 |
| `examples/delta_critic/policy_data.py` | delta 广播、全训练集统计、behavior/reference 对齐、dense 与 response batch | padding 零权重不进入统计；mask/weight 校验；不静默重算 stored logprob |
| `examples/delta_critic/policy_loss.py` | 每样本 PG/KL、V1 nested 输出响应提取、DP/真实行权重归一化 | callback 必须收到整个逻辑更新的 `global_valid_sample_weight`；拒绝 nested 边界错位 |
| `examples/delta_critic/policy_training_config.py` | 独立训练运行参数 | 新落盘，尚无 GPU 验收 |
| `examples/delta_critic/policy_engine.py` | 基于 `FSDPEngineWithLMHead` 注册 `delta_policy`，线性 LR | 新落盘，尚无 GPU 验收 |
| `examples/delta_critic/policy_train.py` | 独立文件数据入口、冻结评分/reference、V1 TrainingWorker/FSDP2 更新、checkpoint/`final_hf` | **初稿，仍须审查和实际执行，不能称可用训练已验收** |
| `examples/delta_critic/policy_online.py` | 同步 V1 配置校验、uncertainty selection、构造 delta examples、冻结 scorer/契约校验 | **只有 helper，未接进 trainer/TQ/actor 更新** |
| `examples/delta_critic/policy_iterations.py` | 分阶段采样、split、critic、policy、固定 reference、轮间权重继承与指纹恢复 | 模拟两轮通过；真实执行仅完成第 0 轮采样/MC/critic |
| `examples/delta_critic/config_policy_iterations_qwen3_8b.yaml` | 正式规模编排配置草案 | 不能当作已跑通的正式训练配置 |
| `examples/delta_critic/config_policy_online_qwen3_8b.yaml` / `config_policy_offline_legacy_qwen3_8b.yaml` | 独立入口配置草案，前者完整上下文，后者显式历史窗口 | 不等于同步 PPOTrainer Hydra 配置；均未 GPU 验收 |
| `tests/special_standalone/delta_critic/test_policy_training.py` | 独立入口新补 CPU tests，覆盖尾窗口 action 对齐和 epoch 尾批 | 与 core 合计 19 passed；最终纳入全套 151 passed |
| `examples/delta_critic/smoke_policy_rounds.py` | 随机 tiny Qwen 两轮验收脚本 | 新增 actor 确实变化、critic artifact 不变、reference 哈希不变检查；这些新增末端检查尚未实际到达 |
| `examples/delta_critic/smoke_policy_training.py` | 3-step baseline、1-step interruption/resume、microbatch=2、离线更新、可选两卡对照 | 本轮新写，**尚未执行** |
| `tests/special_standalone/delta_critic/test_policy_core.py` | 源公式/梯度、统计、mask、nested、分母回归 | 最新子代理报告 17 passed |
| `tests/special_standalone/delta_critic/test_policy_iterations.py` | 编排/恢复/固定 reference/输入变更拒绝 | 最新 9 passed |

**不要混淆两条在线路径：** `policy_iterations.py` 调用真实 V1 `LLMServerClient` 收集 JSONL，再启动独立 critic/policy 进程；它不等于同步 PPOTrainer 的 `AgentLoop → TransferQueue → actor` 路径。后一条尚未接线。

同步 V1 本轮**未修改**的关键位置：

- `verl/trainer/ppo/v1/trainer_base.py`：尚无 delta 分支，默认 advantage 仍走 GRPO/GAE；需 setup 校验、冻结 scorer、整轮 stats、专用字段及更新配置。
- `verl/trainer/ppo/v1/agent_loop_tq.py`：尚无 delta sampling 参数和显式 selected/top-logprob/version 字段处理。
- `verl/workers/engine_workers.py`：尚未安装 `delta_policy_loss` 或添加每个逻辑 minibatch 的真实行权重分母。
- 尚无同步 V1 delta 的专用测试或 tiny launch 配置。

### 4. 已执行验证与未执行验证

| 验证 | 结果 | 证据/范围 |
| --- | --- | --- |
| **交接时 delta 全套** | **151 passed, 10 warnings（30.50 秒）** | `/tmp/delta-policy-handoff-tests.log`；收尾代码已纳入，之后仅格式/unused import/许可证头整理 |
| 本轮修改前 delta 全套 | **139 passed, 10 warnings** | `/tmp/delta-policy-current-tests.log`；含真实安装包 TQ 往返/源差分；不是最终修改后的全套结果 |
| 核心 policy 新增边界测试 | **17 passed**（子代理报告） | `test_policy_core.py`；不代表 GPU actor 已跑 |
| 编排/恢复测试 | **9 passed** | `test_policy_iterations.py`；含新加中断阶段输入变更拒绝 |
| 现有 V1 CPU 回归 | **10 passed, 8 warnings** | `/tmp/delta-policy-v1-regression.log`，下方给出命令；未修改 V1 hooks，故不证明 delta 接线正确 |
| `policy_online.py` | `py_compile` 通过 | 未添加专门 CPU tests；未运行同步 V1 GPU |
| Ruff | **policy/两个 smoke/全部 policy tests check 通过，format 已执行** | `/tmp/delta-policy-handoff-ruff.log`；收尾移除 unused import、修复超长行 |
| 许可证 / diff 空白 | **显式新 Python 文件许可证检查、`git diff --check` 通过** | `/tmp/delta-policy-handoff-license.log`；新 trainer test 已补 Apache 头 |
| 真实 tiny 采样 → MC → critic | **第 0 轮完成** | 下节保存路径；16 条 rollout，split 12 train / 4 eval；critic 1 step |
| 真实 policy 更新/两轮结束 | **未验证** | 当时入口不存在而失败；入口之后才落盘，尚未重试 |
| 精确 checkpoint 恢复、microbatch 对照、两卡 DP | **policy 均未验证** | 不可借用此前 scalar critic 的验收结果 |
| 正式 8B 训练/数学效果 | **未运行** | 随机小模型只用于执行链路 |

### 5. 环境与真实烟测产物

实际可用 Python 环境（本轮复用现有环境，未安装依赖）：

```bash
cd /scratch2/zhiyu/code/new_verl/verl_meta_token
export PATH=/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin:$PATH
export VALUE_MODEL_ROOT=/scratch2/zhiyu/code/value_model
```

- 必须让该环境的 `bin` 进入 PATH：旧尝试仅用绝对 Python 路径启动，vLLM 子进程找不到已有的 `ninja`，导致 EngineDeadError。根因来自 Ray worker 日志，不是显存不足。
- `uv` 位于 `/tmp/verl-uv-bin/uv`；Ruff 位于 `/tmp/delta-runtime/bin/ruff`。如需管理 Python 环境按 AGENTS 使用 uv，不要擅自改现有共享环境。
- 本轮 shell 普通 sandbox 启动报 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，后续使用工具提供的 escalation 通过自动审核执行。新 session 若环境恢复正常则直接使用普通工具。
- 本轮最后观察：GPU 0–5 有其他任务，GPU 6 显示 100% 利用率（虽 0 MiB），仅 GPU 7 确认空闲。**这是历史快照，继续前必须重新 `nvidia-smi`，不要覆盖别人的任务。** 本任务的烟测进程已经退出；没有留下正式训练后台任务。

旧失败目录/日志（保留证据，不要混用）：

- `/tmp/delta-policy-rounds-20260930`、`/tmp/delta-policy-rounds-20260930.log`：前一 session 的 ninja/PATH 失败。
- 对应 Ray 日志：`/tmp/ray/session_2026-09-30_15-48-24_851396_2527107/logs/`，worker 输出含 `No such file or directory: 'ninja'`。

**建议继续使用的新目录：**

```text
/tmp/delta-policy-rounds-resumed-20260930/
  tiny/                    # 随机 Qwen3，初始 actor，也是固定 reference
  prompts.jsonl            # 16 个数字玩具 prompt
  sampling.yaml / critic.yaml / policy.yaml / iterations.yaml
  run/run.json
  run/round_000/
    .collect.done.json     # 已完成采样/MC，带输出指纹
    .split.done.json       # 已完成 prompt split
    .critic.done.json      # 已完成 critic，带输出指纹
    data/                  # rollout/state/MC/continuation JSONL
    split.json             # 12 train / 4 eval
    critic/step_00000001/   # COMPLETE、model.pt、engine 状态等
    policy.yaml            # 待使用的新 policy 入口配置
```

日志：`/tmp/delta-policy-rounds-resumed-20260930.log`。真实采样和 critic 在 16:42（机器日志时间）完成；随后 `examples.delta_critic.policy_train` 当时还不存在，子进程退出。**这条日志不代表新 trainer 的运行失败或通过；新 trainer 尚未试跑。** 无 `report.json`/两轮 `COMPLETE.json` 可作为完成证据。

新加 `.stage.started.json` 输入指纹逻辑是在该次进程启动后写入代码的，因此旧目录可能只有 `.done.json`；这不是文件丢失。首次新代码恢复会为未完成阶段建立 started marker，后续拒绝更改其输入。

### 6. 新 session 建议执行顺序

1. `git status --short` 并读本页及 `AGENTS.md`；确认所有未跟踪文件仍在。先审查刚落盘的 `policy_train.py`，不要直接启动 8B。
2. 跑 CPU 回归及静态检查，优先处理下节列出的独立 trainer 未验证风险。同步 V1 helpers 也需补 tests。
3. 重新检查 GPU，仅选择空闲卡；修复 trainer 后，从上面的第 0 轮 policy 阶段恢复真实两轮烟测。不要重新收集已经完成的数据，也不要手工伪造 `.done.json` 或改现有输入绕过 fingerprint。
4. 两轮成功后重跑 `--resume`，确认阶段不重复执行；再跑 `smoke_policy_training.py` 验证精确恢复、microbatch 以及离线更新。两卡验证只在确有两张空闲卡时执行。
5. 实现同步 V1 接线及专门 tests/config，然后独立完成真实 `AgentLoop → TQ → actor` 验收。文件阶段编排成功不能替代这项。
6. 根据实际日志更新本页和迁移计划，再决定正式配置；只有用户要求时再做正式大模型运行或发布动作。

CPU/静态命令：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/special_standalone/delta_critic
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  tests/workers/test_engine_workers_mini_batch_trace_on_cpu.py
/tmp/delta-runtime/bin/ruff check examples/delta_critic/policy*.py \
  examples/delta_critic/smoke_policy*.py tests/special_standalone/delta_critic/test_policy*.py
git diff --check
```

修复/审查之后的真实恢复命令（假设 GPU 7 当时仍空闲）：

```bash
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 python -m examples.delta_critic.smoke_policy_rounds \
  --output /tmp/delta-policy-rounds-resumed-20260930 --resume \
  > /tmp/delta-policy-rounds-next-session.log 2>&1

# 两轮结束后再执行；output 必须是新的不存在目录。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 python -m examples.delta_critic.smoke_policy_training \
  --source-round /tmp/delta-policy-rounds-resumed-20260930/run/round_000 \
  --output /tmp/delta-policy-training-next-session
```

### 7. 未验证风险/审查清单

- `policy_train.py` 刚落盘后发现的两处问题已在代码中修正：reference 温度显式从 loss config 传入；critic engine 使用 `initial_artifact=args.critic`。这些修正尚未 GPU 验证，尤其必须确认 scorer 实际使用 checkpoint scalar head。
- 全部 active delta 相同时，standardize 会因 std 小于 `1e-8` 显式失败，当前没有静默 fallback；玩具数据/零奖励数据可能触发，应按算法约定处理而非绕过。
- `RowIterator` 在收尾时已改为当前 epoch 尾部返回剩余真实行，再补零权重行至 `world_size × microbatch` 倍数，不再混入下一 epoch；有 CPU 测试。其 DP/GPU 轨迹与历史训练的 sampler/尾累积仍未实测等价，不能宣称完整历史轨迹复现。
- 检查独立 trainer 的 initial actor 权重指纹、保存的结构化 stats/provenance、precomputed behavior 的 actor/温度/窗口 provenance。仅路径或数据文件哈希不等于完整身份验证。
- 检查“最后 checkpoint 已 COMPLETE，但 `final_hf` 导出前中断”的恢复：不能因为 global_step 已到终点就无法重建导出；不要覆盖/删除已有 artifact 掩盖错误。
- 训练 `--stop-after`、零更新恢复、完整 RNG/optimizer/scheduler/data cursor 的精确恢复仍须实测。所有 rank 的失败/导出异常也需避免 collective 挂起。
- 真实 nested actor 输出、稀疏 active mask、合成 padding 行的每 minibatch 分母必须对齐。FSDP 的 microbatch backward 不额外平均；callback 使用 `dp_size / global_valid_sample_weight`，不能漏传或重复缩放。
- 同步 V1 client 使用普通 `LLMServerClient`，它保留 extra_fields；`SingleTurnAgentLoop` 也传递 extra_fields。异步 `FullyAsyncLLMServerClient` 的任意 extra_fields 合并另有问题，本轮范围限制 sync，不要误宣称 async 已支持。连续 token merge/response 截断仍需核对 top-logprob 行对齐。
- 最终 CPU 全套、所涉新 Python 文件 Ruff/许可证检查已完成，见上表；没有执行仓库全量 pre-commit/mypy，也没有 policy GPU 结果。不要把 CPU 151 passed 写成端到端完成。


### 8. 收尾时的稳定状态

- 三个子代理都已结束本次工作，修改已经落盘；无需等跨 session 的 agent 状态。
- 独立 trainer 修复了 critic artifact 初始化和 reference 温度参数，补了 epoch 尾批；两个 8B YAML 均经过配置解析，`py_compile` 通过。
- 核心 tests 最终 17 passed，core + training tests 合计 19 passed；主代理最终 delta 全套 151 passed。相关日志路径已列在验证表。
- 没有继续启动 GPU 训练。**下一步是审查独立 trainer 的剩余风险并恢复 tiny policy smoke，而不是直接跑正式 8B。** 同步 V1 仍须实际接线。
- 本页后续各节保留历史配方和目标流程；关于“已完成到哪里”，以上交接快照优先。


## 已确认的配方

2026-09-30 用户确认：在线 clipping 对本轮 old policy，KL 对固定 reference；离线历史复现保留源公式。

| 项目 | 离线历史复现 | 在线 V1 / 交替迭代 |
| --- | --- | --- |
| Clipping anchor | stored 或显式 precomputed behavior | 本轮 behavior / 更新前 actor；下一轮刷新 |
| KL anchor | 与 clipping 相同的 behavior | 固定 reference，默认初始 actor；不随轮次更新 |
| KL 公式 | 源 sampled reverse KL，无最终值 clamp | PPO low_var_kl，最终值 clamp 到 [-10, 10] |
| KL 系数 | 0.01 | 新配方起点 0.001，可配置，未调优 |
| PG/KL mask | active state × policy token × validity | 同左，改变 anchor 不扩大 mask |
| Advantage | scalar delta 反归一化后按 segment 广播 | 同左 |
| Advantage 统计 | 整个训练集拟合一次，整个阶段复用 | 本轮完整训练集合拟合一次，本轮更新期间复用 |
| Policy 上下文 | 2048 尾窗口，最左 token 无前驱 logit | 完整上下文，超过配置/原生上限报错 |
| PG clipping | 对称 0.2，无额外 dual clip | 同左；仅统一 KL anchor，不声称所有 PPO 默认项相同 |
| 样本权重 | 历史逻辑 batch=1；尾累积单独处理 | 真实样本均值；合成 padding 行权重为零 |

Advantage 使用 population std，在 policy mask、padding、窗口裁剪之前拟合；不能替换成当前 minibatch whitening。原始 response mask 保留，不能靠 advantage 置零代替 loss mask。

训练 logprob 温度默认 1，独立于 rollout 温度。Stored logprob 原样读取，不静默重算。Precomputed 模式记录 actor 身份、温度、上下文并拒绝缺失行。同一轮多个 PPO epoch 固定 old logprob 和 advantage。

## value_model 6–8 月实验对照

来源为 `/scratch2/zhiyu/code/value_model/delta_value_llm_exp/parallel_runs/requested/` 下运行 YAML、final metadata 与 stdout。日期来自运行目录；当前源脚本不自动代表历史执行版本。

| 运行批次 | Policy / 信号 | Advantage / old logprob | 已核对训练参数及产物 |
| --- | --- | --- | --- |
| policy_grpo_final_20260628_102533 | state_only / scalar delta，seed 42–45 | segment + standardize / stored | RL epochs 20，batch 1，累积 128，长度 2048；各 final step 20 |
| policy_grpo_final_20260630_013904 | state_only / scalar delta，seed 42–45 | segment + standardize / stored | RL epochs 40，累积 128，长度 2048；各 final step 40 |
| policy_grpo_final_20260630_125137 | state_only / 17-atom distributional delta | expectation 后 segment + standardize / stored | 两个 final step 40；当前 scalar 导入器不支持该 head |
| policy_grpo_final_20260705_222042 | state_only / TD delta budget 1/2/4/8/16/32 | segment + standardize / stored | 六个 final step 40；不从其他批次推断其余预算 |
| policy_grpo_final_20260710_011646 | state_only / composition h2 delta | segment + standardize / stored | RL epochs 40，累积 8，长度 2048；各 final step 600；该 critic objective 不在首批范围 |
| policy_grpo_final_hybrid_b32_best_20260717_run | state_only / hybrid scalar delta | segment + standardize / stored | RL epochs 40，累积 128，长度 2048；final step 40 |
| policy_grpo_final_hybrid_current_code_budget8_replica_20260718_011500 | state_only / hybrid scalar delta | segment + standardize / stored | RL epochs 20，累积 128，长度 2048；final step 20，每步 eval，每 5 步 save |
| continuation_grpo_budget_20260801_233623 | response_grpo / continuation reward | 无 state advantage / precomputed | b32：RL epochs 2，累积 176，长度 4096；budget 8/16/32 均 final step 640 |

已核对 delta final metadata 共同为 clip 0.2、KL 0.01。代表性配置为 Qwen3-8B、BF16、LR 1e-5、weight decay 0、grad clip 1。7/17 日志实际为 460 train / 52 eval rollouts，不能以 YAML 的 2048 prompts 替代；policy 使用 rl_epochs=40，不使用 train.epochs=100。

`batch × accumulation × world_size` 是 nominal capacity。历史 sampler 补齐、epoch 尾累积、实际 LR 序列必须单独核对；源 loss 差分通过不能证明历史整条多卡轨迹完全一致。

8 月 continuation GRPO 本轮仅作实验对照，不实现该 objective。8 月 value MSE/centering 属于其他 critic 配方，不作为 delta 默认值。源在线脚本强制 selected-state critic 监督，且未显式开启 policy 标准化；新在线按用户选择采用 response + balanced MSE critic 和标准化 advantage，仅参考源阶段编排。

## 验收顺序

1. 源 CPU 差分：广播、统计、窗口、old/reference、PG/KL、梯度。
2. 真实 TQ 字段往返与 V1 nested actor 输出对齐。
3. 真实 FSDP2 小模型更新、microbatch/DP 分母、保存恢复。
4. 真实 AgentLoop 生成到 actor 更新及下一轮权重同步。
5. 至少两轮交替训练，阶段间/阶段内恢复，reference 不刷新。

模拟 runner 的两轮测试只验证编排，不代表真实两轮训练已完成。每项结果在总迁移计划中分别登记。

## 交替迭代运行

从仓库根目录、具备 V1 训练与 vLLM 依赖的环境执行：

```bash
python -m examples.delta_critic.policy_iterations \
  --config examples/delta_critic/config_policy_iterations_qwen3_8b.yaml \
  --output outputs/delta_policy_rounds --dry-run

CUDA_VISIBLE_DEVICES=0 python -m examples.delta_critic.policy_iterations \
  --config examples/delta_critic/config_policy_iterations_qwen3_8b.yaml \
  --output outputs/delta_policy_rounds

CUDA_VISIBLE_DEVICES=0 python -m examples.delta_critic.policy_iterations \
  --config examples/delta_critic/config_policy_iterations_qwen3_8b.yaml \
  --output outputs/delta_policy_rounds --resume
```

Dry run 输出解析配置及命令，不加载模型或创建运行目录。配置路径相对于启动目录。每轮按稳定 prompt hash 划分 train/eval；同 prompt 的所有 responses 保持同 split。数据过少缺少任一 split 时显式报错。

`ours_segment_legacy` 使用 selected-segment 标签；`ours_local` 使用 paired 标签但 learned delta 仍按 segment 广播；`mc_local_oracle` 跳过 critic，仅使用 selected token 信号。

阶段顺序为采样/MC、critic、policy。下一轮继承上一轮权重并重建优化器和统计；阶段内 resume 旨在恢复完整状态（policy 尚待实测，见交接快照）。阶段输入/输出指纹不匹配时拒绝跳过。固定 reference 默认初始 actor，永不替换为下一轮 actor。

采样入口为单 GPU 服务；训练 nproc 单独配置。各阶段顺序释放资源，采样显存比例需匹配可用设备。

## 小模型复现命令

在已安装依赖的环境中先激活环境（或把它的 `bin` 放到 `PATH`），以便 vLLM 编译进程能找到 `ninja`。仅用绝对路径调用 Python 并不等价于激活环境。

```bash
# 真实采样 → MC → critic → policy，两轮，随机小 Qwen 模型。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 python -m examples.delta_critic.smoke_policy_rounds \
  --output /tmp/delta-policy-rounds

# 原地恢复；完成阶段校验输入/输出指纹后跳过。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 python -m examples.delta_critic.smoke_policy_rounds \
  --output /tmp/delta-policy-rounds --resume

# 重用第一轮数据，验证精确恢复、microbatch 一致性及离线更新。
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 python -m examples.delta_critic.smoke_policy_training \
  --source-round /tmp/delta-policy-rounds/run/round_000 \
  --output /tmp/delta-policy-training
```

显卡编号按实际空闲设备调整。`smoke_policy_training --two-gpu` 需要两张可见 GPU，额外比较单卡与两卡最终权重。两轮烟测验证每轮 actor 参数发生变化、下一轮配置使用上一轮导出的 actor、reference 文件哈希保持不变。它使用随机模型与玩具奖励，仅验证执行和数据契约，不衡量数学能力。

中断阶段在启动前保存输入指纹；恢复时若数据或权重被改动会拒绝继续。训练阶段的优化器、学习率、数据游标及统计量恢复还需通过训练入口的 checkpoint 校验。

## 评分和退化诊断（2026-10-03）

比较历史实验时，应在同一套规则下重评两边保存的 validation 回答。
训练 step N 的 rollout 来自更新前的 actor；validation step N 在第 N 次更新后执行。
修改评分代码不会重训历史 checkpoint，也不会改变已启动任务的冻结 source。

同步 reward manager 在线程池内调用评分函数。`sampling_io.score_text` 现在将
线程中的 math-verify 计算转交给可复用的 spawn 子进程主线程，保留解析超时；
进程执行失败向上传播，避免作为错误答案记零。原回答和 continuation 的
MC 评分原本已在子进程中执行，因此历史验证线程漏判不能直接解释短答 actor
优势。新评分入口和明确最终答案的配置见 `examples/delta_critic/README.md`。

continuation 的 `rm_scores` 现在记录实际 `completion.reward`。历史 expansion
日志把这 192 行记为零，因此 `critic/score/mean` 是 original 的奖励和除以 320，
既不是 128 条 original 的准确率，也不是整个扩展 batch 的准确率。
历史 MC labels 中的 continuation reward 仍然可用；短答使用独立的
`delta_true_reward`，此次分数字段修正不会改变既有 delta 优势公式。

`algorithm.delta_policy.kl_mask_scope` 默认仍为 `policy`；现在也支持显式
设置 `response`，以覆盖首个选点前的 response token。两者始终使用固定初始
reference。禁用的 continuation 与 padding 在两种 scope 下均不产生 KL 梯度。
这是新增的可选优化范围，历史正常启动路径会拒绝 `response`。

每步额外记录 `delta_policy/policy_token_coverage`，以及更新前的
`preupdate_ref_kl_{response,policy,outside_policy}_row_mean` 和对应行数。
这些是采样 token 上的 k3 估计，沿用 low-var KL 的截断；不能等同于完整词表
KL 或将其接近零解释为生成质量恢复。`zero_reward_row_advantage_mean`、
`zero_reward_positive_row_advantage_fraction` 及 positive-reward 对照项用于
检查 frozen critic 是否仍在强化失败回答；有 `delta_true_reward` 时优先使用它。

## O-TD / O-H 在线诊断（2026-10-04）

同步训练现在直接把以下指标写入原有 metrics logger；不额外采样、不增加
critic forward，也不改变 loss、advantage、优化器或数据划分。必须用新 source
启动任务；已运行的 Ray worker 和历史 source 快照不会自动加载这些修改。

| 指标前缀 | 用途 | 关键读法 |
| --- | --- | --- |
| `delta_policy/health/{all,original,continuation}/` | 分开看原始回答和续写的奖励、长度、选点、PG 覆盖率 | `reward_mean` 与 `response_length_mean` 同时下降时，先检查输出退化；不同训练 batch 的奖励不是固定验证集准确率 |
| `delta_policy/health/{positive_reward,zero_reward}/` | 对照真实奖励与最终送入 actor 的 advantage | `positive_row_advantage_fraction`、`advantage_mean/std/max_abs` 揭示失败回答仍获正优势或信号失控；这是相关性，成功轨迹也可以包含负的局部 delta |
| `delta_policy/health/{critic,short_outcome}/` | 区分 critic 信号与短答 outcome 替换 | `raw_delta_*` 按选点计数，`advantage_*` 按实际 PG token 计数；二者分母不同 |
| `actor/delta_forward_*` | actor 当前 minibatch 相对本轮 old policy 的 ratio、clip 与 sampled KL | 单个 optimizer step 前 ratio 可以仍为 1；这些不是完整更新后的 policy 测量 |
| `delta_mc/{all,train,diagnostic}/` | MC 数据是否还有可学习信号 | `all_zero_reward_state_fraction`、`mixed_reward_state_fraction`、`td_target_zero_fraction` 与 `branches`、`td_pairs` 一起看 |
| `delta_critic/diagnostic_{prefit,postfit}/` | 同一批独立 critic 诊断数据上的预测质量 | TD 与 terminal 分别记录 RMSE、bias、相关性、非零标签符号正确率、零预测基线 |
| `delta_critic/diagnostic_update/` | 一次 critic 更新本身的影响 | `*_rmse_change` 为 postfit − prefit，正值表示此次更新增大留出误差；`*_prediction_drift_rmse` 记录同批输出变化 |

`diagnostic_{prefit,postfit}` 的 `td_zero_baseline_skill` 定义为
`1 - mean((prediction - target)^2) / mean(target^2)`，小于零表示不如始终预测零。
标签全零时该比值和非零符号正确率没有意义，必须检查相应 `*_valid`、
`*_nonzero_target_count`、`*_target_zero_fraction`。相关性还要求预测和标签都有
非零方差。所有空总体均有显式计数；零占位值不能当作健康信号。critic 更新窗口
默认只有 8 个留出 TD pair，单窗方向正确率不能作稳定结论。

MC 的 `branch_at_cap_fraction` 使用 max_tokens、有效 response 上限和有效总长度
上限中的最小值，并给出 `branches_with_known_cap`；到达上限不等同于确认被截断。
8 次采样全部失败也不证明真实成功概率为零。只有 full actor grid 中相邻的两个
MC anchor 才贡献 `td_pairs`，未观测位置不会作为零标签计入。MC train/diagnostic
分组遵循原有 prompt 角色，不影响训练隔离。

Actor minibatch 指标沿用 loss 的真实行分母及 DP/SUM 聚合。每行先对 PG token
取均值，再按 row weight 累积；没有 PG token 的行贡献零，需结合
`delta_forward_active_pg_row_fraction` 解读。clip 指标分开报告 ratio 超出区间
和按 advantage 符号真正触发 clipped objective 的比例。它们都使用 detach，
不参与梯度。microbatch 内使用 SUM、DP 间补偿并平均，再由现有 trainer 对各次
actor minibatch/epoch 的观测取均值；不能当作最后一次更新后的值。

判断顺序：先看 MC 是否退化为全零、原始回答是否变短，再比较同批 critic
prefit/postfit 的方向与零基线 skill，最后看真实奖励分组的 advantage 和 actor
ratio/clip/KL。跨窗口的 critic RMSE 同时受到 policy、题目和标签分布变化影响，
不能单独用来判断 critic 是否改善。进一步区分初始化 critic、online 更新和
policy loss 的因果影响，仍需 matched frozen-critic 对照或固定数据上的 checkpoint
重评分；新增日志提供定位证据，不自动给出因果结论。
