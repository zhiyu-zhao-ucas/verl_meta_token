# Delta 训练迁移计划与进度

更新日期：2026-09-25。工作分支：`delta`，从 `main` 创建。

本文记录从 `value_model` 向当前 verl 仓库迁移 delta 训练的整体计划、已确认的算法选择、当前实现及待决事项，供后续开发接续。详细接口用法见 [delta 数据契约 README](../examples/delta_critic/README.md)。

## 1. 目标与参考仓库

- 目标仓库：`/scratch2/zhiyu/code/new_verl/verl_meta_token`。
- 算法来源：`~/code/value_model`，重点为 `delta_value_llm_exp/`。
- 工程参考：`~/code/verl-agent/zhiyu-zhao-ucas-verl-agent`，重点为 `recipe/delta_critic/`。
- 目标：先复现数学推理任务中的 token/segment delta 语义，再接入 critic 推理、policy 更新和 actor–critic 交替迭代。
- 当前 verl 默认使用 V1 trainer 与 TransferQueue。旧版 `verl-agent` 的 worker/trainer 接入方式需要适配，不能整段直接移植。

**决策规则：两个参考仓库出现实现冲突时，明确列出双方行为、影响和可选方案，由用户决定；不得静默选择。无冲突部分可以继续推进。**

## 2. 已确认的选择与未决冲突

| 项目 | `value_model` 行为 | `verl-agent` 参考行为 | 当前决定 |
| --- | --- | --- | --- |
| Delta 读取与传播 | 多个 selected token 位置读取；learned delta 广播到对应 segment | 每行 action 的最后有效 token 读取，广播到整个 action | 用户已指定与 `value_model` 一致 |
| Policy advantage 归一化 | critic 驱动 final 运行采用训练集广播后 token 的标准化 | 支持当前 batch 的 action 标准化、仅缩放等 | 按用户要求核对实际运行记录；提供 `value_model_final_grpo` 标准化配置 |
| Checkpoint 接受范围 | 支持 `selected_state` 等监督配置及多种目标 | 当前 frozen worker 只接受其中一部分配置 | 待用户决定支持范围；未实现 reader/validator |
| 未归一化 checkpoint 元数据 | `none` 可以保存 `mean/std=None` | 参考 worker 要求其支持的 identity/standardize 格式 | 待用户决定兼容规则；数值恒等变换已支持，metadata 转换未实现 |
| 超长输入上下文 | critic 训练取完整 rollout 尾窗口；逐状态推理取 selected prefix 尾窗口 | 对完整 action 行截取尾窗口 | 待用户决定；当前仅提供无截断位置映射及 padding |

代码位置从原计划的 `recipe/delta_critic/` 调整为 `examples/delta_critic/`：当前仓库的 `recipe/` 是独立 Git 子模块，在其中新增文件不能直接纳入根仓库的 `delta` 分支。子模块配置未修改。

## 3. 整体迁移拆分：七个部分

### 第 1 部分：语义契约与数据接口

**状态：已完成基础实现和 CPU 数值验证；checkpoint 元数据兼容与截断策略仍待选择。**

工作内容：

1. 固定状态、action、response 下标及模型输出读取位置。
2. 独立表达 label mode、critic supervision mask、policy advantage scope 和两套 normalization。
3. 定义 rollout、selected state、target、advantage、mask 与 provenance 数据结构。
4. 兼容已有 rollout/MC 行，保留 token IDs、数据划分和原始字段。
5. 实现 target 构造、广播、归一化、反归一化、位置及 padding 的纯函数。
6. 用可手算样例与源函数差分测试验证。

验收：给定相同旧数据、原始 critic 预测及配置，在受支持的无截断范围内，target、mask、advantage 和统计量与源实现一致。

### 第 2 部分：rollout、状态选择与 MC 标注

**状态：数据链路、V1 生成接口适配及 Ray V1 批量采样入口已实现；CPU 差分与缓存 Qwen3-8B 的旧单样本 V1 client 烟测均通过。新的批量入口尚未完成真实 GPU 验收；TransferQueue 训练数据流尚未端到端验收。**

参考源文件：`01_collect_rollouts_vllm*.py`、`02_select_states.py`、`03_estimate_values_mc_vllm.py`。

- 先复用现有 rollout/MC 数据，再接入当前 verl 的生成接口。
- 支持 prefix/next continuation 与状态选择，明确 terminal shortcut。
- 保留原 token IDs、behavior logprobs、状态关联 ID、actor 版本、采样配置和 MC budget。
- 对新实验制定 prompt 级训练/评估划分；导入旧实验时保留原划分，不能声称源项目已统一采用 prompt 级切分。

当前实现：`collection.py` 接收 V1 `AgentLoopOutput.as_dict()`/TransferQueue 的未 padding 行，保留生成时 `rollout_log_probs` 为 behavior logprobs，记录 actor 版本和采样配置；新实验用稳定 hash 按 prompt ID 切分。`state_selection.py` 对含 top-k 的行提供 uncertainty 选择，对普通 V1 行提供显式 token indices 选择（普通 AgentLoop 输出不含源项目的 top-k 分布）。`mc_labeling.py` 去重 prefix/next 请求、执行明确的 terminal shortcut、记录 MC budget/采样配置和每个状态标签；`v1_sampler.py` 调用 V1 `LLMServerClient.generate`。CPU 模拟服务和真实 Qwen3-8B client 均完成了新生成行 → 状态 → MC → 第 1 部分校验的闭环。真实烟测从 client 输出构造未 padding 行，未经过 AgentLoop/TransferQueue 的生产数据流。

新增 `batch_sample.py` 使用当前 verl 的 Ray V1 vLLM 服务批量读取源格式 prompt、生成 rollout、执行 uncertainty 选点与 MC 标签、保存源命名的三份 JSONL。V1 vLLM 在请求 `delta_top_logprobs` 时将 top-k 候选放入 `TokenOutput.extra_fields`；普通生成不携带该附加数据。`config_sampling_qwen3_8b.yaml` 与源 aligned 配置的有效采样字段对齐，默认 `prefix_only`，`ours_local` 使用显式 `--mc-mode paired_next_state`。新链路保留源 split，不采用 `collection.py` 的新实验哈希切分。CPU 模拟 V1 client 测试与源 prompt、reward、配置差分已通过；生产 GPU 批量运行尚未验证。运行命令和无法保证逐样本一致的差异见 [README](../examples/delta_critic/README.md#batch-sampling-with-the-verl-v1-server)。

限制：V1 `TokenOutput` 只有 token IDs、没有源 vLLM 的 completion text，所以适配器要求显式指定 continuation 解码时是否跳过特殊 token。当前 V1 vLLM/SGLang 后端已将原始 `finish_reason` 放入 `extra_fields`，避免 vLLM 的 `stop_reason="completed"` 混淆 stop 和 length；旧输出若缺少该字段，paired 模式的最后 token 仍要求显式 legacy terminal 选择。当前只支持单轮、全部 response token 由 actor 生成的 MC；含工具/观察 token 的行会拒绝。

验收：状态关联和 prefix 一致，标签可追溯；小规模新采样生成的数据可直接通过第 1 部分校验。

### 第 3 部分：delta 模型、loss 与 checkpoint

**状态：未实现。**

参考源文件：`05_train_token_scalar_model_accelerate_local.py`、`modeling_token_scalar.py`。

**3.1 语义基线和模型。**先固定本轮要复现的配置组合：`target_type=delta`、`output_dim=1`、`delta_objective=local_td0`，以及 `mse`/`nonzero_balanced_mse`、`selected`/`response` 监督、`none`/`standardize` target normalization。源模型是预训练 causal LM 的 transformer backbone 加一个 `Linear(hidden_size, 1)`；`value_layer=-1` 取末层，其他层按源 `hidden_states` 下标解释，移除不用的 LM 输出头。预测读取位置是第 4.1 节的 `P+t`，**不**使用 next-token logprob 的左移位置。将第 1 部分的未 padding token ID、target、`critic_target_mask`、`critic_signal_mask` 转成模型 batch；保留选中位置和 prompt 末尾等源字段的独立校验，截断后重新映射绝对位置，不能把无效位置夹到边界。

**3.2 损失和统计。**在 `examples/delta_critic/` 新增可单独调用的 PyTorch loss：源 `_masked_loss` 的 MSE 为 `sum(0.5*(pred-target)^2*mask)/max(sum(mask),1)`；balanced MSE 以**显式 signal mask** 分成 signal/background，各自先按 mask 求均值，再各乘 `0.5`，缺一组时回退普通 MSE。这样选中但 delta 为零的状态仍属于 signal。训练集、holdout 之后、collate/截断之前拟合 target mean 和 population std；训练、评估、checkpoint 均复用同一统计，推理按 `raw=pred*std+mean` 反归一化。`none` 模式保持数值恒等，元数据按第 2 节待决兼容规则处理。将 target mask、signal mask 与 padding mask 分开，检查 `loss_mask=response` 的非选中零 target 仍参加训练。

**3.3 接入 verl。**先在单卡 PyTorch 上与源函数逐 token 对齐，再接入 V1 的 `TrainingWorker`/engine。V1 trainer 在 `verl/trainer/ppo/v1/trainer_base.py` 已创建 `model_type="value_model"` 的 critic worker，并用 `set_loss_fn(value_loss)` 注入普通 value loss；delta 模式应有显式配置和专用 loss/输出适配，不得复用 GAE 的 `returns`/value clipping 语义。首个 engine 选与 checkpoint backbone 兼容的实现，核对其 `value_model` head 结构、forward 输出、padding/sequence parallel 和 DP loss 分母；不兼容时提供 delta 模型适配或专用 worker，不能仅改配置名。保持普通 PPO/value 路径原有行为。随后检查有效 global batch、梯度累积、混合精度、gradient checkpointing、优化器与学习率调度和源 run 的有效配置相符。

**3.4 Checkpoint。**定义一个显式导入器读取源 `.pt` 中的 `model`、`config`、`target_type`、`step`、`target_normalization`、`training_method`；校验 backbone/tokenizer、`value_layer`、head 形状、目标类型、loss/objective、统计量及训练最大长度。导入后用相同 token ID 做源/verl forward 差分，再转换成 verl engine checkpoint；保存源 checkpoint 路径/哈希和语义元数据。恢复训练另存 optimizer、scheduler、global step、统计量及模型版本，并验证恢复前后下一步更新一致。第 2 节未决的 checkpoint 接受范围和 `none` metadata 兼容规则决定导入器允许哪些配置；未获明确支持的类型（例如 distributional、value centering）应报错，不静默降级。

**3.5 Hybrid 顺序。**基础 `local_td0` 通过后，再实现源 `hybrid_terminal_composition`：在原始 delta 单位下对 suffix 位置求和，与 terminal target 比较，按独立 terminal std 缩放；总 loss 为 `td_weight*td_loss + terminal_weight*terminal_loss`。其 terminal 统计、suffix mask、有效标记和 loss 权重单独保存，不与 delta target normalization 混用。其他源 objective、分类/分布式 head 逐个登记为后续兼容范围；每增加一种都重复以下等价门禁。

**验收门禁：**固定权重、token ID 和 batch，逐位置比较源与 verl 的 head 输出、归一化前后 target、mask、两项 loss 及梯度；覆盖零 delta、空 signal/background、padding、首末选中位置和超长拒绝/窗口行为。单卡训练至少完成一次参数更新；保存/恢复后 forward 与下一步参数更新一致；再做所选 engine 的多卡短程运行。记录容差、dtype、数据集 hash、命令和实际结果，未通过时不进入第 4 部分。

### 第 4 部分：冻结 critic 推理 worker

**状态：未实现。**

工程参考：`verl-agent/recipe/delta_critic/critic_core.py`、`critic_worker.py`。

**4.1 输入与输出契约。**worker 输入从 V1 `AgentLoopOutput.as_dict()` / TransferQueue 的未 padding 行构造，包含原始 prompt/response token IDs、选中下标、response validity、行 ID、actor/critic 版本和必要的采样 provenance。每个选中 `t` 读取包含当前 action token 的 `prompt+response[:t+1]`；输出按原 response 长度对齐的 `delta_pred_normalized`、`delta_pred_raw`、`critic_signal_mask`，并保存源 checkpoint 与归一化版本。未选中及 padding 位置不作为预测；真实预测为零时 mask 仍为一。按行 ID 和 selected index 还原批处理前顺序，检查重复、缺失和越界。

**4.2 模型和窗口。**冻结第 3 部分的 critic（`eval`、`inference_mode`、无 optimizer 更新），按长度分桶、microbatch 和 engine 允许的动态 token budget 推理，并验证 offload/colocation 后权重不变。当前 `TrainingWorker.infer_batch` 可复用执行和资源管理，但 V1 `_compute_values` 将结果写为普通 `values`，再按 `response_mask` 转换；delta 应单独写出上述三个字段，不能借普通 value 的输出映射或 logprob 的左移提取。截断策略取决于第 2 节决定：逐状态 selected prefix 的尾窗口与完整 action 行尾窗口是不同语义；实现选定规则并记录截断起点、重算读取位置，选中 token 被截掉时显式失败。

**4.3 Checkpoint 边界。**仅加载第 3 部分导入器认可的 head/objective/metadata；从 checkpoint 的 critic 训练配置读取最大长度与 normalization，验证 tokenizer/token IDs 一致。源 `09_train_policy_grpo_offline.py` 对 scalar checkpoint 将未启用 normalization 当作 `mean=0,std=1` 使用；该数值规则可以先验证，原始 `None` metadata 的接受范围仍按第 2 节决定。任何统计、head 维度或截断方式不匹配均停止评分，并在 batch metadata 中记录具体原因。

**验收门禁：**用同一旧 `.pt`、相同 token ID 和 selected index，与源 `_critic_build_selected_eval_rows` / `_predict_critic_selected_states` / 反归一化链路逐行对照；比较单行与混合长度 microbatch、不同 padding、重排、跨设备及 worker 恢复后的输出。确认 raw 与 normalized 只差固定 checkpoint 统计，mask/位置完全一致，冻结 worker 无梯度和参数变化；再在真实 V1 TransferQueue 行上通过一次评分，不把当前 V1 client 烟测当作此项已完成。

### 第 5 部分：delta advantage 与 policy 更新

**状态：仅第 1 部分的纯函数准备完成；训练链路未接入。**

参考源文件：`offline_grpo.py`、`09_train_policy_grpo_offline.py`。

**5.1 V1 数据流。**以第 2 部分的 rollout/selected state/MC 行及第 4 部分的 raw delta 为输入，在 V1 `_step_once` 的采样和 `_compute_old_log_prob` 之后、actor 更新之前加入专用 delta 评分与 advantage 阶段。TransferQueue 按同一行 ID 保存 selected index、critic 版本、raw delta、mask、归一化统计 ID 与最终 advantage；嵌套 response 字段按 V1 `response_to_nested` / padding 规则往返，补齐/负载均衡产生的 padding 行不参与统计或 loss。默认 V1 `_compute_advantage` 调用 GRPO/GAE，不能用其默认白化与分组返回值代替 delta 算法；delta 模式显式绕开该步骤。第 3 部分的 critic 更新也不应在同一次冻结评分/actor 更新中意外触发，交替训练由第 6 部分编排。

**5.2 与源算法逐项对齐。**`value_model` 的 `state_only` final 路径用 `state_advantage_source=delta`：每个 selected token 的 raw delta 广播至下一 selected token 之前，保留源 `state_mask`；`ours_local` 的 label 虽是局部差分，learned signal 仍按 segment 广播。`mc_local_oracle` 单独按 selected-token mask 运行，不能复用 learned 广播。将第 1 部分 `state_advantages`、`state_advantage_stats`、`normalize_state_advantages` 的结果与源逐行比对；训练集广播后 active state token 拟合一次 mean/population std，评估和同一训练阶段复用，统计发生在最终 policy mask 与截断之前。在线交替时必须明确“本轮 rollout 训练集合”或其他可复现的拟合范围及何时冻结；没确定范围前不以当前 mini-batch 白化替代。

**5.3 Logprob 和损失映射。**源 `old_logprobs` 是选定的 behavior logprob（stored 或 precomputed），同时是该 final `state_only` loss 中 sampled reverse KL 的参照；其 `clipped_policy_loss` 使用 current/old logprob、advantage 和 `state_mask`。V1 的 `rollout_log_probs` 是生成时 behavior，`old_log_probs` 在默认模式下由当前 actor 重算，bypass mode 才直接取 rollout 值；V1 的 `ref_log_prob` 是另一条 reference-policy 路径。为复现源 final run，应在 delta 配置中显式选择 stored/precomputed behavior 作为 clipping 与 sampled KL 的参照，检查 temperature、token 对齐及缺失 logprob，不将 V1 默认重算结果或 `ref_log_prob` 偷换成源值。先按源公式实现 `state_only` 的 PG clipping 和 sampled reverse KL；若复用 verl 的 `compute_policy_loss_vanilla`，须差分证明 clip、KL 公式、`loss_agg_mode` 与分母相同，不相同则使用专用 loss。源其他 `objective`、KL 参照或 rollout correction 组合需单独配置和验证，不自动宣称等价。

**5.4 Mask 真正进入 actor 更新。**最终 `policy_loss_mask = state_advantage_mask ∩ policy_token_mask ∩ response_valid_mask`。V1 当前 `verl/workers/utils/losses.py::ppo_loss` 使用 `response_mask` 计算 PG、entropy 和 KL，`AgentLoopWorkerTQ` 还把 `loss_mask` 设为 `response_mask`；因此 delta actor loss 必须读取专用 `policy_loss_mask`，使 PG、源所需 KL 和各自分母只计 active token。保留原始 `response_mask` 用于序列有效性和 nested/padded 转换；不要为了筛 token 改写它，也不要只把 masked advantage 置零。对源未使用的 entropy、额外 reference KL、reward-KL、rollout correction 设置显式关闭/兼容条件，防止默认训练配置悄悄改变目标函数。

**第 3–5 部分的接轨与等价边界：**

| 环节 | `value_model` 做法 | 当前 verl 接入点 | 必须保持的等价语义 |
| --- | --- | --- | --- |
| Critic 训练 | JSONL dataset、Accelerate、token scalar head、源 `.pt` | `TrainingWorker`、`model_type=value_model` engine、专用 delta loss 和 checkpoint 转换 | 同 token 上的 scalar、监督位置、target/terminal 统计、loss 与梯度 |
| 冻结评分 | selected eval rows、源 `.pt` 加载、逐状态读取 scalar | V1 critic worker `infer_batch`、TransferQueue delta 字段 | selected index 的 `P+t` 读取、窗口、反归一化与零值 signal mask |
| Actor 更新 | offline `state_only`、behavior `old_logprobs`、segment advantage | V1 `_step_once` 的 delta advantage 分支与专用 actor loss | 广播/训练集标准化、old/behavior 角色、PG/KL 公式和 active-token 分母 |

此表只承诺第 3.1 节固定的 scalar delta 基础组合及随后单独验收的 hybrid/其他模式。源项目支持的其他 objective 或 head 不会因为 verl 已有同名 `value_model` 或 PPO 能力就自动获得等价性；不支持的 checkpoint/配置应明确拒绝。所有等价结论以同一输入、权重、dtype、有效 batch 和差分测试结果为准。

**验收门禁：**固定小 batch 对照源 `offline_grpo.py` 的 advantage、训练集统计、old logprob 解析、clipped PG 与 sampled KL，以及 `09_train_policy_grpo_offline.py::_losses` 总损失和梯度；覆盖不同 segment 长度、零 delta、受 mask 排除 token、stored/precomputed logprob、padding 与多轮 PPO epoch。检查关闭 active token 后 PG/KL 分母同步变化；小模型在真实 V1 TransferQueue 路径完成一次 actor 更新，确认 actor 参数变化、critic 参数不变，记录与源的数值容差和配置。仅当这些门禁通过，才将第 6 部分的交替迭代接入。

### 第 6 部分：actor–critic 交替迭代

**状态：未实现。**

参考源文件：`run_online_iterations.py`。

流程：

```text
当前 actor rollout → 状态选择与 MC 标注 → critic 更新
→ 冻结 critic → actor 更新 → 下一轮
```

- 保存每轮 actor/critic 版本、数据来源、配置和恢复状态。
- 固定有效 global batch、更新步数及 MC budget；GPU 数变化时检查这些比较条件。
- 明确每轮归一化统计的拟合范围和复用范围。

验收：至少完成一个小规模迭代，并能从阶段 checkpoint 恢复而不混用版本或数据。

### 第 7 部分：评估、配置与端到端验收

**状态：第 1、2 部分的 CPU 验证及第 2 部分真实 V1 client 接口烟测完成；TransferQueue 数据流、模型训练和端到端效果评估未验证。**

- 数据与数值：target、mask、反归一化、advantage 和 loss 对齐。
- 训练链路：小模型短程运行，检查 actor 更新、critic 冻结和恢复。
- 实验效果：delta 误差、相关性、符号/排序指标，以及最终数学任务准确率。
- 同数据、同预算比较 GRPO、MC delta oracle 与 learned delta。
- 训练配置按具体实验来源记录，不将不同阶段的 epochs、梯度累积或长度混成一个默认值。

## 4. 第 1 部分已实现的具体语义

### 4.1 状态和位置

设 prompt token 数为 `P`，response 为 `y[0:L]`，选中下标为 `t`：

- 状态：`s_t = prompt + response[:t]`。
- 当前 action：`response[t]`。
- Delta scalar 输出位置：`P+t`，输入包含当前 token。
- 当前 token 的 policy next-token logit 位置：`P+t-1`。

这里交换的是 **scalar 输出**，不是 hidden vector。源模型内部执行 `scalar_head(hidden)`，调用方再选择当前 token 位置的 scalar。出处为源 `modeling_token_scalar.py` 的 `forward`，以及 `09_train_policy_grpo_offline.py` 的 `_critic_build_selected_eval_rows` 和 `_predict_critic_selected_states`。

### 4.2 标签与传播独立

| 路径 | 标签 | policy 信号 | 作用范围 |
| --- | --- | --- | --- |
| `ours_segment_legacy` | 下一个选中状态 value 减当前 value；最后使用 terminal reward | critic 预测 | segment |
| `ours_local` | 当前 token 前后 value difference | critic 预测 | **仍然是 segment** |
| `mc_local_oracle` | 当前 token 前后 value difference | MC 标签 | selected token |

paired 标签读取优先使用已有 `delta`，否则用 `v_next-v_prefix`。二者同时存在但不一致时，适配器保留原始字段并输出诊断，不静默重算标签。

### 4.3 Mask 与归一化

- 区分 `critic_target_mask`、`critic_signal_mask`、`state_advantage_mask`、`policy_token_mask` 和 `response_valid_mask`。
- 选中且 delta 为零仍是 signal；`loss_mask=response` 下非选中位置的零 target 也参加监督。
- `policy_loss_mask` 为 state、policy、validity mask 的交集。
- Critic normalization 按训练集监督 target 统计，发生在 collate/截断之前。
- Policy normalization 按训练集广播后的 active state token 统计，发生在 policy mask 相乘及 collate/截断之前，因此长 segment 权重更大。
- 两套统计独立，均为 population std；空统计集或 `std < 1e-8` 报错。
- 评估复用训练统计，不重新拟合；预测反归一化使用 critic target 统计。
- 当前 mask 契约为二值；fractional weights、value/centering、distributional 和 hybrid 辅助 target 不在本次基础实现范围。

## 5. 实际运行配置依据

### 5.1 已核实的 critic 驱动 policy 记录

以下路径均相对于 `~/code/value_model/delta_value_llm_exp/parallel_runs/requested/`，读取的是小型 `offline_grpo_metadata.json`，未加载模型权重：

1. `policy_grpo_final_hybrid_b32_best_20260717_run/ckpt/hybrid_b32_best/offline_grpo_final_hybrid_b32_best/final/offline_grpo_metadata.json`
2. `policy_grpo_final_hybrid_current_code_budget8_replica_20260718_011500/ckpt/offline_grpo_final_hybrid_current_code_budget8_replica/final/offline_grpo_metadata.json`

两份实际完成记录均为：

```yaml
objective: state_only
state_advantage_source: delta
state_advantage_normalization: standardize
state_advantage_scope: selected_token_to_before_next_selected_state
```

| 记录 | 统计 token 数 | mean | std | 保存 step |
| --- | ---: | ---: | ---: | ---: |
| 7 月 17 日 hybrid b32 | 513134 | -0.000526664631712179 | 0.015950032255528268 | 40 |
| 7 月 18 日 hybrid b8 replica | 513134 | -0.0038557279886972874 | 0.021601428190504198 | 20 |

这些统计量仅是配置依据，**不能作为新数据的默认数值**。此处“最新同类记录”限于本次检查的 `parallel_runs/requested` 中保留的 final metadata，不代表正在运行的训练或全仓所有实验。

### 5.2 Continuation GRPO 是另一条路径

实际核实的 budget 32 final metadata：

`continuation_grpo_budget_20260801_233623/ckpt/budget_32/continuation_grpo_final_budget_32/final/offline_grpo_metadata.json`

该记录为 `response_grpo`、`behavior_logprob_source=precomputed`、`state_advantage_source=null`、`state_advantage_stats=null`，最终 step 为 640。其 `state_advantage_normalization=none` 表示这条运行未使用 state advantage 路径，不能用来覆盖 delta final-GRPO 配置。

### 5.3 用户提供的训练配置参考

以下是后续训练阶段的参考，不是本次已接入的训练参数；真正启动前需核对选定 run 的配置、命令行覆盖和有效 batch。

| 实验类型 | 用户提供的运行设置与注意点 |
| --- | --- |
| 7 月常规 critic | Qwen3-8B token scalar，MC budget 1/2/4/8/16/32；通常 20 epochs、batch 1、LR 1e-5、长度 2048、每 10 步评估、不早停；样本量匹配实验可改变梯度累积，例如 hybrid b8 使用 19 |
| 7 月 critic 驱动 final policy | `state_only` 与 advantage 标准化；典型 40 RL epochs、batch 1、LR 1e-5、每 50 步评估、保存 final；梯度累积有 8 和 128 等实际配置 |
| 7 月下旬至 8 月 continuation policy | `response_grpo`、预计算 logprob、长度 4096；budget 32 的实际配置为 2 RL epochs、累积 176、最多 640 更新步，并保存 step 30 和 final |
| 8 月 25 日 value critic 对照 | `value_resp_mse`、target 标准化、20 epochs、batch 1、累积 8、LR 1e-5、长度 2048；属于 value 对照，不能当作 delta 目标的默认配置 |

配置入口包括源 `config_qwen3_8b_densecritic_aligned_sweeps.yaml`、各 run 的 `configs/`，以及 `run_continuation_grpo_budget_sweep.sh`。不要把 aligned YAML 中记录的其他实验说明当作当前 run 的有效参数。

## 6. 已落地文件

| 文件 | 内容 |
| --- | --- |
| [contracts.py](../examples/delta_critic/contracts.py) | Schema v1、显式算法配置、rollout/state/vector/stats 类型与校验 |
| [legacy_adapter.py](../examples/delta_critic/legacy_adapter.py) | 已加载旧数据行的转换、metadata 保留、兼容诊断、显式 terminal helper |
| [target_ops.py](../examples/delta_critic/target_ops.py) | 标签、mask、advantage 广播、两套归一化及 final-GRPO 配置 |
| [batch_adapter.py](../examples/delta_critic/batch_adapter.py) | 无截断读取位置、selected prefix、response padding 和最终 policy loss mask |
| [collection.py](../examples/delta_critic/collection.py) | V1 未 padding rollout 导出、behavior logprob 与 prompt 级 split |
| [state_selection.py](../examples/delta_critic/state_selection.py) | 源 uncertainty 或显式 token indices 状态选择 |
| [mc_labeling.py](../examples/delta_critic/mc_labeling.py) | prefix/next 请求去重、MC 标签与 provenance |
| [v1_sampler.py](../examples/delta_critic/v1_sampler.py) | V1 LLMServerClient continuation 采样适配 |
| [batch_sample.py](../examples/delta_critic/batch_sample.py) | Ray V1 批量生成、uncertainty 选点、MC 和源命名 JSONL 输出 |
| [sampling_io.py](../examples/delta_critic/sampling_io.py) | 源格式 prompt 读取、模板与数学奖励 |
| [config_sampling_qwen3_8b.yaml](../examples/delta_critic/config_sampling_qwen3_8b.yaml) | 与源 aligned 配置一致的有效采样参数 |
| [smoke_v1_gpu.py](../examples/delta_critic/smoke_v1_gpu.py) | 单 GPU 真模型 V1 rollout → MC → 契约手动验收脚本 |
| V1 vLLM/SGLang async server | 在 `TokenOutput.extra_fields` 保留原始 `finish_reason`；vLLM 按请求返回 top-k 候选 |
| [README.md](../examples/delta_critic/README.md) | 数据接口语义、使用示例、实际运行出处及未决范围 |
| [test_contracts.py](../tests/examples/delta_critic/test_contracts.py) | CPU golden、输入校验、边界行为及可选源实现差分测试 |
| [test_collection.py](../tests/examples/delta_critic/test_collection.py) | V1 行导出、状态选择、模拟 MC 服务端到端及失败路径 |
| [test_batch_sample.py](../tests/examples/delta_critic/test_batch_sample.py) | 模拟 V1 top-k → uncertainty → paired MC，以及源配置/prompt/reward 差分 |
| [check_license.py](../tests/special_sanity/check_license.py) | 增加 2026 年个人贡献者版权标头识别，原规则保留 |

源仓库未修改；当前 PPO trainer 和训练循环未修改。V1 vLLM rollout 后端除结束原因 metadata 外，还在显式请求时返回 top-k 候选。本次批量入口尚未进行 GPU 运行或创建 PR。

## 7. 已执行验证

包含源实现差分的测试命令：

```bash
VALUE_MODEL_ROOT=/scratch2/zhiyu/code/value_model \
  /scratch2/zhiyu/miniconda3/envs/verl/bin/python -m pytest -q tests/examples/delta_critic
```

第 1 部分当时结果：**23 passed**。第 2 部分初始结果为 **30 passed**。加入批量入口的模拟 V1 top-k/MC 和源 prompt、reward、配置差分测试后，当前整组结果为 **32 passed**（提供 `VALUE_MODEL_ROOT`）。不提供该变量会跳过三个外部源仓库差分测试。

差分测试直接导入源 `offline_grpo.py`，并从源训练文件 AST 中选择 dataset/statistics 定义，以免引入训练依赖；第 2 部分还从源文件 AST 读取状态间隔选择与 MC request 函数比较。CPU 测试只验证纯函数；下面的 GPU 烟测另行验证 V1 client 的真实生成，不覆盖模型 forward、checkpoint、TransferQueue 训练数据流或分布式训练。

V1 后端文件通过 Python 编译检查；当前 `verl` 环境有 vLLM 0.22 但缺少 Ray，另一现有环境有 Ray 但 vLLM 0.11 低于本仓要求。真实验收使用仅在临时进程生效的依赖路径补充和旧 CUDA 12 `flash_attn` 屏蔽，未更改安装环境。另一环境还缺少 `transfer_queue` 和 `pytest-asyncio`，因此 V1 extra_fields/TransferQueue 原有测试未完整运行。正式复现应使用一个同时具备兼容 Ray、vLLM 与 CUDA 扩展的环境。

真实 V1 小采样使用本机缓存 Qwen3-8B snapshot `b968826d9c46dd6066d109eabc6255188de91218`、单张 RTX PRO 6000、`LLMServerManager.create` 和 `LLMServerClient.generate`，用 [smoke_v1_gpu.py](../examples/delta_critic/smoke_v1_gpu.py) 连续运行两次，均退出码 0。每次产生 16 个 response token、16 个 behavior logprob、selected token `[0, 15]`、两行 MC 标签（各 2 次采样），均通过 `adapt_legacy_rows`，诊断为空。原始 `finish_reason=length` 与 `stop_reason=completed` 同时出现，验证了保留原始结束原因的必要性。烟测奖励仅为文本是否含 `2`，只验证接口和数据链路，不代表数学任务准确率。兼容环境中的复跑入口为：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python examples/delta_critic/smoke_v1_gpu.py \
  --model-path /scratch2/zhiyu/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --actor-version Qwen3-8B-cache-smoke
```

上述命令假定环境已具备兼容的 Ray、vLLM 和 CUDA 扩展。本次实际运行额外使用了 `/tmp` 中的进程级依赖兼容 shim；它不是仓库的一部分，也不应作为正式环境配置。

覆盖重点：

- 两种标签模式、selected/response supervision 和零值 signal。
- learned segment 广播与 MC selected-token 路径。
- 两套归一化统计、反归一化、policy mask 交集。
- 首末 token、padding、错误索引、重复状态、prefix 不一致、缺失预测。
- terminal/length 显式选择、空统计集和零方差。

可手算的基准样例：

```text
response 长度 = 6，selected = [1, 4]
v_prefix = [0.2, 0.7]，paired v_next = [0.3, 0.4]，terminal_reward = 1

segment target        = [0, 0.5, 0,   0,   0.3,  0]
paired target         = [0, 0.1, 0,   0,  -0.3,  0]
selected critic mask  = [0, 1,   0,   0,   1,    0]

若 critic 预测恰好等于 paired 标签：
ours_local advantage  = [0, 0.1, 0.1, 0.1, -0.3, -0.3]
mc_local advantage    = [0, 0.1, 0,   0,   -0.3,  0]
```

许可检查显式传入新 Python 文件及检查器自身，结果通过。`git diff --check` 通过；该命令不覆盖未跟踪新文件，agent 另检查了新增 Python 的 120 列限制。环境未找到 uv/Ruff，本次使用已有 Python 环境执行测试，不涉及环境安装；**未执行 Ruff**。

## 8. 建议后续顺序

1. 确定 checkpoint 接受范围、normalization metadata 兼容和超长窗口策略；同时在正式兼容环境中验收 AgentLoop/TransferQueue 未 padding 行到 delta adapter 的生产数据流。
2. 按第 3 部分完成 scalar delta 模型、loss、旧 checkpoint 导入和数值门禁；优先用已有 critic checkpoint 验证转换和推理，随后完成训练及恢复。
3. 按第 4、5 部分完成冻结评分 → V1 delta advantage → actor 更新的最小闭环，逐阶段执行源函数差分和真实 TransferQueue 验收。
4. 基础组合验收后实现 hybrid，接入第 6 部分交替迭代，最后完成同预算基线与端到端效果评估。

后续发现新的跨仓库差异时，更新第 2 节，记录用户选择后再实现依赖部分。每完成一个阶段，更新本文件中的状态、证据与实际验证命令。
