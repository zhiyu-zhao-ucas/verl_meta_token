# TD 和 Hybrid 复现及 online 更新交接说明

Last updated: 10/08/2026

接收人使用 [统一 Bash 入口](../examples/delta_critic/run_td_hybrid_handoff.sh) 完成从初始 critic 训练、TD policy、Hybrid policy，到 Hybrid step5 online critic 更新、policy续训和正式评测的全流程。脚本提供 CPU 预检查、独立运行目录、阶段日志、checkpoint 导出和正式评分。

**将脚本、`handoff_assets.tgz`、`handoff_assets.sha256.json` 和本说明一起提交到 GitHub。** 代码包包含之前复现的源码、驱动、完整配置、固定题目以及两个 parquet，首次运行自动校验并解压。接收人不需要原机器的 `outputs`、私有源码或私有 critic checkpoint。历史实验目录保持原样，新结果写入接收人指定的目录。

## Clone 后从头运行

取得仓库及兼容的训练环境后，运行：

```bash
cd /path/to/cloned/repository
export UV_BIN=/path/to/uv
export PYTHON_BIN=/path/to/compatible_env/bin/python
export WORK_DIR="$PWD/outputs/my_td_hybrid_run"
export BOOTSTRAP_DIR="$PWD/outputs/my_initial_critics"
export SAMPLE_GPUS=0,1,2,3,4,5,6,7
export TD_INIT_GPUS=0,1,2,3,4,5,6,7
export HYBRID_INIT_GPUS=0,1,2,3
export TD_GPUS=0,1,2,3
export HYBRID_GPUS=0,1
export CRITIC_GPU=0
export EVAL_GPU=0

bash examples/delta_critic/run_td_hybrid_handoff.sh fresh-all
```

`fresh-all` 自动下载固定 revision 的公开 Qwen3-4B-Base（若已设置 `BASE_MODEL` 则使用本地完整snapshot），从提交的512道初始critic题目重新采样，训练TD80和Hybrid80，再执行全部policy训练及评测。两个parquet和全部固定prompt已随提交文件提供，不需访问原数据目录或重新下载DAPO等评测集。下载Base需要网络；后续流程使用离线模式。

**从头运行会产生新的初始critic权重。** 它复用历史采样预算、P95筛选和训练配方，但不会要求新权重SHA256等于历史权重；新身份写入 `critic_identity.json`。需要严格使用原TD80／Hybrid80权重时，使用下文的 `historical` 模式并提供那两份artifact。两种模式不能混称为同一权重身份的复现。

## 流程与范围

```text
Base + TD80     → TD policy step1–10     → 导出并评测 step5、step10
Base + Hybrid80 → Hybrid policy step1–10 → 导出并评测 step5、step10
                         │
                         └─ 完整 step5 checkpoint
                            → 512题原回答，每个 state16条MC续写
                            → 461题训练、51题critic验证
                            → Hybrid80权重初始化，新增critic训练80步
                            → 在20/40/60/80中按验证总loss选优
                            → 恢复policy step5，step6重算优势统计
                            → policy step6–10 → 导出并评测step10
```

`fresh-all` 先构建初始TD80和Hybrid80，再执行上述流程。`all` 使用已有的TD80和Hybrid80。online 更新只对 Hybrid 实施；两个引用任务没有实现 TD online 更新。online 是一次独立数据收集和 critic 更新，随后冻结选出的 critic 完成五个 policy step。

## 需要交接的文件

| 文件或目录 | 用途 |
| --- | --- |
| `examples/delta_critic/run_td_hybrid_handoff.sh` | 唯一 Bash 操作入口 |
| 本说明 | 路径配置、运行步骤和验收方法 |
| `examples/delta_critic/handoff_assets.tgz` | 可提交的原复现代码、完整配置、prompt、parquet及评分器 |
| `examples/delta_critic/handoff_assets.sha256.json` | 提交代码包的SHA256 |
| 自动解压的 `td_hybrid_handoff_assets/` | 本机运行素材缓存，不必提交 |
| Qwen3-4B-Base 完整 snapshot | Base actor、固定 reference、tokenizer、critic backbone |
| `train.parquet` 和 `validation.parquet` | 已随代码包提供：4096题训练数据及228题训练内验证数据 |
| TD80 artifact | `fresh-all`自动训练；历史权重模式另需 `model.pt`、`metadata.json` |
| Hybrid80 artifact | `fresh-all`自动训练；历史权重模式另需 `model.pt`、`metadata.json` |
| 可选完整 Hybrid step5 checkpoint | 用已有 step5 直接执行 online；须保留模型、optimizer、scheduler、RNG、dataloader状态 |
| step5 所属 `checkpoints/delta_advantage_stats.json` | 使用外部 step5 时同时交接，供旧统计对照 |

Base snapshot 身份为 `906bfd4b4dc7f14ee4320094d8b41684abff8539`。脚本校验 Base 三个权重分片、两个 parquet 和两个 critic 的 SHA256；不能换成同名其他权重或重新下载后不同内容的数据。

素材中保留各阶段源码，避免当前工作区修改改变历史行为：

| 素材目录 | 代码来源与作用 |
| --- | --- |
| `source_td/` | 历史 TD revision `b8c42ad836e300be8e9566ad78a5468204f4ec0a` 的源码快照 |
| `source_hybrid/` | 已完成复现使用的 Hybrid 历史源码快照 |
| `source_online/` | 已运行 online 更新的源码快照，包含允许 step6 首次优势校准的检查 |
| `source_initial_hybrid/` | 初始Hybrid critic训练的历史源码快照 |
| `evaluation_tools/source/` | 独立评测使用的修正评分源码 |

此外包含原 `driver.py`、`collect.py`、`policy_driver.py`、critic 数据准备/head转换/选优/验收函数，以及评测导出、生成、评分、报告代码。`prepare` 将它们放入新的运行目录，只迁移路径和驱动校验入口；不会编辑原复现代码。旧机器的 SSH 调度、审批记录和临时恢复脚本不参与运行。

## 更新提交素材

普通接收人无需执行这一步。维护者要从原实验重新生成提交代码包时，在拥有历史 `outputs` 的原仓库运行：

```bash
export PYTHON_BIN=/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin/python
export UV_BIN=/tmp/verl-uv-bin/uv
export ASSET_DIR="$PWD/outputs/td_hybrid_handoff_assets"
bash examples/delta_critic/run_td_hybrid_handoff.sh export-assets
bash examples/delta_critic/run_td_hybrid_handoff.sh pack-assets
```

如果素材目录已存在，要重新导出时指定新的 `ASSET_DIR`。导出命令拒绝覆盖已有目录。素材附有逐文件哈希 `manifest.json`，包含题目和parquet，不含模型权重。`pack-assets`更新仓库中的 `.tgz` 及校验文件。

原机器输入位置如下，供交接人定位文件；接收人可将其复制到其他绝对路径。

| 输入 | 原位置 |
| --- | --- |
| Base | `/scratch2/zhiyu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| 训练数据 | `outputs/delta_policy_4b_online_mc_20261001_3467_cap10k/train.parquet` |
| 训练内验证 | `outputs/delta_policy_4b_online_mc_20261001_3467_cap10k/validation.parquet` |
| TD80 | `outputs/delta_qwen3_4b_base_dapo_20260927/train/runs/balanced/step_00000080` |
| Hybrid80 | `outputs/delta_critic_4b_hybrid_shared_td_20261002/checkpoints/step_00000080` |
| 已有 Hybrid step5 | `outputs/delta_historical_reproduction_20261005/hybrid/checkpoints/global_step_5` |

**复制目录时解引用符号链接。** Hugging Face snapshot、critic checkpoint 或旧实验目录可能指向原机器缓存、`/tmp` 或其他输出目录。使用例如 `rsync -aL <源目录>/ <目的目录>/`，确认接收目录里的权重是真实文件。新运行目录也包含本机输入别名，迁移后需要重新 `prepare`。

## 环境与资源

使用 Linux、Bash、`uv`、`flock`、`realpath`、`nvidia-smi` 和已安装训练依赖的 Python 环境。Python 环境管理使用 `uv`。完整包版本保存在素材 `environment.json` 和 `requirements-frozen.txt`。

原环境主要版本为 Python3.12.13、torch2.13.0+cu130、vLLM0.29.0、transformers5.12.1、Ray2.55.1、Hydra1.3.3、OmegaConf2.3.1、torchdata0.11、math-verify0.9。`check` 严格比较 Python 和核心依赖版本；其他包差异记录在新运行目录的 `environment.json`。

`requirements-frozen.txt` 是交接的版本清单，其中 GPU wheel 和本地包可能需要原 wheel 源或环境副本，不能保证只通过公开 PyPI 安装齐全。接收人应取得原兼容环境或相应 wheels，然后指定 `PYTHON_BIN`；所有 Python 命令均经 `uv run --no-project --python ...` 执行。脚本使用离线模式，运行前须准备完整模型、tokenizer 和数据。

| 阶段 | GPU 配置 | 主要内存压力 |
| --- | --- | --- |
| TD policy | 4卡，DP4、TP1 | 4B全参数actor/reference、rollout和delta critic |
| 初始TD critic | 8卡，全局batch128、microbatch16 | P95筛选后的完整序列 |
| 初始Hybrid critic | 4卡，全局batch256、microbatch1 | TD128＋terminal128，序列最长32768 |
| Hybrid policy及续训 | 2卡，DP2、TP1 | 同上，保留历史并行度 |
| 独立采样 | 一张或多张卡，每张TP1副本 | 4B模型与rollout KV cache |
| online critic | 单卡，全局batch256、microbatch1 | 完整序列和activation，配置上限32768 |
| 正式评测 | 单卡，TP1 | 8192 token生成预算 |

确认 GPU 显存、CPU 内存和磁盘足够。一次 online 历史采样产出512条原回答、10,663个state、170,608条续写；新采样的state数会随生成结果变化。JSONL、四份critic checkpoint、policy完整checkpoint及多个推理导出均占空间。`prepare` 不计算磁盘需求，启动前用 `df -h` 检查目标盘，并为完整产物预留空间。checkpoint 放持久存储，避免依赖 `/tmp` 长期保存。

## 使用历史权重运行

要固定使用原历史critic权重，在当前仓库布局下运行，将下列路径改为接收人本机实际位置。`WORK_DIR` 必须是尚不存在的新目录。提交代码包自动解压，`ASSET_DIR`可省略；两个parquet的路径也可省略，默认使用代码包中的文件。

```bash
export UV_BIN=/path/to/uv
export PYTHON_BIN=/path/to/compatible_env/bin/python
export ASSET_DIR=/data/td_hybrid_handoff_assets
export WORK_DIR=/data/runs/td_hybrid_run_001

export BASE_MODEL=/data/models/Qwen3-4B-Base
export TRAIN_PARQUET=/data/datasets/train.parquet
export VAL_PARQUET=/data/datasets/validation.parquet
export TD_CRITIC=/data/critics/td80
export HYBRID_CRITIC=/data/critics/hybrid80
export CRITIC_MODE=historical

export TD_GPUS=0,1,2,3
export HYBRID_GPUS=0,1
export SAMPLE_GPUS=0,1,2,3
export CRITIC_GPU=0
export EVAL_GPU=0

# 不启动GPU：检查输入身份/环境，再生成新运行目录和驱动。
bash examples/delta_critic/run_td_hybrid_handoff.sh check
bash examples/delta_critic/run_td_hybrid_handoff.sh prepare

# 顺序完成TD、Hybrid及Hybrid online，包含所有正式评测。
bash examples/delta_critic/run_td_hybrid_handoff.sh all
```

GPU 编号是本机 `nvidia-smi` 显示的物理编号。脚本在每个 GPU 阶段开始前检查编号、数量和显存占用，所选卡显存已用超过512MiB时退出；它不会自动等待空卡。运行前在作业调度系统中取得这些卡，阶段检查不能替代资源预留。脚本不执行 `ray stop`、不结束其他用户进程，也不使用旧机器的远端节点。

各阶段顺序运行，TD、Hybrid、采样、critic 和评测可以复用物理卡。只有原独立采样中的TP1副本会并发生成。交接版的评测依次在单卡完成，沿用每次128题的评测批次、采样和评分配置。

## 分阶段运行

从提交文件构建初始critic、但暂不执行policy时：

```bash
bash examples/delta_critic/run_td_hybrid_handoff.sh bootstrap
# 后续分阶段运行要显式绑定本次新critic。
export CRITIC_MODE=rebuilt
export TD_CRITIC="$BOOTSTRAP_DIR/td_checkpoints/step_00000080"
export HYBRID_CRITIC="$BOOTSTRAP_DIR/hybrid_checkpoints/step_00000080"
export BASE_MODEL="${BASE_MODEL:-$PWD/outputs/td_hybrid_base_model}"
bash examples/delta_critic/run_td_hybrid_handoff.sh prepare
```

初始采样使用封存的512题、每题一次原回答、最多64个state、每state16条MC、原回答和新续写最大30720 token，模型上下文32768。保留总输入长度的P95以内回答及相应label/continuation，按文件顺序将末尾10%作为critic验证集。TD80从Base新建scalar head训练80步，随后Hybrid80也从Base和新head独立训练80步；不是从TD80继续训练。Hybrid只使用前8条terminal分支，但TD标签仍用16分支。初始采样允许长答，与policy及online阶段的2048预算不同。

初始critic采样集中为连续512题，并使用接收人的TP1副本数分配请求；历史采样使用16题分片，因此请求分配及生成轨迹可不同。这属于从头构建模式的执行差异，新数据和权重身份会保存，不能称为历史MC或权重字节一致。

如果只想复现 TD：

```bash
bash examples/delta_critic/run_td_hybrid_handoff.sh td
bash examples/delta_critic/run_td_hybrid_handoff.sh eval-td
```

如果想从 Base 完成 Hybrid 和 online：

```bash
bash examples/delta_critic/run_td_hybrid_handoff.sh hybrid
bash examples/delta_critic/run_td_hybrid_handoff.sh eval-hybrid
bash examples/delta_critic/run_td_hybrid_handoff.sh online
```

如果已有完整 Hybrid step5，可在 `prepare` **之前**设置 `STEP5_DIR`，然后直接跑 online：

```bash
export STEP5_DIR=/data/previous_hybrid/checkpoints/global_step_5
# 同时保留 /data/previous_hybrid/checkpoints/delta_advantage_stats.json。
bash examples/delta_critic/run_td_hybrid_handoff.sh check
bash examples/delta_critic/run_td_hybrid_handoff.sh prepare
bash examples/delta_critic/run_td_hybrid_handoff.sh online
```

外部 checkpoint 应来自同一历史 Hybrid 配方。`policy_driver` 检查恢复步数为5、optimizer horizon为20、两卡并行以及恢复后实际prompt顺序；格式正确但来源不同的checkpoint不应混入该实验。仅 HF 推理模型不包含optimizer或dataloader状态，不能替代完整step5。`collect` 会从指定完整checkpoint重新导出推理模型。

online 还可拆为以下阶段，便于逐步检查：

```bash
bash examples/delta_critic/run_td_hybrid_handoff.sh collect
bash examples/delta_critic/run_td_hybrid_handoff.sh critic
bash examples/delta_critic/run_td_hybrid_handoff.sh policy
bash examples/delta_critic/run_td_hybrid_handoff.sh eval-online
```

所有环境变量应保持一致，尤其是 `WORK_DIR` 和 `STEP5_DIR`。路径已在 `prepare` 中绑定；要换输入或移动运行目录，应指定新的 `WORK_DIR` 并重新准备。

## 保留的训练规则

| 项目 | TD与Hybrid policy | Hybrid online critic |
| --- | --- | --- |
| 数据规模 | 每step128条原回答＋24个state×8条续写＝320行 | 512题；461训练、51验证 |
| actor更新 | minibatch160、每卡microbatch1；每step两次更新 | 不更新actor |
| 采样温度 | T1、top_p0.95、top_k20、seed42 | 同左；每state16条MC |
| 回答预算 | 原回答≤2048；前缀＋续写≤2048，至少留512新token | 同左 |
| learning rate | actor1e-5，constant；reference KL0.001 | critic1e-5；新建optimizer |
| critic | 各自固定TD80/Hybrid80 | Hybrid80全部权重初始化，再新增80步 |
| critic监督 | policy期间不更新 | TD标签用16分支；terminal监督取前8分支 |
| 标签尺度 | 保存于各自critic artifact | 仅新训练集拟合；scalar head转换保持初始化raw输出 |
| policy优势尺度 | 首次rollout拟合，固定；clip±5 | 选出critic后在policy step6重新拟合，固定至10 |
| 保存 | policy step5/10 | critic新增step20/40/60/80，按验证总loss选优 |

短答阈值为回答总长≤100 token。TD的短答raw outcome与其余优势一起标准化；Hybrid短答从critic校准人群排除，原回答正确为+1、错误为−1，续写按同state兄弟结果做leave-one-out baseline。长答采用selected-segment critic优势。两套规则由各自封存源码执行。

policy 初始化仍保留历史 TD horizon100、Hybrid horizon20；optimizer创建后，driver只将训练循环停止点设为10。online恢复完整step5并保留horizon20，在step6首次actor更新前重新拟合统计，不复制旧优势统计到新checkpoint目录。

训练的2048回答预算、critic的32768总输入上限和评测的8192生成预算各有独立用途。完整序列超出critic上限时按历史 `window_policy=full` 报错，不静默截断。

## 输出与验收

```text
WORK_DIR/
  logs/                         每阶段.log及成功后的.done
  input_paths.json               本次输入路径
  environment.json               本次环境及完整包差异
  code_manifest.json             运行代码和配置哈希
  source_td/ source_hybrid/ source_online/
  td/ hybrid/                    policy配置、config_changes、指标、rollout、checkpoint
  online/
    collection/                  原回答、state、MC label、continuation、split
    data_preparation.json        数据核验、新标签统计、head转换不变性检查
    critic/checkpoints/          新增20/40/60/80 checkpoint与验证指标
    critic_selection.json        候选及最终选择
    policy/runtime_schedule.json 恢复step5、horizon20、停止10、校准step6
    policy/policy_complete.json   step6–10优势统计固定和prompt顺序验收
  evaluation_tools/
    models/                      FSDP合并后的推理模型与EXPORT_VALIDATED
    <model_name>/summary.json    单模型正式评分及两个main_eval交叉检查
    results.csv results.json    所有注册模型的汇总
```

检查 `online/data_preparation.json` 中 `passed=true` 和head转换误差，检查 `critic_selection.json` 候选为20/40/60/80且选择最低有限验证loss，检查 `online/policy/policy_complete.json` 中 `verified=true`。冻结policy还保留 `runtime_prompt_checks.jsonl`，验证每步128道原问题的有序token身份。

正式评测为 DAPO128、MATH500、AIME2024/2025各30题，共688题，每题16次，共11008条回答；T1、top_p1、top_k−1、生成上限8192，seed=`42 + prompt_index × 16 + sample_index`。pass@1是全部16次回答的平均正确率，pass@16是至少有一次正确的题目比例。

每份正式 `summary.json` 的 `repository_main_eval_crosschecks` 应包含两项均通过的核验。比较模型时统一 `full_8192` 中同一个评分器，例如 `repository_math_verify`；`explicit_final_math_verify` 是另一评分口径。`prefix_2048_diagnostic` 是同一回答截取前2048 token后的诊断，不能当作独立2048预算评测。训练内 `math500` 仅前100题greedy验证，不能与完整500题正式评测混用。

只运行部分阶段时，`results.json` 会标记其他注册模型为pending，这不代表已经完成的单模型评测失败。只有运行完 `all` 且全部评分交叉检查通过，汇总才完整。

## 中断与常见错误

已成功阶段带有 `logs/<阶段>.done`，再次调用会跳过。日志已存在而没有成功标记时，脚本拒绝再次执行该阶段，防止重复optimizer更新或覆盖JSONL。不要只删除日志或伪造 `.done`。失败训练请检查日志，修复环境/资源后换新的 `WORK_DIR`；初始critic构建失败则换新的 `BOOTSTRAP_DIR`。本入口不自动接续训练到一半的policy或critic。

| 错误 | 处理 |
| --- | --- |
| 输入SHA256不匹配 | 核对权重snapshot、critic及parquet，重新复制正确文件 |
| 素材或代码校验失败 | 检查拷贝是否完整；重新从原机器导出素材，在新运行目录准备 |
| 核心环境版本不一致 | 使用与封存环境匹配的Python/依赖，不直接跳过校验 |
| 所选GPU占用或数量错误 | 预留空闲GPU，TD4卡、Hybrid2卡、critic和eval单卡 |
| 找不到step5或旧优势统计 | 先完成Hybrid，或交接完整checkpoint及相邻统计文件 |
| 超出输入长度/显存不足 | 检查实际序列和资源，保留报错数据；调整预算属于新实验配置 |
| 正式评分仅部分完成 | 查看 `audit_<model>.log` 和对应 `main_eval_*.log`，不可把部分结果当最终结果 |

同seed和输入身份对齐不保证生成轨迹或指标逐条一致；GPU、驱动、批处理和重启均可能改变生成。step5 checkpoint没有保存vLLM server RNG，续训也不能保证与连续训练step6完全一致。原online实验在新critic验证loss改善后，policy正式整体成绩仍下降；本流程不预设online更新会提升效果。

历史规则与结果可参阅 [历史复现案例](historical_td_hybrid_reproduction_case_20261005_zh.md)。本次已验证Bash语法、封存包全部文件哈希、仅提交文件的独立目录准备、原输入CPU身份校验、初始critic配置及各生成驱动；新增4项CPU测试通过，没有重新运行全量GPU训练或正式评测。运行测试命令为 `uv run --no-project --python "$PYTHON_BIN" python -m pytest -q tests/special_standalone/delta_critic/test_handoff_entrypoint.py`。交接脚本及说明由AI辅助整理，正式执行前请运行者审阅代码和完整配置。
