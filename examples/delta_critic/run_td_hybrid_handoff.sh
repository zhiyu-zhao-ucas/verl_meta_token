#!/usr/bin/env bash
# TD / Hybrid 历史复现及 Hybrid step5 online 更新的统一入口。
# 用法与交接素材说明：docs/td_hybrid_handoff_zh.md
# 本文件编排已有 Python 训练代码；完整配置和源码由 export-assets 封存。
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# ==================== 只需集中修改这一组路径和 GPU ====================
# Python 必须指向已经安装训练依赖的 Python 3.12 环境；全部经 uv 启动。
PYTHON_BIN=${PYTHON_BIN:-python3}
UV_BIN=${UV_BIN:-uv}
ASSET_DIR=${ASSET_DIR:-$REPO_ROOT/outputs/td_hybrid_handoff_assets}
WORK_DIR=${WORK_DIR:-$REPO_ROOT/outputs/td_hybrid_handoff_run}
BASE_MODEL=${BASE_MODEL:-}
TRAIN_PARQUET=${TRAIN_PARQUET:-$ASSET_DIR/inputs/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-$ASSET_DIR/inputs/validation.parquet}
TD_CRITIC=${TD_CRITIC:-}
HYBRID_CRITIC=${HYBRID_CRITIC:-}
# 可选：已有 Hybrid global_step_5 完整 checkpoint；空值时用本次复现产物。
STEP5_DIR=${STEP5_DIR:-$WORK_DIR/hybrid/checkpoints/global_step_5}
TD_GPUS=${TD_GPUS:-0,1,2,3}       # TD 历史配方：4 卡，TP1。
HYBRID_GPUS=${HYBRID_GPUS:-0,1}   # Hybrid 历史配方及续训：2 卡，TP1。
SAMPLE_GPUS=${SAMPLE_GPUS:-0}     # 任意数量单卡 TP1 副本；曾运行 5 副本。
CRITIC_GPU=${CRITIC_GPU:-0}       # online critic 保持单卡/global batch256。
EVAL_GPU=${EVAL_GPU:-0}           # 正式评测：单卡 TP1，依次评测。
BOOTSTRAP_DIR=${BOOTSTRAP_DIR:-$REPO_ROOT/outputs/td_hybrid_initial_critics}
TD_INIT_GPUS=${TD_INIT_GPUS:-0,1,2,3,4,5,6,7}  # 历史初始TD训练8卡。
HYBRID_INIT_GPUS=${HYBRID_INIT_GPUS:-0,1,2,3}   # 历史初始Hybrid训练4卡。
CRITIC_MODE=${CRITIC_MODE:-historical}         # historical或rebuilt。
# ====================================================================

usage() {
    cat <<'HELP'
用法：bash examples/delta_critic/run_td_hybrid_handoff.sh <阶段>

  export-assets   在原仓库导出源码、配置、prompt、parquet和驱动；不含模型权重
  pack-assets     把导出素材压缩为仓库内的handoff_assets.tgz，供GitHub提交
  bootstrap       从Base采样并训练新的TD80和Hybrid80；不需要私有critic
  prepare-bootstrap 仅准备初始critic采样/训练配置，不启动GPU
  fresh-all       bootstrap -> prepare -> all；从提交文件跑全流程（产生新权重）
  check           CPU 校验素材、环境及输入身份；不启动训练（默认）
  prepare         创建新的运行目录，迁移路径，生成驱动；不启动 GPU
  td              Base -> TD policy step10，保存 step5/10
  hybrid          Base -> Hybrid policy step10，保存 step5/10
  collect         Hybrid step5 -> 512 原回答及每 state16 条 MC 续写
  critic          划分461/51题、重拟合标签尺度、转换 head、更新 critic80步、选优
  policy          恢复 Hybrid step5，step6重拟合优势统计，续训至累计step10
  eval-td         正式评测 TD step5/10
  eval-hybrid     正式评测 Hybrid step5/10
  eval-online     正式评测 online Hybrid step10
  online          collect -> critic -> policy -> eval-online
  all             td -> eval-td -> hybrid -> eval-hybrid -> online

先 export-assets 并交接素材，再设置顶部路径环境变量，执行 check 和 prepare。
已完成阶段可重复调用并跳过；失败的训练阶段须使用新的 WORK_DIR。
HELP
}

STAGE=${1:-check}
[[ $# -le 1 ]] || { usage; exit 2; }
case "$STAGE" in
    -h|--help|help) usage; exit 0 ;;
    export-assets|pack-assets|bootstrap|prepare-bootstrap|fresh-all|check|prepare|td|hybrid|collect|critic|policy|eval-td|eval-hybrid|eval-online|online|all) ;;
    *) usage; exit 2 ;;
esac
command -v "$UV_BIN" >/dev/null || { echo "找不到 uv：$UV_BIN" >&2; exit 1; }
PYTHON_BIN=$(command -v "$PYTHON_BIN")
UV_BIN=$(command -v "$UV_BIN")
ASSET_DIR=$(realpath -m -- "$ASSET_DIR")
WORK_DIR=$(realpath -m -- "$WORK_DIR")
STEP5_DIR=$(realpath -m -- "$STEP5_DIR")
BOOTSTRAP_DIR=$(realpath -m -- "$BOOTSTRAP_DIR")
# 路径通过环境传入 Python，避免拼接 shell 命令或执行用户输入。
export REPO_ROOT ASSET_DIR WORK_DIR BASE_MODEL TRAIN_PARQUET VAL_PARQUET
export TD_CRITIC HYBRID_CRITIC STEP5_DIR PYTHON_BIN UV_BIN
export BOOTSTRAP_DIR CRITIC_MODE
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset RAY_ADDRESS
py() { "$UV_BIN" run --no-project --python "$PYTHON_BIN" python "$@"; }

# GitHub clone自带封存代码包，首次使用时自动校验并解压，无需原机器outputs。
# .tgz不受仓库全局*.tar.gz忽略规则影响；它不含模型权重。
if [[ "$STAGE" != export-assets && "$STAGE" != pack-assets && ! -d "$ASSET_DIR" ]]; then
    py - "$SCRIPT_DIR/handoff_assets.tgz" "$ASSET_DIR" <<'PY'
import hashlib, json, os, shutil, sys, tarfile, tempfile
from pathlib import Path
archive, target = map(Path, sys.argv[1:])
expected = json.loads(archive.with_suffix('.sha256.json').read_text())['sha256']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, '提交素材包的SHA256不匹配'
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='.handoff-unpack-', dir=target.parent) as stage:
    with tarfile.open(archive, 'r:gz') as stream:
        stream.extractall(stage, filter='data')
    shutil.move(str(Path(stage) / 'assets'), str(target))
print(f'已从提交文件解压封存代码：{target}')
PY
fi

if [[ "$STAGE" == pack-assets ]]; then
    py - "$ASSET_DIR" "$SCRIPT_DIR/handoff_assets.tgz" <<'PY'
import gzip, hashlib, json, sys, tarfile
from pathlib import Path
assets, archive = map(Path, sys.argv[1:])
assert (assets / 'manifest.json').is_file(), '请先在原机器执行export-assets'
with archive.open('wb') as output, gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as compressed:
    with tarfile.open(fileobj=compressed, mode='w') as stream:
        for path in sorted(assets.rglob('*')):
            if not path.is_file():
                continue
            info = stream.gettarinfo(str(path), arcname='assets/' + str(path.relative_to(assets)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            with path.open('rb') as content:
                stream.addfile(info, content)
archive.with_suffix('.sha256.json').write_text(json.dumps({
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size,
    'contents': 'TD, Hybrid, online and evaluation source snapshots; drivers; configs; audited prompts',
}, indent=2) + '\n')
print(f'可提交代码包：{archive} ({archive.stat().st_size} bytes)')
PY
    exit 0
fi

# 素材导出、CPU 校验、路径迁移共用一段 Python；GPU 阶段在下方清楚展开。
prepare_or_check() {
    py - "$1" <<'PY'
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

action = sys.argv[1]
repo = Path(os.environ['REPO_ROOT'])
assets = Path(os.environ['ASSET_DIR']).resolve()
work = Path(os.environ['WORK_DIR']).resolve()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()

def copy_tree(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '.pytest_cache', '.git', 'AGENTS.md', 'CLAUDE.md'))

if action == 'export-assets':
    if assets.exists():
        raise FileExistsError(f'素材目录已存在，请换 ASSET_DIR：{assets}')
    origin = repo / 'outputs/delta_historical_reproduction_20261005'
    online = repo / 'outputs/delta_historical_hybrid_step5_online_20261005'
    assets.mkdir(parents=True)
    for label, source in [('source_td', origin / 'source_td'),
                          ('source_hybrid', origin / 'source_hybrid'),
                          ('source_online', online / 'source')]:
        copy_tree(source, assets / label)
    # 初始Hybrid训练也使用自己的历史源码；无需接收人取得旧critic产物。
    copy_tree(repo / 'outputs/delta_critic_4b_hybrid_shared_td_20261002/source',
              assets / 'source_initial_hybrid')
    bootstrap = assets / 'bootstrap'
    bootstrap.mkdir()
    raw = repo / 'outputs/delta_qwen3_4b_base_dapo_20260927'
    shutil.copy2(raw / 'config_base512.yaml', bootstrap / 'sampling.yaml')
    shutil.copy2(raw / 'train/runs/balanced/step_00000080/metadata.json', bootstrap / 'td_metadata.json')
    shutil.copy2(raw / 'train/make_p95_subset.py', bootstrap / 'make_p95_subset.py')
    # 只提交问题，不提交旧模型生成的回答/label/MC，bootstrap会全部重采样。
    with (raw / 'data/rollouts_train_regen.jsonl').open() as source, (bootstrap / 'prompts.jsonl').open('w') as sink:
        count = 0
        for line in source:
            row = json.loads(line)
            prompt = {k: row[k] for k in ['prompt_id', 'prompt', 'raw_prompt', 'prompt_token_ids', 'gold_answer', 'data_source']}
            sink.write(json.dumps(prompt, ensure_ascii=False) + '\n')
            count += 1
    assert count == 512
    for arm in ['td', 'hybrid']:
        branch = assets / arm
        branch.mkdir()
        for name in ['config.yaml', 'reward.py', 'prompt_audit.json', 'prompt_snapshot.jsonl']:
            shutil.copy2(origin / arm / name, branch / name)
    for name in ['environment.json', 'protocol.json', 'driver.py']:
        shutil.copy2(origin / name, assets / name)
    # 环境清单用于交接与核验；GPU wheel 仍需由原环境/对应wheel源提供。
    packages = read(assets / 'environment.json')['packages']
    (assets / 'requirements-frozen.txt').write_text(''.join(
        f"{p['name']}=={p['version']}\n" for p in packages))
    # 原 driver 只依赖 prompt_audit.digest，交接版保留同一个实现。
    text = (origin / 'prompt_audit.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'digest')
    (assets / 'prompt_audit.py').write_text('import hashlib\nimport json\n\n' + ast.get_source_segment(text, node) + '\n')
    baseline = assets / 'baseline'
    baseline.mkdir()
    for name in ['critic_metadata.json', 'sampling.yaml', 'input_identity.json', 'evaluation_model.json']:
        shutil.copy2(online / 'baseline' / name, baseline / name)
    inputs = assets / 'inputs'
    inputs.mkdir()
    # 两个parquet合计不到1MB，随代码提交，接收人无需原数据outputs目录。
    original = read(online / 'baseline/input_identity.json')['parquet_hashes']
    from omegaconf import OmegaConf
    config = OmegaConf.load(origin / 'hybrid/config.yaml')
    for name, path in [('train.parquet', config.data.train_files), ('validation.parquet', config.data.val_files)]:
        assert sha(path) == original[name]
        shutil.copy2(path, inputs / name)
    for name in ['prompts.jsonl', 'policy_prompt_schedule.json', 'collect.py', 'policy_driver.py', 'reward.py']:
        shutil.copy2(online / name, assets / ('online_' + name))
    # 从已验证的 runner 提取纯数据/校验函数，不携带历史审批、SSH或队列。
    names = {'read', 'save', 'sha', 'jsonl', 'digest', 'rebase_head',
             'data_and_initialization', 'select_critic', 'verify_policy'}
    text = (online / 'runner.py').read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    header = 'import hashlib\nimport json\nimport math\nfrom pathlib import Path\nROOT = Path(__file__).resolve().parent\n\n'
    (assets / 'online_runner.py').write_text(header + '\n\n'.join(ast.get_source_segment(text, n) for n in nodes) + '\n')
    evaluation = assets / 'evaluation_tools'
    evaluation.mkdir()
    for name in ['prepare.py', 'generate.py', 'audit.py', 'scorers.py', 'report.py',
                 'prompts.jsonl', 'protocol.json', 'accelerated_inference.json']:
        shutil.copy2(online / 'evaluation_tools' / name, evaluation / name)
    copy_tree(online / 'evaluation_tools/source', evaluation / 'source')
    save(assets / 'manifest.json', {str(p.relative_to(assets)): sha(p)
         for p in sorted(assets.rglob('*')) if p.is_file()})
    print(f'素材导出完成：{assets}；含parquet和初始critic训练题目，不含模型权重。')
    sys.exit(0)

# 校验素材字节身份，任何丢文件或误改都会在 GPU 启动前失败。
for relative, expected in read(assets / 'manifest.json').items():
    if sha(assets / relative) != expected:
        raise RuntimeError(f'素材校验失败：{relative}')

from omegaconf import OmegaConf
import yaml

paths = {}
required = ['BASE_MODEL', 'TRAIN_PARQUET', 'VAL_PARQUET']
if action not in ['bootstrap-check', 'bootstrap-prepare']:
    required += ['TD_CRITIC', 'HYBRID_CRITIC']
for name in required:
    value = os.environ[name]
    if not value or not Path(value).exists():
        raise FileNotFoundError(f'请设置存在的 {name} 路径：{value!r}')
    paths[name] = Path(value).resolve()
identity = read(assets / 'baseline/input_identity.json')
checks = [(paths['BASE_MODEL'] / n, h) for n, h in identity['actor_weight_hashes'].items()]
checks += [(paths['TRAIN_PARQUET'], identity['parquet_hashes']['train.parquet']),
           (paths['VAL_PARQUET'], identity['parquet_hashes']['validation.parquet'])]
for arm, key in [('td', 'TD_CRITIC'), ('hybrid', 'HYBRID_CRITIC')] if 'TD_CRITIC' in paths else []:
    expected = read(assets / 'protocol.json')['arms'][arm]['critic_weights_sha256']
    if os.environ['CRITIC_MODE'] == 'rebuilt':
        # 仅接受本入口成功训练的产物，记录新身份，不能伪称历史权重一致。
        root = Path(os.environ['BOOTSTRAP_DIR'])
        assert (root / f'logs/initial_{arm}.done').exists(), f'{arm}初始critic尚未完成'
        assert paths[key] == (root / f'{arm}_checkpoints/step_00000080').resolve()
        expected = read(paths[key] / 'metadata.json')['weights_sha256']
    else:
        assert os.environ['CRITIC_MODE'] == 'historical', 'CRITIC_MODE须为historical或rebuilt'
    checks.append((paths[key] / 'model.pt', expected))
    assert read(paths[key] / 'metadata.json')['weights_sha256'] == expected
for path, expected in checks:
    if sha(path) != expected:
        raise RuntimeError(f'输入 SHA256 不匹配：{path}')

# 核心依赖严格匹配保存的版本。完整环境差异写报告；额外工具包可存在。
def canonical(name):
    return name.lower().replace('_', '-').replace('.', '-')
expected = {canonical(p['name']): p['version'] for p in read(assets / 'environment.json')['packages']}
actual = {canonical(d.metadata['Name']): d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
critical = ['torch', 'vllm', 'transformers', 'ray', 'hydra-core', 'omegaconf', 'torchdata',
            'math-verify', 'transferqueue', 'tensordict', 'triton', 'flashinfer-python',
            'tokenizers', 'numpy', 'pyarrow', 'datasets', 'safetensors', 'accelerate']
drift = {n: {'expected': expected.get(n), 'actual': actual.get(n)}
         for n in critical if actual.get(n) is None or expected.get(n) != actual.get(n)}
if platform.python_version() != '3.12.13':
    drift['python'] = {'expected': '3.12.13', 'actual': platform.python_version()}
if drift:
    raise RuntimeError(f'核心环境版本不一致：{drift}')
print('CPU 校验通过：素材、Base 权重、数据、critic 权重及核心环境。')
if action in ['check', 'bootstrap-check']:
    sys.exit(0)
if action == 'bootstrap-prepare':
    root = Path(os.environ['BOOTSTRAP_DIR'])
    if root.exists():
        assert (root / 'PREPARED').exists(), '初始critic目录准备未完成，请换BOOTSTRAP_DIR'
        assert read(root / 'input_identity.json') == {str(p): h for p, h in checks}, '初始critic输入路径发生变化，请换BOOTSTRAP_DIR'
        sys.exit(0)
    root.mkdir(parents=True)
    (root / 'logs').mkdir()
    (root / 'collection').mkdir()
    (root / 'train').mkdir()
    sampling = yaml.safe_load((assets / 'bootstrap/sampling.yaml').read_text())
    sampling['model']['actor_model'] = str(paths['BASE_MODEL'])
    # 包内已经是历史最终prompt/token，不可再次apply_chat_template。
    sampling['prompt']['use_model_chat_template'] = False
    (root / 'sampling.yaml').write_text(yaml.safe_dump(sampling))
    for arm, metadata in [('td', assets / 'bootstrap/td_metadata.json'),
                          ('hybrid', assets / 'baseline/critic_metadata.json')]:
        config = read(metadata)['config']
        config.update(model_path=str(paths['BASE_MODEL']), tokenizer_path=str(paths['BASE_MODEL']))
        (root / f'{arm}.yaml').write_text(yaml.safe_dump(config))
    # P95脚本的目录约定保持不变，data指向这次新采样的collection。
    (root / 'data').symlink_to(root / 'collection')
    shutil.copy2(assets / 'bootstrap/make_p95_subset.py', root / 'train/make_p95_subset.py')
    # 复用TP1副本管理，采样本身采用初始30720预算，不应用online的2048限制。
    text = (assets / 'online_collect.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'configure_replica_pool')
    header = '''import asyncio
import json
import os
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent
'''
    body = '''
def main():
    from examples.delta_critic import batch_sample
    count = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if count > 1:
        configure_replica_pool(count)
    config = yaml.safe_load((ROOT / 'sampling.yaml').read_text())
    with (Path(os.environ['ASSET_DIR']) / 'bootstrap/prompts.jsonl').open() as source:
        prompts = [json.loads(line) for line in source]
    batch_sample.load_prompt_rows = lambda *args, **kwargs: prompts
    result = asyncio.run(batch_sample.run(
        config, split='train', output_dir=ROOT / 'collection',
        prompts_jsonl='sealed bootstrap prompts', limit=512,
        model_path=config['model']['actor_model'], actor_version=config['model']['actor_model'],
        mc_mode='prefix_only', rollout_concurrency=8*count, mc_concurrency=128*count,
        gpu_memory_utilization=0.9, prompt_source='value_model', verl_overrides=[],
        mc_sampling_mode='requests', mc_native_batch_size=None))
    (ROOT / 'collection/result.json').write_text(json.dumps(result, indent=2)+'\\n')
if __name__ == '__main__':
    main()
'''
    (root / 'collect.py').write_text(header + ast.get_source_segment(text, node) + '\n' + body)
    save(root / 'input_identity.json', {str(p): h for p, h in checks})
    save(root / 'code_manifest.json', {str(p.relative_to(root)): sha(p) for p in
         [root / 'collect.py', root / 'sampling.yaml', root / 'td.yaml', root / 'hybrid.yaml',
          root / 'train/make_p95_subset.py']})
    (root / 'PREPARED').write_text('new critic construction; historical weights not guaranteed\n')
    print(f'初始critic构建目录准备完成：{root}')
    sys.exit(0)
if work.exists():
    raise FileExistsError(f'运行目录已存在，请换 WORK_DIR：{work}')
work.mkdir(parents=True)
(work / 'logs').mkdir()
save(work / 'environment.json', {'python': sys.version, 'packages': actual,
     'all_package_differences': {n: [expected.get(n), actual.get(n)] for n in expected.keys() | actual.keys()
                               if expected.get(n) != actual.get(n)}})
save(work / 'input_paths.json', {k: str(v) for k, v in paths.items()})
save(work / 'critic_identity.json', {'mode': os.environ['CRITIC_MODE'],
     'td_sha256': read(paths['TD_CRITIC'] / 'metadata.json')['weights_sha256'],
     'hybrid_sha256': read(paths['HYBRID_CRITIC'] / 'metadata.json')['weights_sha256']})
shutil.copy2(assets / 'driver.py', work / 'driver.py')
shutil.copy2(assets / 'prompt_audit.py', work / 'prompt_audit.py')

# 每个 arm 使用自己的封存源码，避免当前工作区修改改变历史行为。
for arm in ['td', 'hybrid', 'online']:
    copy_tree(assets / ('source_' + arm), work / ('source_' + arm))

def critic_view(source, destination):
    destination.mkdir(parents=True)
    (destination / 'model.pt').symlink_to(source / 'model.pt')
    meta = read(source / 'metadata.json')
    meta['config'].update(model_path=str(paths['BASE_MODEL']), tokenizer_path=str(paths['BASE_MODEL']))
    save(destination / 'metadata.json', meta)
    return destination

def bind_policy(arm, source_config, critic, resume=None):
    cfg = OmegaConf.to_container(OmegaConf.load(source_config), resolve=True)
    branch = work / arm
    branch.mkdir(exist_ok=True)
    cfg['actor_rollout_ref']['model']['path'] = str(paths['BASE_MODEL'])
    cfg['data'].update(train_files=str(paths['TRAIN_PARQUET']), val_files=str(paths['VAL_PARQUET']))
    cfg['algorithm']['delta_policy']['critic_artifact'] = str(critic)
    cfg['reward']['custom_reward_function']['path'] = str(branch / 'reward.py')
    cfg['trainer'].update(project_name='td_hybrid_handoff', experiment_name=arm,
        default_local_dir=str(branch / 'checkpoints'), rollout_data_dir=str(branch / 'rollouts'),
        validation_data_dir=str(branch / 'validation'), resume_mode='resume_path' if resume else 'disable',
        resume_from_path=str(resume) if resume else None)
    runtime = cfg['ray_kwargs']['ray_init']['runtime_env']
    runtime.update(working_dir=str(work / ('source_' + arm)))
    runtime['env_vars']['PYTHONPATH'] = str(work / ('source_' + arm)) + ':' + str(work)
    # horizon 保留历史20/100。driver 在 optimizer 初始化后才把循环停止点改到10。
    assert cfg['trainer']['total_training_steps'] == (100 if arm == 'td' else 20)
    assert cfg['algorithm']['delta_policy']['critic_update'] is None
    OmegaConf.save(OmegaConf.create(cfg), branch / ('policy/config.yaml' if arm == 'online' else 'config.yaml'))
    def flatten(value, prefix=''):
        if isinstance(value, dict):
            return {k: v for name, item in value.items()
                    for k, v in flatten(item, f'{prefix}.{name}' if prefix else name).items()}
        return {prefix: value}
    before = flatten(OmegaConf.to_container(OmegaConf.load(source_config), resolve=True))
    after = flatten(cfg)
    save(branch / 'config_changes.json', {k: {'before': before.get(k), 'after': after.get(k)}
         for k in before.keys() | after.keys() if before.get(k) != after.get(k)})

for arm, key in [('td', 'TD_CRITIC'), ('hybrid', 'HYBRID_CRITIC')]:
    branch = work / arm
    branch.mkdir()
    critic = critic_view(paths[key], work / 'critic_inputs' / arm)
    for name in ['reward.py', 'prompt_audit.json', 'prompt_snapshot.jsonl']:
        shutil.copy2(assets / arm / name, branch / name)
    bind_policy(arm, assets / arm / 'config.yaml', critic)

online = work / 'online'
online.mkdir()
for directory in ['inputs/checkpoints', 'policy', 'critic', 'baseline', 'collection']:
    (online / directory).mkdir(parents=True, exist_ok=True)
for alias, key in [('base', 'BASE_MODEL'), ('train.parquet', 'TRAIN_PARQUET'), ('validation.parquet', 'VAL_PARQUET')]:
    (online / 'inputs' / alias).symlink_to(paths[key])
(online / 'inputs/critic80').symlink_to(work / 'critic_inputs/hybrid')
step5 = Path(os.environ['STEP5_DIR']).resolve()
(online / 'inputs/checkpoints/global_step_5').symlink_to(step5)
# 总入口只接受完整 step5 checkpoint，并在 collect 时由它重新导出推理权重。
(online / 'inputs/step5_model').symlink_to(work / 'evaluation_tools/models/hybrid_step_5')
for name in ['critic_metadata.json', 'evaluation_model.json']:
    shutil.copy2(assets / 'baseline' / name, online / 'baseline' / name)
for name in ['prompts.jsonl', 'policy_prompt_schedule.json', 'reward.py']:
    shutil.copy2(assets / ('online_' + name), online / name)
save(online / 'protocol.json', {'critic_start_weights_sha256': read(paths['HYBRID_CRITIC'] / 'metadata.json')['weights_sha256']})
critic = read(assets / 'baseline/critic_metadata.json')['config']
critic.update(model_path=str(paths['BASE_MODEL']), tokenizer_path=str(paths['BASE_MODEL']))
(online / 'critic/config.yaml').write_text(yaml.safe_dump(critic, sort_keys=True))
sampling = yaml.safe_load((assets / 'baseline/sampling.yaml').read_text())
sampling['model'].update(actor_model=str(online / 'inputs/step5_model'), gpu_memory_utilization=0.6)
sampling['prompt']['use_model_chat_template'] = False
sampling['smoke'].update(train_prompts=512, responses_per_prompt=1, states_per_response=64)
sampling['rollout']['max_tokens'] = 2048
sampling['mc'].update(max_continuation_tokens=2048, continuations_per_state=16,
                      save_continuations=True, save_individual_rewards=True)
(online / 'sampling.yaml').write_text(yaml.safe_dump(sampling, sort_keys=True))
# online 目录的层次对应原 policy_driver 中 branch.parent 的约定。
bind_policy('online', assets / 'hybrid/config.yaml', online / 'critic/selected',
            online / 'inputs/checkpoints/global_step_5')
cfg = OmegaConf.load(online / 'policy/config.yaml')
cfg.trainer.default_local_dir = str(online / 'policy/checkpoints')
cfg.trainer.rollout_data_dir = str(online / 'policy/rollouts')
cfg.trainer.validation_data_dir = str(online / 'policy/validation')
cfg.reward.custom_reward_function.path = str(online / 'reward.py')
OmegaConf.save(cfg, online / 'policy/config.yaml')
changes = read(online / 'config_changes.json')
for key in ['default_local_dir', 'rollout_data_dir', 'validation_data_dir']:
    changes['trainer.' + key]['after'] = str(cfg.trainer[key])
changes['reward.custom_reward_function.path']['after'] = str(cfg.reward.custom_reward_function.path)
save(online / 'config_changes.json', changes)

# 迁移后的驱动执行本次素材校验；历史审批记录/远端队列不进入交接流程。
runner = (assets / 'online_runner.py').read_text()
runner += '''\n\ndef verify_assets():
    for relative, expected in read(ROOT / "source_manifest.json").items():
        assert sha(ROOT.parent / relative) == expected, f"Source changed: {relative}"
'''
(online / 'runner.py').write_text(runner)
guard = 'from runner import require_review, verify_seal\n    require_review()\n    verify_seal()'
for name in ['collect.py', 'policy_driver.py']:
    text = (assets / ('online_' + name)).read_text()
    assert text.count(guard) == 1
    text = text.replace(guard, 'from runner import verify_assets\n    verify_assets()')
    (online / name).write_text(text)

copy_tree(assets / 'evaluation_tools', work / 'evaluation_tools')
evaluation = work / 'evaluation_tools'
protocol = read(evaluation / 'protocol.json')
protocol['actor'] = str(paths['BASE_MODEL'])
save(evaluation / 'protocol.json', protocol)
names = ['td_step_5', 'td_step_10', 'hybrid_step_5', 'hybrid_step_10', 'hybrid_online_step_10']
registry = {}
for name in names:
    arm = 'online/policy' if name.startswith('hybrid_online') else name.split('_step_')[0]
    step = int(name.rsplit('_', 1)[1])
    entry = read(assets / 'baseline/evaluation_model.json')
    entry.update(label=arm, step=step, run=str(work / arm),
                 source=str(work / arm / f'checkpoints/global_step_{step}/actor'),
                 train_files=[str(paths['TRAIN_PARQUET'])], world_size=4 if arm == 'td' else 2)
    registry[name] = entry
# 若外部 step5 指定，baseline 评测和 online 采样共同导出这个 checkpoint。
registry['hybrid_step_5']['run'] = str(step5.parent.parent)
registry['hybrid_step_5']['source'] = str(step5 / 'actor')
save(evaluation / 'historical_models.json', registry)
acceleration = read(evaluation / 'accelerated_inference.json')
acceleration['models'] = names
save(evaluation / 'accelerated_inference.json', acceleration)
# audit 的两个 main_eval 子进程也使用接收人的 uv/Python 环境。
audit = (evaluation / 'audit.py').read_text()
audit = audit.replace("'/tmp/verl-uv-bin/uv'", "os.environ['UV_BIN']")
audit = audit.replace("'/scratch2/zhiyu/miniconda3/envs/verl-delta-vllm-20260925/bin/python'", "os.environ['PYTHON_BIN']")
assert '/scratch2/zhiyu/' not in audit and '/tmp/verl-uv-bin/uv' not in audit
(evaluation / 'audit.py').write_text(audit)

# 运行目录的不可变代码单独封存，后续每阶段检查，配置/产物不在清单中。
files = [p for directory in ['source_td', 'source_hybrid', 'source_online', 'evaluation_tools/source']
         for p in (work / directory).rglob('*') if p.is_file()]
files += [work / 'driver.py', work / 'prompt_audit.py', online / 'runner.py',
          online / 'collect.py', online / 'policy_driver.py']
files += [work / 'td/config.yaml', work / 'hybrid/config.yaml', online / 'policy/config.yaml',
          online / 'critic/config.yaml', online / 'sampling.yaml', online / 'prompts.jsonl',
          online / 'policy_prompt_schedule.json', online / 'reward.py',
          work / 'td/reward.py', work / 'hybrid/reward.py']
files += [p for p in evaluation.glob('*') if p.is_file()]
manifest = {str(p.relative_to(work)): sha(p) for p in sorted(files)}
save(work / 'code_manifest.json', manifest)
save(online / 'source_manifest.json', manifest)
(work / 'PREPARED').write_text('CPU checks passed; no GPU job launched\n')
print(f'运行目录已准备：{work}')
PY
}

if [[ "$STAGE" == export-assets || "$STAGE" == check || "$STAGE" == prepare ]]; then
    prepare_or_check "$STAGE"
    exit 0
fi
# 从提交文件启动时只下载公开Base snapshot；数据/题目/所有代码均随包提交。
if [[ "$STAGE" == bootstrap || "$STAGE" == prepare-bootstrap || "$STAGE" == fresh-all ]]; then
    BASE_MODEL=${BASE_MODEL:-$REPO_ROOT/outputs/td_hybrid_base_model}
    export BASE_MODEL
    if [[ ! -f "$BASE_MODEL/model-00001-of-00003.safetensors" ||
          ! -f "$BASE_MODEL/model-00002-of-00003.safetensors" ||
          ! -f "$BASE_MODEL/model-00003-of-00003.safetensors" ||
          ! -f "$BASE_MODEL/tokenizer.json" ]]; then
        HF_HUB_OFFLINE=0 py - "$BASE_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-4B-Base',
                  revision='906bfd4b4dc7f14ee4320094d8b41684abff8539', local_dir=sys.argv[1],
                  allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
PY
    fi
fi
if [[ "$STAGE" == prepare-bootstrap ]]; then
    prepare_or_check bootstrap-prepare
    exit 0
fi
if [[ "$STAGE" == fresh-all ]]; then
    bash "$SCRIPT_DIR/run_td_hybrid_handoff.sh" bootstrap
    export CRITIC_MODE=rebuilt
    export TD_CRITIC="$BOOTSTRAP_DIR/td_checkpoints/step_00000080"
    export HYBRID_CRITIC="$BOOTSTRAP_DIR/hybrid_checkpoints/step_00000080"
    # 已prepared的运行目录沿用绑定；all中的done标记控制已完成阶段跳过。
    [[ -f "$WORK_DIR/PREPARED" ]] || bash "$SCRIPT_DIR/run_td_hybrid_handoff.sh" prepare
    bash "$SCRIPT_DIR/run_td_hybrid_handoff.sh" all
    exit 0
fi
if [[ "$STAGE" == bootstrap ]]; then
    prepare_or_check bootstrap-prepare
    WORK_DIR=$BOOTSTRAP_DIR
else
    [[ -f "$WORK_DIR/PREPARED" ]] || { echo "请先执行 check 和 prepare。" >&2; exit 1; }
fi
WORK_DIR=$(cd -- "$WORK_DIR" && pwd)
export WORK_DIR

# 同一 WORK_DIR 只允许运行一个编排进程。不会结束其他人的 Ray 或 GPU 任务。
exec 9>"$WORK_DIR/.pipeline.lock"
flock -n 9 || { echo "此 WORK_DIR 已有流程运行。" >&2; exit 1; }
trap 'echo "阶段失败：$STAGE；查看 $WORK_DIR/logs。" >&2' ERR
py - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['WORK_DIR'])
for relative, expected in json.loads((root / 'code_manifest.json').read_text()).items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
PY

run_logged() {
    local name=$1 source=$2 gpus=$3
    shift 3
    if [[ -f "$WORK_DIR/logs/$name.done" ]]; then
        echo "[$name] 已完成，跳过。"
        return
    fi
    # 日志存在但没有成功标记时拒绝重复训练，避免重复更新或覆盖产物。
    [[ ! -e "$WORK_DIR/logs/$name.log" ]] || {
        echo "[$name] 存在未完成日志，请按交接文档处理后使用新的 WORK_DIR。" >&2; return 1;
    }
    echo "[$name] GPU=$gpus；日志=$WORK_DIR/logs/$name.log"
    (
        cd -- "$source"
        export PYTHONPATH="$source:$WORK_DIR" CUDA_VISIBLE_DEVICES="$gpus"
        py "$@"
    ) 2>&1 | tee "$WORK_DIR/logs/$name.log"
    touch "$WORK_DIR/logs/$name.done"
}

validate_gpus() {
    local gpus=$1 count=$2
    py - "$gpus" "$count" <<'PY'
import json, subprocess, sys
ids = sys.argv[1].split(',')
assert ids and all(x.isdecimal() for x in ids) and len(set(ids)) == len(ids), 'GPU编号须唯一且为整数'
assert not int(sys.argv[2]) or len(ids) == int(sys.argv[2]), 'GPU数量与历史配方不符'
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                               '--format=csv,noheader,nounits'], text=True)
memory = {i.strip(): int(m.strip()) for i, m in (line.split(',') for line in rows.splitlines())}
assert all(i in memory and memory[i] <= 512 for i in ids), f'所选GPU不存在或已占用：{memory}'
print('GPU空闲检查通过：' + ','.join(ids))
PY
}

train_frozen() {
    local arm=$1 gpus=$2 count=$3
    [[ -f "$WORK_DIR/logs/$arm.done" ]] || validate_gpus "$gpus" "$count"
    run_logged "$arm" "$WORK_DIR/source_$arm" "$gpus" "$WORK_DIR/driver.py" --arm "$arm"
}

# 导出使用原评测 merger：合并FSDP分片，检查全部tensor有限、键名和形状匹配Base。
export_model() {
    local name=$1 step=$2 run=$3
    # registry 记录prepare时绑定的实际来源，含可选外部step5路径。
    run=$(py - "$WORK_DIR/evaluation_tools/historical_models.json" "$name" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())[sys.argv[2]]['run'])
PY
)
    run_logged "export_$name" "$WORK_DIR/evaluation_tools/source" "" \
        "$WORK_DIR/evaluation_tools/prepare.py" --export "$step" --run "$run" --name "$name"
}

collect() {
    export_model hybrid_step_5 5 "$(dirname -- "$(dirname -- "$STEP5_DIR")")"
    [[ -f "$WORK_DIR/logs/collect.done" ]] || validate_gpus "$SAMPLE_GPUS" 0
    run_logged collect "$WORK_DIR/source_online" "$SAMPLE_GPUS" "$WORK_DIR/online/collect.py"
}

critic() {
    # 原 runner 中的 data_and_initialization 包含完整512题/16分支预算校验、
    # 461/51划分、仅训练集统计拟合、scalar head重映射及原始预测不变检查。
    run_logged critic_prepare "$WORK_DIR/source_online" "" -c \
        'import sys; sys.path.insert(0, sys.argv[1]); import runner; runner.data_and_initialization()' "$WORK_DIR/online"
    [[ -f "$WORK_DIR/logs/critic.done" ]] || validate_gpus "$CRITIC_GPU" 1
    run_logged critic "$WORK_DIR/source_online" "$CRITIC_GPU" -m torch.distributed.run \
        --standalone --nproc_per_node=1 -m examples.delta_critic.train \
        --config "$WORK_DIR/online/critic/config.yaml" \
        --rollouts "$WORK_DIR/online/collection/rollouts_train_regen.jsonl" \
        --labels "$WORK_DIR/online/collection/mc_labels_train.jsonl" \
        --split "$WORK_DIR/online/collection/split.json" \
        --continuations "$WORK_DIR/online/collection/continuations_train.jsonl" \
        --initialize "$WORK_DIR/online/critic/initial" \
        --output "$WORK_DIR/online/critic/checkpoints" \
        --global-batch 256 --td-rows-per-batch 128 --microbatch 1 --max-steps 80 --save-every 20
    # 选择20/40/60/80中 held-out 总 Hybrid loss 最小者；同值取较早checkpoint。
    run_logged critic_select "$WORK_DIR/source_online" "" -c \
        'import sys; sys.path.insert(0, sys.argv[1]); import runner; runner.select_critic()' "$WORK_DIR/online"
}

policy() {
    [[ -f "$WORK_DIR/logs/policy.done" ]] || validate_gpus "$HYBRID_GPUS" 2
    # 旧统计仅作对照；新checkpoints目录不复制旧advantage stats，让step6重新拟合。
    if [[ ! -f "$WORK_DIR/online/baseline/policy_old_stats.json" ]]; then
        cp -- "$(dirname -- "$STEP5_DIR")/delta_advantage_stats.json" \
            "$WORK_DIR/online/baseline/policy_old_stats.json"
    fi
    run_logged policy "$WORK_DIR/source_online" "$HYBRID_GPUS" "$WORK_DIR/online/policy_driver.py"
    run_logged policy_verify "$WORK_DIR/source_online" "" -c \
        'import sys; sys.path.insert(0, sys.argv[1]); import runner; runner.verify_policy()' "$WORK_DIR/online"
}

evaluate() {
    local name=$1 step=$2 run=$3
    export_model "$name" "$step" "$run"
    [[ -f "$WORK_DIR/logs/generate_$name.done" ]] || validate_gpus "$EVAL_GPU" 1
    run_logged "generate_$name" "$WORK_DIR/evaluation_tools/source" "$EVAL_GPU" \
        "$WORK_DIR/evaluation_tools/generate.py" --step "$step" --name "$name"
    # CPU评分并运行两种repository main_eval交叉检查，成功后才计入results.csv。
    run_logged "audit_$name" "$WORK_DIR/evaluation_tools/source" "" \
        "$WORK_DIR/evaluation_tools/audit.py" --step "$step" --name "$name"
}

eval_arm() {
    local arm=$1
    evaluate "${arm}_step_5" 5 "$WORK_DIR/$arm"
    evaluate "${arm}_step_10" 10 "$WORK_DIR/$arm"
}
online() { collect; critic; policy; evaluate hybrid_online_step_10 10 "$WORK_DIR/online/policy"; }

bootstrap() {
    [[ -f "$WORK_DIR/logs/initial_collect.done" ]] || validate_gpus "$SAMPLE_GPUS" 0
    run_logged initial_collect "$ASSET_DIR/source_online" "$SAMPLE_GPUS" "$WORK_DIR/collect.py"
    run_logged initial_p95 "$ASSET_DIR/source_td" "" "$WORK_DIR/train/make_p95_subset.py"
    [[ -f "$WORK_DIR/logs/initial_td.done" ]] || validate_gpus "$TD_INIT_GPUS" 8
    # 原TD初始配方：P95数据、128全局batch、16每卡microbatch、8卡、80步。
    run_logged initial_td "$ASSET_DIR/source_td" "$TD_INIT_GPUS" -m torch.distributed.run \
        --standalone --nproc_per_node=8 -m examples.delta_critic.train \
        --config "$WORK_DIR/td.yaml" \
        --rollouts "$WORK_DIR/train/data_p95/rollouts_train_regen.jsonl" \
        --labels "$WORK_DIR/train/data_p95/mc_labels_train.jsonl" \
        --split "$WORK_DIR/train/data_p95/split.json" \
        --output "$WORK_DIR/td_checkpoints" \
        --global-batch 128 --microbatch 16 --max-steps 80 --save-every 20
    [[ -f "$WORK_DIR/logs/initial_hybrid.done" ]] || validate_gpus "$HYBRID_INIT_GPUS" 4
    # 原Hybrid初始配方：同一P95 pool；TD128+terminal128；terminal取前8分支。
    run_logged initial_hybrid "$ASSET_DIR/source_initial_hybrid" "$HYBRID_INIT_GPUS" -m torch.distributed.run \
        --standalone --nproc_per_node=4 -m examples.delta_critic.train \
        --config "$WORK_DIR/hybrid.yaml" \
        --rollouts "$WORK_DIR/train/data_p95/rollouts_train_regen.jsonl" \
        --labels "$WORK_DIR/train/data_p95/mc_labels_train.jsonl" \
        --split "$WORK_DIR/train/data_p95/split.json" \
        --continuations "$WORK_DIR/train/data_p95/continuations_train.jsonl" \
        --output "$WORK_DIR/hybrid_checkpoints" \
        --global-batch 256 --td-rows-per-batch 128 --microbatch 1 --max-steps 80 --save-every 20
    echo "初始critic完成；这是新采样/新训练的权重，记录新SHA256。"
}

case "$STAGE" in
    bootstrap) bootstrap ;;
    td) train_frozen td "$TD_GPUS" 4 ;;
    hybrid) train_frozen hybrid "$HYBRID_GPUS" 2 ;;
    collect) collect ;;
    critic) critic ;;
    policy) policy ;;
    eval-td) eval_arm td ;;
    eval-hybrid) eval_arm hybrid ;;
    eval-online) evaluate hybrid_online_step_10 10 "$WORK_DIR/online/policy" ;;
    online) online ;;
    all)
        train_frozen td "$TD_GPUS" 4
        eval_arm td
        train_frozen hybrid "$HYBRID_GPUS" 2
        eval_arm hybrid
        online
        ;;
esac
echo "阶段完成：$STAGE；结果汇总：$WORK_DIR/evaluation_tools/results.csv"
