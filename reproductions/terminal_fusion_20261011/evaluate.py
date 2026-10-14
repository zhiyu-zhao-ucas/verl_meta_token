# Copyright 2026 Individual Contributor: zhiyu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export and evaluate one terminal-fusion checkpoint with the frozen pass@16 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    "prepare.py": "1d4ae48653ef59649dda9be55fa37c3cff988980659001c83a7d513df24c7214",
    "generate.py": "02b84e15bb6458d300d8dfb35a5d020c5d786c438c0306b526782635b06ee4ca",
    "audit.py": "b580aa98a8584f30e833c2fd27af5c3ea325619e053b9a88871a1d4e17fbff8f",
    "scorers.py": "67331923a2a49ae79edf850c12f905077b267d574d6d5364eb2de0cb6396a5a7",
    "report.py": "64cefc5409dd8d17cdb3603ff5c25d604145a8af5c37d5a6f2b00dd41f395e4b",
}
PROMPTS_SHA256 = "b34a8ce86f02a648edb621255637cdcc25e999213db0cee0c00ccebd5eec98ca"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage(name: str, command: list[str], directory: Path, environment: dict[str, str]) -> None:
    log = directory / f"{name}.log"
    with log.open("x") as stream:
        subprocess.run(
            command,
            cwd=directory / "source",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"{name}: complete; log {log}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="directory created by prepare.py")
    parser.add_argument("--assets-dir", type=Path, required=True, help="extracted handoff assets")
    parser.add_argument("--run", required=True, choices=("C_run_1", "D_run_1", "C_run_2", "D_run_2"))
    parser.add_argument("--step", required=True, type=int, choices=(3, 5, 6, 7, 10))
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU ID for one vLLM replica")
    parser.add_argument("--prepare-only", action="store_true", help="check and stage inputs without GPU work")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve(strict=True)
    assets = args.assets_dir.resolve(strict=True) / "evaluation_tools"
    branch = run_dir / args.run
    config = json.loads((branch / "config.yaml").read_text())
    base_model = Path(config["actor_rollout_ref"]["model"]["path"])
    checkpoint = branch / "checkpoints" / f"global_step_{args.step}" / "actor"
    if not (checkpoint / "fsdp_config.json").is_file():
        parser.error(f"missing completed actor checkpoint: {checkpoint}")
    for filename, expected in SCRIPTS.items():
        if sha256(assets / filename) != expected:
            parser.error(f"evaluation script hash mismatch: {filename}")
    if sha256(assets / "prompts.jsonl") != PROMPTS_SHA256:
        parser.error("evaluation prompts hash mismatch")
    name = f"{args.run}_step_{args.step}"
    destination = run_dir / "evaluation" / name
    if destination.exists():
        parser.error(f"evaluation directory already exists: {destination}")
    destination.mkdir(parents=True)
    for filename in (*SCRIPTS, "prompts.jsonl"):
        shutil.copy2(assets / filename, destination / filename)
    original = (destination / "audit.py").read_text()
    old_command = (
        "command = ['/tmp/verl-uv-bin/uv', 'run', '--python',\n"
        "                   '/scratch2/zhiyu/miniconda3/envs/"
        "verl-delta-vllm-20260925/bin/python', '--no-project',\n"
        "                   'python', '-m', 'verl.trainer.main_eval',"
    )
    if original.count(old_command) != 1:
        raise ValueError("Frozen audit script changed; expected path adapter not found")
    adapted = original.replace(old_command, "command = [sys.executable, '-m', 'verl.trainer.main_eval',")
    (destination / "audit.py").write_text(adapted)
    (destination / "source").symlink_to((assets / "source").resolve(strict=True), target_is_directory=True)
    protocol = json.loads((assets / "protocol.json").read_text())
    protocol["actor"] = str(base_model)
    protocol["checkpoints"] = [args.step]
    protocol["physical_gpus"] = [args.gpu]
    protocol["sources"]["dapo_holdout"] = str(Path(config["data"]["val_files"]))
    protocol["sources"]["math500"] = "frozen prompts.jsonl in the public handoff assets"
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    registry = {
        name: {
            "label": name,
            "run": str(branch),
            "source": str(checkpoint),
            "step": args.step,
            "world_size": 2,
            "train_files": config["data"]["train_files"],
            "overlapping_prompt_indices": [],
            "exact_training_overlap_counts": {key: 0 for key in protocol["dataset_counts"]},
        }
    }
    (destination / "historical_models.json").write_text(json.dumps(registry, indent=2) + "\n")
    if args.prepare_only:
        print(f"Evaluation inputs prepared: {destination}")
        return
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(destination / "source"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="2",
    )
    environment.pop("RAY_ADDRESS", None)
    run_stage(
        "export",
        [
            sys.executable,
            str(destination / "prepare.py"),
            "--export",
            str(args.step),
            "--run",
            str(branch),
            "--name",
            name,
        ],
        destination,
        environment,
    )
    run_stage(
        "generate",
        [sys.executable, str(destination / "generate.py"), "--step", str(args.step), "--name", name],
        destination,
        environment,
    )
    run_stage(
        "score",
        [sys.executable, str(destination / "audit.py"), "--step", str(args.step), "--name", name],
        destination,
        environment,
    )
    summary = json.loads((destination / name / "summary.json").read_text())
    assert summary["samples_per_prompt"] == 16
    assert len(summary["repository_main_eval_crosschecks"]) == 2
    print(f"Evaluation complete: {destination / 'results.csv'}")


if __name__ == "__main__":
    main()
