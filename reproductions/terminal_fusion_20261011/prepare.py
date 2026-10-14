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
"""Rebuild the frozen C/D terminal-fusion runs from a public Git checkout.

This command only writes a new run directory. It never starts policy training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE_COMMIT = "5b6b0db8a6114341c9264cae7df3bded372785b9"
RUNS = ("C_run_1", "D_run_1", "C_run_2", "D_run_2")
DATA_SHA256 = {
    "train.parquet": "6e020de627a86126f79e344a07e318d75c61a5797e25cf99d5b9deeb73f91942",
    "validation.parquet": "753e608c85527689dfac6dd27895c1cb859884b5fd6c95413521510f35dc2db7",
}
BASE_TOKENIZER_SHA256 = "5c89ba5026ac05ebca167030f5359176783f8268cd7b85e74713654c91682ee4"
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bytes(repo: Path, relative: str, variant: str) -> bytes:
    override = HERE / "overrides" / variant / relative
    if override.is_file():
        return override.read_bytes()
    if variant == "source_run_2":
        common_override = HERE / "overrides" / "source" / relative
        if common_override.is_file():
            return common_override.read_bytes()
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{BASE_COMMIT}:{relative}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def build_source(repo: Path, output: Path, variant: str) -> None:
    manifest_file = "source_manifest.json" if variant == "source" else "source_run_2_manifest.json"
    manifest = json.loads((HERE / manifest_file).read_text())
    for relative, expected in manifest.items():
        content = source_bytes(repo, relative, variant)
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source mismatch: {relative}: {actual} != {expected}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    shutil.copy2(HERE / manifest_file, output.parent / manifest_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="checkout of the public repository")
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent run directory")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1", help="two visible physical GPU IDs, default 0,1")
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    output = args.output.resolve()
    base_model = args.base_model.resolve(strict=True)
    critic = args.critic.resolve(strict=True)
    train = args.train.resolve(strict=True)
    validation = args.validation.resolve(strict=True)
    if output.exists():
        parser.error(f"output already exists: {output}")
    available = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}"],
        capture_output=True,
    )
    if available.returncode:
        parser.error(f"checkout does not contain required public commit {BASE_COMMIT}")
    if not (base_model / "config.json").is_file() or not (base_model / "tokenizer.json").is_file():
        parser.error("base model is missing config.json or tokenizer.json")
    if not (critic / "model.pt").is_file() or not (critic / "metadata.json").is_file():
        parser.error("critic needs model.pt and metadata.json")
    metadata = json.loads((critic / "metadata.json").read_text())
    if sha256(critic / "model.pt") != metadata.get("weights_sha256"):
        parser.error("critic weight SHA256 does not match metadata")
    if metadata.get("config", {}).get("delta_label_mode") != "selected_segment":
        parser.error("critic label mode must be selected_segment")
    if metadata.get("config", {}).get("objective") != "hybrid_terminal_composition":
        parser.error("critic objective must be hybrid_terminal_composition")
    if metadata.get("tokenizer_sha256") != BASE_TOKENIZER_SHA256:
        parser.error("critic tokenizer fingerprint differs from the pinned Qwen Base")
    for file, expected in ((train, DATA_SHA256["train.parquet"]), (validation, DATA_SHA256["validation.parquet"])):
        if sha256(file) != expected:
            parser.error(f"historical parquet SHA256 mismatch: {file}")
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not set(gpu_ids) <= {0, 1, 5, 6, 7}:
        parser.error("the frozen launcher needs two distinct GPU IDs from {0,1,5,6,7}")

    output.mkdir(parents=True)
    shutil.copy2(HERE / "train_once.py", output / "train_once.py")
    variants = {"source" if run.endswith("_1") else "source_run_2" for run in args.runs}
    for variant in variants:
        build_source(repo, output / variant, variant)
    manifest = {
        "gpu_plan": gpu_ids,
        "gpu_plan_by_run": {run: gpu_ids for run in args.runs},
        "source_by_run": {run: str(output / ("source" if run.endswith("_1") else "source_run_2")) for run in args.runs},
        "source_manifests_by_run": {
            run: str(output / ("source_manifest.json" if run.endswith("_1") else "source_run_2_manifest.json"))
            for run in args.runs
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for run in args.runs:
        branch = output / run
        branch.mkdir()
        shutil.copy2(HERE / "original" / run / "reward.py", branch / "reward.py")
        config = json.loads((HERE / "original" / run / "config.yaml").read_text())
        source = output / ("source" if run.endswith("_1") else "source_run_2")
        config["data"]["train_files"] = str(train)
        config["data"]["val_files"] = str(validation)
        config["actor_rollout_ref"]["model"]["path"] = str(base_model)
        config["actor_rollout_ref"]["rollout"]["prometheus"]["served_model_name"] = str(base_model)
        config["algorithm"]["delta_policy"]["critic_artifact"] = str(critic)
        config["reward"]["custom_reward_function"]["path"] = str(branch / "reward.py")
        trainer = config["trainer"]
        trainer["default_local_dir"] = str(branch / "checkpoints")
        trainer["rollout_data_dir"] = str(branch / "rollouts")
        trainer["validation_data_dir"] = str(branch / "validation")
        ray = config["ray_kwargs"]["ray_init"]
        ray_dir = Path("/tmp") / f"terminal_fusion_{hashlib.sha256(str(output / run).encode()).hexdigest()[:12]}"
        ray["_temp_dir"] = str(ray_dir)
        env = ray["runtime_env"]["env_vars"]
        env["PYTHONPATH"] = str(source)
        env["TMPDIR"] = str(ray_dir / "tmp")
        env["VLLM_RPC_BASE_PATH"] = str(ray_dir / "rpc")
        env["VERL_WARMUP_LR_TRACE_DIR"] = str(branch / "lr_trace")
        (branch / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "public_base_commit": BASE_COMMIT,
        "model_path": str(base_model),
        "critic_path": str(critic),
        "critic_weights_sha256": metadata["weights_sha256"],
        "critic_tokenizer_sha256": metadata.get("tokenizer_sha256"),
        "train_sha256": DATA_SHA256["train.parquet"],
        "validation_sha256": DATA_SHA256["validation.parquet"],
        "runs": args.runs,
    }
    (output / "reproduction_inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {output}")
    for run in args.runs:
        print(f"Preflight: python {output / 'train_once.py'} {run} --preflight")
        print(f"Training:  python {output / 'train_once.py'} {run}")


if __name__ == "__main__":
    main()
