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
"""Run sealed 688-prompt evaluation for boundary value/delta policies and Base."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml


def require(condition, message):
    """Fail closed when evaluation inputs or readiness checks differ."""
    if not condition:
        raise ValueError(message)


def load_module(path, name):
    """Load the frozen operational helper without importing workspace code."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(work):
    """Restore exact evaluation code and prompts, adapting model identities only."""
    from .value_policy_pipeline import REPO, check_inputs, load_reproduction, read, save, sha

    require(read(work / "status.json")["state"] == "complete", "Training pipeline must complete first")
    require((work / "exit_code").read_text().strip() == "0", "Training exited unsuccessfully")
    settings = read(work / "settings.json")
    check_inputs(work, settings)
    reproduction = load_reproduction(REPO)
    bundle = reproduction.verify_bundle()
    root = work / "evaluation"
    require(not root.exists(), f"Refusing to overwrite {root}")
    root.mkdir()
    with tarfile.open(reproduction.PACKAGE / bundle["archive"], "r:gz") as stream:
        for member in stream:
            if not member.name.startswith("evaluation_tools/"):
                continue
            relative = Path(member.name).relative_to("evaluation_tools")
            # The bundle contains code/assets, not previous generation results.
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.extractfile(member).read())
    inherited = read(root / "assets_revision.json")
    for name, digest in inherited["files"].items():
        require(sha(root / name) == digest, f"Sealed evaluation asset differs: {name}")
    protocol = read(root / "protocol.json")
    require(
        protocol["sampling"]["n"] == 16 and protocol["sampling"]["max_tokens"] == 8192,
        "Historical sampling budget differs",
    )
    require(
        protocol["dataset_counts"] == {"math500": 500, "aime2024": 30, "aime2025": 30, "dapo_holdout": 128},
        "Historical dataset counts differ",
    )
    require(sha(root / "prompts.jsonl") == protocol["prompts_sha256"], "Evaluation prompts differ")
    require(protocol["actor"] == settings["base"], "Base identity differs")
    for name, digest in settings["base_weights"].items():
        require(sha(Path(settings["base"]) / name) == digest, f"Base weights changed: {name}")
    for arm in ("value", "direct_delta"):
        config = yaml.safe_load((work / f"policy_{arm}/td/config.yaml").read_text())
        require(
            sha(config["data"]["train_files"]) == read(root / "provenance.json")["legacy_train_sha256"],
            "Policy training data differs from frozen overlap audit",
        )
    original_models = read(root / "historical_models.json")
    registry, checkpoints = {}, {}
    registry["base_step_0"] = {
        "label": "base",
        "step": 0,
        "source": settings["base"],
        "overlapping_prompt_indices": [],
        "exact_training_overlap_counts": {key: 0 for key in protocol["dataset_counts"]},
    }
    for arm in ("value", "direct_delta"):
        run = work / f"policy_{arm}/td"
        for step in (5, 10):
            name = f"{arm}_step_{step}"
            model = dict(original_models[f"td_step_{step}"])
            source = run / f"checkpoints/global_step_{step}/actor"
            model.update(
                label=arm,
                run=str(run),
                source=str(source),
                train_files=[str(work / "inputs/train.parquet")],
                world_size=4,
            )
            registry[name] = model
            require(read(source / "fsdp_config.json")["world_size"] == 4, "Actor checkpoint topology differs")
            files = [source / f"model_world_size_4_rank_{rank}.pt" for rank in range(4)]
            files += [
                source / "fsdp_config.json",
                source / "huggingface/config.json",
                source / "huggingface/tokenizer_config.json",
            ]
            for path in files:
                require(path.is_file() and path.stat().st_size > 0, f"Incomplete actor checkpoint: {path}")
            checkpoints[name] = {str(path): sha(path) for path in files}
    save(root / "historical_models.json", registry)
    # All new model names use the original_v1 engine/batching profile.
    acceleration = read(root / "accelerated_inference.json")
    acceleration.update(enabled=False, models=[])
    save(root / "accelerated_inference.json", acceleration)
    protocol["notes"] = [
        "Controlled value-difference and direct-delta policies, steps5/10, plus the common Base.",
        "Same sealed 688 prompts, 16 independent responses, 8192-token cap and original_v1 inference profile.",
        "Same frozen scoring scripts; 2048-token results are prefix diagnostics of the same generated responses.",
    ]
    save(root / "protocol.json", protocol)
    adapter = root / "evaluate_boundary.py"
    shutil.copyfile(__file__, adapter)
    revision = {name: sha(root / name) for name in inherited["files"]}
    revision[adapter.name] = sha(adapter)
    save(root / "assets_revision.json", {"files": revision})
    for name in revision:
        (root / name).chmod(0o444)
    save(
        root / "evaluation_identity.json",
        {
            "parent_bundle_sha256": bundle["archive_sha256"],
            "original_assets_revision": inherited,
            "adapted_assets": revision,
            "checkpoints": checkpoints,
            "training_work": str(work),
            "prompts_sha256": protocol["prompts_sha256"],
            "model_names": list(registry),
            "sample_count_per_model": 688 * 16,
            "total_sample_count": len(registry) * 688 * 16,
            "changes": [
                "model registry paths/labels",
                "protocol explanatory notes",
                "disable optional accelerated profile",
                "readiness adapter for the completed controlled training pipeline",
            ],
        },
    )
    return root


def run(root, gpus):
    """Delegate export, leased single-GPU generation and scoring to frozen code."""
    evaluator = load_module(root / "run_stage.py", "sealed_boundary_evaluator")
    identity = evaluator.read(root / "evaluation_identity.json")
    work = Path(identity["training_work"])

    def readiness(stage, include_base=False):
        require(stage == "all" and not include_base, "Boundary evaluation registry already includes Base")
        require(evaluator.read(work / "status.json")["state"] == "complete", "Training is incomplete")
        registry = evaluator.read(root / "historical_models.json")
        require(set(registry) == set(identity["model_names"]), "Evaluation model registry changed")
        pending = []
        for name, hashes in identity["checkpoints"].items():
            for path, digest in hashes.items():
                # Avoid materializing multi-GB checkpoint files in sha's read_bytes helper.
                import hashlib

                hasher = hashlib.sha256()
                with Path(path).open("rb") as stream:
                    for block in iter(lambda: stream.read(8 << 20), b""):
                        hasher.update(block)
                require(hasher.hexdigest() == digest, f"Checkpoint changed: {name} {path}")
        for name, model in registry.items():
            ready, reason = evaluator.checkpoint_check(model)
            if not ready:
                pending.append(f"{name}: {reason}")
        return registry, pending

    evaluator.stage_readiness = readiness
    sys.argv = [str(root / "run_stage.py"), "--stage", "all", "--gpus", ",".join(map(str, gpus))]
    try:
        evaluator.main()
    except Exception:
        (root / "exit_code").write_text("1\n")
        raise
    (root / "exit_code").write_text("0\n")


def main():
    """Prepare or detach a formal evaluation without changing training outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "start", "run"])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    args = parser.parse_args()
    require(args.gpus and len(set(args.gpus)) == len(args.gpus) and set(args.gpus) <= set(range(8)), "Invalid GPU list")
    work = args.work_dir.resolve()
    if args.action == "run":
        run(work / "evaluation", args.gpus)
        return
    root = prepare(work)
    if args.action == "start":
        with (root / "evaluation.log").open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "evaluate_boundary.py"),
                    "run",
                    "--work-dir",
                    str(work),
                    "--gpus",
                    *map(str, args.gpus),
                ],
                cwd=root / "source",
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH=str(root / "source")),
                start_new_session=True,
            )
        (root / "launcher.json").write_text(json.dumps({"pid": process.pid, "gpus": args.gpus}) + "\n")
        print(json.dumps({"pid": process.pid, "evaluation_dir": str(root), "gpus": args.gpus}), flush=True)
    else:
        print(str(root), flush=True)


if __name__ == "__main__":
    main()
