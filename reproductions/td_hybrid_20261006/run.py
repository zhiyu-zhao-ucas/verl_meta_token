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
"""Recover, verify, rerun, or resume the sealed October 6 TD/Hybrid sources."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from contextlib import ExitStack
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def read(path):
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text())


def save(path, value):
    """Write a JSON artifact without changing frozen snapshots."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    """Hash an artifact using bounded memory, including large model weights."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    """Reject invalid artifacts even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def verify_bundle():
    """Check the committed archive and every member before restoring anything."""
    manifest = read(PACKAGE / "bundle.json")
    archive = PACKAGE / manifest["archive"]
    require(sha(archive) == manifest["archive_sha256"], "Archive SHA256 differs from the committed manifest")
    names = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = member.name
            require(member.isfile() and name in manifest["files"] and name not in names, f"Unexpected member: {name}")
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, f"Unsafe archive path: {name}")
            data = stream.extractfile(member).read()
            require(hashlib.sha256(data).hexdigest() == manifest["files"][name], f"Member hash mismatch: {name}")
            names.add(name)
    require(names == set(manifest["files"]), "Archive is missing sealed files")
    return manifest


def map_paths(value, replacements):
    """Relocate artifact paths while retaining every numerical training setting."""
    if isinstance(value, dict):
        return {key: map_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [map_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for before, after in replacements.items():
            value = value.replace(before, after)
    return value


def restore(args):
    """Extract exact sources into a new run, then derive relocated configs."""
    from omegaconf import OmegaConf

    bundle = verify_bundle()
    work = args.work_dir.resolve()
    require(not work.exists(), f"Refusing to overwrite an existing run: {work}")
    work.mkdir(parents=True)
    with tarfile.open(PACKAGE / bundle["archive"], "r:gz") as stream:
        for member in stream:
            target = work / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.extractfile(member).read())
            target.chmod(0o444)
    for name in ["driver.py", "runner.py", "prompt_audit.py", "prepare_runs.py", "verify_driver_hooks.py"]:
        shutil.copyfile(work / "runtime" / name, work / name)
        (work / name).chmod(0o444)
    shutil.copyfile(work / "evidence/environment.json", work / "environment.json")
    seal = read(work / "evidence/seal.json")
    replacements = {bundle["original_run"]: str(work)}
    for old in seal["arms"]["td"]["data"]:
        replacements[old] = str(work / "inputs" / Path(old).name)
    for arm in ("td", "hybrid"):
        branch = work / arm
        branch.mkdir()
        for name in ["reward.py", "historical_config.yaml", "prompt_audit.json", "manifest.json"]:
            shutil.copyfile(work / "snapshots" / arm / name, branch / name)
        shutil.copyfile(work / "snapshots" / f"source_{arm}_revision.json", work / f"source_{arm}_revision.json")
        cfg = OmegaConf.to_container(OmegaConf.load(work / "snapshots" / arm / "config.yaml"), resolve=True)
        cfg = map_paths(cfg, replacements)
        cfg["trainer"]["project_name"] = "sealed_td_hybrid_20261006"
        cfg["trainer"]["experiment_name"] = f"{arm}_{work.name}"
        ray_init = cfg["ray_kwargs"]["ray_init"]
        ray_init["_temp_dir"] = f"/tmp/th26-{hashlib.sha256(str(work).encode()).hexdigest()[:10]}-{arm}"
        ray_init["dashboard_port"] = args.port_base + (0 if arm == "td" else 2)
        ray_init["_metrics_export_port"] = args.port_base + (1 if arm == "td" else 3)
        OmegaConf.save(OmegaConf.create(cfg), branch / "config.yaml")
        training = map_paths(read(branch / "manifest.json"), replacements)
        save(branch / "manifest.json", training)
        asset = seal["arms"][arm]
        asset["data"] = {replacements[old]: digest for old, digest in asset["data"].items()}
        asset["config_sha256"] = sha(branch / "config.yaml")
        asset["source_manifest_sha256"] = sha(work / f"source_{arm}_revision.json")
        asset["reward_sha256"] = sha(branch / "reward.py")
        asset["prompt_audit_sha256"] = sha(branch / "prompt_audit.json")
        for name in ["config.yaml", "reward.py", "historical_config.yaml", "prompt_audit.json"]:
            (branch / name).chmod(0o444)
    evaluation = work / "evaluation_tools"
    for name in ["historical_models.json", "protocol.json"]:
        path = evaluation / name
        path.chmod(0o644)
        save(path, map_paths(read(path), replacements))
        path.chmod(0o444)
    revision_path = evaluation / "assets_revision.json"
    revision = read(revision_path)
    revision["files"] = {name: sha(evaluation / name) for name in revision["files"]}
    revision_path.chmod(0o644)
    save(revision_path, revision)
    revision_path.chmod(0o444)
    seal["runtime_assets"] = {name: sha(work / name) for name in seal["runtime_assets"]}
    save(work / "seal.json", seal)
    save(
        work / "restoration.json",
        {
            "archive_sha256": bundle["archive_sha256"],
            "source": str(PACKAGE),
            "work_dir": str(work),
            "mode": "fresh",
            "stop_step": 10,
            "start_step": 0,
            "note": (
                "Numerical settings and frozen training sources retained; "
                "only artifact paths, names and Ray ports relocated."
            ),
        },
    )
    check(work)
    return work


def check(work, arm=None, weights=False):
    """Validate restored code/config; optionally hash all external model weights."""
    from omegaconf import OmegaConf

    bundle = verify_bundle()
    for name, expected in bundle["files"].items():
        if name.startswith(("source_td/", "source_hybrid/", "snapshots/", "runtime/", "inputs/")):
            require(sha(work / name) == expected, f"Frozen restored artifact changed: {name}")
    seal = read(work / "seal.json")
    for name, expected in seal["runtime_assets"].items():
        require(sha(work / name) == expected, f"Runtime artifact changed: {name}")
    for selected in [arm] if arm else ["td", "hybrid"]:
        assets = seal["arms"][selected]
        branch = work / selected
        for filename, key in [
            ("config.yaml", "config_sha256"),
            ("reward.py", "reward_sha256"),
            ("prompt_audit.json", "prompt_audit_sha256"),
        ]:
            require(sha(branch / filename) == assets[key], f"Changed {selected}/{filename}")
        require(
            sha(work / f"source_{selected}_revision.json") == assets["source_manifest_sha256"],
            "Source manifest changed",
        )
        cfg = OmegaConf.load(branch / "config.yaml")
        require(cfg.trainer.n_gpus_per_node == (4 if selected == "td" else 2), "Historical DP topology changed")
        require(cfg.algorithm.delta_policy.critic_update is None, "Frozen critic setting changed")
        if weights:
            model = Path(cfg.actor_rollout_ref.model.path)
            for name, digest in bundle["external_inputs"]["model"]["weights"].items():
                require(sha(model / name) == digest, f"Base weight mismatch: {name}")
            for name, digest in read(branch / "prompt_audit.json")["tokenizer_files"].items():
                require(sha(model / name) == digest, f"Tokenizer file changed: {name}")
            critic = Path(cfg.algorithm.delta_policy.critic_artifact)
            identity = bundle["external_inputs"]["critics"][selected]
            require(sha(critic / "model.pt") == identity["model_sha256"], "Frozen critic weights changed")
            require(sha(critic / "metadata.json") == identity["metadata_sha256"], "Frozen critic metadata changed")
    revision = read(work / "evaluation_tools/assets_revision.json")
    for name, expected in revision["files"].items():
        require(sha(work / "evaluation_tools" / name) == expected, f"Evaluation artifact changed: {name}")
    continuation = work / "continuation_driver.json"
    if continuation.exists():
        require(sha(work / "continue_driver.py") == read(continuation)["sha256"], "Continuation driver changed")


def validate_checkpoint(source, arm, step):
    """Require model/optimizer/RNG/dataloader state and fixed advantage statistics."""
    checkpoint = source / arm / "checkpoints" / f"global_step_{step}"
    world = 4 if arm == "td" else 2
    require(read(checkpoint / "actor/fsdp_config.json")["world_size"] == world, "Checkpoint DP topology differs")
    paths = [checkpoint / "data.pt", source / arm / "checkpoints/delta_advantage_stats.json"]
    for rank in range(world):
        paths.extend(
            checkpoint / "actor" / f"{kind}_world_size_{world}_rank_{rank}.pt"
            for kind in ["model", "optim", "extra_state"]
        )
    require(all(path.is_file() and path.stat().st_size > 0 for path in paths), "Incomplete continuation checkpoint")
    return checkpoint


def verify_environment(work):
    """Reject distribution-version drift before launching the frozen trainer."""

    def normalize(name):
        return name.lower().replace("_", "-")

    expected = {normalize(item["name"]): item["version"] for item in read(work / "environment.json")["packages"]}
    actual = {normalize(dist.metadata["Name"]): dist.version for dist in importlib.metadata.distributions()}
    differences = {
        name: [expected.get(name), actual.get(name)]
        for name in set(expected) | set(actual)
        if expected.get(name) != actual.get(name)
    }
    require(not differences, f"Runtime distribution versions differ from the recorded environment: {differences}")


def prepare_resume(args):
    """Create a separate continuation run without editing frozen training sources."""
    from omegaconf import OmegaConf

    source = args.from_run.resolve()
    checkpoint = validate_checkpoint(source, args.arm, args.step)
    horizon = 100 if args.arm == "td" else 20
    require(args.step < args.stop_step <= horizon, f"Stop step must be after the checkpoint and <= {horizon}")
    stats = read(source / args.arm / "checkpoints/delta_advantage_stats.json")
    prototype = verify_bundle()
    identity = prototype["external_inputs"]["critics"][args.arm]
    require(stats["critic_weights_sha256"] == identity["model_sha256"], "Checkpoint uses a different critic")
    work = restore(args)
    branch = work / args.arm
    cfg = OmegaConf.load(branch / "config.yaml")
    require(
        stats["reference_policy_id"] == cfg.algorithm.delta_policy.reference_policy_id,
        "Reference policy identity differs",
    )
    expected_stats = read(work / "evidence" / args.arm / "delta_advantage_stats.json")
    require(stats["advantage_contract"] == expected_stats["advantage_contract"], "Checkpoint advantage recipe differs")
    cfg.trainer.resume_mode = "resume_path"
    cfg.trainer.resume_from_path = str(checkpoint)
    cfg.trainer.del_local_ckpt_after_load = False
    (branch / "config.yaml").chmod(0o644)
    OmegaConf.save(cfg, branch / "config.yaml")
    (branch / "config.yaml").chmod(0o444)
    (branch / "checkpoints").mkdir()
    shutil.copyfile(
        source / args.arm / "checkpoints/delta_advantage_stats.json", branch / "checkpoints/delta_advantage_stats.json"
    )
    original = (work / "driver.py").read_text()
    expected_prompt = 'expected = audit["ordered_training_prompt_schedule"][step]["token_hashes"]'
    fresh_stats = 'assert not (branch / "checkpoints/delta_advantage_stats.json").exists()'
    replacements = {
        "assert trainer.global_steps == 0": f"assert trainer.global_steps == {args.step}",
        "trainer.total_training_steps = 10": f"trainer.total_training_steps = {args.stop_step}",
        expected_prompt: (
            'known = audit["ordered_training_prompt_schedule"].get(step)\n'
            '                expected = known["token_hashes"] if known else token_hashes'
        ),
        "ordered_prompt_tokens_equal=True,": "ordered_prompt_tokens_equal=True if known else None,",
        fresh_stats: 'assert (branch / "checkpoints/delta_advantage_stats.json").is_file()',
    }
    continued = original
    for before, after in replacements.items():
        require(continued.count(before) == 1, f"Continuation adapter no longer matches the frozen driver: {before}")
        continued = continued.replace(before, after)
    (work / "continue_driver.py").write_text(continued)
    (work / "continue_driver.py").chmod(0o444)
    save(
        work / "continuation_driver.json",
        {
            "sha256": sha(work / "continue_driver.py"),
            "original_driver_sha256": sha(work / "driver.py"),
            "replacements": replacements,
            "arm": args.arm,
            "checkpoint": str(checkpoint),
            "note": (
                "Frozen algorithm sources and original fresh driver remain byte-identical. "
                "Beyond step10 prompt hashes are recorded without claiming historical schedule equality."
            ),
        },
    )
    state = read(work / "restoration.json")
    state.update(mode="resume", arm=args.arm, start_step=args.step, stop_step=args.stop_step)
    save(work / "restoration.json", state)
    seal = read(work / "seal.json")
    seal["arms"][args.arm]["config_sha256"] = sha(branch / "config.yaml")
    save(work / "seal.json", seal)
    check(work)
    return work


def load_runner(work):
    """Load operational helpers from the frozen runner rather than the workspace."""
    spec = importlib.util.spec_from_file_location("sealed_runner", work / "runner.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the frozen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(args):
    """Run a fresh or continued arm only after verification and idle GPU leases."""
    work = args.work_dir.resolve()
    check(work, args.arm, weights=True)
    verify_environment(work)
    require(len(args.gpus) == (4 if args.arm == "td" else 2), "GPU count must match historical DP")
    require(len(set(args.gpus)) == len(args.gpus), "GPU indices must be unique")
    runner = load_runner(work)
    branch = work / args.arm
    state = read(work / "restoration.json")
    resumed = state["mode"] == "resume"
    require(not resumed or state["arm"] == args.arm, "Resume arm differs from prepared checkpoint")
    require(not (branch / "train.log").exists(), "This arm already started; prepare another run to retry safely")
    with ExitStack() as stack:
        for gpu in args.gpus:
            lock = stack.enter_context(Path(f"/tmp/delta_old4_evaluation_gpu_{gpu}.lock").open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        usage = runner.snapshot()
        require(
            all(g in usage and not usage[g]["compute_pids"] and usage[g]["memory_used_mib"] <= 512 for g in args.gpus),
            "Requested GPUs are occupied",
        )
        env = runner.environment(work / f"source_{args.arm}")
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(map(str, args.gpus)), VERL_FILE_LOGGER_PATH=str(branch / "metrics.jsonl")
        )
        from omegaconf import OmegaConf

        env["RAY_TMPDIR"] = OmegaConf.load(branch / "config.yaml").ray_kwargs.ray_init._temp_dir
        env["UV_CACHE_DIR"] = os.environ.get("UV_CACHE_DIR", "/tmp/uv-codex-audit")
        command = [sys.executable, str(work / ("continue_driver.py" if resumed else "driver.py")), "--arm", args.arm]
        save(branch / "started.json", {"physical_gpus": args.gpus, "mode": state["mode"], "command": command})
        with (branch / "train.log").open("x") as logfile:
            proc = subprocess.Popen(
                command, cwd=work / f"source_{args.arm}", env=env, stdout=logfile, stderr=subprocess.STDOUT
            )
            save(branch / "launch_status.json", {"state": "running", "pid": proc.pid, "command": command})
            rc = proc.wait()
        (branch / "exit_code").write_text(str(rc) + "\n")
        save(branch / "launch_status.json", {"state": "complete" if rc == 0 else "failed", "returncode": rc})
        require(rc == 0, f"Training exited {rc}; inspect {branch / 'train.log'}")
        if not resumed:
            runner.verify_training(args.arm)
        else:
            rows = [json.loads(line) for line in (branch / "metrics.jsonl").read_text().splitlines() if line.strip()]
            steps = {row["step"] for row in rows if row["step"] > state["start_step"]}
            require(steps == set(range(state["start_step"] + 1, state["stop_step"] + 1)), "Missing continued updates")
            require(
                all(math.isfinite(row["data"]["actor/loss"]) for row in rows if row["step"] in steps),
                "Nonfinite actor loss",
            )
            validate_checkpoint(work, args.arm, state["stop_step"])
            save(branch / "policy_complete.json", {"verified": True, "mode": "resume", "steps": sorted(steps)})


def evaluate(args):
    """Use the exact frozen step5/step10 evaluator on a completed fresh arm."""
    work = args.work_dir.resolve()
    check(work, args.arm)
    require(
        read(work / "restoration.json")["mode"] == "fresh", "This evaluator retains the original step5/step10 protocol"
    )
    runner = load_runner(work)
    command = [
        sys.executable,
        str(work / "evaluation_tools/evaluate.py"),
        "--arm",
        args.arm,
        "--stage",
        "all",
        "--gpus",
        ",".join(map(str, args.gpus)),
    ]
    return subprocess.call(
        command, cwd=work / "evaluation_tools/source", env=runner.environment(work / "evaluation_tools/source")
    )


def main():
    """Expose recovery, preflight, training, continuation and evaluation commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-bundle")
    for name in ["restore", "prepare-resume", "check", "launch", "evaluate"]:
        sub = commands.add_parser(name)
        sub.add_argument("--work-dir", type=Path, required=True)
        if name in ["restore", "prepare-resume"]:
            sub.add_argument("--port-base", type=int, default=35200)
        if name in ["prepare-resume", "launch", "evaluate"]:
            sub.add_argument("--arm", choices=["td", "hybrid"], required=True)
        if name in ["launch", "evaluate"]:
            sub.add_argument("--gpus", type=lambda v: [int(x) for x in v.split(",")], required=True)
        if name == "check":
            sub.add_argument("--weights", action="store_true")
        if name == "prepare-resume":
            sub.add_argument("--from-run", type=Path, required=True)
            sub.add_argument("--step", type=int, default=10)
            sub.add_argument("--stop-step", type=int, required=True)
    args = parser.parse_args()
    if args.command == "verify-bundle":
        print(f"Verified {len(verify_bundle()['files'])} archived files")
    elif args.command == "restore":
        print(restore(args))
    elif args.command == "prepare-resume":
        print(prepare_resume(args))
    elif args.command == "check":
        check(args.work_dir.resolve(), weights=args.weights)
        print("Restored code, config and input checks passed")
    elif args.command == "launch":
        launch(args)
    else:
        raise SystemExit(evaluate(args))


if __name__ == "__main__":
    main()
