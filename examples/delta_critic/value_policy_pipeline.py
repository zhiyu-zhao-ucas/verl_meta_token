# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Freeze, audit and run value -> policy, followed by a matched delta control."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MODULES = ("value_difference.py", "value_difference_score.py", "train_boundary_scalar.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reproduction(repo):
    path = repo / "reproductions/td_hybrid_20261006/run.py"
    spec = importlib.util.spec_from_file_location("value_pipeline_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(work, gpus):
    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
        PYTHONPATH=str(work / "source") + ":" + str(work),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled",
    )
    return env


def prepare(args):
    from .checkpoint import read_artifact
    from .legacy_adapter import adapt_legacy_rows
    from .train import read_rows
    from .value_difference import SEMANTICS, boundary_rows

    work = args.work_dir.resolve()
    require(not work.exists(), f"Refusing to overwrite {work}")
    require(len(args.gpus) == 4 and len(set(args.gpus)) == 4, "Both arms use TD DP4; specify four distinct GPUs")
    reproduction = load_reproduction(REPO)
    bundle = reproduction.verify_bundle()
    original = Path(bundle["original_run"])
    base = Path(args.base_model or bundle["external_inputs"]["model"]["original_path"]).resolve()
    for name, expected in bundle["external_inputs"]["model"]["weights"].items():
        require(sha(base / name) == expected, f"Base identity mismatch: {name}")
    td80 = read_artifact(bundle["external_inputs"]["critics"]["td"]["original_path"])
    data = (args.data_dir or REPO / "outputs/delta_qwen3_4b_base_dapo_20260927/train/data_p95").resolve()
    data_names = {"rollouts": "rollouts_train_regen.jsonl", "labels": "mc_labels_train.jsonl", "split": "split.json"}
    hashes = {key: sha(data / name) for key, name in data_names.items()}
    require(hashes == td80["data_fingerprints"], "Data/split must match the actual TD80 training fingerprints")
    examples, _ = adapt_legacy_rows(read_rows(data / data_names["rollouts"]), read_rows(data / data_names["labels"]))
    split = read(data / "split.json")
    require(
        len(examples) == 486
        and list(split.values()).count("train") == 438
        and list(split.values()).count("eval") == 48,
        "Expected the exact 438/48 TD80 split",
    )
    require(set(split) == {example.rollout.rollout_id for example in examples}, "Split IDs differ")
    value, delta = boundary_rows(examples, "value"), boundary_rows(examples, "direct_delta")
    for left, right in zip(value, delta, strict=True):
        require(
            {k: v for k, v in left.items() if k != "boundary_targets"}
            == {k: v for k, v in right.items() if k != "boundary_targets"},
            "Arms differ beyond target values",
        )
    work.mkdir(parents=True)
    source = work / "source"
    with tarfile.open(reproduction.PACKAGE / bundle["archive"], "r:gz") as stream:
        for member in stream:
            name = member.name
            if name.startswith("source_td/"):
                dest = source / Path(name).relative_to("source_td")
            elif name.startswith(("snapshots/td/", "inputs/")) or name in (
                "runtime/driver.py",
                "runtime/prompt_audit.py",
                "evidence/environment.json",
            ):
                dest = work / name
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(stream.extractfile(member).read())
    for name in MODULES:
        shutil.copyfile(REPO / "examples/delta_critic" / name, source / "examples/delta_critic" / name)
    factory = source / "examples/delta_critic/policy_online.py"
    text = factory.read_text()
    anchor = "    worker_class = FrozenDeltaWorker\n"
    require(text.count(anchor) == 1, "Frozen scorer factory changed")
    replacement = (
        "    from .checkpoint import read_artifact\n"
        "    metadata = read_artifact(policy['critic_artifact'])\n"
        "    if metadata.get('artifact_kind') == 'boundary_scalar':\n"
        "        if policy.get('critic_update') is not None:\n"
        "            raise ValueError('Boundary comparison requires a frozen critic')\n"
        "        from .value_difference_score import FrozenBoundaryWorker\n"
        "        return FrozenBoundaryWorker(policy['critic_artifact'],\n"
        "            device=policy.get('scorer_device', 'cpu'),\n"
        "            microbatch=int(policy.get('scorer_microbatch', 1)),\n"
        "            scoring_max_length=policy.get('scoring_max_length'))\n" + anchor
    )
    factory.write_text(text.replace(anchor, replacement))
    source_files = {str(path.relative_to(source)): sha(path) for path in sorted(source.rglob("*")) if path.is_file()}
    save(source / "source_manifest.json", {"files": source_files})
    for path in source.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    critic_inputs = work / "critic_inputs"
    critic_inputs.mkdir()
    for name in data_names.values():
        shutil.copyfile(data / name, critic_inputs / name)
        (critic_inputs / name).chmod(0o444)
    # Starting from the recorded TD80 config retains architecture, LR, clipping,
    # dtype and context. Both controlled targets use the same required changes.
    config = dict(td80["config"])
    config.update(
        model_path=str(base),
        tokenizer_path=str(base),
        loss_type="mse",
        loss_mask="selected_state",
        target_normalization="none",
    )
    (work / "critic_config.yaml").write_text(yaml.safe_dump(config))
    (work / "critic_config.yaml").chmod(0o444)
    coordinator = work / "pipeline.py"
    shutil.copyfile(__file__, coordinator)
    coordinator.chmod(0o444)
    common_changes = {
        key: {"historical": td80["config"].get(key), "controlled": config[key]}
        for key in ("loss_type", "loss_mask", "target_normalization")
    }
    policy_configs = {}
    for index, arm in enumerate(("value", "direct_delta")):
        root = work / f"policy_{arm}"
        branch = root / "td"
        branch.mkdir(parents=True)
        cfg = yaml.safe_load((work / "snapshots/td/config.yaml").read_text())
        cfg = reproduction.map_paths(cfg, {str(original / "source_td"): str(source), str(original): str(root)})
        for key in ("train_files", "val_files"):
            cfg["data"][key] = str(work / "inputs" / Path(cfg["data"][key]).name)
        cfg["actor_rollout_ref"]["model"]["path"] = str(base)
        cfg["algorithm"]["delta_policy"]["critic_artifact"] = str(work / f"critic_{arm}/best")
        require(cfg["algorithm"]["delta_policy"]["critic_update"] is None, "Critic must remain frozen")
        cfg["trainer"]["project_name"] = "controlled_value_difference"
        cfg["trainer"]["experiment_name"] = f"{arm}_{work.name}"
        ray = cfg["ray_kwargs"]["ray_init"]
        ray["runtime_env"]["working_dir"] = str(source)
        ray["runtime_env"]["env_vars"]["PYTHONPATH"] = str(source) + ":" + str(root)
        ray["_temp_dir"] = f"/tmp/vdiff-{hashlib.sha256(str(work).encode()).hexdigest()[:8]}-{index}"
        ray["dashboard_port"], ray["_metrics_export_port"] = args.port_base + index * 2, args.port_base + index * 2 + 1
        (branch / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        policy_configs[arm] = cfg
        for name in ("reward.py", "prompt_audit.json"):
            shutil.copyfile(work / "snapshots/td" / name, branch / name)
        driver = (work / "runtime/driver.py").read_text()
        require(driver.count("trainer.total_training_steps = 10") == 1, "Historical policy driver changed")
        driver = driver.replace(
            "trainer.total_training_steps = 10", f"trainer.total_training_steps = {args.policy_steps}"
        )
        driver = driver.replace("loop_stop_step=10", f"loop_stop_step={args.policy_steps}")
        (root / "driver.py").write_text(driver)
        shutil.copyfile(work / "runtime/prompt_audit.py", root / "prompt_audit.py")
        save(
            root / "source_td_revision.json",
            {"files": source_files, "note": "sealed TD sources plus common boundary scorer"},
        )

    # Numerical policy settings must remain exactly equal; only identity/path
    # strings and isolated Ray ports differ.
    def canonical(value):
        if isinstance(value, dict):
            return {
                key: canonical(item)
                for key, item in value.items()
                if key not in {"dashboard_port", "_metrics_export_port", "_temp_dir", "experiment_name"}
            }
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, str):
            return (
                value.replace("policy_value", "policy_ARM")
                .replace("policy_direct_delta", "policy_ARM")
                .replace("critic_value", "critic_ARM")
                .replace("critic_direct_delta", "critic_ARM")
            )
        return value

    require(
        canonical(policy_configs["value"]) == canonical(policy_configs["direct_delta"]), "Policy numerical config drift"
    )
    reproduction.verify_environment(work / "evidence")
    settings = {
        "repo": str(REPO),
        "work_dir": str(work),
        "gpus": args.gpus,
        "python": sys.executable,
        "critic_steps": args.critic_steps,
        "policy_steps": args.policy_steps,
        "eval_every": args.eval_every,
        "global_batch": 128,
        "microbatch": 16,
        "base": str(base),
        "base_weights": bundle["external_inputs"]["model"]["weights"],
        "data_hashes": hashes,
        "data_names": data_names,
        "source_files": source_files,
        "policy_data_hashes": {
            str(path.relative_to(work)): sha(path) for path in (work / "inputs").iterdir() if path.is_file()
        },
        "critic_config_sha256": sha(work / "critic_config.yaml"),
        "coordinator_sha256": sha(coordinator),
        "policy_files": {
            str(path.relative_to(work)): sha(path)
            for arm in ("value", "direct_delta")
            for path in (work / f"policy_{arm}").rglob("*")
            if path.is_file()
        },
        "sealed_archive_sha256": bundle["archive_sha256"],
    }
    save(work / "settings.json", settings)
    save(
        work / "alignment.json",
        {
            "preflight_passed": True,
            "boundary_semantics": SEMANTICS,
            "data_identical_to_td80": hashes,
            "train_rows": 438,
            "eval_rows": 48,
            "paired_inputs_support_equal": True,
            "policy_numerical_configs_equal": True,
            "common_changes_from_historical_td80": common_changes,
            "historical_position": "prompt_length + i (after token i)",
            "controlled_position": "prompt_length + i - 1 (before token i)",
            "historical_background": "response-wide zero delta targets",
            "controlled_background": "unsampled prefix values unknown; no background supervision",
            "claim": (
                "Controlled arms differ in regression targets and their required inference mapping; "
                "not target-only versus historical TD80"
            ),
            "inference_information": (
                "value difference reads the later selected prefix; direct delta reads only the current prefix"
            ),
            "policy_short_answer_rule": (
                "historical <=100-token realized-reward override retained identically in both arms"
            ),
            "policy_schedule": "historical optimizer horizon 100; loop stops at configured policy_steps",
            "runtime_limits": (
                "GPU generation can be nondeterministic; "
                "later trajectories are consequences of different trained policies"
            ),
        },
    )
    save(work / "status.json", {"state": "prepared"})
    return work


def check_inputs(work, settings):
    require(sha(work / "pipeline.py") == settings["coordinator_sha256"], "Frozen coordinator changed")
    for name, digest in settings["source_files"].items():
        require(sha(work / "source" / name) == digest, f"Frozen source changed: {name}")
    for key, name in settings["data_names"].items():
        require(sha(work / "critic_inputs" / name) == settings["data_hashes"][key], f"Critic input changed: {name}")
    require(sha(work / "critic_config.yaml") == settings["critic_config_sha256"], "Critic config changed")
    for name, digest in settings["policy_files"].items():
        require(sha(work / name) == digest, f"Policy input changed: {name}")
    for name, digest in settings["policy_data_hashes"].items():
        require(sha(work / name) == digest, f"Policy data changed: {name}")


def stage(work, name, command, env):
    save(work / "status.json", {"state": "running", "stage": name, "command": command})
    print(f"Starting {name}; log: {work / (name + '.log')}", flush=True)
    with (work / (name + ".log")).open("x") as log:
        result = subprocess.run(command, cwd=work / "source", env=env, stdout=log, stderr=subprocess.STDOUT)
    (work / (name + ".exit_code")).write_text(str(result.returncode) + "\n")
    require(result.returncode == 0, f"{name} exited {result.returncode}; see {name}.log")


def run(work):
    settings = read(work / "settings.json")
    require(read(work / "status.json")["state"] == "prepared", "Run already started; create a new output directory")
    with ExitStack() as stack:
        for gpu in settings["gpus"]:
            lock = stack.enter_context(Path(f"/tmp/delta_old4_evaluation_gpu_{gpu}.lock").open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
        )
        memories = {int(line.split(",")[0]): int(line.split(",")[1]) for line in usage.splitlines()}
        require(all(memories.get(gpu, 999999) <= 512 for gpu in settings["gpus"]), "Requested GPUs are occupied")
        check_inputs(work, settings)
        env = environment(work, settings["gpus"])
        for arm in ("value", "direct_delta"):
            check_inputs(work, settings)
            command = [
                settings["python"],
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                "-m",
                "examples.delta_critic.train_boundary_scalar",
                "--target",
                arm,
                "--config",
                str(work / "critic_config.yaml"),
                "--output",
                str(work / f"critic_{arm}"),
                "--max-steps",
                str(settings["critic_steps"]),
                "--eval-every",
                str(settings["eval_every"]),
                "--global-batch",
                "128",
                "--microbatch",
                "16",
            ]
            for key, name in settings["data_names"].items():
                command.extend([f"--{key}", str(work / "critic_inputs" / name)])
            stage(work, f"critic_{arm}", command, env)
            require((work / f"critic_{arm}/best/COMPLETE").exists(), "Selected checkpoint is incomplete")
            complete = read(work / f"critic_{arm}/complete.json")
            require(complete["steps"] == settings["critic_steps"], "Critic training ended before configured horizon")
            if arm == "direct_delta":
                require(
                    read(work / "critic_value/controls.json") == read(work / "critic_direct_delta/controls.json"),
                    "Actual initial weights or controlled training configuration differ",
                )
                require(
                    read(work / "critic_value/complete.json")["batch_schedule_sha256"]
                    == complete["batch_schedule_sha256"],
                    "Optimizer batch order differs",
                )
                alignment = read(work / "alignment.json")
                alignment.update(
                    actual_initial_weights_equal=True,
                    actual_optimizer_batch_order_equal=True,
                    controlled_training_equal=True,
                )
                save(work / "alignment.json", alignment)
            root = work / f"policy_{arm}"
            policy_env = dict(
                env,
                PYTHONPATH=str(work / "source") + ":" + str(root),
                VERL_FILE_LOGGER_PATH=str(root / "td/metrics.jsonl"),
            )
            cfg = yaml.safe_load((root / "td/config.yaml").read_text())
            policy_env["RAY_TMPDIR"] = cfg["ray_kwargs"]["ray_init"]["_temp_dir"]
            stage(work, f"policy_{arm}", [settings["python"], str(root / "driver.py"), "--arm", "td"], policy_env)
            rows = [json.loads(line) for line in (root / "td/metrics.jsonl").read_text().splitlines() if line.strip()]
            require(
                set(range(1, settings["policy_steps"] + 1)).issubset({row["step"] for row in rows}),
                "Policy did not complete every requested update",
            )
        save(
            work / "status.json",
            {
                "state": "complete",
                "stages": ["critic_value", "policy_value", "critic_direct_delta", "policy_direct_delta"],
            },
        )
        (work / "exit_code").write_text("0\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "start", "run"])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--critic-steps", type=int, default=80)
    parser.add_argument("--policy-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--port-base", type=int, default=36520)
    args = parser.parse_args()
    if args.work_dir is None:
        args.work_dir = REPO / "outputs" / ("value_difference_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    args.work_dir = args.work_dir.resolve()
    require(args.critic_steps > 0 and 0 < args.policy_steps <= 10 and args.eval_every > 0, "Invalid step counts")
    if args.action == "run":
        frozen = args.work_dir / "pipeline.py"
        if frozen.exists() and Path(__file__).resolve() != frozen:
            os.execv(sys.executable, [sys.executable, str(frozen), "run", "--work-dir", str(args.work_dir)])
        try:
            run(args.work_dir)
        except Exception as exc:
            save(args.work_dir / "status.json", {"state": "failed", "error": repr(exc)})
            (args.work_dir / "exit_code").write_text("1\n")
            raise
        return
    work = prepare(args)
    if args.action == "start":
        frozen = work / "pipeline.py"
        with (work / "pipeline.log").open("x") as log:
            process = subprocess.Popen(
                [sys.executable, str(frozen), "run", "--work-dir", str(work)],
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        save(work / "launcher.json", {"pid": process.pid, "work_dir": str(work)})
        print(json.dumps({"pid": process.pid, "work_dir": str(work), "status": str(work / "status.json")}), flush=True)
    else:
        print(str(work), flush=True)


if __name__ == "__main__":
    main()
