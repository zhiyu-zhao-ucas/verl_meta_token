"""Adapt a fresh TD/Hybrid handoff work directory to the October 7 Hybrid warmup recipe.

Run after run_td_hybrid_handoff.sh prepare and before its hybrid stage. This
copies the two October 6 scorer fixes from the checked-in sealed archive,
applies the recorded optimizer-step warmup patch, and records all new hashes.
"""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from omegaconf import OmegaConf, open_dict


REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "reproductions/td_hybrid_20261006/assets.tgz"
BUNDLE = ARCHIVE.with_name("bundle.json")
PATCH = Path(__file__).with_name("warmup3_engine_workers.patch")
FILES = {
    "examples/delta_critic/sampling_io.py": (
        "93f9868ca315833c33d8fc75045629e2d0ebfd84d1df8f0ccb61ab674a9b6786",
        "b9ad51bec0d4ae396d0ba7d0c27af2f8823c0e15bf6823fd1dd900e05b6e52fd",
    ),
    "verl/trainer/ppo/v1/agent_loop_tq.py": (
        "e0c0d0fe3c9431f92097ebe0dc3d3a02d90ee1612e6d4346ee87d2b61c2cc990",
        "6f6d3a4c68454eb08f48219667a0b2bcf7ff639fad4559c5baf81655fe5028c0",
    ),
    "verl/workers/engine_workers.py": (
        "7d0d29dfdbf575c3b837c0fc54cfdd680a7c121fb11f241216794641b59ebc15",
        "ab0aff4dc39b3952740e681d00213cbbef7f250dd3eafdf19067ebc0821dedbf",
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def check_file(path, expected):
    actual = digest(path.read_bytes())
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")


def prepare(work):
    if not (work / "PREPARED").is_file() or not (work / "code_manifest.json").is_file():
        raise ValueError("Run handoff check and prepare first")
    if (work / "warmup_hybrid_preparation.json").exists() or (work / "logs/hybrid.log").exists():
        raise FileExistsError("Warmup was prepared or Hybrid already started; use a new WORK_DIR")
    source = work / "source_hybrid"
    config_path = work / "hybrid/config.yaml"
    manifest = read_json(work / "code_manifest.json")
    for name, (before, _) in FILES.items():
        path = source / name
        check_file(path, before)
        if manifest.get(f"source_hybrid/{name}") != before:
            raise ValueError(f"Prepared source manifest differs: {name}")
    if manifest.get("hybrid/config.yaml") != digest(config_path.read_bytes()):
        raise ValueError("Prepared Hybrid config differs from code_manifest.json")
    config = OmegaConf.load(config_path)
    optim = config.actor_rollout_ref.actor.optim
    if not (
        optim.lr == 1e-5
        and optim.lr_scheduler_type == "constant"
        and optim.lr_warmup_steps == -1
        and optim.lr_warmup_steps_ratio == 0.0
        and config.trainer.total_training_steps == 20
        and config.trainer.n_gpus_per_node == 2
        and config.algorithm.delta_policy.critic_update is None
        and config.trainer.resume_mode == "disable"
    ):
        raise ValueError("Prepared Hybrid is not the historical frozen-critic recipe")
    archive_expected = read_json(BUNDLE)["archive_sha256"]
    check_file(ARCHIVE, archive_expected)
    sealed = {}
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for name in list(FILES)[:2]:
            member = tar.getmember(f"source_hybrid/{name}")
            if not member.isfile():
                raise ValueError(f"Sealed source member is not a regular file: {name}")
            with tar.extractfile(member) as stream:
                data = stream.read()
            if digest(data) != FILES[name][1]:
                raise ValueError(f"Sealed source hash differs: {name}")
            sealed[name] = data
    # All preconditions are checked before changing the prepared directory.
    for name, data in sealed.items():
        (source / name).write_bytes(data)
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(source), "-i", str(PATCH)], check=True)
    for name, (_, after) in FILES.items():
        check_file(source / name, after)
        manifest[f"source_hybrid/{name}"] = after

    optim.lr_warmup_steps = 3
    config.trainer.project_name = "delta_policy_warmup3_reproduction"
    config.trainer.experiment_name = "hybrid_warmup3_optimizer_updates_seed42"
    trace = work / "hybrid/lr_trace"
    trace.mkdir()
    with open_dict(config.ray_kwargs.ray_init.runtime_env.env_vars):
        config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_WARMUP_LR_TRACE_DIR = str(trace)
    OmegaConf.save(config, config_path)
    manifest["hybrid/config.yaml"] = digest(config_path.read_bytes())
    save_json(work / "code_manifest.json", manifest)
    identity = read_json(work / "critic_identity.json")
    record = {
        "recipe": "2026-10-07 Hybrid actor warmup, three optimizer updates",
        "source_baseline": "2026-10-06 scorer-fixed Hybrid plus recorded worker patch",
        "archive_sha256": archive_expected,
        "source_sha256": {name: after for name, (_, after) in FILES.items()},
        "config_sha256": manifest["hybrid/config.yaml"],
        "critic_mode": identity["mode"],
        "critic_weights_sha256": identity["hybrid_sha256"],
        "learning_rates_used_first_five_updates": [0.0, 1e-5 / 3, 2e-5 / 3, 1e-5, 1e-5],
        "policy_steps": 10,
        "optimizer_updates": 20,
    }
    save_json(work / "warmup_hybrid_preparation.json", record)
    print(json.dumps(record, indent=2, ensure_ascii=False))


def verify(work):
    record = read_json(work / "warmup_hybrid_preparation.json")
    manifest = read_json(work / "code_manifest.json")
    for name, (_, after) in FILES.items():
        check_file(work / "source_hybrid" / name, after)
        if manifest.get(f"source_hybrid/{name}") != after:
            raise ValueError(f"Runtime manifest differs: {name}")
    config_path = work / "hybrid/config.yaml"
    check_file(config_path, record["config_sha256"])
    if manifest.get("hybrid/config.yaml") != record["config_sha256"]:
        raise ValueError("Runtime config manifest differs")
    trace = work / "hybrid/lr_trace"
    for rank in range(2):
        path = trace / f"rank_{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 20:
            raise ValueError(f"Rank {rank}: expected 20 optimizer updates, got {len(rows)}")
        for index, row in enumerate(rows):
            expected = 1e-5 * min(index / 3, 1.0)
            expected_after = 1e-5 * min((index + 1) / 3, 1.0)
            if not (
                row["optimizer_update"] == index + 1
                and row["scheduler_advanced"] is True
                and abs(row["lr_used"] - expected) < 1e-12
                and abs(row["lr_after_update"] - expected_after) < 1e-12
            ):
                raise ValueError(f"Rank {rank}, update {index + 1}: wrong LR trace")
    if not (work / "logs/hybrid.done").is_file():
        raise ValueError("Hybrid stage has no success marker")
    branch = work / "hybrid"
    rows = [json.loads(line) for line in (branch / "metrics.jsonl").read_text().splitlines() if line.strip()]
    metrics = {row["step"]: row["data"] for row in rows if row["step"] > 0}
    if set(metrics) != set(range(1, 11)):
        raise ValueError(f"Expected policy steps 1–10, got {sorted(metrics)}")
    for step, data in metrics.items():
        if not (
            data["delta_policy/original_rows"] == 128
            and data["delta_policy/continuation_rows"] == 192
            and data["delta_policy/actor_loss_continuation_rows"] == 192
            and data["delta_critic/update_step"] == 0
        ):
            raise ValueError(f"Policy step {step} has unexpected data layout or critic update")
    if not (branch / "checkpoints/delta_advantage_stats.json").is_file():
        raise ValueError("Frozen policy advantage statistics are missing")
    for step in (5, 10):
        if not (branch / f"checkpoints/global_step_{step}/actor/fsdp_config.json").is_file():
            raise ValueError(f"Complete actor checkpoint missing at policy step {step}")
    print("Hybrid warmup verified: 10 policy steps, 20 optimizer updates, two matching LR traces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else verify)(args.work_dir.resolve())


if __name__ == "__main__":
    main()
