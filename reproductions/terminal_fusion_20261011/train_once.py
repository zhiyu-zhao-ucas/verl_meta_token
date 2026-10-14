"""Train one fresh MC GRPO repeat using its verified frozen source and config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKPOINT_STEPS = [3, 5, 6, 7, 10]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=('C_run_1', 'D_run_1', 'C_run_2', 'D_run_2'))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    branch = ROOT / args.run
    manifest = json.loads((ROOT / "manifest.json").read_text())
    source = Path(manifest.get("source_by_run", {}).get(args.run, str(ROOT / "source")))
    source_manifest = Path(manifest.get("source_manifests_by_run", {}).get(args.run, str(ROOT / "source_manifest.json")))
    if any((branch / name).exists() for name in ("started.json", "metrics.jsonl", "checkpoints", "finished.json")):
        raise RuntimeError(f"Refusing to overwrite existing training: {branch}")
    for relative, expected in json.loads(source_manifest.read_text()).items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source mismatch: {relative}")
    config_bytes = (branch / "config.yaml").read_bytes()
    config = json.loads(config_bytes)
    policy = config["algorithm"]["delta_policy"]
    assert policy["advantage_stats_mode"] == "raw_scale_50" and policy["advantage_center"] is False
    assert policy["terminal_fusion"] == "full_trajectory" and policy["terminal_advantage_mode"] == "mc_grpo"
    assert policy["terminal_include_delta"] == args.run.startswith("D_")
    assert policy["expansion"]["sampling_mode"] == "independent_full_response"
    assert policy["terminal_mc_target_rms_ratio"] == 0.8 and policy["terminal_mc_max_abs_rms_ratio"] == 1.0
    assert policy["expansion"].get("short_response_tokens") is None
    assert policy["rollout_mode"] == "selected_prefix_mc" and policy["mc"]["continuations_per_state"] == 8
    assert config["trainer"]["total_training_steps"] == 10
    assert config["trainer"]["save_steps"] == CHECKPOINT_STEPS and config["trainer"]["save_freq"] == -1
    assert config["trainer"]["max_actor_ckpt_to_keep"] == 5
    assert config["trainer"]["resume_mode"] == "disable"
    assert config["trainer"]["n_gpus_per_node"] == config["ray_kwargs"]["ray_init"]["num_gpus"] == 2
    assert config["actor_rollout_ref"]["rollout"]["data_parallel_size"] == 2
    physical_gpus = manifest.get("gpu_plan_by_run", {}).get(args.run, manifest["gpu_plan"])
    assert len(physical_gpus) == 2 and len(set(physical_gpus)) == 2
    assert set(physical_gpus) <= {0, 1, 5, 6, 7}
    ray_init = config["ray_kwargs"]["ray_init"]
    env_vars = ray_init["runtime_env"]["env_vars"]
    for name in ("TMPDIR", "VLLM_RPC_BASE_PATH"):
        if env_vars.get(name):
            Path(env_vars[name]).mkdir(parents=True, exist_ok=True)
    os.environ.update({name: env_vars[name] for name in ("TMPDIR", "VLLM_RPC_BASE_PATH", "VLLM_DP_MASTER_PORT") if name in env_vars})
    os.environ.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, physical_gpus)),
        RAY_TMPDIR=ray_init["_temp_dir"],
        OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", VLLM_WORKER_MULTIPROC_METHOD="spawn",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline",
        RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(source),
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", VERL_FILE_LOGGER_PATH=str(branch / "metrics.jsonl"),
        VERL_WARMUP_LR_TRACE_DIR=env_vars["VERL_WARMUP_LR_TRACE_DIR"],
    )
    os.environ.pop("RAY_ADDRESS", None)
    os.chdir(source)
    sys.path.insert(0, str(source))
    import ray
    from omegaconf import OmegaConf
    import examples.delta_critic.policy_config as policy_config
    import examples.delta_critic.policy_data as policy_data
    import examples.delta_critic.policy_online as policy_online
    import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq
    import verl.trainer.ppo.v1.trainer_base as trainer_base
    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
    from verl.trainer.ppo.utils import need_reference_policy
    from verl.utils.config import validate_config

    resolved = OmegaConf.create(config)
    policy_online.validate_online_v1_settings(resolved)
    validate_config(config=resolved, use_reference_policy=need_reference_policy(resolved), use_critic=False)
    for module in (policy_config, policy_data, policy_online, agent_loop_tq, trainer_base):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"Module escaped frozen source: {module.__file__}")
    if args.preflight:
        print(json.dumps({"run": args.run, "preflight": "passed", "source": str(source),
                          "physical_gpus": physical_gpus, "checkpoint_steps": CHECKPOINT_STEPS}))
        return
    start = time.time()
    write_json(branch / "started.json", {
        "pid": os.getpid(), "run": args.run, "source": str(source), "physical_gpus": physical_gpus,
        "started_at": start, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "initial_model": config["actor_rollout_ref"]["model"]["path"],
        "advantage_stats_mode": "raw_scale_50", "advantage_center": False,
        "terminal_advantage_mode": "mc_grpo", "checkpoint_steps": CHECKPOINT_STEPS,
        "sampling_mode": "independent_full_response",
        "terminal_include_delta": policy["terminal_include_delta"],
    })
    error = error_traceback = None
    try:
        run_ppo(resolved, TaskRunnerV1)
    except BaseException as exc:
        error, error_traceback = f"{type(exc).__name__}: {exc}", traceback.format_exc()
        raise
    finally:
        try:
            ray.shutdown()
        finally:
            write_json(branch / "finished.json", {
                "run": args.run, "elapsed_seconds": time.time() - start,
                "error": error, "traceback": error_traceback,
            })


if __name__ == "__main__":
    main()
