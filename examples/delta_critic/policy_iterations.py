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
"""Sequential V1 collection, delta critic training and online policy rounds.

The reference actor is fixed across all rounds. Generation and training run in
separate processes so a finished sampling stage releases its model resources.
Use --dry-run to inspect every command and resolved config without writing files.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .collection import prompt_split


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def artifact_digest(path):
    """Hash exact stage outputs, including model weights, for safe stage resume."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"Empty stage artifact: {path}")
    result = {}
    for item in paths:
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(item.relative_to(path)) if path.is_dir() else path.name] = digest.hexdigest()
    return fingerprint(result)


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    inputs: tuple[Path, ...] = ()


def execute_stage(stage, directory, *, signature, resume, runner=subprocess.run):
    """Only skip a stage when both its inputs and outputs still match its marker."""
    marker = Path(directory) / f".{stage.name}.done.json"
    key = {
        "signature": signature,
        "command": stage.command,
        "inputs": {str(p): artifact_digest(p) for p in stage.inputs},
    }
    stage_key = fingerprint(key)
    if marker.exists():
        if not resume:
            raise FileExistsError(f"Completed stage exists; use --resume: {marker}")
        previous = json.loads(marker.read_text())
        current = {str(p): artifact_digest(p) for p in stage.outputs}
        if previous.get("key") != stage_key or previous.get("outputs") != current:
            raise ValueError(f"Stage {stage.name} configuration, inputs or outputs changed")
        return previous
    # Bind partial checkpoints to their original inputs before launching work.
    # A completion marker alone cannot protect an interrupted stage from being
    # resumed with changed data, model weights, or configuration.
    started = Path(directory) / f".{stage.name}.started.json"
    if started.exists():
        if not resume:
            raise FileExistsError(f"Started stage exists; use --resume: {started}")
        if json.loads(started.read_text()).get("key") != stage_key:
            raise ValueError(f"Stage {stage.name} configuration or inputs changed since it started")
    else:
        atomic_json(started, {"key": stage_key})
    command = list(stage.command)
    if resume and stage.name in {"critic", "policy"}:
        output = Path(command[command.index("--output") + 1])
        complete = sorted(p.parent for p in output.glob("step_*/COMPLETE"))
        if complete:
            # Weight initialization and full-state resume are mutually exclusive.
            if "--initialize" in command:
                i = command.index("--initialize")
                del command[i : i + 2]
            command.extend(["--resume", str(complete[-1])])
    runner(command, check=True)
    result = {"key": stage_key, "outputs": {str(p): artifact_digest(p) for p in stage.outputs}}
    atomic_json(marker, result)
    return result


def build_split(rollouts, output, *, eval_fraction, seed):
    rows = [json.loads(line) for line in Path(rollouts).read_text().splitlines() if line.strip()]
    split = {}
    for row in rows:
        row_id, prompt_id = str(row["id"]), str(row["prompt_id"])
        if row_id in split:
            raise ValueError(f"Duplicate rollout ID: {row_id}")
        split[row_id] = prompt_split(prompt_id, eval_fraction=eval_fraction, seed=str(seed))
    if set(split.values()) != {"train", "eval"}:
        raise ValueError("Both train and eval prompts are required; collect more prompts or change split seed")
    atomic_json(output, split)


def load_settings(path):
    settings = yaml.safe_load(Path(path).read_text())
    required = {"sampling_config", "critic_config", "policy_config", "rounds", "critic_steps"}
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"Missing iteration settings: {sorted(missing)}")
    settings = dict(settings)
    defaults = {
        "algorithm": "ours_segment_legacy",
        "nproc": 1,
        "critic_global_batch": 8,
        "policy_global_batch": 128,
        "microbatch": 1,
        "eval_fraction": 0.1,
        "seed": 42,
        "prompt_source": "value_model",
        "rollout_concurrency": 8,
        "mc_concurrency": 16,
        "gpu_memory_utilization": 0.9,
    }
    for key, value in defaults.items():
        settings.setdefault(key, value)
    if settings["algorithm"] not in {"ours_segment_legacy", "ours_local", "mc_local_oracle"}:
        raise ValueError("Unsupported alternating algorithm")
    for name in ("rounds", "critic_steps", "nproc", "critic_global_batch", "policy_global_batch", "microbatch"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < settings["eval_fraction"] < 1:
        raise ValueError("eval_fraction must be between zero and one")
    if settings["critic_global_batch"] % (settings["nproc"] * settings["microbatch"]):
        raise ValueError("critic_global_batch must divide across nproc * microbatch")
    if settings["prompt_source"] not in {"value_model", "verl_v1"}:
        raise ValueError("Unsupported prompt_source")
    for name in ("sampling_config", "critic_config", "policy_config"):
        settings[name] = str(Path(settings[name]).resolve())
    return settings


def plan_rounds(settings, output):
    """Pure plan: commands and configs for every round, without model loading."""
    output = Path(output).resolve()
    sampling = yaml.safe_load(Path(settings["sampling_config"]).read_text())
    critic = yaml.safe_load(Path(settings["critic_config"]).read_text())
    policy = yaml.safe_load(Path(settings["policy_config"]).read_text())
    actor = settings.get("initial_actor") or sampling["model"]["actor_model"]
    reference = settings.get("reference_model") or actor
    previous_critic = settings.get("initial_critic")
    algorithm = settings["algorithm"]
    label_mode = "selected_segment" if algorithm == "ours_segment_legacy" else "paired_next_state"
    if critic.get("objective", "local_td0") == "hybrid_terminal_composition":
        sampling.setdefault("mc", {})["save_continuations"] = True
        sampling["mc"]["save_individual_rewards"] = True
    plans = []
    torchrun = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={settings['nproc']}",
        "--module",
    )
    for index in range(settings["rounds"]):
        directory = output / f"round_{index:03d}"
        sample_config, critic_config, policy_config = copy.deepcopy((sampling, critic, policy))
        sample_config["model"]["actor_model"] = actor
        # Critic's backbone identity is stable; initialize weights from preceding
        # critic rather than interpreting an exported actor as a scalar artifact.
        critic_config.update(delta_label_mode=label_mode)
        policy_config.update(
            model_path=actor,
            reference_model_path=reference,
            mode="online_ppo",
            label_mode=label_mode,
            advantage_source="mc_local_oracle" if algorithm == "mc_local_oracle" else "critic",
            kl_reference="reference",
            kl_estimator="low_var_kl",
            kl_mask_scope="policy",
        )
        if policy_config.get("window", "error") != "error":
            raise ValueError("Online rounds require a full-context policy configuration")
        policy_config.setdefault("kl_coef", 0.001)
        policy_config.setdefault("rl_epochs", 1)
        configs = {"sampling": sample_config, "critic": critic_config, "policy": policy_config}
        config_paths = {key: directory / f"{key}.yaml" for key in configs}
        data = directory / "data"
        rollouts = data / "rollouts_train_regen.jsonl"
        labels = data / "mc_labels_train.jsonl"
        continuations = data / "continuations_train.jsonl"
        split = directory / "split.json"
        collect = [
            sys.executable,
            "-m",
            "examples.delta_critic.batch_sample",
            "--config",
            str(config_paths["sampling"]),
            "--output-dir",
            str(data),
            "--model-path",
            str(actor),
            "--actor-version",
            f"round_{index:03d}:{actor}",
            "--split",
            "train",
            "--prompt-source",
            settings["prompt_source"],
            "--mc-mode",
            "prefix_only" if label_mode == "selected_segment" else "paired_next_state",
            "--rollout-concurrency",
            str(settings["rollout_concurrency"]),
            "--mc-concurrency",
            str(settings["mc_concurrency"]),
            "--gpu-memory-utilization",
            str(settings["gpu_memory_utilization"]),
        ]
        for key, flag in (("prompts_jsonl", "--prompts-jsonl"), ("prompt_limit", "--limit")):
            if settings.get(key) is not None:
                collect.extend([flag, str(settings[key])])
        for override in settings.get("verl_overrides", []):
            collect.extend(["--verl-override", override])
        collect_outputs = (rollouts, labels, data / "states_train.jsonl", continuations)
        collect_inputs = [config_paths["sampling"]]
        if Path(actor).exists() or index > 0:
            collect_inputs.append(Path(actor))
        if settings.get("prompts_jsonl"):
            collect_inputs.append(Path(settings["prompts_jsonl"]))
        stages = [Stage("collect", tuple(collect), collect_outputs, tuple(collect_inputs))]
        stages.append(
            Stage(
                "split",
                (
                    sys.executable,
                    "-m",
                    "examples.delta_critic.policy_iterations",
                    "--make-split",
                    str(rollouts),
                    "--output",
                    str(split),
                    "--eval-fraction",
                    str(settings["eval_fraction"]),
                    "--seed",
                    str(settings["seed"]),
                ),
                (split,),
                (rollouts,),
            )
        )
        critic_output = directory / "critic"
        artifact = critic_output / f"step_{settings['critic_steps']:08d}"
        if algorithm != "mc_local_oracle":
            command = [
                *torchrun,
                "examples.delta_critic.train",
                "--config",
                str(config_paths["critic"]),
                "--rollouts",
                str(rollouts),
                "--labels",
                str(labels),
                "--split",
                str(split),
                "--output",
                str(critic_output),
                "--global-batch",
                str(settings["critic_global_batch"]),
                "--max-steps",
                str(settings["critic_steps"]),
                "--microbatch",
                str(settings["microbatch"]),
            ]
            inputs = [rollouts, labels, split, config_paths["critic"]]
            if critic_config.get("objective") == "hybrid_terminal_composition":
                command.extend(["--continuations", str(continuations)])
                inputs.append(continuations)
            if previous_critic:
                command.extend(["--initialize", str(previous_critic)])
                inputs.append(Path(previous_critic))
            stages.append(Stage("critic", tuple(command), (artifact,), tuple(inputs)))
        policy_output = directory / "policy"
        command = [
            *torchrun,
            "examples.delta_critic.policy_train",
            "--config",
            str(config_paths["policy"]),
            "--rollouts",
            str(rollouts),
            "--labels",
            str(labels),
            "--split",
            str(split),
            "--output",
            str(policy_output),
            "--global-batch",
            str(settings["policy_global_batch"]),
            "--microbatch",
            str(settings["microbatch"]),
            "--mode",
            "online",
        ]
        if settings.get("policy_steps") is not None:
            command.extend(["--max-steps", str(settings["policy_steps"])])
        inputs = [rollouts, labels, split, config_paths["policy"]]
        if Path(reference).exists():
            inputs.append(Path(reference))
        if Path(actor).exists() or index > 0:
            inputs.append(Path(actor))
        if algorithm != "mc_local_oracle":
            command.extend(["--critic", str(artifact)])
            inputs.append(artifact)
        actor_output = policy_output / "final_hf"
        stages.append(Stage("policy", tuple(command), (actor_output,), tuple(inputs)))
        plans.append(
            {
                "round": index,
                "directory": directory,
                "configs": configs,
                "stages": stages,
                "actor_input": actor,
                "actor_output": str(actor_output),
                "reference_model": reference,
                "critic_input": previous_critic,
                "critic_output": str(artifact) if algorithm != "mc_local_oracle" else None,
            }
        )
        actor = str(actor_output)
        if algorithm != "mc_local_oracle":
            previous_critic = str(artifact)
    return plans


def run(settings, output, *, resume=False, dry_run=False, runner=subprocess.run):
    plans = plan_rounds(settings, output)
    signature = fingerprint({"settings": settings, "configs": plans[0]["configs"]})
    if dry_run:
        return [
            {
                "round": p["round"],
                "reference_model": p["reference_model"],
                "configs": p["configs"],
                "commands": {s.name: list(s.command) for s in p["stages"]},
            }
            for p in plans
        ]
    output = Path(output)
    marker = output / "run.json"
    if marker.exists():
        if not resume:
            raise FileExistsError("Run exists; select a new output directory or use --resume")
        if json.loads(marker.read_text())["signature"] != signature:
            raise ValueError("Run settings changed; resume requires identical configs and fixed reference")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Nonempty run directory has no valid run manifest")
        atomic_json(marker, {"signature": signature, "settings": settings})
    for plan in plans:
        directory = plan["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        for name, config in plan["configs"].items():
            path = directory / f"{name}.yaml"
            if path.exists():
                if yaml.safe_load(path.read_text()) != config:
                    raise ValueError(f"Generated config was modified: {path}")
            else:
                path.write_text(yaml.safe_dump(config, sort_keys=False))
        for stage in plan["stages"]:
            print(json.dumps({"round": plan["round"], "stage": stage.name, "command": stage.command}), flush=True)
            execute_stage(stage, directory, signature=signature, resume=resume, runner=runner)
        atomic_json(
            directory / "manifest.json",
            {
                key: plan[key]
                for key in ("round", "actor_input", "actor_output", "reference_model", "critic_input", "critic_output")
            },
        )
    atomic_json(
        output / "COMPLETE.json", {"signature": signature, "rounds": len(plans), "actor": plans[-1]["actor_output"]}
    )
    return {"rounds": len(plans), "actor": plans[-1]["actor_output"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--make-split", help=argparse.SUPPRESS)
    parser.add_argument("--eval-fraction", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--seed", default="42", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.make_split:
        build_split(args.make_split, args.output, eval_fraction=args.eval_fraction, seed=args.seed)
        return
    if not args.config:
        parser.error("--config is required")
    result = run(load_settings(args.config), args.output, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
