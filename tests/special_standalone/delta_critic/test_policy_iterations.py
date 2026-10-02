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
"""Round manifests must freeze reference identity and detect stale artifacts."""

import json
from pathlib import Path

import pytest
import yaml

from examples.delta_critic.policy_iterations import Stage, build_split, execute_stage, load_settings, plan_rounds, run


def settings(tmp_path):
    configs = {
        "sampling": {"model": {"actor_model": "initial-actor"}, "mc": {}},
        "critic": {"model_path": "initial-critic", "objective": "local_td0"},
        "policy": {"model_path": "initial-actor"},
    }
    config = {"rounds": 2, "critic_steps": 2}
    for name, value in configs.items():
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(value))
        config[f"{name}_config"] = str(path)
    path = tmp_path / "iterations.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_settings(path)


def test_rounds_refresh_actor_and_critic_but_never_reference(tmp_path):
    config = settings(tmp_path)
    plans = plan_rounds(config, tmp_path / "run")
    assert plans[1]["actor_input"] == plans[0]["actor_output"]
    assert plans[1]["critic_input"] == plans[0]["critic_output"]
    assert plans[0]["reference_model"] == plans[1]["reference_model"] == "initial-actor"
    assert plans[1]["configs"]["policy"]["reference_model_path"] == "initial-actor"
    assert "--initialize" in next(s for s in plans[1]["stages"] if s.name == "critic").command


def test_dry_run_does_not_create_outputs(tmp_path):
    output = tmp_path / "run"
    result = run(settings(tmp_path), output, dry_run=True)
    assert len(result) == 2
    assert not output.exists()


def test_stage_resume_checks_input_and_output_bytes(tmp_path):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.write_text("data")
    stage = Stage("collect", ("fake",), (destination,), (source,))
    calls = []

    def runner(command, check):
        calls.append(command)
        destination.write_text("generated")

    execute_stage(stage, tmp_path, signature="run", resume=False, runner=runner)
    execute_stage(stage, tmp_path, signature="run", resume=True, runner=runner)
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        execute_stage(stage, tmp_path, signature="run", resume=False, runner=runner)
    destination.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        execute_stage(stage, tmp_path, signature="run", resume=True, runner=runner)
    destination.write_text("generated")
    source.write_text("other data")
    with pytest.raises(ValueError, match="changed"):
        execute_stage(stage, tmp_path, signature="run", resume=True, runner=runner)


def test_failed_stage_does_not_mark_completion(tmp_path):
    stage = Stage("collect", ("fake",), (tmp_path / "output",))

    def fail(command, check):
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        execute_stage(stage, tmp_path, signature="run", resume=False, runner=fail)
    assert not (tmp_path / ".collect.done.json").exists()


def test_partial_training_resume_removes_initialization(tmp_path):
    output = tmp_path / "critic"
    completed = output / "step_00000001"
    completed.mkdir(parents=True)
    (completed / "COMPLETE").write_text("done")
    stage = Stage("critic", ("fake", "--output", str(output), "--initialize", "previous"), (output,))
    commands = []

    def runner(command, check):
        commands.append(command)

    execute_stage(stage, tmp_path, signature="run", resume=True, runner=runner)
    assert "--initialize" not in commands[0]
    assert commands[0][-2:] == ["--resume", str(completed)]


def test_prompt_split_stays_stable_across_responses_and_rounds(tmp_path):
    rows = [{"id": f"r{i}-{j}", "prompt_id": f"p{i}"} for i in range(30) for j in range(2)]
    data = tmp_path / "rollouts.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "split.json"
    build_split(data, output, eval_fraction=0.3, seed="42")
    split = json.loads(output.read_text())
    assert set(split.values()) == {"train", "eval"}
    for i in range(30):
        assert split[f"r{i}-0"] == split[f"r{i}-1"]


def test_oracle_skips_critic_and_selects_paired_mc(tmp_path):
    config = settings(tmp_path)
    config["algorithm"] = "mc_local_oracle"
    plan = plan_rounds(config, tmp_path / "run")[0]
    assert "critic" not in [stage.name for stage in plan["stages"]]
    assert "paired_next_state" in plan["stages"][0].command
    assert plan["critic_output"] is None


def test_two_round_pipeline_resume_and_fixed_reference(tmp_path):
    config = settings(tmp_path)
    output = tmp_path / "run"
    commands = []

    def runner(command, check):
        commands.append(command)
        if "examples.delta_critic.batch_sample" in command:
            directory = Path(command[command.index("--output-dir") + 1])
            directory.mkdir(parents=True)
            rows = [{"id": f"r{i}", "prompt_id": f"p{i}"} for i in range(60)]
            (directory / "rollouts_train_regen.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            for name in ("states_train.jsonl", "mc_labels_train.jsonl", "continuations_train.jsonl"):
                (directory / name).write_text("{}\n")
        elif "--make-split" in command:
            build_split(
                command[command.index("--make-split") + 1],
                command[command.index("--output") + 1],
                eval_fraction=0.1,
                seed="42",
            )
        else:
            directory = Path(command[command.index("--output") + 1])
            if "examples.delta_critic.train" in command:
                directory /= "step_00000002"
                directory.mkdir(parents=True)
                (directory / "COMPLETE").write_text("done")
            else:
                directory /= "final_hf"
                directory.mkdir(parents=True)
                (directory / "weights").write_text("updated actor")

    result = run(config, output, runner=runner)
    assert result["rounds"] == 2
    assert len(commands) == 8
    run(config, output, resume=True, runner=runner)
    assert len(commands) == 8
    manifests = [json.loads((output / f"round_{i:03d}" / "manifest.json").read_text()) for i in range(2)]
    assert manifests[1]["actor_input"] == manifests[0]["actor_output"]
    assert manifests[0]["reference_model"] == manifests[1]["reference_model"]
    config["reference_model"] = "other-reference"
    with pytest.raises(ValueError, match="fixed reference"):
        run(config, output, resume=True, runner=runner)


def test_interrupted_stage_rejects_changed_inputs(tmp_path):
    source = tmp_path / "input"
    source.write_text("original")
    stage = Stage("critic", ("fake", "--output", str(tmp_path / "model")), (), (source,))

    def fail(command, check):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        execute_stage(stage, tmp_path, signature="run", resume=False, runner=fail)
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed since it started"):
        execute_stage(stage, tmp_path, signature="run", resume=True, runner=fail)
