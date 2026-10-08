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

"""CPU checks for the GitHub handoff package; no private outputs or GPU required."""

import ast
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/delta_critic"
ARCHIVE = EXAMPLE / "handoff_assets.tgz"


def test_bash_syntax_and_help_without_runtime_environment():
    script = EXAMPLE / "run_td_hybrid_handoff.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(
        ["bash", str(script), "--help"],
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "fresh-all" in result.stdout
    assert "prepare-bootstrap" in result.stdout
    assert "eval-online" in result.stdout


def test_committed_package_checksum_and_all_member_identities():
    expected = json.loads((EXAMPLE / "handoff_assets.sha256.json").read_text())
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == expected["sha256"]
    assert ARCHIVE.stat().st_size == expected["bytes"]
    with tarfile.open(ARCHIVE) as archive:
        manifest = json.load(archive.extractfile("assets/manifest.json"))
        names = archive.getnames()
        assert set(names) == {"assets/manifest.json", *("assets/" + name for name in manifest)}
        for name, digest in manifest.items():
            content = archive.extractfile("assets/" + name).read()
            assert hashlib.sha256(content).hexdigest() == digest, name


def test_package_contains_the_entire_workflow_and_fixed_inputs():
    with tarfile.open(ARCHIVE) as archive:
        names = set(archive.getnames())
        for source in ["source_td", "source_hybrid", "source_online", "source_initial_hybrid"]:
            assert f"assets/{source}/examples/delta_critic/train.py" in names
            assert f"assets/{source}/verl/trainer/ppo/v1/trainer_base.py" in names
        for name in [
            "inputs/train.parquet",
            "inputs/validation.parquet",
            "bootstrap/prompts.jsonl",
            "driver.py",
            "online_collect.py",
            "online_policy_driver.py",
            "online_runner.py",
            "evaluation_tools/prepare.py",
            "evaluation_tools/generate.py",
            "evaluation_tools/audit.py",
            "environment.json",
            "requirements-frozen.txt",
        ]:
            assert "assets/" + name in names
        assert not any(name.endswith("model.pt") for name in names)
        prompts = [json.loads(line) for line in archive.extractfile("assets/bootstrap/prompts.jsonl")]
        assert len(prompts) == len({row["prompt_id"] for row in prompts}) == 512
        assert all(0 < len(row["prompt_token_ids"]) <= 2048 for row in prompts)
        for name in ["driver.py", "online_collect.py", "online_policy_driver.py", "online_runner.py"]:
            ast.parse(archive.extractfile("assets/" + name).read(), filename=name)


def test_archive_is_not_ignored_by_git():
    result = subprocess.run(["git", "check-ignore", "--no-index", str(ARCHIVE)], cwd=ROOT, capture_output=True)
    assert result.returncode == 1, result.stdout.decode()
