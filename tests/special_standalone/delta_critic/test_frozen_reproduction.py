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
"""CPU recovery and continuation checks for the committed reproduction bundle."""

import argparse
import ast
import hashlib
import importlib.util
import io
import shutil
import tarfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ENTRY = Path(__file__).resolve().parents[3] / "reproductions/td_hybrid_20261006/run.py"
SPEC = importlib.util.spec_from_file_location("frozen_reproduction_entry", ENTRY)
assert SPEC is not None and SPEC.loader is not None
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def restore_args(path):
    """Build an isolated new-run request."""
    return argparse.Namespace(work_dir=path, port_base=35400)


def tiny_package(path, members):
    """Construct archive fixtures with deliberate integrity errors."""
    path.mkdir()
    with tarfile.open(path / "assets.tgz", "w:gz") as stream:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    entry.save(
        path / "bundle.json",
        {
            "archive": "assets.tgz",
            "archive_sha256": entry.sha(path / "assets.tgz"),
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()},
        },
    )


def test_restore_is_independent_of_mutable_workspace(tmp_path):
    """Recover complete sources and relocate only runtime paths and names."""
    work = entry.restore(restore_args(tmp_path / "recovered"))
    manifest = entry.verify_bundle()
    for name, digest in manifest["files"].items():
        if name.startswith(("source_td/", "source_hybrid/", "runtime/")):
            assert entry.sha(work / name) == digest
    for arm in ["td", "hybrid"]:
        cfg = OmegaConf.load(work / arm / "config.yaml")
        original = OmegaConf.load(work / "snapshots" / arm / "config.yaml")
        assert cfg.algorithm.delta_policy == original.algorithm.delta_policy
        assert cfg.actor_rollout_ref.actor == original.actor_rollout_ref.actor
        assert cfg.trainer.total_training_steps == original.trainer.total_training_steps
        assert cfg.data.train_files == str(work / "inputs/train.parquet")
    with pytest.raises(ValueError, match="overwrite"):
        entry.restore(restore_args(work))


def test_changed_frozen_source_blocks_launch_preflight(tmp_path):
    """A later edit cannot silently enter this run's loss implementation."""
    work = entry.restore(restore_args(tmp_path / "run"))
    source = work / "source_td/verl/workers/engine_workers.py"
    source.chmod(0o644)
    source.write_text(source.read_text() + "\n# later edit\n")
    with pytest.raises(ValueError, match="Frozen restored artifact changed"):
        entry.check(work)


def test_archive_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    """Changing the archive is detected before extraction."""
    package = tmp_path / "package"
    tiny_package(package, {"source_td/a.py": b"original"})
    with (package / "assets.tgz").open("ab") as stream:
        stream.write(b"later edit")
    monkeypatch.setattr(entry, "PACKAGE", package)
    with pytest.raises(ValueError, match="Archive SHA256"):
        entry.verify_bundle()


def test_member_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    """A valid archive checksum alone cannot override member-level identity."""
    package = tmp_path / "package"
    tiny_package(package, {"source_td/a.py": b"original"})
    manifest = entry.read(package / "bundle.json")
    manifest["files"]["source_td/a.py"] = "0" * 64
    entry.save(package / "bundle.json", manifest)
    monkeypatch.setattr(entry, "PACKAGE", package)
    with pytest.raises(ValueError, match="Member hash mismatch"):
        entry.verify_bundle()


def test_archive_path_traversal_is_rejected(tmp_path, monkeypatch):
    """Even a listed and correctly hashed member cannot escape the new run."""
    package = tmp_path / "package"
    tiny_package(package, {"../escape": b"payload"})
    monkeypatch.setattr(entry, "PACKAGE", package)
    with pytest.raises(ValueError, match="Unsafe archive path"):
        entry.verify_bundle()


@pytest.mark.parametrize("arm,world", [("td", 4), ("hybrid", 2)])
def test_resume_preserves_sources_and_fixed_statistics(tmp_path, arm, world):
    """Only the continuation harness/config changes, with complete state required."""
    source = entry.restore(restore_args(tmp_path / "source"))
    checkpoint = source / arm / "checkpoints/global_step_10"
    (checkpoint / "actor").mkdir(parents=True)
    entry.save(checkpoint / "actor/fsdp_config.json", {"world_size": world})
    (checkpoint / "data.pt").write_bytes(b"dataloader fixture")
    for rank in range(world):
        for kind in ["model", "optim", "extra_state"]:
            (checkpoint / "actor" / f"{kind}_world_size_{world}_rank_{rank}.pt").write_bytes(b"checkpoint fixture")
    stats = source / arm / "checkpoints/delta_advantage_stats.json"
    shutil.copyfile(source / "evidence" / arm / "delta_advantage_stats.json", stats)
    args = restore_args(tmp_path / "continued")
    args.from_run, args.arm, args.step, args.stop_step = source, arm, 10, 20
    continued = entry.prepare_resume(args)
    cfg = OmegaConf.load(continued / arm / "config.yaml")
    assert cfg.trainer.resume_mode == "resume_path"
    assert cfg.trainer.resume_from_path == str(checkpoint)
    assert cfg.trainer.del_local_ckpt_after_load is False
    assert (continued / arm / "checkpoints/delta_advantage_stats.json").read_bytes() == stats.read_bytes()
    assert (continued / "driver.py").read_bytes() == (source / "driver.py").read_bytes()
    driver = (continued / "continue_driver.py").read_text()
    ast.parse(driver)
    assert "assert trainer.global_steps == 10" in driver
    assert "trainer.total_training_steps = 20" in driver
    assert "ordered_prompt_tokens_equal=True if known else None" in driver
    entry.check(continued)
    missing = checkpoint / "actor" / f"optim_world_size_{world}_rank_0.pt"
    missing.unlink()
    with pytest.raises(ValueError, match="Incomplete continuation checkpoint"):
        entry.validate_checkpoint(source, arm, 10)
