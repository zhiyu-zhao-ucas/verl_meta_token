# Copyright 2026 Individual Contributor: zhiyu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate and merge completed MC-relabel shards in the original row order."""

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from .legacy_adapter import adapt_legacy_rows
from .mc_diagnostics import impossible_near_end_labels, mc_endpoint_calibration
from .sampling_io import read_jsonl

SHARD_FILES = {
    "rollouts": "rollouts_relabel.jsonl",
    "states": "states_relabel.jsonl",
    "mc_labels": "mc_labels_relabel.jsonl",
    "continuations": "continuations_relabel.jsonl",
}
OUTPUT_FILES = {
    "rollouts": "rollouts_train_regen.jsonl",
    "states": "states_train.jsonl",
    "mc_labels": "mc_labels_train.jsonl",
    "continuations": "continuations_train.jsonl",
}
SOURCE_FILES = (
    "examples/delta_critic/legacy_adapter.py",
    "examples/delta_critic/mc_diagnostics.py",
    "examples/delta_critic/merge_relabel_shards.py",
)


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_row(stream, row) -> None:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _unique_by(rows, key, description):
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {description}={value}")
        result[value] = row
    return result


def _check_manifest(shard_dir: Path) -> dict:
    manifest_path = shard_dir / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed shard manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("scoring", {}).get("unexpected_errors", -1)) != 0:
        raise ValueError(f"Shard reports scorer errors: {shard_dir}")
    error_path = shard_dir / "scoring_errors.jsonl"
    if error_path.exists() and error_path.stat().st_size:
        raise ValueError(f"Shard has a nonempty scorer error journal: {error_path}")
    outputs = manifest.get("outputs", {})
    for name, filename in SHARD_FILES.items():
        path = shard_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Completed shard output is missing: {path}")
        expected = outputs.get(name)
        if expected is None or _file_hash(path) != expected:
            raise ValueError(f"Shard output hash mismatch for {path}")
    return manifest


def _validate_continuation(row, *, state_id: str, continuation_index: int) -> float | None:
    if str(row.get("state_id")) != state_id or int(row.get("continuation_index", -1)) != continuation_index:
        raise ValueError(f"Continuation identity mismatch for state_id={state_id}")
    token_ids = row.get("continuation_token_ids")
    sampled_logprobs = row.get("sampled_logprobs")
    if not isinstance(token_ids, list) or not isinstance(row.get("continuation_text"), str):
        raise ValueError(f"Continuation payload is incomplete for state_id={state_id}")
    if not isinstance(sampled_logprobs, list) or len(sampled_logprobs) != len(token_ids):
        raise ValueError(f"Sampled log-probabilities are not token-aligned for state_id={state_id}")
    if any(
        isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value > 0
        for value in sampled_logprobs
    ):
        raise ValueError(f"Invalid sampled log-probability for state_id={state_id}")
    status = row.get("score_status")
    reward = row.get("reward")
    if status == "timeout":
        if reward is not None:
            raise ValueError(f"Timeout carries a reward for state_id={state_id}")
        return None
    if status not in {"correct", "incorrect", "unparseable"}:
        raise ValueError(f"Unsupported score status={status!r} for state_id={state_id}")
    if isinstance(reward, bool) or not isinstance(reward, int | float) or float(reward) not in {0.0, 1.0}:
        raise ValueError(f"Invalid binary reward for state_id={state_id}")
    if (status == "correct") != (float(reward) == 1.0):
        raise ValueError(f"Score status/reward disagreement for state_id={state_id}")
    return float(reward)


def _index_continuations(shard_dirs, expected_state_ids, continuations_per_state):
    locations = defaultdict(dict)
    total = 0
    for shard_dir in shard_dirs:
        path = shard_dir / SHARD_FILES["continuations"]
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                state_id = str(row.get("state_id"))
                if state_id not in expected_state_ids:
                    raise ValueError(f"Unknown continuation state_id={state_id}")
                index = int(row.get("continuation_index", -1))
                if not 0 <= index < continuations_per_state:
                    raise ValueError(f"Invalid continuation_index={index} for state_id={state_id}")
                if index in locations[state_id]:
                    raise ValueError(f"Duplicate continuation state_id={state_id}, index={index}")
                locations[state_id][index] = (path, offset)
                total += 1
    expected_indices = set(range(continuations_per_state))
    bad = [state_id for state_id in expected_state_ids if set(locations[state_id]) != expected_indices]
    if bad:
        raise ValueError(f"Incomplete continuation groups for {len(bad)} states; first={bad[0]}")
    return locations, total


def merge_shards(
    *,
    source_rollouts: Path,
    source_states: Path,
    source_split: Path,
    shard_dirs: list[Path],
    output_dir: Path,
    continuations_per_state: int = 16,
    expected_rollouts: int | None = 486,
    expected_states: int | None = 8593,
    max_timeout_fraction: float = 0.0001,
    max_timeouts_per_state: int = 1,
) -> dict:
    """Merge only complete, hashed shards and enforce population-level gates."""
    if continuations_per_state < 1:
        raise ValueError("continuations_per_state must be positive")
    if not 0 <= max_timeout_fraction <= 1 or max_timeouts_per_state < 0:
        raise ValueError("Timeout budgets must be nonnegative")
    if not shard_dirs:
        raise ValueError("At least one shard is required")
    source_rollout_rows = read_jsonl(source_rollouts)
    source_state_rows = read_jsonl(source_states)
    source_rollout_map = _unique_by(source_rollout_rows, "id", "source rollout_id")
    source_state_map = _unique_by(source_state_rows, "state_id", "source state_id")
    rollout_order = [str(row["id"]) for row in source_rollout_rows]
    state_order = [str(row["state_id"]) for row in source_state_rows]
    if expected_rollouts is not None and len(rollout_order) != expected_rollouts:
        raise ValueError(f"Expected {expected_rollouts} rollouts, found {len(rollout_order)}")
    if expected_states is not None and len(state_order) != expected_states:
        raise ValueError(f"Expected {expected_states} states, found {len(state_order)}")

    manifests = [_check_manifest(path) for path in shard_dirs]
    requested_sampling = [manifest.get("sampling", {}).get("requested") for manifest in manifests]
    actor_versions = [manifest.get("sampling", {}).get("actor_version") for manifest in manifests]
    if any(value != requested_sampling[0] for value in requested_sampling[1:]):
        raise ValueError("Shard sampling parameters differ")
    if any(value != actor_versions[0] for value in actor_versions[1:]):
        raise ValueError("Shard actor versions differ")

    rollout_rows, state_rows, label_rows = [], [], []
    for shard_dir in shard_dirs:
        rollout_rows.extend(read_jsonl(shard_dir / SHARD_FILES["rollouts"]))
        state_rows.extend(read_jsonl(shard_dir / SHARD_FILES["states"]))
        label_rows.extend(read_jsonl(shard_dir / SHARD_FILES["mc_labels"]))
    rollout_map = _unique_by(rollout_rows, "id", "rollout_id")
    state_map = _unique_by(state_rows, "state_id", "state_id")
    label_map = _unique_by(label_rows, "state_id", "label state_id")
    if set(rollout_map) != set(source_rollout_map):
        raise ValueError("Merged rollout IDs do not exactly match the immutable source")
    if set(state_map) != set(source_state_map) or set(label_map) != set(source_state_map):
        raise ValueError("Merged state/label IDs do not exactly match the immutable source")
    for state_id, source in source_state_map.items():
        state = state_map[state_id]
        for key in ("rollout_id", "token_index", "prompt_token_ids", "prefix_response_token_ids"):
            if state.get(key) != source.get(key):
                raise ValueError(f"State changed immutable field {key!r} for state_id={state_id}")
        label = label_map[state_id]
        if str(label.get("rollout_id")) != str(source.get("rollout_id")):
            raise ValueError(f"Label rollout mismatch for state_id={state_id}")
        if int(label.get("token_index", -1)) != int(source.get("token_index", -2)):
            raise ValueError(f"Label token index mismatch for state_id={state_id}")

    continuation_locations, continuation_count = _index_continuations(
        shard_dirs, set(source_state_map), continuations_per_state
    )
    expected_continuations = len(state_order) * continuations_per_state
    if continuation_count != expected_continuations:
        raise ValueError(f"Expected {expected_continuations} continuations, found {continuation_count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    final = {name: output_dir / filename for name, filename in OUTPUT_FILES.items()}
    final["split"] = output_dir / "split.json"
    final["manifest"] = output_dir / "source_manifest.json"
    occupied = [str(path) for path in final.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite merged outputs: {occupied}")
    temporary = {name: path.with_name("." + path.name + ".incomplete") for name, path in final.items()}
    occupied = [str(path) for path in temporary.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite incomplete merge outputs: {occupied}")

    timeout_count = 0
    try:
        with (
            temporary["rollouts"].open("w", encoding="utf-8") as rollout_stream,
            temporary["states"].open("w", encoding="utf-8") as state_stream,
            temporary["mc_labels"].open("w", encoding="utf-8") as label_stream,
            temporary["continuations"].open("w", encoding="utf-8") as continuation_stream,
            ExitStack() as source_stack,
        ):
            continuation_sources = {
                path: source_stack.enter_context(path.open("rb"))
                for path in {location[0] for rows in continuation_locations.values() for location in rows.values()}
            }
            for rollout_id in rollout_order:
                _write_row(rollout_stream, rollout_map[rollout_id])
            for state_id in state_order:
                state = state_map[state_id]
                label = label_map[state_id]
                _write_row(state_stream, state)
                rewards = []
                state_timeouts = 0
                for index in range(continuations_per_state):
                    path, offset = continuation_locations[state_id][index]
                    source = continuation_sources[path]
                    source.seek(offset)
                    row = json.loads(source.readline())
                    reward = _validate_continuation(row, state_id=state_id, continuation_index=index)
                    if reward is None:
                        state_timeouts += 1
                    else:
                        rewards.append(reward)
                    _write_row(continuation_stream, row)
                if state_timeouts > max_timeouts_per_state:
                    raise ValueError(f"State {state_id} exceeded its timeout budget")
                timeout_count += state_timeouts
                if int(label.get("mc_requested_samples", -1)) != continuations_per_state:
                    raise ValueError(f"Label request count mismatch for state_id={state_id}")
                if int(label.get("mc_num_samples", -1)) != len(rewards):
                    raise ValueError(f"Label valid-trial count mismatch for state_id={state_id}")
                if int(label.get("mc_timeout_count", -1)) != state_timeouts:
                    raise ValueError(f"Label timeout count mismatch for state_id={state_id}")
                if int(label.get("mc_scorer_error_count", -1)) != 0:
                    raise ValueError(f"Label scorer errors are nonzero for state_id={state_id}")
                if label.get("prefix_continuation_rewards") != rewards:
                    raise ValueError(f"Label rewards disagree with continuations for state_id={state_id}")
                if not rewards or not math.isclose(float(label["v_prefix"]), sum(rewards) / len(rewards)):
                    raise ValueError(f"Label value disagrees with continuations for state_id={state_id}")
                _write_row(label_stream, label)
        allowed_timeouts = math.floor(expected_continuations * max_timeout_fraction)
        if timeout_count > allowed_timeouts:
            raise ValueError(f"Global timeout budget exceeded: {timeout_count} > {allowed_timeouts}")

        ordered_rollouts = [rollout_map[rollout_id] for rollout_id in rollout_order]
        ordered_labels = [label_map[state_id] for state_id in state_order]
        examples, diagnostics = adapt_legacy_rows(ordered_rollouts, ordered_labels)
        if len(examples) != len(ordered_rollouts) or diagnostics:
            raise ValueError(f"Merged labels failed compatibility validation: {diagnostics}")
        labels_by_rollout = defaultdict(list)
        for label in ordered_labels:
            labels_by_rollout[str(label["rollout_id"])].append(label)
        records = [
            {"rollout": rollout, "labels": labels_by_rollout[str(rollout["id"])]} for rollout in ordered_rollouts
        ]
        calibration = mc_endpoint_calibration(records)
        impossible = impossible_near_end_labels(records)
        if not calibration["passed"]:
            raise ValueError(f"Merged labels failed endpoint calibration: {calibration}")
        if not impossible["passed"]:
            raise ValueError(
                "Merged labels failed the near-end suffix-probability check: "
                f"{len(impossible['violations'])} violations"
            )
        shutil.copyfile(source_split, temporary["split"])
        if _file_hash(temporary["split"]) != _file_hash(source_split):
            raise RuntimeError("split.json changed while copying")
        repo = Path(__file__).resolve().parents[2]
        manifest = {
            "files": {relative: _file_hash(repo / relative) for relative in SOURCE_FILES},
            "inputs": {
                "rollouts": _file_hash(source_rollouts),
                "states": _file_hash(source_states),
                "split": _file_hash(source_split),
                "shard_manifests": [_file_hash(path / "source_manifest.json") for path in shard_dirs],
            },
            "counts": {
                "rollouts": len(rollout_order),
                "states": len(state_order),
                "continuations": expected_continuations,
            },
            "sampling": {
                "requested": requested_sampling[0],
                "actor_version": actor_versions[0],
                "prompt_transport": "token_ids",
            },
            "scoring": {
                "timeouts": timeout_count,
                "allowed_timeouts": allowed_timeouts,
                "unexpected_errors": 0,
            },
            "validation": {
                "endpoint_calibration": calibration,
                "impossible_near_end_labels": impossible,
                "split_unchanged": True,
            },
        }
        temporary["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name in ("rollouts", "states", "mc_labels", "continuations", "split"):
            manifest.setdefault("outputs", {})[name] = _file_hash(temporary[name])
        temporary["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name in ("rollouts", "states", "mc_labels", "continuations", "split", "manifest"):
            temporary[name].replace(final[name])
    except Exception:
        # Keep incomplete files as an audit trail. They are never accepted as
        # training inputs and a rerun must use a fresh destination directory.
        raise
    return {"output_dir": str(output_dir), **manifest["counts"], "manifest": str(final["manifest"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rollouts", required=True)
    parser.add_argument("--source-states", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--continuations-per-state", type=int, default=16)
    parser.add_argument("--expected-rollouts", type=int, default=486)
    parser.add_argument("--expected-states", type=int, default=8593)
    args = parser.parse_args()
    result = merge_shards(
        source_rollouts=Path(args.source_rollouts),
        source_states=Path(args.source_states),
        source_split=Path(args.source_split),
        shard_dirs=[Path(path) for path in args.shard_dir],
        output_dir=Path(args.output_dir),
        continuations_per_state=args.continuations_per_state,
        expected_rollouts=args.expected_rollouts,
        expected_states=args.expected_states,
    )
    print("DELTA_MC_MERGE_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
