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
"""Freeze an exact-size subset from live, append-only MC collection shards.

The collector is never paused or modified.  Input byte sizes are captured at
startup and only complete newline-terminated records within that boundary are
read.  A rollout is eligible only when its selected-state and MC-label ID sets
match exactly.  The selected population is deterministic and can be constrained
to match both the rollout and state counts of an existing critic dataset.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

INPUT_NAMES = {
    "rollouts": ".rollouts_train_regen.jsonl.incomplete",
    "states": ".states_train.jsonl.incomplete",
    "labels": ".mc_labels_train.jsonl.incomplete",
    "continuations": ".continuations_train.jsonl.incomplete",
}
OUTPUT_NAMES = {
    "rollouts": "rollouts_train_regen.jsonl",
    "states": "states_train.jsonl",
    "labels": "mc_labels_collector.jsonl",
    "continuations": "continuations_collector.jsonl",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input_path(shard: Path, name: str) -> Path:
    data = shard / "data" if (shard / "data").is_dir() else shard
    incomplete = data / INPUT_NAMES[name]
    if incomplete.is_file():
        return incomplete
    final = data / INPUT_NAMES[name].removeprefix(".").removesuffix(".incomplete")
    if final.is_file():
        return final
    raise FileNotFoundError(f"Missing {name} input under {shard}")


def _capture_inputs(shards: list[Path]) -> list[dict]:
    captured = []
    for shard_index, shard in enumerate(shards):
        files = {}
        for name in INPUT_NAMES:
            path = _input_path(shard, name).resolve()
            stat = path.stat()
            files[name] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        captured.append({"shard_index": shard_index, "shard": str(shard.resolve()), "files": files})
    return captured


def _iter_snapshot(path: Path, size: int):
    """Yield only full JSONL records ending at or before the frozen byte size."""
    for start, line in _iter_snapshot_raw(path, size):
        try:
            yield start, line, json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL record before frozen boundary in {path} at {start}") from exc


def _iter_snapshot_raw(path: Path, size: int):
    with path.open("rb") as stream:
        while stream.tell() < size:
            start = stream.tell()
            line = stream.readline()
            end = stream.tell()
            if end > size or not line.endswith(b"\n"):
                break
            if line.strip():
                yield start, line


_STATE_ID_PREFIX = re.compile(rb'^\{"state_id":\s*"([^"]+)"')


def _quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _advance_dp(dp: list[int], count: int, target_k: int, target_sum: int) -> None:
    mask = (1 << (target_sum + 1)) - 1
    for k in range(target_k, 0, -1):
        dp[k] |= (dp[k - 1] << count) & mask


def _exact_subset(candidates: list[dict], target_k: int, target_sum: int, block_size: int = 32) -> list[dict] | None:
    """Cardinality-constrained subset sum using bitsets and block checkpoints."""
    if target_k < 0 or target_sum < 0 or target_k > len(candidates):
        return None
    dp = [0] * (target_k + 1)
    dp[0] = 1
    checkpoints = [tuple(dp)]
    boundaries = [0]
    for index, candidate in enumerate(candidates, start=1):
        _advance_dp(dp, int(candidate["state_count"]), min(target_k, index), target_sum)
        if index % block_size == 0 or index == len(candidates):
            checkpoints.append(tuple(dp))
            boundaries.append(index)
    if not (dp[target_k] >> target_sum) & 1:
        return None

    selected = []
    k, total = target_k, target_sum
    for block in range(len(boundaries) - 1, 0, -1):
        start, end = boundaries[block - 1], boundaries[block]
        local = list(checkpoints[block - 1])
        history = [tuple(local)]
        for index in range(start, end):
            _advance_dp(local, int(candidates[index]["state_count"]), min(target_k, index + 1), target_sum)
            history.append(tuple(local))
        for index in range(end - 1, start - 1, -1):
            before = history[index - start]
            if (before[k] >> total) & 1:
                continue
            candidate = candidates[index]
            count = int(candidate["state_count"])
            if k < 1 or total < count or not (before[k - 1] >> (total - count)) & 1:
                raise RuntimeError("Subset reconstruction disagrees with reachability table")
            selected.append(candidate)
            k -= 1
            total -= count
    if k != 0 or total != 0:
        raise RuntimeError("Subset reconstruction did not reach the origin")
    selected.reverse()
    return selected


def _stable_key(seed: str, namespace: str, rollout_id: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{rollout_id}".encode()).hexdigest()


def build_plan(
    shards: list[Path],
    *,
    expected_rollouts: int,
    expected_states: int,
    continuations_per_state: int,
    width_quantile: float,
    eval_rollouts: int,
    seed: str,
) -> dict:
    captured = _capture_inputs(shards)
    rollouts, rollout_locations = {}, {}
    states_by_rollout, labels_by_rollout = defaultdict(dict), defaultdict(dict)
    for shard in captured:
        shard_index = shard["shard_index"]
        for order, (_, _, row) in enumerate(
            _iter_snapshot(Path(shard["files"]["rollouts"]["path"]), shard["files"]["rollouts"]["size"])
        ):
            rollout_id = str(row["id"])
            if rollout_id in rollouts:
                raise ValueError(f"Duplicate rollout_id={rollout_id}")
            rollouts[rollout_id] = row
            rollout_locations[rollout_id] = {"shard_index": shard_index, "row_order": order}
        for _, _, row in _iter_snapshot(Path(shard["files"]["states"]["path"]), shard["files"]["states"]["size"]):
            rollout_id, state_id = str(row["rollout_id"]), str(row["state_id"])
            if state_id in states_by_rollout[rollout_id]:
                raise ValueError(f"Duplicate state_id={state_id}")
            states_by_rollout[rollout_id][state_id] = row
        for _, _, row in _iter_snapshot(Path(shard["files"]["labels"]["path"]), shard["files"]["labels"]["size"]):
            rollout_id, state_id = str(row["rollout_id"]), str(row["state_id"])
            if state_id in labels_by_rollout[rollout_id]:
                raise ValueError(f"Duplicate label state_id={state_id}")
            labels_by_rollout[rollout_id][state_id] = row

    candidates = []
    rejection_counts = Counter()
    for rollout_id, rollout in rollouts.items():
        states = states_by_rollout.get(rollout_id, {})
        labels = labels_by_rollout.get(rollout_id, {})
        if not states or set(states) != set(labels):
            rejection_counts["state_label_incomplete"] += 1
            continue
        if any(
            int(label.get("mc_num_samples", -1)) != continuations_per_state
            or len(label.get("prefix_continuation_rewards", [])) != continuations_per_state
            for label in labels.values()
        ):
            rejection_counts["mc_budget_incomplete"] += 1
            continue
        response_ids = rollout.get("response_token_ids")
        prompt_ids = rollout.get("prompt_token_ids")
        if not isinstance(response_ids, list) or not isinstance(prompt_ids, list):
            rejection_counts["missing_token_ids"] += 1
            continue
        location = rollout_locations[rollout_id]
        candidates.append(
            {
                "rollout_id": rollout_id,
                "prompt_id": str(rollout.get("prompt_id")),
                "state_count": len(states),
                "state_ids": sorted(states, key=lambda value: int(states[value]["token_index"])),
                "width": len(prompt_ids) + len(response_ids),
                **location,
            }
        )
    if len(candidates) < expected_rollouts:
        raise ValueError(f"Only {len(candidates)} complete rollouts; need {expected_rollouts}")

    cutoff = _quantile([candidate["width"] for candidate in candidates], width_quantile)
    eligible = [candidate for candidate in candidates if candidate["width"] <= cutoff]
    eligible.sort(key=lambda candidate: _stable_key(seed, "subset", candidate["rollout_id"]))
    selected = _exact_subset(eligible, expected_rollouts, expected_states)
    used_width_quantile = width_quantile
    if selected is None:
        # Exact state-count matching has priority.  Widen the p95-like cap in
        # declared 1% increments rather than silently changing the target.
        for step in range(math.floor(width_quantile * 100) + 1, 101):
            trial_quantile = step / 100
            trial_cutoff = _quantile([candidate["width"] for candidate in candidates], trial_quantile)
            eligible = [candidate for candidate in candidates if candidate["width"] <= trial_cutoff]
            eligible.sort(key=lambda candidate: _stable_key(seed, "subset", candidate["rollout_id"]))
            selected = _exact_subset(eligible, expected_rollouts, expected_states)
            if selected is not None:
                cutoff, used_width_quantile = trial_cutoff, trial_quantile
                break
    if selected is None:
        raise ValueError(
            f"No exact subset of {expected_rollouts} rollouts and {expected_states} states exists "
            f"among {len(candidates)} complete rollouts"
        )

    selected.sort(key=lambda candidate: _stable_key(seed, "output", candidate["rollout_id"]))
    eval_ids = {
        candidate["rollout_id"]
        for candidate in sorted(selected, key=lambda row: _stable_key(seed, "eval", row["rollout_id"]))[:eval_rollouts]
    }
    split = {
        candidate["rollout_id"]: ("eval" if candidate["rollout_id"] in eval_ids else "train") for candidate in selected
    }
    return {
        "schema_version": 1,
        "source": {"captured_shards": captured, "append_only_inputs": True},
        "selection": {
            "seed": seed,
            "expected_rollouts": expected_rollouts,
            "expected_states": expected_states,
            "continuations_per_state": continuations_per_state,
            "requested_width_quantile": width_quantile,
            "used_width_quantile": used_width_quantile,
            "width_cutoff": cutoff,
            "complete_candidates": len(candidates),
            "eligible_after_width_cap": len(eligible),
            "rejections": dict(rejection_counts),
            "selected": selected,
        },
        "split": {
            "mapping": split,
            "train": expected_rollouts - eval_rollouts,
            "eval": eval_rollouts,
            "method": "sha256-ranked fixed-count holdout",
        },
    }


def _publish(temporary: dict[str, Path], final: dict[str, Path]) -> None:
    for name, path in temporary.items():
        path.replace(final[name])


def extract_plan(plan: dict, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    selected = plan["selection"]["selected"]
    rollout_ids = {row["rollout_id"] for row in selected}
    state_ids = {state_id for row in selected for state_id in row["state_ids"]}
    expected_rollouts = int(plan["selection"]["expected_rollouts"])
    expected_states = int(plan["selection"]["expected_states"])
    per_state = int(plan["selection"]["continuations_per_state"])
    final = {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}
    final.update(
        {
            "split": output_dir / "split.json",
            "selection": output_dir / "selection_manifest.json",
            "manifest": output_dir / "source_manifest.json",
        }
    )
    temporary = {name: path.with_name("." + path.name + ".incomplete") for name, path in final.items()}
    counts = Counter()
    continuation_indices = defaultdict(set)
    try:
        streams = {name: temporary[name].open("wb") for name in OUTPUT_NAMES}
        try:
            for shard in plan["source"]["captured_shards"]:
                for name in ("rollouts", "states", "labels", "continuations"):
                    path = Path(shard["files"][name]["path"])
                    size = int(shard["files"][name]["size"])
                    iterator = _iter_snapshot_raw(path, size)
                    for offset, raw in iterator:
                        if name == "continuations":
                            match = _STATE_ID_PREFIX.match(raw, 0, min(len(raw), 256))
                            if match is None:
                                try:
                                    state_id = str(json.loads(raw)["state_id"])
                                except (json.JSONDecodeError, KeyError) as exc:
                                    raise ValueError(f"Invalid continuation identity in {path} at {offset}") from exc
                            else:
                                state_id = match.group(1).decode("utf-8")
                            if state_id not in state_ids:
                                continue
                            row = json.loads(raw)
                        else:
                            row = json.loads(raw)
                        keep = (
                            str(row["id"]) in rollout_ids if name == "rollouts" else str(row["state_id"]) in state_ids
                        )
                        if not keep:
                            continue
                        streams[name].write(raw)
                        counts[name] += 1
                        if name == "continuations":
                            state_id = str(row["state_id"])
                            index = int(row.get("continuation_index", -1))
                            if index in continuation_indices[state_id]:
                                raise ValueError(f"Duplicate continuation {state_id}:{index}")
                            continuation_indices[state_id].add(index)
        finally:
            for stream in streams.values():
                stream.close()

        expected = {
            "rollouts": expected_rollouts,
            "states": expected_states,
            "labels": expected_states,
            "continuations": expected_states * per_state,
        }
        if dict(counts) != expected:
            raise ValueError(f"Frozen output counts differ: {dict(counts)} != {expected}")
        required_indices = set(range(per_state))
        incomplete = [state_id for state_id in state_ids if continuation_indices[state_id] != required_indices]
        if incomplete:
            raise ValueError(f"Incomplete continuation groups: {len(incomplete)}; first={incomplete[0]}")
        temporary["split"].write_text(
            json.dumps(plan["split"]["mapping"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary["selection"].write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "frozen_collector_snapshot_requires_strict_rescoring",
            "counts": dict(counts),
            "split": {"train": plan["split"]["train"], "eval": plan["split"]["eval"]},
            "outputs": {
                name: {"path": final[name].name, "sha256": _hash(temporary[name])}
                for name in (*OUTPUT_NAMES, "split", "selection")
            },
        }
        temporary["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish(temporary, final)
    except Exception:
        # Incomplete files intentionally remain auditable but are never valid inputs.
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--from-plan")
    parser.add_argument("--expected-rollouts", type=int, default=486)
    parser.add_argument("--expected-states", type=int, default=8593)
    parser.add_argument("--continuations-per-state", type=int, default=32)
    parser.add_argument("--width-quantile", type=float, default=0.95)
    parser.add_argument("--eval-rollouts", type=int, default=48)
    parser.add_argument("--seed", default="brezel-corrected-critic-v1")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.width_quantile <= 1:
        parser.error("--width-quantile must be in (0, 1]")
    if not 0 < args.eval_rollouts < args.expected_rollouts:
        parser.error("--eval-rollouts must leave both train and eval nonempty")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error("--output-dir already exists")
    if args.from_plan:
        if args.shard_dir:
            parser.error("--from-plan and --shard-dir are mutually exclusive")
        plan = json.loads(Path(args.from_plan).read_text(encoding="utf-8"))
    else:
        if not args.shard_dir:
            parser.error("--shard-dir is required unless --from-plan is used")
        plan = build_plan(
            [Path(path) for path in args.shard_dir],
            expected_rollouts=args.expected_rollouts,
            expected_states=args.expected_states,
            continuations_per_state=args.continuations_per_state,
            width_quantile=args.width_quantile,
            eval_rollouts=args.eval_rollouts,
            seed=args.seed,
        )
    if args.plan_only:
        output_dir.mkdir(parents=True)
        _json_dump(output_dir / "selection_manifest.json", plan)
        print(json.dumps({"output_dir": str(output_dir), "selection": plan["selection"]}, sort_keys=True))
        return
    manifest = extract_plan(plan, output_dir)
    print("DELTA_MC_FREEZE_RESULT=" + json.dumps({"output_dir": str(output_dir), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
