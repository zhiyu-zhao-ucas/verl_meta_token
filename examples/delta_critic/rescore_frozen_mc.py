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
"""Strictly rescore a frozen saved-continuation MC snapshot.

All scoring runs in spawned worker processes, where math-verify's SIGALRM
timeouts are valid.  Parsed-wrong, unparseable, timeout, and unexpected scorer
errors remain distinct.  Unexpected errors abort publication; timeouts are
excluded from MC means and must remain within explicit per-state/global budgets.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from transformers import AutoTokenizer

from .legacy_adapter import adapt_legacy_rows
from .mc_diagnostics import impossible_near_end_labels, mc_endpoint_calibration
from .sampling_io import score_text_result, scorer_metadata

SOURCE_FILES = (
    "examples/delta_critic/freeze_completed_mc_subset.py",
    "examples/delta_critic/replace_frozen_mc_rollout.py",
    "examples/delta_critic/sampling_io.py",
    "examples/delta_critic/mc_diagnostics.py",
    "examples/delta_critic/rescore_frozen_mc.py",
)
REWARD_CONFIG = {
    "method": "math_verify",
    "fallback_on_import_error": False,
    "math_verify_timeout_seconds": 5,
    "math_verify_retry_timeout_seconds": 20,
    "math_verify_require_final_answer": False,
}
_TOP_LOGPROBS_MARKER = b', "delta_top_logprobs": '


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from exc
    return rows


def _unique(rows: list[dict], key: str, description: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {description}={value}")
        result[value] = row
    return result


def _score(payload):
    text, gold_answer, reward_config = payload
    return score_text_result(text, gold_answer, reward_config)


def _decode_continuation(raw: bytes, line_number: int) -> tuple[dict, bytes | None]:
    """Parse the small prefix while retaining large top-k arrays byte-for-byte."""
    marker = raw.find(_TOP_LOGPROBS_MARKER)
    try:
        if marker >= 0:
            return json.loads(raw[:marker] + b"}"), raw[marker:]
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed frozen continuation at line {line_number}") from exc


def _score_rollouts(rollouts: list[dict], executor: ProcessPoolExecutor) -> tuple[list[dict], Counter]:
    payloads = [(str(row.get("response", "")), row.get("gold_answer"), REWARD_CONFIG) for row in rollouts]
    corrected, counts, timeouts = [], Counter(), []
    for row, result in zip(rollouts, executor.map(_score, payloads, chunksize=1), strict=True):
        if result.reward is None:
            timeouts.append(str(row["id"]))
        updated = dict(row)
        updated["terminal_reward_collector"] = row.get("terminal_reward")
        updated["terminal_reward"] = None if result.reward is None else float(result.reward)
        updated["terminal_score_status"] = result.status
        updated["terminal_score_attempts"] = result.attempts
        updated["terminal_score_timeout_seconds"] = result.timeout_seconds
        corrected.append(updated)
        counts[result.status] += 1
    if timeouts:
        raise RuntimeError(
            "Terminal rollout timeouts require population replacement; "
            + json.dumps({"count": len(timeouts), "rollout_ids": timeouts}, sort_keys=True)
        )
    return corrected, counts


def rescore(
    source_dir: Path,
    output_dir: Path,
    *,
    workers: int,
    max_timeout_fraction: float,
    max_timeouts_per_state: int,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    source_manifest_path = source_dir / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen_collector_snapshot_requires_strict_rescoring":
        raise ValueError("Input is not an unrescored frozen collector snapshot")
    expected = source_manifest["counts"]
    expected_rollouts = int(expected["rollouts"])
    expected_states = int(expected["states"])
    expected_continuations = int(expected["continuations"])
    if expected_continuations % expected_states:
        raise ValueError("Continuation count is not divisible by state count")
    continuations_per_state = expected_continuations // expected_states

    rollouts = _read_jsonl(source_dir / "rollouts_train_regen.jsonl")
    states = _read_jsonl(source_dir / "states_train.jsonl")
    collector_labels = _read_jsonl(source_dir / "mc_labels_collector.jsonl")
    split = json.loads((source_dir / "split.json").read_text(encoding="utf-8"))
    if len(rollouts) != expected_rollouts or len(states) != expected_states or len(collector_labels) != expected_states:
        raise ValueError("Frozen metadata counts do not match its manifest")
    rollout_map = _unique(rollouts, "id", "rollout_id")
    state_map = _unique(states, "state_id", "state_id")
    collector_label_map = _unique(collector_labels, "state_id", "collector label state_id")
    if set(split) != set(rollout_map) or Counter(split.values()) != Counter(
        {"train": source_manifest["split"]["train"], "eval": source_manifest["split"]["eval"]}
    ):
        raise ValueError("Frozen split disagrees with rollout IDs or declared counts")
    if set(state_map) != set(collector_label_map):
        raise ValueError("Frozen state and collector-label IDs differ")

    actor_versions = {str(row.get("actor_version", "")).strip() for row in rollouts}
    if len(actor_versions) != 1 or not next(iter(actor_versions)):
        raise ValueError(f"Rescoring requires exactly one declared actor tokenizer: {actor_versions}")
    actor_version = next(iter(actor_versions))
    tokenizer = AutoTokenizer.from_pretrained(actor_version, local_files_only=True)
    for rollout in rollouts:
        response_ids = rollout.get("response_token_ids")
        if not isinstance(response_ids, list):
            raise ValueError(f"Rollout {rollout['id']} lacks response_token_ids")
        decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
        if decoded != str(rollout.get("response", "")):
            raise ValueError(f"Tokenizer reconstruction disagrees with saved response for {rollout['id']}")

    prefix_ids_by_state = {}
    skip_special_by_state = {}
    for state_id, state in state_map.items():
        prefix_ids = state.get("prefix_response_token_ids")
        label_prefix_ids = collector_label_map[state_id].get("prefix_response_token_ids")
        if not isinstance(prefix_ids, list) or prefix_ids != label_prefix_ids:
            raise ValueError(f"State/label prefix token IDs disagree for {state_id}")
        prefix_ids_by_state[state_id] = prefix_ids
        skip_special_by_state[state_id] = bool(
            collector_label_map[state_id].get("mc_continuation_skip_special_tokens", True)
        )

    output_dir.mkdir(parents=True)
    finals = {
        "rollouts": output_dir / "rollouts_train_regen.jsonl",
        "states": output_dir / "states_train.jsonl",
        "labels": output_dir / "mc_labels_train.jsonl",
        "continuations": output_dir / "continuations_train.jsonl",
        "split": output_dir / "split.json",
        "manifest": output_dir / "source_manifest.json",
    }
    temporary = {name: path.with_name("." + path.name + ".incomplete") for name, path in finals.items()}
    score_meta = scorer_metadata()
    continuation_outcomes = defaultdict(list)
    continuation_indices = defaultdict(set)
    status_counts = Counter()
    empty_counts = Counter()
    pending = {}
    scored_rows = 0
    score_started = time.monotonic()

    def consume_completed(stream):
        nonlocal scored_rows
        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in completed:
            row, preserved_tail = pending.pop(future)
            result = future.result()
            state_id = str(row["state_id"])
            updated = dict(row)
            updated["collector_reward"] = row.get("reward")
            updated["reward"] = result.reward
            updated["score_status"] = result.status
            updated["score_attempts"] = result.attempts
            updated["score_timeout_seconds"] = result.timeout_seconds
            encoded = json.dumps(updated, ensure_ascii=False).encode("utf-8")
            stream.write(encoded[:-1] + preserved_tail if preserved_tail is not None else encoded + b"\n")
            continuation_outcomes[state_id].append(
                (int(row["continuation_index"]), result.reward, result.status, result.attempts, result.timeout_seconds)
            )
            status_counts[result.status] += 1
            empty_counts["empty"] += not bool(row.get("continuation_token_ids"))
            scored_rows += 1
            if scored_rows % 1024 == 0:
                elapsed = time.monotonic() - score_started
                print(
                    json.dumps(
                        {
                            "event": "rescore_progress",
                            "continuations": scored_rows,
                            "expected": expected_continuations,
                            "rows_per_second": scored_rows / elapsed,
                            "statuses": dict(status_counts),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            corrected_rollouts, terminal_status_counts = _score_rollouts(rollouts, executor)
            with (
                (source_dir / "continuations_collector.jsonl").open("rb") as source,
                temporary["continuations"].open("wb") as destination,
            ):
                for line_number, raw in enumerate(source, start=1):
                    if not raw.strip():
                        continue
                    row, preserved_tail = _decode_continuation(raw, line_number)
                    state_id = str(row.get("state_id"))
                    if state_id not in state_map:
                        raise ValueError(f"Continuation references unknown state_id={state_id}")
                    rollout_id = str(row.get("rollout_id"))
                    if rollout_id != str(state_map[state_id]["rollout_id"]):
                        raise ValueError(f"Continuation rollout mismatch for state_id={state_id}")
                    index = int(row.get("continuation_index", -1))
                    if not 0 <= index < continuations_per_state or index in continuation_indices[state_id]:
                        raise ValueError(f"Invalid or duplicate continuation {state_id}:{index}")
                    continuation_indices[state_id].add(index)
                    rollout = rollout_map[rollout_id]
                    continuation_ids = row.get("continuation_token_ids")
                    if not isinstance(continuation_ids, list):
                        raise ValueError(f"Continuation lacks token IDs for {state_id}:{index}")
                    skip_special_tokens = skip_special_by_state[state_id]
                    decoded_continuation = tokenizer.decode(continuation_ids, skip_special_tokens=skip_special_tokens)
                    if decoded_continuation != str(row.get("continuation_text", "")):
                        raise ValueError(f"Continuation text/token mismatch for {state_id}:{index}")
                    full_response = tokenizer.decode(
                        prefix_ids_by_state[state_id] + continuation_ids,
                        skip_special_tokens=skip_special_tokens,
                    )
                    future = executor.submit(
                        _score,
                        (full_response, rollout.get("gold_answer"), REWARD_CONFIG),
                    )
                    pending[future] = (row, preserved_tail)
                    if len(pending) >= workers * 2:
                        consume_completed(destination)
                while pending:
                    consume_completed(destination)

        if scored_rows != expected_continuations:
            raise ValueError(f"Expected {expected_continuations} rescored continuations, found {scored_rows}")
        required_indices = set(range(continuations_per_state))
        incomplete = [state_id for state_id in state_map if continuation_indices[state_id] != required_indices]
        if incomplete:
            raise ValueError(f"Incomplete continuation groups after rescore: {len(incomplete)}; first={incomplete[0]}")

        labels = []
        timeout_count = 0
        with temporary["labels"].open("w", encoding="utf-8") as stream:
            for state in states:
                state_id = str(state["state_id"])
                outcomes = sorted(continuation_outcomes[state_id])
                rewards = [float(reward) for _, reward, _, _, _ in outcomes if reward is not None]
                state_timeouts = sum(status == "timeout" for _, _, status, _, _ in outcomes)
                if state_timeouts > max_timeouts_per_state:
                    raise ValueError(f"State {state_id} exceeded timeout budget: {state_timeouts}")
                if not rewards:
                    raise ValueError(f"State {state_id} has no valid scored continuations")
                timeout_count += state_timeouts
                label = dict(collector_label_map[state_id])
                label["collector_v_prefix"] = label.get("v_prefix")
                label["collector_prefix_continuation_rewards"] = label.get("prefix_continuation_rewards")
                label["v_prefix"] = sum(rewards) / len(rewards)
                label["prefix_continuation_rewards"] = rewards
                label["mc_requested_samples"] = continuations_per_state
                label["mc_num_samples"] = len(rewards)
                label["mc_timeout_count"] = state_timeouts
                label["mc_scorer_error_count"] = 0
                label["mc_scorer"] = score_meta
                labels.append(label)
                stream.write(json.dumps(label, ensure_ascii=False) + "\n")
        allowed_timeouts = math.floor(expected_continuations * max_timeout_fraction)
        if timeout_count > allowed_timeouts:
            raise ValueError(f"Global timeout budget exceeded: {timeout_count} > {allowed_timeouts}")

        with temporary["rollouts"].open("w", encoding="utf-8") as stream:
            for row in corrected_rollouts:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        shutil.copyfile(source_dir / "states_train.jsonl", temporary["states"])
        shutil.copyfile(source_dir / "split.json", temporary["split"])

        examples, diagnostics = adapt_legacy_rows(corrected_rollouts, labels)
        if len(examples) != expected_rollouts or diagnostics:
            raise ValueError(f"Corrected labels failed critic-loader compatibility: {diagnostics}")
        labels_by_rollout = defaultdict(list)
        for label in labels:
            labels_by_rollout[str(label["rollout_id"])].append(label)
        records = [
            {"rollout": rollout, "labels": labels_by_rollout[str(rollout["id"])]} for rollout in corrected_rollouts
        ]
        calibration = mc_endpoint_calibration(records)
        impossible = impossible_near_end_labels(records)
        if not calibration["passed"]:
            raise ValueError(f"Corrected labels failed endpoint calibration: {calibration}")
        if not impossible["passed"]:
            raise ValueError(
                f"Corrected labels failed near-end suffix-probability gate: {len(impossible['violations'])} violations"
            )

        repo = Path(__file__).resolve().parents[2]
        manifest = {
            "schema_version": 1,
            "status": "complete_strictly_rescored",
            "counts": {
                "rollouts": expected_rollouts,
                "states": expected_states,
                "continuations": expected_continuations,
                "train_rollouts": source_manifest["split"]["train"],
                "eval_rollouts": source_manifest["split"]["eval"],
            },
            "source": {
                "frozen_snapshot_manifest": _hash(source_manifest_path),
                "selection_manifest": _hash(source_dir / "selection_manifest.json"),
                "collector_outputs": source_manifest["outputs"],
            },
            "scoring": {
                "reward_config": REWARD_CONFIG,
                "scorer": score_meta,
                "scoring_input": "tokenizer-decoded prefix_response_token_ids + continuation_token_ids",
                "tokenizer": actor_version,
                "workers": workers,
                "terminal_status_counts": dict(terminal_status_counts),
                "continuation_status_counts": dict(status_counts),
                "empty_continuations": empty_counts["empty"],
                "timeouts": timeout_count,
                "allowed_timeouts": allowed_timeouts,
                "max_timeouts_per_state": max_timeouts_per_state,
                "unexpected_errors": 0,
            },
            "validation": {
                "endpoint_calibration": calibration,
                "impossible_near_end_labels": impossible,
                "critic_loader_compatible": True,
            },
            "source_files": {relative: _hash(repo / relative) for relative in SOURCE_FILES},
        }
        manifest["outputs"] = {
            name: {"path": finals[name].name, "sha256": _hash(temporary[name])}
            for name in ("rollouts", "states", "labels", "continuations", "split")
        }
        temporary["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name in ("rollouts", "states", "labels", "continuations", "split", "manifest"):
            temporary[name].replace(finals[name])
    except Exception:
        # Keep incomplete files for diagnosis; training loaders only receive final names.
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(32, multiprocessing.cpu_count() // 2)))
    parser.add_argument("--max-timeout-fraction", type=float, default=0.0001)
    parser.add_argument("--max-timeouts-per-state", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 0 <= args.max_timeout_fraction <= 1 or args.max_timeouts_per_state < 0:
        parser.error("timeout budgets must be nonnegative")
    result = rescore(
        Path(args.source_dir),
        Path(args.output_dir),
        workers=args.workers,
        max_timeout_fraction=args.max_timeout_fraction,
        max_timeouts_per_state=args.max_timeouts_per_state,
    )
    print("DELTA_MC_RESCORE_RESULT=" + json.dumps({"output_dir": args.output_dir, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
