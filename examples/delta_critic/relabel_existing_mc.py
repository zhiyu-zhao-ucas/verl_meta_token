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
"""Regenerate MC labels for immutable legacy rollouts and selected states."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import replace
from functools import partial
from pathlib import Path

from .batch_sample import _read_partial_rows, _rewrite_rows, _write_row, sampling_settings
from .legacy_adapter import adapt_legacy_rows
from .mc_diagnostics import impossible_near_end_labels, mc_endpoint_calibration
from .mc_labeling import build_mc_requests, label_mc_states_async
from .sampling_io import read_jsonl, score_text, score_text_results_async
from .v1_sampler import sample_v1_continuations

SOURCE_FILES = (
    "examples/delta_critic/batch_sample.py",
    "examples/delta_critic/legacy_adapter.py",
    "examples/delta_critic/mc_diagnostics.py",
    "examples/delta_critic/mc_labeling.py",
    "examples/delta_critic/relabel_existing_mc.py",
    "examples/delta_critic/sampling_io.py",
    "examples/delta_critic/v1_sampler.py",
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
)

REPAIR_SAMPLING_CONTRACT = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_tokens": 30720,
}


def _json_fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest(repo: Path) -> dict:
    return {relative: _file_hash(repo / relative) for relative in SOURCE_FILES}


def _write_json_atomic(path: Path, value) -> None:
    replacement = path.with_name(path.name + ".incomplete")
    replacement.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replacement.replace(path)


def _validate_sampling_contract(mc_config) -> None:
    actual = dict(mc_config.sampling_config)
    mismatches = {
        key: (actual.get(key), expected)
        for key, expected in REPAIR_SAMPLING_CONTRACT.items()
        if actual.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Legacy repair must match the rollout sampler "
            f"(T=1, top_p=.95, top_k=20, min_p=0, max_tokens=30720): {mismatches}"
        )
    if actual.get("delta_top_logprobs") is not None or actual.get("logprobs") is not True:
        raise ValueError("Relabeling must save sampled-token log-probabilities without top-k log-probability rows")


def _paths(output_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    names = {
        "rollouts": "rollouts_relabel.jsonl",
        "states": "states_relabel.jsonl",
        "mc_labels": "mc_labels_relabel.jsonl",
        "continuations": "continuations_relabel.jsonl",
    }
    final = {name: output_dir / filename for name, filename in names.items()}
    temporary = {name: output_dir / f".{filename}.incomplete" for name, filename in names.items()}
    return final, temporary


def _canonicalize_states(rollouts, rows) -> list[dict]:
    """Extract immutable selected-state inputs from states or legacy MC labels.

    The historical 8,593-row artifact retained token-ID prefixes in the label
    file but did not retain a separate states file. Never copy its old MC
    values into the repaired state journal.
    """
    rollout_map = {str(row["id"]): row for row in rollouts}
    ranks = defaultdict(int)
    states = []
    for row in rows:
        rollout_id = str(row["rollout_id"])
        if rollout_id not in rollout_map:
            raise ValueError(f"Legacy state references missing rollout_id={rollout_id}")
        rollout = rollout_map[rollout_id]
        token_index = int(row["token_index"])
        response_ids = list(rollout["response_token_ids"])
        if not 0 <= token_index < len(response_ids):
            raise ValueError(f"Legacy token_index={token_index} outside rollout_id={rollout_id}")
        prompt_ids = list(row["prompt_token_ids"])
        prefix_ids = list(row["prefix_response_token_ids"])
        if prompt_ids != list(rollout["prompt_token_ids"]) or prefix_ids != response_ids[:token_index]:
            raise ValueError(f"Legacy state token prefix mismatch for rollout_id={rollout_id}")
        token = (rollout.get("tokens") or [{}] * len(response_ids))[token_index]
        state = {
            "state_id": str(row["state_id"]),
            "rollout_id": rollout_id,
            "split": rollout.get("split"),
            "prompt_id": rollout.get("prompt_id"),
            "response_index": rollout.get("response_index"),
            "token_index": token_index,
            "selection_rank": int(row.get("selection_rank", ranks[rollout_id])),
            "selection_score": float(row.get("selection_score", 1.0 - float(token.get("top1_prob", 1.0)))),
            "selection_strategy": row.get("selection_strategy", "legacy_mc_label_positions"),
            "prompt_token_ids": prompt_ids,
            "prefix_response_token_ids": prefix_ids,
            "sampled_token_id": response_ids[token_index],
            "candidates": row.get("candidates", []),
            "candidate_mass": float(row.get("candidate_mass", 0.0)),
            "actor_version": rollout.get("actor_version"),
            "gold_answer": rollout.get("gold_answer"),
        }
        ranks[rollout_id] += 1
        states.append(state)
    return states


def _validate_inputs(rollouts, states, mc_config) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    rollout_map = {str(row["id"]): row for row in rollouts}
    if len(rollout_map) != len(rollouts):
        raise ValueError("Duplicate rollout IDs in legacy input")
    for rollout in rollouts:
        if float(rollout["terminal_reward"]) <= 0:
            continue
        response_ids = rollout.get("response_token_ids")
        behavior_logprobs = rollout.get("behavior_logprobs")
        if (
            not isinstance(response_ids, list)
            or not isinstance(behavior_logprobs, list)
            or len(response_ids) != len(behavior_logprobs)
            or any(
                isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value > 0
                for value in behavior_logprobs
            )
        ):
            raise ValueError(
                f"Correct rollout {rollout.get('id')} needs token-aligned finite behavior_logprobs "
                "for the impossible-label gate"
            )
    grouped = defaultdict(list)
    for state in states:
        grouped[str(state["rollout_id"])].append(state)
    if set(grouped) != set(rollout_map):
        missing = set(rollout_map) - set(grouped)
        unknown = set(grouped) - set(rollout_map)
        raise ValueError(f"Legacy rollout/state coverage mismatch: missing={len(missing)}, unknown={len(unknown)}")
    for rollout_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["token_index"]))
        build_mc_requests([rollout_map[rollout_id]], rows, mc_config)
    return rollout_map, dict(grouped)


def _rescore_rollouts(rollouts, reward_config):
    rescored = []
    for row in rollouts:
        if not isinstance(row.get("response"), str):
            raise ValueError(f"Legacy rollout {row.get('id')} is missing decoded response text")
        updated = dict(row)
        updated["legacy_terminal_reward"] = row.get("terminal_reward")
        updated["terminal_reward"] = score_text(row["response"], row.get("gold_answer"), reward_config)
        rescored.append(updated)
    return rescored


def _recover_labels(temporary, states, *, continuations_per_state):
    ordered_state_ids = list(dict.fromkeys(str(state["state_id"]) for state in states))
    state_ids = set(ordered_state_ids)
    continuations = _read_partial_rows(temporary["continuations"])
    by_state = defaultdict(dict)
    for row in continuations:
        state_id = str(row["state_id"])
        index = int(row["continuation_index"])
        if state_id in state_ids:
            by_state[state_id][index] = row
    expected = set(range(continuations_per_state))
    labels = {}
    for label in _read_partial_rows(temporary["mc_labels"]):
        state_id = str(label["state_id"])
        requested = label.get("mc_requested_samples", label.get("mc_num_samples"))
        if state_id in state_ids and requested == continuations_per_state and set(by_state[state_id]) == expected:
            labels[state_id] = label
    canonical_continuations = [
        by_state[state_id][index]
        for state_id in ordered_state_ids
        if state_id in labels
        for index in sorted(by_state[state_id])
    ]
    _rewrite_rows(temporary["mc_labels"], (labels[state_id] for state_id in ordered_state_ids if state_id in labels))
    _rewrite_rows(temporary["continuations"], canonical_continuations)
    return labels, canonical_continuations


def _continuation_row(rollout, label, continuation, mc_config):
    continuation_index = continuation["continuation_index"]
    token_ids = continuation["token_ids"]
    return {
        "state_id": label["state_id"],
        "rollout_id": label["rollout_id"],
        "prompt_id": label.get("prompt_id"),
        "split": label.get("split"),
        "response_index": rollout.get("response_index"),
        "token_index": label["token_index"],
        "continuation_index": continuation_index,
        "continuation_id": f"{label['state_id']}:k{continuation_index}",
        "continuation_token_ids": token_ids,
        "continuation_text": continuation["text"],
        "continuation_num_tokens": len(token_ids),
        "empty_continuation": not token_ids,
        "reward": continuation["reward"],
        "score_status": continuation["score_status"],
        "score_attempts": continuation["score_attempts"],
        "score_timeout_seconds": continuation["score_timeout_seconds"],
        **(
            {"sampled_logprobs": continuation["sampled_logprobs"]}
            if continuation.get("sampled_logprobs") is not None
            else {}
        ),
        "actor_version": mc_config.actor_version,
        "sampling_config": dict(label["mc_effective_sampling_config"]),
        "requested_sampling_config": dict(mc_config.sampling_config),
        "prompt_transport": "token_ids",
    }


def _check_visible_gpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly; GPUs 6 and 7 are unavailable for relabeling")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError("The relabel entry point owns exactly one TP1 GPU")
    if devices[0] in {"6", "7"}:
        raise RuntimeError("GPUs 6 and 7 are unavailable for MC label regeneration")


async def run(
    config,
    *,
    rollouts_path: Path,
    states_path: Path,
    output_dir: Path,
    model_path: str,
    actor_version: str,
    rollout_concurrency: int,
    mc_concurrency: int,
    mc_native_batch_size: int | None,
    gpu_memory_utilization: float,
    enforce_endpoint_calibration: bool,
    verl_overrides: list[str],
):
    _check_visible_gpu()
    if rollout_concurrency < 1 or mc_concurrency < 1:
        raise ValueError("Concurrency must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1)")
    _, mc_config = sampling_settings(config, "prefix_only")
    mc_config = replace(
        mc_config,
        actor_version=actor_version,
        max_total_tokens=int(config["model"]["max_model_len"]),
    )
    _validate_sampling_contract(mc_config)
    if not mc_config.save_continuations or not mc_config.save_individual_rewards:
        raise ValueError("Relabeling requires save_continuations and save_individual_rewards")
    rollouts = read_jsonl(rollouts_path)
    rescored = _rescore_rollouts(rollouts, config.get("reward") or {})
    states = _canonicalize_states(rescored, read_jsonl(states_path))
    rollout_map, grouped = _validate_inputs(rescored, states, mc_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    final, temporary = _paths(output_dir)
    existing = [str(path) for path in final.values() if path.exists()]
    if existing and len(existing) != len(final):
        # Recover an interruption during the final sequence of atomic renames.
        for name, destination in final.items():
            if destination.exists():
                if temporary[name].exists():
                    raise RuntimeError(f"Both final and temporary relabel output exist for {name}")
                destination.replace(temporary[name])
        existing = []
    if existing:
        raise FileExistsError(f"Relabeling output already exists: {existing}")
    repo = Path(__file__).resolve().parents[2]
    source_files = _source_manifest(repo)
    checkpoint_payload = {
        "version": 1,
        "rollouts_sha256": _file_hash(rollouts_path),
        "states_sha256": _file_hash(states_path),
        "model_path": model_path,
        "actor_version": actor_version,
        "mc_config": {
            "mode": mc_config.mode,
            "continuations_per_state": mc_config.continuations_per_state,
            "sampling_config": dict(mc_config.sampling_config),
            "continuation_skip_special_tokens": mc_config.continuation_skip_special_tokens,
            "max_total_tokens": mc_config.max_total_tokens,
            "scorer_metadata": dict(mc_config.scorer_metadata or {}),
        },
        "reward": config.get("reward") or {},
        "enforce_endpoint_calibration": enforce_endpoint_calibration,
        "source_files": source_files,
        "rescored_rows_fingerprint": _json_fingerprint(rescored),
        "state_rows_fingerprint": _json_fingerprint(states),
    }
    checkpoint_path = output_dir / ".checkpoint_relabel.json"
    checkpoint_fingerprint = _json_fingerprint(checkpoint_payload)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("fingerprint") != checkpoint_fingerprint:
            raise ValueError("Relabel checkpoint does not match inputs, source, scorer, model, or sampling config")
        if _json_fingerprint(_read_partial_rows(temporary["rollouts"])) != _json_fingerprint(rescored):
            raise ValueError("Recovered relabel rollouts differ from current rescoring")
        if _json_fingerprint(_read_partial_rows(temporary["states"])) != _json_fingerprint(states):
            raise ValueError("Recovered relabel states differ from the immutable input")
    else:
        nonempty = [str(path) for path in temporary.values() if path.exists() and path.stat().st_size]
        if nonempty:
            raise ValueError(f"Partial relabel outputs have no compatible checkpoint: {nonempty}")
        _rewrite_rows(temporary["rollouts"], rescored)
        _rewrite_rows(temporary["states"], states)
        _rewrite_rows(temporary["mc_labels"], [])
        _rewrite_rows(temporary["continuations"], [])
        _write_json_atomic(
            checkpoint_path,
            {"fingerprint": checkpoint_fingerprint, "payload": checkpoint_payload},
        )
    labels_by_state, _ = _recover_labels(
        temporary,
        states,
        continuations_per_state=mc_config.continuations_per_state,
    )

    import ray
    from hydra import compose, initialize_config_dir

    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.model import HFModelConfig
    from verl.workers.rollout.llm_server import LLMServerManager

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        verl_config = compose(config_name="ppo_trainer", overrides=verl_overrides)
    verl_config.trainer.n_gpus_per_node = 1
    verl_config.trainer.nnodes = 1
    verl_config.actor_rollout_ref.model.path = model_path
    rollout_config = verl_config.actor_rollout_ref.rollout
    rollout_config.name = "vllm"
    rollout_config.mode = "async"
    rollout_config.nnodes = 1
    rollout_config.tensor_model_parallel_size = 1
    rollout_config.data_parallel_size = 1
    rollout_config.pipeline_model_parallel_size = 1
    rollout_config.load_format = "auto"
    rollout_config.skip_tokenizer_init = False
    rollout_config.dtype = str(config["model"].get("dtype", "auto"))
    rollout_config.max_model_len = int(config["model"]["max_model_len"])
    rollout_config.prompt_length = int(config["model"]["max_model_len"])
    rollout_config.response_length = int(mc_config.sampling_config["max_tokens"])
    rollout_config.seed = int(config.get("seed", 42))
    rollout_config.gpu_memory_utilization = gpu_memory_utilization
    rollout_config.standalone_gpu_memory_utilization = gpu_memory_utilization
    model_config = omega_conf_to_dataclass(verl_config.actor_rollout_ref.model, HFModelConfig)
    tokenizer = model_config.tokenizer

    native_batch_size = min(mc_native_batch_size or mc_concurrency, mc_config.continuations_per_state)
    if not 1 <= native_batch_size <= mc_concurrency:
        raise ValueError("mc_native_batch_size must be between 1 and mc_concurrency")
    request_semaphore = asyncio.Semaphore(max(1, mc_concurrency // native_batch_size))
    score_semaphore = asyncio.Semaphore(min(4, os.cpu_count() or 1))
    rollout_semaphore = asyncio.Semaphore(rollout_concurrency)
    timeout_count = sum(int(label.get("mc_timeout_count", 0)) for label in labels_by_state.values())
    reward_config = config.get("reward") or {}
    max_per_state = int(reward_config.get("mc_max_timeouts_per_state", 1))
    max_fraction = float(reward_config.get("mc_max_timeout_fraction", 0.0001))
    max_timeout_count = reward_config.get("mc_max_timeout_count")
    if max_per_state < 0 or not 0 <= max_fraction <= 1:
        raise ValueError("MC timeout budgets must be nonnegative")
    if max_timeout_count is not None and (
        isinstance(max_timeout_count, bool) or not isinstance(max_timeout_count, int) or max_timeout_count < 0
    ):
        raise ValueError("reward.mc_max_timeout_count must be a nonnegative integer")
    error_path = output_dir / "scoring_errors.jsonl"

    ray.init(
        num_gpus=1,
        num_cpus=min(8, os.cpu_count() or 8),
        object_store_memory=4 * 1024**3,
        include_dashboard=False,
        log_to_driver=False,
    )
    try:
        manager = await LLMServerManager.create(config=verl_config)
        client = manager.get_client()
        with (
            temporary["mc_labels"].open("a", encoding="utf-8") as label_file,
            temporary["continuations"].open("a", encoding="utf-8") as continuation_file,
        ):

            def emit_label(rollout, label):
                nonlocal timeout_count
                label_timeouts = int(label.get("mc_timeout_count", 0))
                if label_timeouts > max_per_state:
                    raise RuntimeError(f"State {label['state_id']} exceeded its timeout budget")
                timeout_count += label_timeouts
                for continuation in label.pop("mc_continuations", []):
                    _write_row(continuation_file, _continuation_row(rollout, label, continuation, mc_config))
                continuation_file.flush()
                _write_row(label_file, label)
                label_file.flush()
                labels_by_state[str(label["state_id"])] = label

            async def label_rollout(rollout_id):
                pending = [row for row in grouped[rollout_id] if str(row["state_id"]) not in labels_by_state]
                if not pending:
                    return
                rollout = rollout_map[rollout_id]
                async with rollout_semaphore:
                    try:
                        await label_mc_states_async(
                            [rollout],
                            pending,
                            mc_config,
                            sample=partial(
                                sample_v1_continuations,
                                server_client=client,
                                tokenizer=tokenizer,
                                request_semaphore=request_semaphore,
                                native_n_batch_size=native_batch_size,
                            ),
                            decode_prefix=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
                            score=lambda response, row: score_text(response, row["gold_answer"], reward_config),
                            request_concurrency=mc_concurrency,
                            score_batch_async=partial(score_text_results_async, reward_config=reward_config),
                            score_semaphore=score_semaphore,
                            on_label=partial(emit_label, rollout),
                        )
                    except Exception as exc:
                        with error_path.open("a", encoding="utf-8") as stream:
                            _write_row(
                                stream,
                                {
                                    "rollout_id": rollout_id,
                                    "pending_state_ids": [row["state_id"] for row in pending],
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                            )
                        raise

            await asyncio.gather(*(label_rollout(rollout_id) for rollout_id in rollout_map))
    finally:
        ray.shutdown()

    labels = _read_partial_rows(temporary["mc_labels"])
    if len(labels) != len(states) or set(labels_by_state) != {str(state["state_id"]) for state in states}:
        raise RuntimeError("Relabeling did not commit exactly one label per state")
    requested = len(states) * mc_config.continuations_per_state
    allowed_timeouts = math.floor(requested * max_fraction)
    if max_timeout_count is not None:
        allowed_timeouts = min(allowed_timeouts, max_timeout_count)
    if timeout_count > allowed_timeouts:
        raise RuntimeError(f"MC timeout budget exceeded: {timeout_count} > {allowed_timeouts}")
    if len(_read_partial_rows(temporary["continuations"])) != requested:
        raise RuntimeError("Relabeling did not persist every requested continuation")
    examples, diagnostics = adapt_legacy_rows(rescored, labels)
    if len(examples) != len(rescored) or diagnostics:
        raise ValueError(f"Regenerated label validation failed: {diagnostics}")
    labels_by_rollout = defaultdict(list)
    for label in labels:
        labels_by_rollout[str(label["rollout_id"])].append(label)
    validation_records = [{"rollout": rollout, "labels": labels_by_rollout[str(rollout["id"])]} for rollout in rescored]
    calibration = mc_endpoint_calibration(validation_records)
    if enforce_endpoint_calibration and not calibration["passed"]:
        raise ValueError(f"Regenerated labels failed endpoint calibration: {calibration}")
    impossible_labels = impossible_near_end_labels(validation_records)
    if not impossible_labels["passed"]:
        raise ValueError(
            "Regenerated labels failed the near-end suffix-probability check: "
            f"{len(impossible_labels['violations'])} violations"
        )
    for name in final:
        temporary[name].replace(final[name])
    manifest = {
        "files": source_files,
        "inputs": {
            "rollouts": _file_hash(rollouts_path),
            "states": _file_hash(states_path),
        },
        "outputs": {name: _file_hash(path) for name, path in final.items()},
        "sampling": {
            "prompt_transport": "token_ids",
            "requested": dict(mc_config.sampling_config),
            "actor_version": actor_version,
        },
        "scoring": {
            "timeouts": timeout_count,
            "allowed_timeouts": allowed_timeouts,
            "unexpected_errors": 0,
        },
        "validation": {
            "endpoint_calibration": calibration,
            "endpoint_calibration_enforced": enforce_endpoint_calibration,
            "impossible_near_end_labels": impossible_labels,
        },
    }
    manifest_path = output_dir / "source_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return {"rollouts": len(rescored), "states": len(states), "manifest": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/delta_critic/config_sampling_qwen3_4b_base_dapo.yaml")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument(
        "--states", required=True, help="Selected states JSONL, or legacy MC labels containing prefixes"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--actor-version")
    parser.add_argument("--rollout-concurrency", type=int, default=8)
    parser.add_argument("--mc-concurrency", type=int, default=16)
    parser.add_argument("--mc-native-batch-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument(
        "--defer-endpoint-calibration",
        action="store_true",
        help="Report but do not gate endpoint calibration in a shard; the merged full population must enforce it",
    )
    parser.add_argument("--verl-override", action="append", default=[])
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    model_path = args.model_path or str(config["model"]["actor_model"])
    actor_version = args.actor_version or model_path
    result = asyncio.run(
        run(
            config,
            rollouts_path=Path(args.rollouts),
            states_path=Path(args.states),
            output_dir=Path(args.output_dir),
            model_path=model_path,
            actor_version=actor_version,
            rollout_concurrency=args.rollout_concurrency,
            mc_concurrency=args.mc_concurrency,
            mc_native_batch_size=args.mc_native_batch_size,
            gpu_memory_utilization=(
                args.gpu_memory_utilization
                if args.gpu_memory_utilization is not None
                else float(config["model"].get("gpu_memory_utilization", 0.9))
            ),
            enforce_endpoint_calibration=not args.defer_endpoint_calibration,
            verl_overrides=args.verl_override,
        )
    )
    print("DELTA_MC_RELABEL_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
