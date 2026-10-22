# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0

import hashlib
import json
from pathlib import Path

import pytest

from examples.delta_critic.merge_relabel_shards import merge_shards


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _source_rows():
    rollouts, states = [], []
    for index, reward in enumerate((1.0, 0.0)):
        rollout_id = f"r{index}"
        rollouts.append(
            {
                "id": rollout_id,
                "prompt_id": f"p{index}",
                "prompt_token_ids": [10 + index],
                "response_token_ids": [20 + index],
                "behavior_logprobs": [-0.1],
                "response": str(index),
                "gold_answer": str(index),
                "terminal_reward": reward,
            }
        )
        states.append(
            {
                "state_id": f"s{index}",
                "rollout_id": rollout_id,
                "token_index": 0,
                "prompt_token_ids": [10 + index],
                "prefix_response_token_ids": [],
            }
        )
    return rollouts, states


def _make_shard(root, rollout, state, reward):
    root.mkdir()
    repaired_rollout = {**rollout, "legacy_terminal_reward": rollout["terminal_reward"]}
    labels = [
        {
            **state,
            "v_prefix": reward,
            "mc_num_samples": 2,
            "mc_requested_samples": 2,
            "mc_timeout_count": 0,
            "mc_scorer_error_count": 0,
            "prefix_continuation_rewards": [reward, reward],
        }
    ]
    continuations = [
        {
            "state_id": state["state_id"],
            "rollout_id": rollout["id"],
            "continuation_index": index,
            "continuation_token_ids": [30 + index],
            "continuation_text": str(index),
            "reward": reward,
            "score_status": "correct" if reward else "incorrect",
            "sampled_logprobs": [-0.2],
        }
        for index in range(2)
    ]
    rows = {"rollouts": [repaired_rollout], "states": [state], "mc_labels": labels, "continuations": continuations}
    outputs = {}
    filenames = {
        "rollouts": "rollouts_relabel.jsonl",
        "states": "states_relabel.jsonl",
        "mc_labels": "mc_labels_relabel.jsonl",
        "continuations": "continuations_relabel.jsonl",
    }
    for name, filename in filenames.items():
        _write_jsonl(root / filename, rows[name])
        outputs[name] = _hash(root / filename)
    (root / "source_manifest.json").write_text(
        json.dumps(
            {
                "outputs": outputs,
                "sampling": {"requested": {"temperature": 1.0}, "actor_version": "actor"},
                "scoring": {"unexpected_errors": 0},
            }
        ),
        encoding="utf-8",
    )


def test_merge_restores_source_order_and_preserves_split(tmp_path):
    rollouts, states = _source_rows()
    source_rollouts = tmp_path / "rollouts.jsonl"
    source_states = tmp_path / "states.jsonl"
    source_split = tmp_path / "split.json"
    _write_jsonl(source_rollouts, rollouts)
    _write_jsonl(source_states, states)
    source_split.write_bytes(b'{"train": ["p0", "p1"]}\n')
    shard0, shard1 = tmp_path / "shard0", tmp_path / "shard1"
    _make_shard(shard0, rollouts[0], states[0], 1.0)
    _make_shard(shard1, rollouts[1], states[1], 0.0)

    output = tmp_path / "merged"
    result = merge_shards(
        source_rollouts=source_rollouts,
        source_states=source_states,
        source_split=source_split,
        shard_dirs=[shard1, shard0],
        output_dir=output,
        continuations_per_state=2,
        expected_rollouts=2,
        expected_states=2,
    )

    assert result["continuations"] == 4
    assert [row["id"] for row in map(json.loads, (output / "rollouts_train_regen.jsonl").read_text().splitlines())] == [
        "r0",
        "r1",
    ]
    assert [
        row["state_id"] for row in map(json.loads, (output / "mc_labels_train.jsonl").read_text().splitlines())
    ] == ["s0", "s1"]
    assert [
        row["state_id"] for row in map(json.loads, (output / "continuations_train.jsonl").read_text().splitlines())
    ] == ["s0", "s0", "s1", "s1"]
    assert (output / "split.json").read_bytes() == source_split.read_bytes()
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert manifest["validation"]["endpoint_calibration"]["passed"]
    assert manifest["validation"]["impossible_near_end_labels"]["passed"]


def test_merge_rejects_tampered_shard(tmp_path):
    rollouts, states = _source_rows()
    source_rollouts = tmp_path / "rollouts.jsonl"
    source_states = tmp_path / "states.jsonl"
    source_split = tmp_path / "split.json"
    _write_jsonl(source_rollouts, rollouts[:1])
    _write_jsonl(source_states, states[:1])
    source_split.write_text("{}\n")
    shard = tmp_path / "shard"
    _make_shard(shard, rollouts[0], states[0], 1.0)
    with (shard / "mc_labels_relabel.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_shards(
            source_rollouts=source_rollouts,
            source_states=source_states,
            source_split=source_split,
            shard_dirs=[shard],
            output_dir=tmp_path / "merged",
            continuations_per_state=2,
            expected_rollouts=1,
            expected_states=1,
        )
