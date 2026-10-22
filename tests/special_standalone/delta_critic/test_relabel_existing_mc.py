# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0

import json
from dataclasses import replace

import pytest

from examples.delta_critic.mc_labeling import MCConfig
from examples.delta_critic.relabel_existing_mc import (
    _canonicalize_states,
    _check_visible_gpu,
    _continuation_row,
    _recover_labels,
    _rescore_rollouts,
    _validate_inputs,
    _validate_sampling_contract,
)


def _fixture():
    rollout = {
        "id": "r",
        "prompt_id": "p",
        "split": "train",
        "prompt_token_ids": [1],
        "response_token_ids": [2, 3],
        "policy_token_mask": [1, 1],
        "actor_version": "actor",
        "response": "Answer: 2",
        "gold_answer": "2",
        "terminal_reward": 0.0,
    }
    state = {
        "state_id": "r:t1",
        "rollout_id": "r",
        "prompt_id": "p",
        "split": "train",
        "token_index": 1,
        "prompt_token_ids": [1],
        "prefix_response_token_ids": [2],
        "actor_version": "actor",
    }
    config = MCConfig(
        mode="prefix_only",
        continuations_per_state=2,
        sampling_config={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 8},
        actor_version="actor",
        continuation_skip_special_tokens=True,
        save_individual_rewards=True,
        save_continuations=True,
    )
    return rollout, state, config


def test_relabel_inputs_preserve_existing_token_prefixes():
    rollout, state, config = _fixture()
    rollout_map, grouped = _validate_inputs([rollout], [state], config)
    assert rollout_map == {"r": rollout}
    assert grouped == {"r": [state]}
    with pytest.raises(ValueError, match="prefix_response"):
        _validate_inputs([rollout], [{**state, "prefix_response_token_ids": []}], config)


def test_legacy_mc_labels_are_reduced_to_pure_state_rows():
    rollout, state, _ = _fixture()
    rollout["tokens"] = [{"top1_prob": 0.8}, {"top1_prob": 0.7}]
    legacy_label = {**state, "v_prefix": 0.0, "mc_num_samples": 16, "prefix_continuation_rewards": [0.0] * 16}
    canonical = _canonicalize_states([rollout], [legacy_label])
    assert canonical == [
        {
            "state_id": state["state_id"],
            "rollout_id": "r",
            "split": "train",
            "prompt_id": "p",
            "response_index": None,
            "token_index": 1,
            "selection_rank": 0,
            "selection_score": pytest.approx(0.3),
            "selection_strategy": "legacy_mc_label_positions",
            "prompt_token_ids": [1],
            "prefix_response_token_ids": [2],
            "sampled_token_id": 3,
            "candidates": [],
            "candidate_mass": 0.0,
            "actor_version": "actor",
            "gold_answer": "2",
        }
    ]
    assert "v_prefix" not in canonical[0]


def test_terminal_rewards_are_rescored_and_legacy_values_retained():
    rollout, _, _ = _fixture()
    rescored = _rescore_rollouts([rollout], {"method": "exact_or_numeric"})
    assert rescored[0]["legacy_terminal_reward"] == 0.0
    assert rescored[0]["terminal_reward"] == 1.0
    assert rollout["terminal_reward"] == 0.0


def test_relabel_resume_keeps_only_fully_committed_states(tmp_path):
    _, state, config = _fixture()
    temporary = {
        name: tmp_path / f".{name}.jsonl.incomplete" for name in ("rollouts", "states", "mc_labels", "continuations")
    }
    temporary["mc_labels"].write_text(
        json.dumps({"state_id": state["state_id"], "mc_requested_samples": 2, "mc_num_samples": 1}) + "\n"
    )
    temporary["continuations"].write_text(json.dumps({"state_id": state["state_id"], "continuation_index": 0}) + "\n")
    labels, continuations = _recover_labels(
        temporary,
        [state],
        continuations_per_state=config.continuations_per_state,
    )
    assert labels == {}
    assert continuations == []

    temporary["mc_labels"].write_text(
        json.dumps({"state_id": state["state_id"], "mc_requested_samples": 2, "mc_num_samples": 1}) + "\n"
    )
    temporary["continuations"].write_text(
        "".join(json.dumps({"state_id": state["state_id"], "continuation_index": index}) + "\n" for index in range(2))
    )
    labels, continuations = _recover_labels(temporary, [state], continuations_per_state=2)
    assert set(labels) == {state["state_id"]}
    assert len(continuations) == 2


def test_relabel_requires_rollout_sampling_contract_and_records_effective_parameters():
    rollout, state, config = _fixture()
    with pytest.raises(ValueError, match="match the rollout sampler"):
        _validate_sampling_contract(config)
    config = replace(
        config,
        sampling_config={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 30720,
            "logprobs": True,
        },
    )
    _validate_sampling_contract(config)
    label = {
        **state,
        "mc_sampling_config": {**config.sampling_config, "max_total_tokens": 32768, "effective_max_tokens": 42},
        "mc_effective_sampling_config": {**config.sampling_config, "max_tokens": 42},
    }
    continuation = {
        "continuation_index": 0,
        "token_ids": [7],
        "text": "x",
        "reward": 0.0,
        "score_status": "incorrect",
        "score_attempts": 1,
        "score_timeout_seconds": 5,
        "sampled_logprobs": [-0.2],
    }
    row = _continuation_row(rollout, label, continuation, config)
    assert row["sampling_config"]["max_tokens"] == 42
    assert row["requested_sampling_config"]["max_tokens"] == 30720
    assert row["prompt_transport"] == "token_ids"
    assert "delta_top_logprobs" not in row["sampling_config"]


def test_relabel_gpu_guard(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="explicitly"):
        _check_visible_gpu()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    with pytest.raises(RuntimeError, match="unavailable"):
        _check_visible_gpu()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    _check_visible_gpu()
