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
"""Optional source differential on an actual legacy artifact and its original window.

The source repository is used only by this verification command, never at runtime.
"""

import argparse
import ast
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .checkpoint import file_hash
from .legacy_adapter import adapt_legacy_rows
from .score import FrozenDeltaWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source_root) / "delta_value_llm_exp"
    worker = FrozenDeltaWorker(args.artifact, microbatch=2)
    if worker.metadata.get("source_sha256") != file_hash(args.legacy_checkpoint):
        raise ValueError("Artifact does not correspond to the supplied legacy checkpoint")
    spec = importlib.util.spec_from_file_location("source_scalar", source / "modeling_token_scalar.py")
    modeling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling)
    model = modeling.LLMTokenScalarModel(
        worker.config.model_path, worker.config.trust_remote_code, worker.config.dtype, worker.config.value_layer
    ).module()
    state = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state["model"])
    model.cuda().eval().requires_grad_(False)
    path = source / "09_train_policy_grpo_offline.py"
    names = {
        "_critic_group_mc_labels",
        "_critic_build_selected_eval_rows",
        "_critic_collate_selected_batch",
        "_predict_critic_selected_states",
    }
    tree = ast.parse(path.read_text())
    namespace = {"Any": Any, "defaultdict": defaultdict}
    selected = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[]
    )
    exec(compile(selected, str(path), "exec"), namespace)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(worker.config.tokenizer_path or worker.config.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = tokenizer.encode("Compute 1 + 1 and explain the result.")
    response = (tokenizer.encode(" Let us compute the sum carefully.") * 400)[:2200]
    rollouts = [
        {"id": "long", "prompt_token_ids": prompt, "response_token_ids": response, "terminal_reward": 1.0},
        {"id": "short", "prompt_token_ids": prompt, "response_token_ids": response[:17], "terminal_reward": 0.0},
    ]
    labels = [
        {"rollout_id": row["id"], "token_index": t, "v_prefix": 0.5}
        for row in rollouts
        for t in (0, len(row["response_token_ids"]) - 1)
    ]
    examples, _ = adapt_legacy_rows(rollouts, labels)
    source_rows = namespace["_critic_build_selected_eval_rows"](rollouts, labels)
    reference = namespace["_predict_critic_selected_states"](
        model,
        tokenizer,
        "cuda",
        source_rows,
        2,
        worker.config.max_length,
        target_type="delta",
        critic_decode_mode="expectation",
        critic_atoms=None,
    )
    actual = worker.score(examples)
    values = [
        row["delta_pred_normalized"][state.token_index]
        for row, example in zip(actual, examples, strict=True)
        for state in example.states
    ]
    torch.testing.assert_close(torch.tensor(values), torch.tensor(reference), atol=1e-6, rtol=1e-6)
    norm = worker.metadata["normalization"]
    raw_ref = [p * norm["std"] + norm["mean"] if norm["enabled"] else p for p in reference]
    raw = [row["delta_pred_raw"][s.token_index] for row, e in zip(actual, examples, strict=True) for s in e.states]
    torch.testing.assert_close(torch.tensor(raw), torch.tensor(raw_ref), atol=1e-7, rtol=1e-6)
    report = {
        "source_checkpoint_sha256": worker.metadata["source_sha256"],
        "window": worker.config.max_length,
        "window_policy": worker.config.window_policy,
        "autocast": worker.config.scoring_autocast,
        "normalized_max_abs": max(abs(a - b) for a, b in zip(values, reference, strict=True)),
        "raw_max_abs": max(abs(a - b) for a, b in zip(raw, raw_ref, strict=True)),
        "selected_positions": [(r["rollout_id"], r["token_index"]) for r in source_rows],
        "source_functions": sorted(names),
        "atol": 1e-6,
        "rtol": 1e-6,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
