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
"""Tiny real two-round V1 generation/MC/critic/policy acceptance, on one GPU.

Uses a randomly initialized Qwen3 and numeric toy prompts. This tests execution,
checkpoint chaining and fixed-reference identity, not mathematical accuracy.
"""

import argparse
import json
from pathlib import Path

import yaml

from .policy_iterations import artifact_digest, load_settings, run


def prepare(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    model = directory / "tiny"
    torch.manual_seed(3)
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=128,
            tie_word_embeddings=False,
            eos_token_id=1,
        )
    ).save_pretrained(model)
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="0",
        pad_token="0",
        eos_token="1",
    ).save_pretrained(model)
    prompts = directory / "prompts.jsonl"
    prompts.write_text(
        "".join(json.dumps({"id": f"p{i}", "prompt": str(i % 20 + 2), "answer": "2"}) + "\n" for i in range(16))
    )
    sampling = yaml.safe_load(Path("examples/delta_critic/config_sampling_qwen3_8b.yaml").read_text())
    sampling["model"].update(actor_model=str(model), max_model_len=128, gpu_memory_utilization=0.05)
    sampling["reward"] = {"method": "exact_or_numeric"}
    sampling["smoke"].update(train_prompts=16, states_per_response=2)
    sampling["rollout"].update(max_tokens=8, top_k_logprobs=5, stop=[])
    sampling["mc"].update(continuations_per_state=2, max_continuation_tokens=8)
    sampling["state_selection"].update(min_token_gap=1)
    critic = {
        "model_path": str(model),
        "dtype": "float32",
        "objective": "local_td0",
        "loss_type": "nonzero_balanced_mse",
        "loss_mask": "response",
        "target_normalization": "none",
        "max_length": 128,
        "learning_rate": 1e-3,
        "gradient_checkpointing": False,
    }
    policy = {
        "model_path": str(model),
        "reference_model_path": str(model),
        "mode": "online_ppo",
        "dtype": "float32",
        "rl_epochs": 1,
        "max_length": 128,
        "learning_rate": 1e-3,
        "gradient_checkpointing": False,
        "save_every_steps": 1,
    }
    settings = {
        "rounds": 2,
        "critic_steps": 1,
        "policy_steps": 1,
        "critic_global_batch": 4,
        "policy_global_batch": 4,
        "microbatch": 1,
        "nproc": 1,
        "initial_actor": str(model),
        "reference_model": str(model),
        "eval_fraction": 0.25,
        "prompts_jsonl": str(prompts),
        "prompt_source": "value_model",
        "rollout_concurrency": 2,
        "mc_concurrency": 2,
        "gpu_memory_utilization": 0.05,
    }
    for name, config in (("sampling", sampling), ("critic", critic), ("policy", policy)):
        path = directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        settings[f"{name}_config"] = str(path)
    config_path = directory / "iterations.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    config_path = directory / "iterations.yaml" if args.resume else prepare(directory)
    if args.prepare_only:
        print(config_path)
        return
    settings = load_settings(config_path)
    reference_before = artifact_digest(directory / "tiny")
    result = run(settings, directory / "run", resume=args.resume)
    assert artifact_digest(directory / "tiny") == reference_before, "Reference weights changed"
    first = json.loads((directory / "run/round_000/manifest.json").read_text())
    second = json.loads((directory / "run/round_001/manifest.json").read_text())
    assert first["actor_output"] == second["actor_input"]
    assert first["critic_output"] == second["critic_input"]
    assert first["reference_model"] == second["reference_model"] == settings["reference_model"]
    # A completed command is not enough: both policy updates must actually
    # change parameters, and round two must sample from round one's export.
    from safetensors.torch import load_file

    def weights(path):
        result = {}
        for shard in sorted(Path(path).glob("*.safetensors")):
            result.update(load_file(str(shard)))
        if not result:
            raise AssertionError(f"No exported actor weights: {path}")
        return result

    # Critic artifacts must remain byte-identical while policy scoring/updating
    # and the following round's warm start read them.
    for index, manifest in enumerate((first, second)):
        marker = json.loads((directory / f"run/round_{index:03d}/.critic.done.json").read_text())
        artifact = manifest["critic_output"]
        assert artifact_digest(artifact) == marker["outputs"][artifact], "Frozen critic artifact changed"
    changes = []
    for manifest in (first, second):
        before, after = weights(manifest["actor_input"]), weights(manifest["actor_output"])
        assert before.keys() == after.keys(), "Export changed actor parameter names"
        maximum = max((before[key].float() - after[key].float()).abs().max().item() for key in before)
        assert maximum > 0, "Policy update did not change actor parameters"
        changes.append(maximum)
    sampling = yaml.safe_load((directory / "run/round_001/sampling.yaml").read_text())
    assert sampling["model"]["actor_model"] == first["actor_output"]
    result["actor_max_abs_changes"] = changes
    result["fixed_reference_sha256"] = reference_before
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
