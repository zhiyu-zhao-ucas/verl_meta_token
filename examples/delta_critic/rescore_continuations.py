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
"""Add collection-actor top-k probabilities to existing Qwen3 continuations.

This writes a new JSONL and leaves the collected tokens, rewards and labels
unchanged. Use the weights of the actor that collected these continuations.
The LM head is evaluated in chunks so long rows do not allocate sequence-wide
vocabulary logits. This utility does not generate tokens or train either model.
"""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import torch


def continuation_top_logprobs(model, prefix_ids, response_ids, *, top_k=20, logits_chunk_size=64):
    """Predict each stored response token from the prefix before that token."""
    if not prefix_ids or not response_ids:
        raise ValueError("Scoring requires a nonempty conditioning prefix and continuation")
    if top_k < 1 or logits_chunk_size < 1:
        raise ValueError("top_k and logits_chunk_size must be positive")
    if model.config.model_type != "qwen3":
        raise ValueError("Chunked LM-head scoring currently supports Qwen3 decoder-only actors")
    sequence = [*prefix_ids, *response_ids[:-1]]
    if len(sequence) > model.config.max_position_embeddings:
        raise ValueError("Continuation exceeds the actor context; rescoring never truncates")
    device = next(model.parameters()).device
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.base_model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        # The last conditioning hidden state predicts response[0]. A scalar
        # critic's token-inclusive position convention does not apply here.
        hidden = output.last_hidden_state[0, len(prefix_ids) - 1 :]
        if hidden.shape[0] != len(response_ids):
            raise ValueError("Actor hidden states do not align with continuation tokens")
        rows = []
        head = model.get_output_embeddings()
        for start in range(0, len(response_ids), logits_chunk_size):
            # The collector uses vLLM's default raw_logprobs, computed before
            # sampling temperature, top-p/top-k, and repetition penalties.
            logits = head(hidden[start : start + logits_chunk_size]).float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite actor logits")
            log_probs = logits.log_softmax(-1)
            values, indices = log_probs.topk(min(top_k, log_probs.shape[-1]), dim=-1)
            for token_ids, log_values in zip(indices.cpu().tolist(), values.cpu().tolist(), strict=True):
                rows.append(
                    [
                        {"token_id": token_id, "logprob": logprob, "prob": math.exp(logprob)}
                        for token_id, logprob in zip(token_ids, log_values, strict=True)
                    ]
                )
    return rows


def write_rescored_rows(continuations, labels, output, model, *, model_path, actor_version, top_k=20, chunk_size=64):
    """Stream a new file, publishing it only after every row has been scored."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    by_state = {}
    by_token = {}
    with Path(labels).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            label = json.loads(line)
            if label.get("state_id") is not None:
                key = str(label["state_id"])
                if key in by_state:
                    raise ValueError(f"Duplicate state_id in labels: {key}")
                by_state[key] = label
            by_token[(str(label["rollout_id"]), int(label["token_index"]))] = label
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False) as dest:
            temporary = Path(dest.name)
            with Path(continuations).open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if str(row.get("actor_version")) != str(actor_version):
                        raise ValueError("Continuation actor_version differs from the explicitly selected actor")
                    label = by_state.get(str(row.get("state_id")))
                    if label is None:
                        label = by_token.get((str(row["rollout_id"]), int(row["token_index"])))
                    if label is None:
                        raise ValueError(f"Missing prefix label for continuation {row.get('continuation_id')}")
                    prompt = label.get("prompt_token_ids") or row.get("prompt_token_ids") or []
                    prefix = label.get("prefix_response_token_ids") or row.get("prefix_response_token_ids") or []
                    response = row.get("continuation_token_ids") or []
                    if response:
                        row["delta_top_logprobs"] = continuation_top_logprobs(
                            model,
                            [*prompt, *prefix],
                            response,
                            top_k=top_k,
                            logits_chunk_size=chunk_size,
                        )
                    else:
                        row["delta_top_logprobs"] = []
                    row["delta_top_logprobs_provenance"] = {
                        "model_path": str(model_path),
                        "actor_version": str(actor_version),
                        "method": "teacher_forcing",
                        "logprobs_mode": "raw_logprobs",
                        "top_k": top_k,
                    }
                    dest.write(json.dumps(row) + "\n")
                    count += 1
        # An atomic hard-link refuses an existing destination even if another
        # writer created it after the initial check. Both paths share a filesystem.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "labels", "continuations", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--actor-version", help="Collection actor identity; defaults to --model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--logits-chunk-size", type=int, default=64)
    args = parser.parse_args()
    if args.top_k < 1 or args.logits_chunk_size < 1:
        parser.error("top-k and logits-chunk-size must be positive")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new file")
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(args.device)
        .eval()
    )
    count = write_rescored_rows(
        args.continuations,
        args.labels,
        args.output,
        model,
        model_path=args.model,
        actor_version=args.actor_version or args.model,
        top_k=args.top_k,
        chunk_size=args.logits_chunk_size,
    )
    print(json.dumps({"rows": count, "output": args.output}))


if __name__ == "__main__":
    main()
