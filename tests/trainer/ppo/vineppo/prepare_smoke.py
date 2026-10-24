# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
"""Create a tiny local model and toy verifier for the opt-in two-GPU integration smoke.

This fixture tests runtime behavior, not mathematical reasoning quality. All assets
stay in the explicitly supplied output directory; no model download is needed.
"""

import argparse
import string
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    chars = list(string.ascii_letters + string.digits + " .\n")
    vocab = {c: i for i, c in enumerate(["<pad>", "<unk>", "</s>"] + chars)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    hf.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    hf.save_pretrained(root / "model")
    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=None,
            tie_word_embeddings=False,
        )
    )
    model.save_pretrained(root / "model")
    pd.DataFrame(
        [
            dict(
                data_source="smoke",
                prompt=[{"role": "user", "content": "Q"}],
                reward_model={"ground_truth": "1", "style": "rule"},
                extra_info={"index": i},
            )
            for i in range(8)
        ]
    ).to_parquet(root / "data.parquet")
    signature = "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
    (root / "reward.py").write_text(signature + "    return float(bool(solution_str.strip()))\n")
    (root / "failing_reward.py").write_text(
        signature + "    raise RuntimeError('intentional VinePPO MC verifier failure')\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    prepare(parser.parse_args().output_dir)
