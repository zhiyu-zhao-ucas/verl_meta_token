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
"""Source-compatible backbone and biased scalar head, without an LM head."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM


class DeltaScalarModel(nn.Module):
    """Causal transformer backbone followed by the source FP32 scalar head.

    The state-dict layout intentionally matches ``LLMTokenScalarModel.module()``:
    ``backbone.*`` and ``scalar_head.*``. The scalar at sequence position ``i``
    reads the hidden state after consuming token ``i``; callers do not shift it
    to the preceding next-token-logit position.
    """

    def __init__(self, backbone, value_layer=-1):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.value_layer = int(value_layer)
        self.scalar_head = nn.Linear(self.config.hidden_size, 1)  # source head is FP32
        self.max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
        if self.max_position_embeddings is not None:
            self.max_position_embeddings = int(self.max_position_embeddings)
            if self.max_position_embeddings < 1:
                raise ValueError("backbone max_position_embeddings must be positive")
        self._no_split_modules = getattr(backbone, "_no_split_modules", None)
        for name in ("lm_head", "embed_out", "output_layer"):
            if isinstance(getattr(backbone, name, None), nn.Module):
                setattr(backbone, name, nn.Identity())
        count = self.config.num_hidden_layers + 1
        if not -count <= self.value_layer < count:
            raise ValueError(f"value_layer={value_layer} outside hidden_states [-{count}, {count - 1}]")

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        dtype="bfloat16",
        value_layer=-1,
        trust_remote_code=False,
        attn_implementation="sdpa",
        gradient_checkpointing=False,
    ):
        """Load a source-compatible scalar critic from a causal LM checkpoint."""
        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype {dtype!r}") from exc
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"Unsupported torch dtype {dtype!r}")
        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        model = cls(backbone, value_layer)
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model

    @classmethod
    def from_config(cls, config):
        return cls.from_pretrained(
            config.model_path,
            dtype=config.dtype,
            value_layer=config.value_layer,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attention_implementation,
            gradient_checkpointing=config.gradient_checkpointing,
        )

    def can_generate(self):
        return False

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] < 1:
            raise ValueError("input sequence must contain at least one token")
        if self.max_position_embeddings is not None and input_ids.shape[1] > self.max_position_embeddings:
            raise ValueError(
                f"Input length {input_ids.shape[1]} exceeds backbone max_position_embeddings "
                f"{self.max_position_embeddings}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same [batch, sequence] shape as input_ids")
        decoder = getattr(self.backbone, "get_decoder", None)
        transformer = decoder() if callable(decoder) else None
        if transformer is None:
            for name in ("model", "transformer", "gpt_neox", "bert", "roberta"):
                transformer = getattr(self.backbone, name, None)
                if transformer is not None:
                    break
        outputs = (transformer if transformer is not None else self.backbone)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=self.value_layer != -1 or transformer is None,
        )
        if self.value_layer != -1 or transformer is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise ValueError("Model outputs do not include hidden_states for value_layer selection")
            try:
                hidden = hidden_states[self.value_layer]
            except IndexError as exc:
                raise ValueError(
                    f"value_layer={self.value_layer} outside returned hidden_states ["
                    f"-{len(hidden_states)}, {len(hidden_states) - 1}]"
                ) from exc
        else:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                try:
                    hidden = outputs[0]
                except (IndexError, TypeError) as exc:
                    raise ValueError("Backbone outputs do not contain last_hidden_state") from exc
        return self.scalar_head(hidden.to(self.scalar_head.weight.dtype)).squeeze(-1)
