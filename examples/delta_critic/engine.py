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
"""V1 delta engine using FSDP2 execution, optimizer and checkpoint infrastructure."""

from contextlib import nullcontext

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from verl.utils import tensordict_utils as tu
from verl.utils.fsdp_utils import apply_fsdp2
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

from .scalar_model import DeltaScalarModel
from .training_config import ScalarConfig


def microbatch_padding_width(attention_mask):
    """Width this micro-batch actually needs, i.e. its longest real row.

    Collate builds one padded width for the whole global batch. Running every
    micro-batch at that width wastes a forward and backward column for every
    padding token; trimming to the micro-batch's own longest row removes that
    without changing any masked-out value.
    """
    if attention_mask.ndim != 2 or attention_mask.shape[0] < 1:
        raise ValueError(
            f"attention_mask must be a nonempty [batch, sequence] tensor, got {tuple(attention_mask.shape)}"
        )
    width = int(attention_mask.sum(-1).max().item())
    if width < 1:
        raise ValueError("Every micro-batch row is empty")
    return width


# Dense [batch, sequence] columns that share the padding width. A training batch
# carries all three; the frozen scoring batch carries only `target` (plus
# `loss_mask`), so a missing key is expected rather than an error.
PADDED_COLUMNS = ("target", "target_mask", "signal_mask")


def trim_dense_columns(batch, width):
    """Trim the padded columns of `batch` to `width`, skipping absent ones."""
    if width < 1:
        raise ValueError("width must be positive")
    for key in PADDED_COLUMNS:
        value = batch.get(key)
        if value is not None:
            batch[key] = value[:, :width]
    return batch


def resolve_autocast_dtype(autocast, model_dtype):
    """Map a ScalarConfig autocast name to the torch dtype used for the forward."""
    if autocast == "model_dtype":
        autocast = model_dtype
    if autocast == "none":
        return torch.float32
    try:
        return getattr(torch, autocast)
    except AttributeError as exc:
        raise ValueError(f"Unsupported autocast dtype {autocast!r}") from exc


@EngineRegistry.register(model_type="delta_scalar", backend="fsdp2", device="cuda")
class DeltaFSDPEngine(FSDPEngine):
    def _build_module(self):
        self.delta_config = ScalarConfig(**self.model_config.hf_config.delta_scalar_config)
        if (
            self.use_remove_padding
            or self.engine_config.ulysses_sequence_parallel_size != 1
            or self.model_config.use_fused_kernels
            or self.model_config.lora_rank
            or self.model_config.use_liger
            or self._qat_enabled
        ):
            raise ValueError("Delta engine requires padding, SP=1, no fused kernels/LoRA/Liger/QAT")
        module = DeltaScalarModel.from_config(self.delta_config)
        artifact = getattr(self.model_config.hf_config, "delta_initial_artifact", None)
        if artifact:
            from .checkpoint import load_weights, read_artifact

            read_artifact(artifact)
            load_weights(module, torch.load(f"{artifact}/model.pt", map_location="cpu", weights_only=True))
        return module

    def _build_fsdp_module(self, module):
        # Never replace heterogeneous source parameter dtypes with a uniform cast.
        # Head has its own FSDP group because its FP32 parameters differ from BF16 backbone.
        expected_backbone_dtype = getattr(torch, self.delta_config.dtype)
        if module.scalar_head.weight.dtype != torch.float32 or module.scalar_head.bias.dtype != torch.float32:
            raise ValueError("The biased scalar head must remain FP32 before FSDP2 wrapping")
        backbone_parameter = next(module.backbone.parameters())
        if backbone_parameter.dtype != expected_backbone_dtype:
            raise ValueError(
                f"Backbone dtype {backbone_parameter.dtype} does not match configured {expected_backbone_dtype}"
            )
        module.to(torch.cuda.current_device())
        kwargs = dict(
            mesh=self.device_mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32, cast_forward_inputs=False),
            reshard_after_forward=self.engine_config.reshard_after_forward,
        )
        fully_shard(module.scalar_head, **kwargs)
        apply_fsdp2(module, kwargs, self.engine_config)
        if module.scalar_head.weight.dtype != torch.float32 or module.scalar_head.bias.dtype != torch.float32:
            raise RuntimeError("FSDP2 changed the scalar head parameter dtype; expected FP32 weight and bias")
        self._autocast_dtype = resolve_autocast_dtype(self.delta_config.training_autocast, self.delta_config.dtype)
        # Frozen scoring must use the scoring semantics, not the training ones:
        # the standalone FrozenDeltaWorker scores under `scoring_autocast`
        # (default "none"), so reusing the training autocast here would make the
        # two scoring paths disagree numerically.
        self._scoring_autocast_dtype = resolve_autocast_dtype(
            self.delta_config.scoring_autocast, self.delta_config.dtype
        )
        self.scaler = None
        return module

    def _build_lr_scheduler(self, optimizer):
        # Source scalar trainer: linear decay, zero warmup.
        total = self.optimizer_config.total_training_steps
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / total))

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        sample_valid = data.get("sample_valid_mask")
        if sample_valid is None:
            sample_valid = torch.ones(data.shape[0], dtype=torch.float32, device=data.device)
        count = sample_valid.float().sum().to(torch.cuda.current_device())
        torch.distributed.all_reduce(count, group=self.get_data_parallel_group())
        if not forward_only and count.item() == 0:
            raise ValueError("Update has no real samples")
        tu.assign_non_tensor(data, valid_sample_count=count.item())
        return super().forward_backward_batch(data, loss_function, forward_only)

    def forward_step(self, micro_batch, loss_function, forward_only):
        micro_batch = micro_batch.to(torch.cuda.current_device())
        # Collate right-pads the whole global batch to its longest row, so every
        # micro-batch would otherwise run at that width even though only one row
        # is padded to it. Trim to this micro-batch's own longest row instead:
        # at microbatch=1 that leaves no padding at all, which is what the source
        # trainer's batch_size=1 collate produced. Predicted values are unchanged
        # because padding is masked out of attention, but the forward and backward
        # no longer run over ~1.7x wasted columns on this data.
        width = microbatch_padding_width(micro_batch["attention_mask"])
        attention_mask = micro_batch["attention_mask"][:, :width]
        trim_dense_columns(micro_batch, width)
        ids = torch.nested.to_padded_tensor(
            micro_batch["input_ids"], tu.get(micro_batch, "pad_token_id"), output_size=(len(micro_batch), width)
        )
        autocast_dtype = self._scoring_autocast_dtype if forward_only else self._autocast_dtype
        autocast = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype != torch.float32 else nullcontext()
        # The base engine already wraps this call in `torch.no_grad()` when
        # forward_only. Do NOT add `torch.inference_mode()` here: FSDP2's
        # pre-forward hook calls `_unsafe_preserve_version_counter` on tensors it
        # all-gathers, which raises on inference tensors.
        with autocast:
            pred = self.module(input_ids=ids, attention_mask=attention_mask)
            model_output = {"delta_scalar": pred}
            if loss_function is None:
                loss, metrics = pred.new_zeros(()), {}
            else:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
        return loss, {"model_output": {"delta_scalar": pred.detach()}, "loss": loss.detach().item(), "metrics": metrics}
