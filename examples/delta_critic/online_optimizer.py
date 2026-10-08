# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU FP32 AdamW masters and accumulation for a BF16 inference model."""

import torch


class FP32MasterAdamW:
    """Offload each completed parameter gradient instead of a full FP32 GPU copy.

    Masters, gradients and Adam moments remain FP32 on CPU. The model retains
    its original parameter dtypes. Small updates accumulate in the masters even
    when one update is below a BF16 representable increment.
    """

    def __init__(self, model, *, learning_rate, gradient_clip=1.0):
        self.named_parameters = list(model.named_parameters())
        self.masters = [
            torch.nn.Parameter(p.detach().to(device="cpu", dtype=torch.float32).clone())
            for _, p in self.named_parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.masters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False
        )
        self.gradient_clip = float(gradient_clip)
        self.hooks = []
        for (_, parameter), master in zip(self.named_parameters, self.masters, strict=True):
            was_trainable = parameter.requires_grad
            parameter.requires_grad_(True)

            def collect(param, dest=master):
                if param.grad is None:
                    return
                grad = param.grad.detach().to(device="cpu", dtype=torch.float32)
                if dest.grad is None:
                    dest.grad = grad.clone() if grad.data_ptr() == param.grad.data_ptr() else grad
                else:
                    dest.grad.add_(grad)
                param.grad = None

            self.hooks.append(parameter.register_post_accumulate_grad_hook(collect))
            parameter.requires_grad_(was_trainable)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for _, parameter in self.named_parameters:
            parameter.grad = None

    def gradient_norm(self):
        norms = [torch.linalg.vector_norm(master.grad, 2) for master in self.masters if master.grad is not None]
        return float(torch.linalg.vector_norm(torch.stack(norms), 2)) if norms else 0.0

    @torch.no_grad()
    def step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.masters, self.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        squared = {"backbone": 0.0, "head": 0.0}
        changed = {"backbone": 0, "head": 0}
        for (name, parameter), master in zip(self.named_parameters, self.masters, strict=True):
            category = "head" if "scalar_head" in name else "backbone"
            # Compare actual inference-dtype weights, before copying them back.
            before = parameter.detach().to(device="cpu")
            after = master.detach().to(dtype=parameter.dtype)
            diff = after.float() - before.float()
            squared[category] += float(diff.double().square().sum())
            changed[category] += int(torch.count_nonzero(diff))
            parameter.copy_(after)
        return {
            "grad_norm": float(norm),
            **{f"{name}_change_norm": value**0.5 for name, value in squared.items()},
            **{f"{name}_changed_elements": value for name, value in changed.items()},
        }

    def state_dict(self):
        return {
            "format": "fp32_master_cpu_v1",
            "names": [name for name, _ in self.named_parameters],
            "masters": [master.detach() for master in self.masters],
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("format") != "fp32_master_cpu_v1" or state.get("names") != [
            name for name, _ in self.named_parameters
        ]:
            raise ValueError("FP32 master optimizer parameter identity mismatch")
        with torch.no_grad():
            for master, saved in zip(self.masters, state["masters"], strict=True):
                master.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])
