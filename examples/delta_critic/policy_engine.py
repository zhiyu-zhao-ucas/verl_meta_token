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
"""FSDP2 causal actor engine used by delta policy updates."""

import torch

from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@EngineRegistry.register(model_type="delta_policy", backend="fsdp2", device="cuda")
class DeltaPolicyFSDPEngine(FSDPEngineWithLMHead):
    """Use verl's causal-LM forward/logprob path with a linear source schedule."""

    def _build_module(self):
        # The registry needs a distinct model type for this custom loss, while
        # the standard FSDP loader only accepts language_model/value_model.
        previous = self.model_config.model_type
        self.model_config.model_type = "language_model"
        try:
            return super()._build_module()
        finally:
            self.model_config.model_type = previous

    def _build_lr_scheduler(self, optimizer):
        total_steps = max(1, int(self.optimizer_config.total_training_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / total_steps))
