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
"""Trainer reward entry point sharing the critic collector's math scorer."""

from examples.delta_critic.sampling_io import score_text


def compute_score(
    data_source, solution_str, ground_truth, extra_info=None, math_verify_require_final_answer=False, **kwargs
):
    """Use as reward.custom_reward_function with the synchronous reward API."""
    score = score_text(
        solution_str,
        None if ground_truth is None else str(ground_truth),
        {
            "method": "math_verify",
            "fallback_on_import_error": False,
            "math_verify_require_final_answer": math_verify_require_final_answer,
        },
    )
    return {"score": score, "acc": score}
