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

import pytest

from examples.delta_critic import online_critic, score
from examples.delta_critic.policy_online import create_frozen_delta_scorer, online_critic_updates_enabled


@pytest.mark.parametrize(
    "expansion,update,expected",
    [
        (True, {}, False),
        (False, {}, True),
        (True, {"enabled": True}, True),
        (False, {"enabled": False}, False),
        (True, {"enabled": False}, False),
    ],
)
def test_explicit_critic_update_switch_controls_worker(monkeypatch, expansion, update, expected):
    monkeypatch.setattr(score, "FrozenDeltaWorker", lambda *args, **kwargs: "frozen")
    monkeypatch.setattr(online_critic, "OnlineDeltaWorker", lambda *args, **kwargs: "online")
    policy = {
        "rollout_mode": "selected_prefix_mc",
        "critic_artifact": "unused",
        "expansion": {"enabled": expansion},
        "critic_update": update,
    }
    assert online_critic_updates_enabled(policy) is expected
    assert create_frozen_delta_scorer(policy) == ("online" if expected else "frozen")


def test_critic_update_switch_rejects_string_false():
    with pytest.raises(ValueError, match="must be a boolean"):
        online_critic_updates_enabled({"rollout_mode": "selected_prefix_mc", "critic_update": {"enabled": "false"}})
