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
"""Exercise real math parsing through the trainer and MC scoring paths."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from unittest.mock import Mock

import pytest

from examples.delta_critic import reward, sampling_io
from examples.delta_critic.reward import compute_score


@pytest.mark.parametrize(
    "response,gold,expected",
    [
        ("Answer: 7927", "7927", 1.0),
        ("Answer: 7926", "7927", 0.0),
        (r"Answer: $\frac{1}{2}$", "0.5", 1.0),
        (r"The answer is \boxed{\sqrt{4}}.", "2", 1.0),
        (r"The answer is \boxed{7927}.", "7927", 1.0),
        (r"\boxed{\frac{3}{2}}", r"\frac{1}{2}", 0.0),
        (r"\boxed{3/2}", "1/2", 0.0),
        (r"\boxed{1+2}", "2", 0.0),
        (r"\boxed{\frac{1}{2}}", "2", 0.0),
        (r"\boxed{2.0000001}", "2", 1.0),
        (r"\boxed{5e-1}", "0.5", 0.0),
        (r"\boxed{X}", "x", 1.0),  # math-verify's LaTeX parser normalizes symbol case.
        (r"\boxed{1 2}", "12", 0.0),
        (r"\boxed{100000000000000000001}", "100000000000000000000", 0.0),
        (r"\boxed{2k + 2}", "2k+2", 1.0),
        (r"\boxed{3k + 2}", "2k+2", 0.0),
        (r"\boxed{11\sqrt{2}}", r"11\sqrt2", 1.0),
        (r"\boxed{12\sqrt{2}}", r"11\sqrt2", 0.0),
        (r"\boxed{6 + 9i}", "6+9i", 1.0),
        (r"\boxed{(1, 2, 3)}", "(1,2,3)", 1.0),
        (r"\boxed{(4, 5, 3)}", "(1,2,3)", 0.0),
        (
            r"\boxed{\begin{pmatrix}-\frac{1}{3}\\\frac{2}{3}\\\frac{5}{3}\end{pmatrix}}",
            r"\begin{pmatrix}-1/3\\2/3\\5/3\end{pmatrix}",
            1.0,
        ),
        (
            r"\boxed{\begin{pmatrix}\frac{1}{3}\\\frac{2}{3}\\\frac{5}{3}\end{pmatrix}}",
            r"\begin{pmatrix}-1/3\\2/3\\5/3\end{pmatrix}",
            0.0,
        ),
        ("Answer:\n220", "220", 1.0),
        ("Answer:\nNo", "No", 1.0),
        ("Answer:\nYes", "No", 0.0),
        ("Answer: no solution", "no solution", 1.0),
        ("The answer is 16.", "16", 1.0),
        ("\\boxed{-2+7i}\nThe answer is: -2+7i", "-2 + 7i", 1.0),
        ("**Final Answer:**\n\\[\n2k+2\n\\]", "2k+2", 1.0),
        ("Answer: 2\nAnswer: 3", "2", 0.0),
        ("Answer: 2\nAnswer:\n", "2", 0.0),
        ("\\boxed{2}\nFinal answer:\n", "2", 0.0),
        ("\\boxed{1}\nAnswer: 2", "1", 0.0),
        ("\\boxed{2}\nAnswer: 1", "1", 1.0),
        ("Answer: 2\n\\boxed{", "2", 0.0),
        ("Answer:\nWe first compute 2.\nThen check our reasoning.", "2", 0.0),
        ("1. 1. 1. 1. 1. 1.", "1", 0.0),
        ("There is no answer.", "7927", 0.0),
    ],
)
def test_math_reward_matches_thread_and_mc(response, gold, expected):
    pytest.importorskip("math_verify")
    sampling_io._parse_math.cache_clear()
    reward_config = {
        "method": "math_verify",
        "fallback_on_import_error": False,
        "math_verify_require_final_answer": True,
    }

    async def score_online():
        loop = asyncio.get_running_loop()
        # This is how the reward manager invokes a synchronous custom scorer.
        trainer_reward = await loop.run_in_executor(
            None, partial(compute_score, "math_dapo", response, gold, math_verify_require_final_answer=True)
        )
        mc_rewards = await sampling_io.score_texts_async([response], {"gold_answer": gold}, reward_config=reward_config)
        return trainer_reward, mc_rewards

    # Run the worker thread first, so a main-thread parse cache cannot mask it.
    trainer_reward, mc_rewards = asyncio.run(score_online())
    assert trainer_reward == {"score": expected, "acc": expected}
    assert mc_rewards == [expected]
    assert sampling_io.score_text(response, gold, reward_config) == expected


def test_concurrent_first_rewards_share_one_process_pool(monkeypatch):
    pytest.importorskip("math_verify")
    # Start with no pool even when the parity tests have already created one.
    monkeypatch.setattr(sampling_io, "_reward_pool", None)
    try:
        with ThreadPoolExecutor(max_workers=8) as threads:
            futures = [threads.submit(compute_score, "math_dapo", f"Answer: {value}", str(value)) for value in range(8)]
            assert [future.result(timeout=60) for future in futures] == [{"score": 1.0, "acc": 1.0}] * 8
            pools = list(threads.map(lambda _: sampling_io._reward_executor(), range(8)))
            assert all(pool is pools[0] for pool in pools)
    finally:
        if sampling_io._reward_pool is not None:
            sampling_io._reward_pool.shutdown()


def test_missing_gold_and_unsupported_method():
    assert compute_score("math_dapo", "Answer: 7927", None) == {"score": 0.0, "acc": 0.0}
    with pytest.raises(ValueError, match="Unsupported reward method"):
        sampling_io.score_text("Answer: 7927", "7927", {"method": "unknown"})


def test_scoring_worker_failure_is_not_a_wrong_answer(monkeypatch):
    pool = Mock()
    pool.submit.return_value.result.side_effect = RuntimeError("scoring worker failed")
    monkeypatch.setattr(sampling_io, "_reward_pool", pool)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(compute_score, "math_dapo", "Answer: 7927", "7927")
        with pytest.raises(RuntimeError, match="scoring worker failed"):
            future.result(timeout=5)


def test_dynamically_loaded_reward_can_score_from_thread():
    pytest.importorskip("math_verify")
    from verl.utils.import_utils import load_extern_object

    # The dynamic module is not importable by a spawned child. Only the
    # module-level sampling_io scorer should cross the process boundary.
    scorer = load_extern_object(reward.__file__, "compute_score")
    with ThreadPoolExecutor(max_workers=1) as threads:
        result = threads.submit(scorer, "math_dapo", "Answer: 7927", "7927").result(timeout=60)
    assert result == {"score": 1.0, "acc": 1.0}


@pytest.mark.parametrize("require_final,expected", [(False, 1.0), (True, 0.0)])
def test_custom_reward_kwargs_match_mc_configuration(require_final, expected):
    pytest.importorskip("math_verify")
    from omegaconf import OmegaConf

    from verl.trainer.ppo.reward import get_custom_reward_fn

    scorer = get_custom_reward_fn(
        OmegaConf.create(
            {
                "reward": {
                    "custom_reward_function": {
                        "path": reward.__file__,
                        "name": "compute_score",
                        "reward_kwargs": {"math_verify_require_final_answer": require_final},
                    }
                }
            }
        )
    )
    response = "1. 1. 1. 1. 1. 1."
    with ThreadPoolExecutor(max_workers=1) as threads:
        result = threads.submit(scorer, "math_dapo", response, "1").result(timeout=60)
    assert result == {"score": expected, "acc": expected}
    mc_rewards = asyncio.run(
        sampling_io.score_texts_async(
            [response], {"gold_answer": "1"}, reward_config={"math_verify_require_final_answer": require_final}
        )
    )
    assert mc_rewards == [expected]


def test_final_answer_requirement_is_opt_in_for_existing_collectors():
    pytest.importorskip("math_verify")
    response = "1. 1. 1. 1. 1. 1."
    assert sampling_io.score_text(response, "1", {}) == 1.0
    assert compute_score("math_dapo", response, "1") == {"score": 1.0, "acc": 1.0}
    assert sampling_io.score_text(response, "1", {"math_verify_require_final_answer": True}) == 0.0
    assert compute_score("math_dapo", response, "1", math_verify_require_final_answer=True) == {
        "score": 0.0,
        "acc": 0.0,
    }


@pytest.mark.parametrize("gold", ["$2k+2$", r"\(2k+2\)", r"\boxed{2k+2}"])
def test_gold_already_delimited(gold):
    pytest.importorskip("math_verify")
    assert sampling_io.score_text(r"\boxed{2k + 2}", gold, {}) == 1.0
