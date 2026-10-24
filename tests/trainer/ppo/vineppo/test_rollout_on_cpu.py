# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.vineppo.alignment import align_steps
from verl.trainer.ppo.vineppo.config import VinePPOConfig
from verl.trainer.ppo.vineppo.rollout import VinePPOCollector


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class CrossingTokenizer(CharacterTokenizer):
    # A generated token straddles the official newline boundary. Prefix re-encoding differs.
    def encode(self, text, **kwargs):
        return [999 if c == "\uffff" else ord(c) for c in text.replace("a\n", "\uffff")]

    def decode(self, ids, **kwargs):
        return "".join("a\n" if i == 999 else chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        ids = self.encode(text)
        offsets, start = [], 0
        for token in ids:
            end = start + (2 if token == 999 else 1)
            offsets.append((start, end))
            start = end
        return {"input_ids": ids, "offset_mapping": offsets}


def main_output(text="a\nb\nc", **extra):
    return SimpleNamespace(
        prompt_ids=[ord("Q")],
        response_ids=CharacterTokenizer().encode(text) + [0],
        response_mask=[1] * (len(text) + 1),
        num_turns=2,
        multi_modal_data=None,
        extra_fields=dict(global_steps=0, finish_reason="stop", **extra),
        reward_score=1.0,
    )


def test_cross_boundary_token_preserved_and_mc_text_prefix_reencoded():
    tokenizer = CrossingTokenizer()
    ids = tokenizer.encode("a\nb") + [0]
    aligned = align_steps(tokenizer, [ord("Q")], ids, "gsm8k")
    assert aligned.boundaries == [0, 1, 3]
    assert aligned.token_steps == [0, 1, 1]  # crossing token stays in step 0; EOS in last step
    assert aligned.prefix_ids(tokenizer, 1) == [ord("Q"), ord("a")]
    assert ids == [999, ord("b"), 0]


class ByteTokenizer:
    """Canonical tokens are characters; a sampler may also emit raw UTF-8 bytes (ids 1..255)."""

    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [1000 + ord(c) for c in text]

    def decode(self, ids, **kwargs):
        out, pending = [], b""
        for i in ids:
            if i < 256:
                pending += bytes([i])
                continue
            out.append(pending.decode("utf-8", errors="replace"))
            pending = b""
            out.append(chr(i - 1000))
        out.append(pending.decode("utf-8", errors="replace"))
        return "".join(out)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_unicode_whitespace_and_noncanonical_tokens_keep_actor_ids():
    tok = CharacterTokenizer()
    text = "α β\n下一步\n#### 2"
    aligned = align_steps(tok, [ord("Q")], tok.encode(text) + [0], "gsm8k")
    assert aligned.response == text and not aligned.noncanonical
    assert aligned.token_steps[-1] == len(aligned.boundaries) - 2
    # Sampled "a", "\n" where the canonical encoding is one crossing token.
    aligned = align_steps(CrossingTokenizer(), [ord("Q")], tok.encode("a\nb"), "gsm8k")
    assert aligned.noncanonical and aligned.boundaries == [0, 1, 3] and aligned.token_steps == [0, 1, 1]


def test_noncanonical_split_utf8_matches_canonical_character_starts():
    tok = ByteTokenizer()
    prompt = tok.encode("Q")
    canonical = tok.encode("x\n下y\n#### 2") + [0]
    expected = align_steps(tok, prompt, canonical, "gsm8k")
    split = canonical[:2] + list("下".encode()) + canonical[3:]
    aligned = align_steps(tok, prompt, split, "gsm8k")
    assert aligned.noncanonical and not expected.noncanonical
    assert aligned.boundaries == expected.boundaries
    # Each byte of the split character is assigned to the step containing that character.
    assert aligned.token_steps == expected.token_steps[:2] + [expected.token_steps[2]] * 3 + expected.token_steps[3:]


@pytest.mark.parametrize(
    ("text", "splitter"),
    [(" ", "math"), ("Let $f(x)=\\", "math"), ("a\n#### 1\n#### 2", "gsm8k")],
)
def test_splitter_failure_falls_back_to_one_step(text, splitter):
    tok = CharacterTokenizer()
    aligned = align_steps(tok, [ord("Q")], tok.encode(text) + [0], splitter)
    assert aligned.split_fallback and aligned.boundaries == [0, len(text)]
    assert aligned.token_steps == [0] * (len(text) + 1)


def test_empty_response_is_root_only_state():
    aligned = align_steps(CharacterTokenizer(), [ord("Q")], [0], "math")
    assert (
        aligned.boundaries == [0, 0]
        and aligned.token_steps == [0]
        and aligned.prefix_ids(CharacterTokenizer(), 0) == [ord("Q")]
    )


def test_math_formula_protection_and_exact_reconstruction():
    tok = CharacterTokenizer()
    text = "First use $x=1.5$.\nThen compute \\[x+2=3.5\\]. Done."
    aligned = align_steps(tok, [ord("Q")], tok.encode(text), "math")
    assert "".join(text[a:b] for a, b in zip(aligned.boundaries, aligned.boundaries[1:], strict=False)) == text
    formula_start, formula_end = text.index("$"), text.index("$", text.index("$") + 1)
    assert not any(formula_start < b <= formula_end for b in aligned.boundaries)
    unbreakable = "x" * 150
    assert align_steps(tok, [ord("Q")], tok.encode(unbreakable), "math").boundaries == [0, 150]


def test_step_merging_keeps_official_boundaries():
    tok = CharacterTokenizer()
    text = (
        "The product can be written as:\n\n\\[\n\\frac{2 \\cdot 3}{1 \\cdot 5} \\cdot \\frac{2 \\cdot 4}{3 \\cdot 7}"
        "\n\\]\n\nThis product can be rewritten as:\n\n\\[\ny = 3\n\\]\n\nAnswer: 3"
    )
    ids = tok.encode(text) + [0]
    official = align_steps(tok, [ord("Q")], ids, "math")
    steps = lambda a: [a.response[x:y] for x, y in zip(a.boundaries, a.boundaries[1:], strict=False)]
    assert "\n\\]" in steps(official)

    closed = align_steps(tok, [ord("Q")], ids, "math", merge_closing_steps=True)
    assert set(closed.boundaries) < set(official.boundaries)
    assert "\n\\]" not in steps(closed) and "".join(steps(closed)) == text

    merged = align_steps(tok, [ord("Q")], ids, "math", merge_closing_steps=True, min_step_tokens=40)
    assert set(merged.boundaries) <= set(closed.boundaries) and len(merged.boundaries) < len(closed.boundaries)
    assert all(len(step) >= 40 for step in steps(merged)[:-1])
    assert merged.token_steps == sorted(merged.token_steps) and merged.token_steps[-1] == len(merged.boundaries) - 2
    assert len(merged.token_steps) == len(ids)


def test_step_merging_leaves_split_fallback_alone():
    aligned = align_steps(CharacterTokenizer(), [ord("Q")], [ord(c) for c in "Answer: 2"], "math", min_step_tokens=32)
    assert aligned.split_fallback and aligned.boundaries == [0, 9]
    with pytest.raises(ValueError, match="min_step_tokens"):
        VinePPOConfig(min_step_tokens=-1)


def test_official_gsm8k_state_cap_includes_root():
    assert VinePPOConfig(splitter="gsm8k").state_cap == 26
    assert VinePPOConfig(splitter="math").state_cap is None
    assert VinePPOConfig(max_step_for_value_estimation=0).state_cap == 1


def test_score_errors_default_to_incorrect():
    assert VinePPOConfig().score_error_reward == 0.0


def test_collection_horizon_exact_k_duplicates_cap_and_transient_retry():
    async def run():
        calls, attempts, scored = [], {}, []
        active = peak = 0

        async def generate(**kwargs):
            nonlocal active, peak
            calls.append(kwargs)
            key = kwargs["request_id"]
            attempts[key] = attempts.get(key, 0) + 1
            if len(attempts) == 1 and attempts[key] == 1:
                raise TimeoutError("transient")
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return SimpleNamespace(token_ids=[ord("!"), 0], extra_fields={"global_steps": 0, "finish_reason": "stop"})

        async def score(prompt, text, kwargs):
            scored.append(text)
            return 0.5

        config = VinePPOConfig(
            mc_samples=3,
            splitter="gsm8k",
            max_step_for_value_estimation=1,
            mc_max_new_tokens=10,
            concurrency=2,
            retry_delay=0,
        )
        collector = VinePPOCollector(
            config, CharacterTokenizer(), generate, score, context_length=11, sampling_params={}, seed=42
        )
        main = main_output()
        saved_ids = list(main.response_ids)
        fields = await collector.collect(main, expected_version=0, identity=["row", 0], score_kwargs={})
        assert len(calls) == 7 and len(scored) == 6  # repeated identical completions still count
        assert fields["vineppo_mc_counts"].tolist() == [3, 3, 0]
        assert fields["vineppo_values"].tolist() == [0.5, 0.5, 1.0]
        assert fields["vineppo_generated_tokens"] == 12 and peak <= 2
        assert {c["sampling_params"]["max_tokens"] for c in calls} == {10, 9}
        first = [c for c in calls if c["request_id"] == calls[0]["request_id"]]
        assert first[0] == first[1]  # same branch, seed, prefix, version
        assert len({c["sampling_params"]["seed"] for c in calls}) == 6
        assert main.response_ids == saved_ids

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["exhausted", "scoring", "version", "aborted", "nonfinite", "context"])
def test_failed_mc_never_becomes_zero_or_partial_success(failure):
    async def run():
        calls = 0

        async def generate(**kwargs):
            nonlocal calls
            calls += 1
            if failure == "exhausted":
                raise TimeoutError("exhausted")
            return SimpleNamespace(
                token_ids=[ord("!")],
                extra_fields={
                    "global_steps": 1 if failure == "version" else 0,
                    "finish_reason": "abort" if failure == "aborted" else "stop",
                },
            )

        async def score(*args):
            if failure == "scoring":
                raise RuntimeError("verifier bug")
            return float("nan") if failure == "nonfinite" else 0.0

        collector = VinePPOCollector(
            VinePPOConfig(mc_samples=1, concurrency=1, splitter="gsm8k", retry_delay=0, score_error_reward=None),
            CharacterTokenizer(),
            generate,
            score,
            context_length=1 if failure == "context" else 100,
            sampling_params={},
        )
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            await collector.collect(main_output("a"), expected_version=0, identity=[1, 0], score_kwargs={})
        assert calls == (3 if failure == "exhausted" else 0 if failure == "context" else 1)

    asyncio.run(run())


def test_verifier_error_is_graded_incorrect_like_upstream():
    async def run():
        async def generate(**kwargs):
            return SimpleNamespace(token_ids=[ord("!"), 0], extra_fields={"global_steps": 0, "finish_reason": "stop"})

        calls = 0

        async def score(*args):
            nonlocal calls
            calls += 1
            if calls % 2:
                raise RuntimeError("math-verify could not parse the nonempty gold answer")
            return 1.0

        collector = VinePPOCollector(
            VinePPOConfig(mc_samples=4, concurrency=1, splitter="gsm8k", score_error_reward=0.0),
            CharacterTokenizer(),
            generate,
            score,
            context_length=100,
            sampling_params={},
        )
        main = main_output("a")
        main.extra_fields["vineppo_score_error"] = True
        result = await collector.collect(main, expected_version=0, identity=[1, 0], score_kwargs={})
        assert result["vineppo_values"].tolist() == [0.5]
        assert result["vineppo_mc_score_errors"] == 2 and result["vineppo_main_score_error"]

    asyncio.run(run())


def test_length_reward_is_task_penalty_not_generation_failure():
    async def run():
        async def generate(**kwargs):
            return SimpleNamespace(token_ids=[ord("!")], extra_fields={"global_steps": 0, "finish_reason": "length"})

        async def score(*args):
            raise AssertionError("truncated MC must not invoke verifier")

        collector = VinePPOCollector(
            VinePPOConfig(mc_samples=2, splitter="gsm8k"),
            CharacterTokenizer(),
            generate,
            score,
            context_length=100,
            sampling_params={},
        )
        main = main_output("a")
        main.extra_fields["finish_reason"] = "length"
        result = await collector.collect(main, expected_version=0, identity=[1, 0], score_kwargs={})
        assert main.reward_score == 0 and torch.equal(result["vineppo_values"], torch.zeros(1))
        assert result["vineppo_mc_truncated"] == 2

    asyncio.run(run())
