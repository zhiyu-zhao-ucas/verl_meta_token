# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
"""Official text boundaries, mapped onto the actor's generated token IDs."""

import re
from bisect import bisect_right
from dataclasses import dataclass

from .gsm8k_steps import split_gsm8k
from .math_steps import split_solution_inplace


@dataclass(frozen=True)
class StepAlignment:
    query: str
    response: str
    boundaries: list[int]
    token_steps: list[int]
    # The official splitter rejected this text; the response is one step (upstream drops it).
    split_fallback: bool = False
    # The sampled IDs are not the canonical tokenization of their text.
    noncanonical: bool = False

    def prefix_ids(self, tokenizer, state: int) -> list[int]:
        # A text boundary may lie inside an actor token. Re-encode this text prefix;
        # never snap the boundary, and never replace the actor's generated IDs.
        return tokenizer.encode(self.query + self.response[: self.boundaries[state]], add_special_tokens=False)


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _common_prefix_length(a: str, b: str) -> int:
    if b.startswith(a):
        return len(a)
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _token_starts(tokenizer, prompt_ids, body, query, response):
    """Character offset (within response) where each generated token begins.

    Upstream takes offsets from re-tokenizing the text. That is exact when the sampled IDs
    are canonical. Otherwise, decode growing prefixes of the actor's own IDs: a token starts
    where the cleanly decoded text before it ends (a token completing a split UTF-8 character
    starts at that character, as tokenizer offset mappings report it).
    """
    encoded = tokenizer(query + response, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) == prompt_ids + body:
        return [start - len(query) for start, _ in encoded["offset_mapping"][len(prompt_ids) :]], False
    full = query + response
    starts = []
    for i in range(len(body)):
        text = decode(tokenizer, prompt_ids + body[:i])
        starts.append(min(_common_prefix_length(text, full) - len(query), len(response) - 1))
    return starts, True


def _split(response, splitter):
    if splitter not in ("math", "gsm8k"):
        raise ValueError(f"Unknown VinePPO splitter: {splitter}")
    try:
        boundaries = split_solution_inplace(response) if splitter == "math" else split_gsm8k(response)
    except Exception:
        # Upstream logs parse_failed and drops the trajectory; keep it as a single step.
        return [0, len(response)], True
    if (
        len(boundaries) < 2
        or boundaries[0] != 0
        or boundaries[-1] != len(response)
        or any(a > b for a, b in zip(boundaries, boundaries[1:], strict=False))
    ):
        return [0, len(response)], True
    return boundaries, False


# A step holding only a display-math or environment terminator, e.g. "\n\\]".
_CLOSING_STEP = re.compile(r"\s*(\\\]|\$\$|\\end\{[^}]*\}|\\\)|\[/asy\])[\s.,;:]*")


def _merge_steps(response, boundaries, starts, merge_closing_steps, min_step_tokens):
    """Coarsen official boundaries (a variant, not upstream): every kept boundary is an official one."""
    if merge_closing_steps:
        # The state before a lone terminator has the value of the state after it.
        boundaries = [
            boundaries[0],
            *(
                boundaries[i]
                for i in range(1, len(boundaries) - 1)
                if not _CLOSING_STEP.fullmatch(response[boundaries[i] : boundaries[i + 1]])
            ),
            boundaries[-1],
        ]
    if min_step_tokens > 1:
        counts = [0] * (len(boundaries) - 1)
        for start in starts:
            counts[bisect_right(boundaries, start) - 1] += 1
        merged, pending = [boundaries[0]], 0
        for i, count in enumerate(counts):
            pending += count
            # Close a step once it holds enough generated tokens; the final step may be shorter.
            if pending >= min_step_tokens or i == len(counts) - 1:
                merged.append(boundaries[i + 1])
                pending = 0
        boundaries = merged
    return boundaries


def align_steps(
    tokenizer, prompt_ids, response_ids, splitter: str, *, merge_closing_steps=False, min_step_tokens=0
) -> StepAlignment:
    eos = tokenizer.eos_token_id
    has_eos = bool(response_ids) and response_ids[-1] == eos
    body = response_ids[:-1] if has_eos else response_ids
    query, response = decode(tokenizer, prompt_ids), decode(tokenizer, body)
    if not response_ids or decode(tokenizer, prompt_ids + body) != query + response:
        raise ValueError("VinePPO requires a nonempty, losslessly decoded response")
    boundaries, split_fallback = _split(response, splitter)
    starts, noncanonical = [], False
    if body:
        starts, noncanonical = _token_starts(tokenizer, prompt_ids, body, query, response)
        previous = -1
        for start in starts:
            if not (0 <= start < len(response)) or start < previous:
                raise ValueError("VinePPO tokenizer has invalid response character offsets")
            previous = start
    if not split_fallback:
        boundaries = _merge_steps(response, boundaries, starts, merge_closing_steps, min_step_tokens)
    token_steps = [bisect_right(boundaries, start) - 1 for start in starts]
    if has_eos:
        token_steps.append(len(boundaries) - 2)
    if len(token_steps) != len(response_ids):
        raise ValueError("VinePPO token-to-step mapping is incomplete")
    return StepAlignment(query, response, boundaries, token_steps, split_fallback, noncanonical)
