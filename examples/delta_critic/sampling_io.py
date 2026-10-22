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
"""Prompt and math reward handling for the source-compatible sampling path."""

import asyncio
import importlib.metadata
import json
import multiprocessing
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

DEFAULT_MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem carefully.\n"
    "Show your reasoning and put the final answer in \\boxed{{}}.\n\n"
    "Problem:\n{problem}"
)


ScoreStatus = Literal["correct", "incorrect", "unparseable", "timeout"]


@dataclass(frozen=True)
class ScoreResult:
    """One observable scorer outcome; timeouts are never converted to reward zero."""

    reward: float | None
    status: ScoreStatus
    attempts: int
    timeout_seconds: int | None


class ScoringError(RuntimeError):
    """Unexpected scorer failure that must abort collection."""


class ScoringTimeoutError(TimeoutError):
    """A terminal reward could not be scored after the configured retry."""


def scorer_metadata() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "name": "math_verify",
        "math_verify_version": version,
        "execution_backend": "spawned_process",
        "raise_on_error": True,
    }


def _first(row: dict, *keys: str):
    for key in keys:
        if row.get(key) is not None and row[key] != "":
            return row[key]
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict | list) else str(value)


def _prompt_text(value: Any) -> str:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        contents = [item.get("content") for item in value if item.get("content")]
        if contents:
            return "\n".join(str(content) for content in contents)
    return _stringify(value)


def normalize_prompt_row(row: dict, index: int, prompt_config: dict | None = None) -> dict:
    prompt_config = prompt_config or {}
    base = _first(row, "prompt", "question", "problem")
    if base is None or not _stringify(base).strip():
        raise ValueError("Prompt row must contain nonempty prompt, question, or problem")
    base_text = _prompt_text(base).strip()
    template = prompt_config.get("template")
    if template and ("prompt" not in row or prompt_config.get("apply_template_to_existing_prompt", False)):
        fields = {key: _stringify(value) for key, value in row.items()}
        answer = _first(row, "gold_answer", "answer", "target", "final_answer", "solution")
        if answer is not None:
            for key in ("answer", "gold_answer", "target"):
                fields.setdefault(key, _stringify(answer))
        for key in ("prompt", "question", "problem"):
            fields.setdefault(key, base_text)
        prompt = str(template).format(**fields).strip()
    elif "problem" in row and "prompt" not in row:
        prompt = DEFAULT_MATH_PROMPT_TEMPLATE.format(problem=base_text)
    else:
        prompt = base_text
    answer = _first(row, "gold_answer", "answer", "target", "final_answer", "solution")
    reward_model = row.get("reward_model")
    if answer is None and isinstance(reward_model, dict):
        answer = _first(reward_model, "ground_truth", "answer", "target")
    identifier = _first(row, "id", "unique_id")
    extra_info = row.get("extra_info")
    if identifier is None and isinstance(extra_info, dict):
        identifier = _first(extra_info, "index", "id", "unique_id")
    return {
        "prompt_id": str(identifier or index),
        "prompt": prompt,
        "gold_answer": None if answer is None else _stringify(answer),
    }


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_prompt_rows(
    config: dict, split: str, *, prompts_jsonl: str | None = None, limit: int | None = None
) -> list[dict]:
    if split not in {"train", "test", "rank_eval"}:
        raise ValueError("split must be train, test, or rank_eval")
    data = config.get("data") or {}
    smoke = config.get("smoke") or {}
    source_path = prompts_jsonl or data.get(f"{split}_prompts_path")
    hf_source = data.get(f"{split}_hf_dataset")
    if split == "rank_eval" and not source_path and not hf_source:
        source_path = data.get("train_prompts_path")
        hf_source = data.get("train_hf_dataset")
    if prompts_jsonl:
        hf_source = None
    if bool(source_path) == bool(hf_source):
        raise ValueError("Exactly one JSONL or Hugging Face prompt source is required")
    configured_limit = smoke.get(f"{split}_prompts")
    count = limit if limit is not None else (int(configured_limit) if configured_limit else None)
    offset = smoke.get(f"{split}_prompt_offset")
    if offset is None and split == "rank_eval":
        offset = smoke.get("train_prompts", 0)
    offset = int(offset or 0)
    if offset < 0 or (count is not None and count < 0):
        raise ValueError("Prompt offset and limit must be nonnegative")
    if source_path:
        rows = read_jsonl(source_path)[offset:]
        if count is not None:
            rows = rows[:count]
    else:
        from datasets import load_dataset

        spec = (
            {"path": hf_source}
            if isinstance(hf_source, str)
            else {
                "path": hf_source.get("repo_id") or hf_source.get("path"),
                "name": hf_source.get("config_name"),
                "split": hf_source.get("split", split),
                "revision": hf_source.get("revision"),
                "data_files": hf_source.get("data_files"),
                "streaming": bool(hf_source.get("streaming", False)),
            }
        )
        if isinstance(hf_source, str):
            spec["split"] = split
        if not spec["path"]:
            raise ValueError("Hugging Face dataset requires repo_id or path")
        kwargs = {key: value for key, value in spec.items() if value is not None}
        try:
            datasets = [load_dataset(**kwargs)]
        except ValueError as exc:
            if "Config name is missing" not in str(exc) or spec.get("name"):
                raise
            from datasets import get_dataset_config_names

            datasets = [load_dataset(**{**kwargs, "name": name}) for name in get_dataset_config_names(spec["path"])]
        rows = []
        for dataset in datasets:
            for index, row in enumerate(dataset):
                if index < offset:
                    continue
                rows.append(dict(row))
                if count is not None and len(rows) >= count:
                    break
            if count is not None and len(rows) >= count:
                break
    return [normalize_prompt_row(row, offset + index, config.get("prompt")) for index, row in enumerate(rows)]


def _last_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def _boxed_answer(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth, result = 1, []
    for char in text[start + len(marker) :]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(result).strip()
        result.append(char)
    return None


@lru_cache(maxsize=20000)
def _parse_math(
    text: str,
    fallback: str,
    extraction: str,
    parsing_timeout: int,
    raise_on_error: bool,
):
    from math_verify import parse

    return tuple(
        parse(
            text,
            fallback_mode=fallback,
            extraction_mode=extraction,
            parsing_timeout=parsing_timeout,
            raise_on_error=raise_on_error,
        )
    )


def _parse_math_answer(
    text: str,
    fallback: str,
    extraction: str,
    parsing_timeout: int,
    raise_on_error: bool,
):
    """Parse a dataset answer as one expression, retaining textual fallback."""
    delimited = any(marker in text for marker in ("$", r"\(", r"\[", r"\boxed{"))
    is_math = re.search(r"[\\0-9=+*/^_(){}\[\],<>-]", text) or re.fullmatch(r"[A-Za-z]", text.strip())
    if not delimited and is_math:
        parsed = _parse_math("$" + text + "$", "no_fallback", extraction, parsing_timeout, raise_on_error)
        if parsed:
            return parsed
    return _parse_math(text, fallback, extraction, parsing_timeout, raise_on_error)


def _final_answer_field(response: str) -> tuple[int, str]:
    """Extract the last anchored answer, including one following math block."""
    pattern = (
        r"(?im)^[ \t]*(?:\#{1,6}[ \t]+)?(?:\*{1,2})?"
        r"(?:the[ \t]+)?(?:final[ \t]+)?answer(?:\*{1,2})?[ \t]*"
        r"(?::|is\b[ \t]*:?)(?:\*{1,2})?[ \t]*([^\n]*)"
    )
    matches = list(re.finditer(pattern, response))
    if not matches:
        return -1, ""
    match = matches[-1]
    tail = match.group(1).strip()
    if not tail:
        tail = response[match.end() :].lstrip()
        # A multiline field may contain one immediate math answer, never
        # a later number harvested from explanatory prose.
        math_start = re.match(r"(?:[-+.\d\\$({\[]|[A-Za-z](?:$|[ \t]*[=+*/^_()-]))", tail)
        literal_text = re.fullmatch(r"[A-Za-z]+(?:[ \t]+[A-Za-z]+)*", tail.splitlines()[0]) if tail else None
        if not math_start and not literal_text:
            return match.start(), ""
    for opening, closing in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$")):
        if tail.startswith(opening):
            end = tail.find(closing, len(opening))
            return match.start(), tail[: end + len(closing)] if end >= 0 else ""
    return match.start(), tail.splitlines()[0].strip() if tail else ""


def _positive_timeout(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _score_math_once(
    response: str,
    gold_answer: str,
    reward_config: dict,
    *,
    timeout_seconds: int,
) -> tuple[float, ScoreStatus]:
    from math_verify import verify
    from sympy import SympifyError

    boxed = _boxed_answer(response)
    final_answer = None
    if reward_config.get("math_verify_require_final_answer", False):
        boxed_position = response.rfind(r"\boxed{")
        answer_position, answer_field = _final_answer_field(response)
        if answer_position > boxed_position:
            final_answer = answer_field
            boxed = None
        else:
            final_answer = boxed
        if not final_answer:
            return 0.0, "unparseable"
    gold_text = str(gold_answer)
    if reward_config.get("math_verify_simple_first", True):
        literal_answer = boxed or final_answer
        if literal_answer and literal_answer.strip() == gold_text.strip():
            return 1.0, "correct"
        number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        if boxed and re.fullmatch(number_pattern, boxed.strip()) and re.fullmatch(number_pattern, gold_text.strip()):
            if abs(Decimal(boxed.strip()) - Decimal(gold_text.strip())) <= Decimal("1e-6"):
                return 1.0, "correct"
    max_boxed_chars = int(reward_config.get("math_verify_max_boxed_chars", 512))
    if boxed and max_boxed_chars > 0 and len(boxed) > max_boxed_chars:
        return 0.0, "unparseable"
    extraction = str(reward_config.get("math_verify_extraction_mode", "first_match"))
    try:
        if final_answer is not None:
            prediction = _parse_math_answer(final_answer, "no_fallback", extraction, timeout_seconds, True)
        else:
            prediction = _parse_math(
                "$" + boxed + "$" if boxed else response,
                "no_fallback",
                extraction,
                timeout_seconds,
                True,
            )
    except SympifyError:
        # This is a property of an invalid model answer, not a scorer outage.
        # In particular, SymPy rejects decimal integers beyond Python's
        # configured digit limit. Gold parsing and verify failures remain
        # fail-closed below and are never converted into wrong answers.
        return 0.0, "unparseable"
    if not prediction:
        return 0.0, "unparseable"
    gold = _parse_math_answer(gold_text, "first_match", extraction, timeout_seconds, True)
    if not gold:
        raise ScoringError("math-verify could not parse the nonempty gold answer")
    correct = bool(
        verify(
            gold[0],
            prediction[0],
            timeout_seconds=timeout_seconds,
            raise_on_error=True,
        )
    )
    return float(correct), "correct" if correct else "incorrect"


def score_text_result(response: str, gold_answer: str | None, reward_config: dict) -> ScoreResult:
    """Score text with explicit wrong, unparseable, and timeout outcomes."""
    method = reward_config.get("method", "math_verify")
    if method == "exact_or_numeric":
        if gold_answer is None or str(gold_answer).strip() == "":
            return ScoreResult(0.0, "unparseable", 1, None)
        if str(gold_answer).strip().lower() in response.strip().lower():
            return ScoreResult(1.0, "correct", 1, None)
        prediction, gold = _last_number(response), _last_number(str(gold_answer))
        if prediction is None or gold is None:
            return ScoreResult(0.0, "unparseable", 1, None)
        correct = abs(float(prediction) - float(gold)) <= 1e-6
        return ScoreResult(float(correct), "correct" if correct else "incorrect", 1, None)
    if method != "math_verify":
        raise ValueError(f"Unsupported reward method={method!r}")
    if gold_answer is None or not str(gold_answer).strip():
        return ScoreResult(0.0, "unparseable", 1, None)
    if threading.current_thread() is not threading.main_thread():
        # math-verify's SIGALRM timeouts require the spawned worker's main thread.
        return _reward_executor().submit(score_text_result, response, gold_answer, dict(reward_config)).result()
    try:
        from math_verify.errors import TimeoutException
    except ImportError as exc:
        if reward_config.get("fallback_on_import_error", False):
            return score_text_result(response, gold_answer, {"method": "exact_or_numeric"})
        raise RuntimeError("reward.method=math_verify requires math-verify") from exc
    first_timeout = _positive_timeout(
        reward_config.get("math_verify_timeout_seconds", 5), "math_verify_timeout_seconds"
    )
    retry_timeout = _positive_timeout(
        reward_config.get("math_verify_retry_timeout_seconds", 20), "math_verify_retry_timeout_seconds"
    )
    for attempt, timeout_seconds in enumerate((first_timeout, retry_timeout), start=1):
        try:
            reward, status = _score_math_once(
                response,
                str(gold_answer),
                reward_config,
                timeout_seconds=timeout_seconds,
            )
            return ScoreResult(reward, status, attempt, timeout_seconds)
        except TimeoutException:
            if attempt == 2:
                return ScoreResult(None, "timeout", attempt, timeout_seconds)
        except ScoringError:
            raise
        except Exception as exc:
            raise ScoringError(f"Unexpected math-verify {type(exc).__name__}") from exc
    raise AssertionError("unreachable scorer retry state")


def score_text(response: str, gold_answer: str | None, reward_config: dict) -> float:
    """Backward-compatible scalar API; terminal scoring cannot exclude timeouts."""
    result = score_text_result(response, gold_answer, reward_config)
    if result.reward is None:
        raise ScoringTimeoutError(
            f"math-verify timed out after {result.attempts} attempts (last timeout={result.timeout_seconds}s)"
        )
    return result.reward


_reward_pool = None
_reward_pool_lock = threading.Lock()


def _reward_executor():
    # One reusable CPU worker per collector/AgentLoop process. Spawn avoids
    # inheriting CUDA/Ray state, and math-verify runs on its main thread so its
    # signal-based timeouts work. Python shuts the executor down on exit.
    global _reward_pool
    # lru_cache can evaluate a first call more than once under concurrency.
    # A lock keeps synchronous reward threads and async MC on one worker.
    with _reward_pool_lock:
        if _reward_pool is None:
            _reward_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        return _reward_pool


def _score_text_results(responses, gold_answer, reward_config):
    return [score_text_result(response, gold_answer, reward_config) for response in responses]


async def score_text_results_async(responses, row, *, reward_config):
    """Return structured MC outcomes from the spawned scorer process."""
    return await asyncio.get_running_loop().run_in_executor(
        _reward_executor(), _score_text_results, list(responses), row["gold_answer"], dict(reward_config)
    )


async def score_texts_async(responses, row, *, reward_config):
    """Backward-compatible float API; timeout-aware MC callers use results."""
    results = await score_text_results_async(responses, row, reward_config=reward_config)
    rewards = []
    for result in results:
        if result.reward is None:
            raise ScoringTimeoutError(
                f"math-verify timed out after {result.attempts} attempts (last timeout={result.timeout_seconds}s)"
            )
        rewards.append(result.reward)
    return rewards
