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

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem carefully.\n"
    "Show your reasoning and put the final answer in \\boxed{{}}.\n\n"
    "Problem:\n{problem}"
)


def _first(row: dict, *keys: str):
    for key in keys:
        if row.get(key) is not None and row[key] != "":
            return row[key]
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def normalize_prompt_row(row: dict, index: int, prompt_config: dict | None = None) -> dict:
    prompt_config = prompt_config or {}
    base = _first(row, "prompt", "question", "problem")
    if base is None or not _stringify(base).strip():
        raise ValueError("Prompt row must contain nonempty prompt, question, or problem")
    base_text = _stringify(base).strip()
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
    identifier = _first(row, "id", "unique_id")
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

        spec = {"path": hf_source} if isinstance(hf_source, str) else {
            "path": hf_source.get("repo_id") or hf_source.get("path"),
            "name": hf_source.get("config_name"),
            "split": hf_source.get("split", split),
            "revision": hf_source.get("revision"),
            "data_files": hf_source.get("data_files"),
            "streaming": bool(hf_source.get("streaming", False)),
        }
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
    for char in text[start + len(marker):]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(result).strip()
        result.append(char)
    return None


@lru_cache(maxsize=20000)
def _parse_math(text: str, fallback: str, extraction: str):
    from math_verify import parse

    return tuple(parse(text, fallback_mode=fallback, extraction_mode=extraction))


def score_text(response: str, gold_answer: str | None, reward_config: dict) -> float:
    method = reward_config.get("method", "math_verify")
    if method == "exact_or_numeric":
        if gold_answer is None or str(gold_answer).strip() == "":
            return 0.0
        if str(gold_answer).strip().lower() in response.strip().lower():
            return 1.0
        prediction, gold = _last_number(response), _last_number(str(gold_answer))
        return float(prediction is not None and gold is not None and abs(float(prediction) - float(gold)) <= 1e-6)
    if method != "math_verify":
        raise ValueError(f"Unsupported reward method={method!r}")
    if gold_answer is None or not str(gold_answer).strip():
        return 0.0
    try:
        from math_verify import verify
    except ImportError as exc:
        if reward_config.get("fallback_on_import_error", False):
            return score_text(response, gold_answer, {"method": "exact_or_numeric"})
        raise RuntimeError("reward.method=math_verify requires math-verify") from exc
    try:
        boxed = _boxed_answer(response)
        gold_text = str(gold_answer)
        if boxed and reward_config.get("math_verify_simple_first", True):
            left, right = re.sub(r"\s+", "", boxed.lower()), re.sub(r"\s+", "", gold_text.lower())
            if left and left == right:
                return 1.0
            left_number, right_number = _last_number(boxed), _last_number(right)
            if left_number is not None and right_number is not None:
                if abs(float(left_number) - float(right_number)) <= 1e-6:
                    return 1.0
        max_boxed_chars = int(reward_config.get("math_verify_max_boxed_chars", 512))
        if boxed and max_boxed_chars > 0 and len(boxed) > max_boxed_chars:
            return 0.0
        extraction = str(reward_config.get("math_verify_extraction_mode", "first_match"))
        prediction = _parse_math("$" + boxed + "$" if boxed else response, "no_fallback", extraction)
        gold = _parse_math(gold_text, "first_match", extraction)
        return float(bool(prediction and gold and verify(gold[0], prediction[0])))
    except Exception:
        return 0.0
