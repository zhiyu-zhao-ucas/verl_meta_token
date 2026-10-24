# Copyright 2026 Individual Contributor: zhiyu
# SPDX-License-Identifier: Apache-2.0
"""Bounded, ephemeral Monte Carlo branches on the active rollout policy."""

import asyncio
import hashlib
import json
import logging
import math
import time

import torch

from .alignment import align_steps, decode

logger = logging.getLogger(__name__)


def check_version(extra, expected):
    for key in ("global_steps", "min_global_steps", "max_global_steps"):
        # A single server only reports global_steps. Min/max, when supplied by a
        # routed client, additionally prove there was no version change mid-call.
        if (key == "global_steps" or key in extra) and extra.get(key) != expected:
            raise ValueError(f"VinePPO policy version mismatch: {key}={extra.get(key)}, expected {expected}")


def finish_reason(output):
    reason = output.extra_fields.get("finish_reason")
    if reason not in ("stop", "length"):
        raise ValueError(f"VinePPO incomplete or aborted generation: {reason!r}")
    return reason


class VinePPOCollector:
    """One instance per agent worker. No branch tokens or incomplete state are persisted."""

    def __init__(self, config, tokenizer, generate, score, *, context_length, sampling_params, seed=0):
        self.config, self.tokenizer = config, tokenizer
        self.generate, self.score = generate, score
        self.context_length = min(context_length, config.mc_context_length or context_length)
        self.sampling_params, self.seed = dict(sampling_params), seed
        self.semaphore = asyncio.Semaphore(config.concurrency)

    async def collect(self, output, *, expected_version, identity, score_kwargs):
        started = time.monotonic()
        if output.num_turns != 2 or output.multi_modal_data or not all(output.response_mask):
            raise ValueError("VinePPO supports text-only single-turn responses")
        if len(output.response_mask) != len(output.response_ids):
            raise ValueError("VinePPO response mask length mismatch")
        check_version(output.extra_fields, expected_version)
        main_finish = finish_reason(output)
        reward = self.config.unfinished_reward if main_finish == "length" else output.reward_score
        if reward is None or not math.isfinite(reward):
            raise ValueError("VinePPO requires a finite main task reward")
        output.reward_score = float(reward)
        alignment = align_steps(
            self.tokenizer,
            output.prompt_ids,
            output.response_ids,
            self.config.splitter,
            merge_closing_steps=self.config.merge_closing_steps,
            min_step_tokens=self.config.min_step_tokens,
        )
        n = len(alignment.boundaries) - 1
        sampled = min(n, self.config.state_cap or n)
        rewards = [[] for _ in range(sampled)]
        generated_tokens = 0
        truncated = 0
        score_errors = 0
        score_seconds = 0.0
        # A fixed number of consumers; do not create K * steps pending tasks.
        branches = iter((state, k) for state in range(sampled) for k in range(self.config.mc_samples))

        async def consume():
            nonlocal generated_tokens, truncated, score_errors, score_seconds
            for state, k in branches:
                prefix = alignment.prefix_ids(self.tokenizer, state)
                horizon = min(self.config.mc_max_new_tokens, self.context_length - len(prefix))
                if horizon < 1:
                    raise ValueError("VinePPO MC prefix exhausts the rollout context window")
                digest = hashlib.sha256(
                    json.dumps(
                        [self.seed, expected_version, identity, alignment.boundaries[state], k],
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                params = dict(self.sampling_params, max_tokens=horizon, seed=int(digest[:8], 16))
                async with self.semaphore:
                    for attempt in range(self.config.max_retries + 1):
                        try:
                            branch = await self.generate(
                                request_id=f"vineppo-{digest}",
                                prompt_ids=prefix,
                                sampling_params=dict(params),
                            )
                            break
                        except (TimeoutError, ConnectionError):
                            if attempt == self.config.max_retries:
                                raise
                            await asyncio.sleep(self.config.retry_delay)
                    check_version(branch.extra_fields, expected_version)
                    reason = finish_reason(branch)
                    if len(branch.token_ids) > horizon:
                        raise ValueError("VinePPO MC generation exceeded its horizon")
                    generated_tokens += len(branch.token_ids)
                    if reason == "length":
                        truncated += 1
                        value = self.config.unfinished_reward
                    else:
                        ids = branch.token_ids
                        if ids and ids[-1] == self.tokenizer.eos_token_id:
                            ids = ids[:-1]
                        text = alignment.response[: alignment.boundaries[state]] + decode(self.tokenizer, ids)
                        # This re-encoding is only for the verifier. It never enters actor training.
                        score_start = time.monotonic()
                        try:
                            value = await self.score(output.prompt_ids, text, score_kwargs)
                        except Exception:
                            if self.config.score_error_reward is None:
                                raise
                            # Upstream grades a verifier error or timeout as an incorrect answer.
                            logger.warning("VinePPO MC verifier failed; using score_error_reward", exc_info=True)
                            value = self.config.score_error_reward
                            score_errors += 1
                        score_seconds += time.monotonic() - score_start
                    if value is None or not math.isfinite(value):
                        raise ValueError("VinePPO MC verifier returned a nonfinite or missing reward")
                    rewards[state].append(float(value))

        tasks = [
            asyncio.create_task(consume())
            for _ in range(min(self.config.concurrency, sampled * self.config.mc_samples))
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if any(len(items) != self.config.mc_samples for items in rewards):
            raise RuntimeError("VinePPO MC collection did not complete exactly K samples per state")
        values = [math.fsum(items) / self.config.mc_samples for items in rewards] + [reward] * (n - sampled)
        return {
            "vineppo_step_ids": torch.tensor(alignment.token_steps, dtype=torch.long),
            "vineppo_values": torch.tensor(values, dtype=torch.float32),
            "vineppo_mc_counts": torch.tensor([self.config.mc_samples] * sampled + [0] * (n - sampled)),
            "vineppo_num_steps": torch.tensor(n),
            "vineppo_actor_version": torch.tensor(expected_version),
            "vineppo_generated_tokens": torch.tensor(generated_tokens),
            "vineppo_boundaries": torch.tensor(alignment.boundaries),
            "vineppo_mc_variance": torch.tensor(
                [math.fsum((r - values[i]) ** 2 for r in items) / len(items) for i, items in enumerate(rewards)]
                + [0.0] * (n - sampled)
            ),
            "vineppo_mc_truncated": torch.tensor(truncated),
            "vineppo_main_truncated": torch.tensor(main_finish == "length"),
            "vineppo_split_fallback": torch.tensor(alignment.split_fallback),
            "vineppo_noncanonical": torch.tensor(alignment.noncanonical),
            "vineppo_main_score_error": torch.tensor(bool(output.extra_fields.get("vineppo_score_error", False))),
            "vineppo_mc_score_errors": torch.tensor(score_errors),
            "vineppo_mc_seconds": torch.tensor(time.monotonic() - started),
            "vineppo_score_seconds": torch.tensor(score_seconds),
            "vineppo_mc_horizons": torch.tensor(
                [
                    min(
                        self.config.mc_max_new_tokens,
                        self.context_length - len(alignment.prefix_ids(self.tokenizer, i)),
                    )
                    for i in range(sampled)
                ]
            ),
        }


async def collect_for_worker(worker, outputs, kwargs):
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.rollout.utils import get_max_position_embeddings

    if len(outputs) != 1:
        raise ValueError("VinePPO requires exactly one actor trajectory per session")
    if not hasattr(worker, "_vineppo_collector"):
        vine = omega_conf_to_dataclass(worker.config.algorithm.vineppo)
        rollout = worker.config.actor_rollout_ref.rollout
        context = rollout.max_model_len or get_max_position_embeddings(worker.model_config.hf_config)
        if not context:
            raise ValueError("VinePPO requires rollout.max_model_len for a model without a context limit")

        async def score(prompt_ids, text, sample_kwargs):
            response_ids = worker.tokenizer.encode(text, add_special_tokens=False)
            if not response_ids:
                # Empty text is an unextractable task answer, not a generation failure.
                response_ids = [worker.tokenizer.eos_token_id]
            candidate = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                num_turns=2,
                metrics=AgentLoopMetrics(),
            )
            await worker._compute_score([candidate], kwargs=sample_kwargs)
            return candidate.reward_score

        worker._vineppo_collector = VinePPOCollector(
            vine,
            worker.tokenizer,
            worker.llm_client.generate,
            score,
            context_length=context,
            sampling_params={"temperature": rollout.temperature, "top_p": rollout.top_p, "top_k": rollout.top_k},
            seed=rollout.get("seed", 0),
        )
    # Dataset index, session, and iteration are stable across replay from a checkpoint.
    if "index" not in kwargs:
        raise ValueError("VinePPO requires a stable dataset index for MC branch identity")
    return await worker._vineppo_collector.collect(
        outputs[0],
        expected_version=int(kwargs["global_steps"]) - 1,
        identity=[str(kwargs["index"]), str(kwargs["session_id"])],
        score_kwargs=kwargs,
    )
