# Delta data contract (migration step 1)

## Batch sampling with the verl V1 server

The batch sampler runs prompt loading, rollout generation, uncertainty state
selection, MC labels, and JSONL export through the Ray-managed V1 vLLM server:

```bash
python -m examples.delta_critic.batch_sample \
  --config examples/delta_critic/config_sampling_qwen3_8b.yaml \
  --split train \
  --output-dir /path/to/new/delta_samples
```

Run from the repository root in an environment with compatible Ray, vLLM,
`datasets`, `math-verify`, and CUDA dependencies. The default config matches
the **effective** sampling settings in
`value_model/delta_value_llm_exp/config_qwen3_8b_densecritic_aligned_sweeps.yaml`:
Qwen3-8B, 2048 DeepMath train prompts, one rollout each, 2048 maximum response
tokens, temperature 0.7, top-p 0.95, top-20 candidate logprobs, up to 64
selected states spaced by 32 tokens, top-mass 0.8, and 32 MC continuations
with temperature 0.7, top-p 0.95, and 2048 maximum new tokens. The default
MC mode is `prefix_only`, matching the source MC script's default. For the
`ours_local` paired label path, add `--mc-mode paired_next_state`. To sample
the configured test or rank-eval split, use `--split test` or
`--split rank_eval`. `--prompts-jsonl` and `--limit` support a small input
check while keeping the other settings.

The command writes `rollouts_<split>_regen.jsonl`,
`states_<split>.jsonl`, and `mc_labels_<split>.jsonl`. An existing final
output file causes an error to prevent accidental replacement. Incomplete
temporary files are not promoted if a run fails. Rows retain the source's
identifiers, token fields, labels, and split, with added actor/version and
behavior-logprob provenance. The V1 backend returns top-k candidates in
`TokenOutput.extra_fields["delta_top_logprobs"]` only when this sampler
requests them; normal rollout responses keep their existing payload.

The data and selection formulas match the source, but generated token streams
are not guaranteed identical. The source calls synchronous vLLM with string
prompts and requests all 32 MC samples in one `SamplingParams(n=32)` call.
This sampler sends token IDs to Ray-managed V1 vLLM, runs one asynchronous
request per MC continuation, and can overlap requests. The source MC path
decodes and retokenizes prefixes; this path sends the original token IDs.
It decodes V1 outputs locally because `TokenOutput` does not carry vLLM
`completion.text`. RNG order, tokenization at prefix boundaries, and backend
version can therefore change exact samples. The source YAML declares rollout
`repetition_penalty` and `stop`, but its rollout script does not pass them
to `SamplingParams`; the equivalent config uses the effective parameters.
The sampler uses the current V1 server but does not enqueue data into
TransferQueue or train a critic.

This package establishes the `value_model` token-delta semantics. It imports
existing token IDs and MC rows and now accepts unpadded V1 AgentLoop/TransferQueue
output rows. It does not train a model, load a checkpoint, or modify the V1 trainer.

The implementation lives under `examples/` because this checkout's `recipe/` is
an uninitialized git submodule. Putting files there would not track them in the
parent repository's `delta` branch. No submodule configuration is changed.

## Contract and selected experiment

For response token `t` (zero based) and prompt length `P`, the pre-action state is
`prompt + response[:t]`. A delta model receives the selected token as well and
reads its **scalar output at `P+t`**; policy next-token logits are read at
`P+t-1`. Hidden vectors are not part of this interchange contract.
`selected_inference_rows` constructs the source's per-state prefix inputs;
`read_positions` also handles an explicit left-padding offset.

| Choice | Semantics |
| --- | --- |
| `selected_segment` label | Next selected state's `v_prefix` minus current `v_prefix`; the last selected state uses terminal reward minus current value. |
| `paired_next_state` label | Stored `delta` if present, otherwise `v_next - v_prefix`. |
| `advantage_source=critic` | Explicit raw predictions keyed by selected token index, separate from MC labels. |
| `advantage_source=mc_label` | Labels computed according to the configured label mode. |
| `policy_advantage_scope=segment` | Broadcast from selected token through the token before the next selected state; final segment includes response end. |
| `policy_advantage_scope=selected_token` | Apply only at the selected token. |

The original `ours_local` path combines paired labels with **segment broadcast**
for learned predictions. The `mc_local` oracle uses paired labels with
`selected_token`. These choices are independent in `DeltaConfig`; no inference
from the word “local” changes the algorithm.

`value_model_final_grpo(...)` selects learned delta, segment broadcast, and
**training-split state advantage standardization**. The critic label mode, loss
mask and target normalization must still be supplied explicitly. This preset is
supported by completed source runs, not CLI defaults:

- `delta_value_llm_exp/parallel_runs/requested/policy_grpo_final_hybrid_b32_best_20260717_run/ckpt/hybrid_b32_best/offline_grpo_final_hybrid_b32_best/final/offline_grpo_metadata.json`
- `delta_value_llm_exp/parallel_runs/requested/policy_grpo_final_hybrid_current_code_budget8_replica_20260718_011500/ckpt/offline_grpo_final_hybrid_current_code_budget8_replica/final/offline_grpo_metadata.json`

Both record `objective=state_only`, `state_advantage_source=delta`,
`state_advantage_normalization=standardize`, and
`state_advantage_scope=selected_token_to_before_next_selected_state`.
Their fitted statistics are **not** defaults for new data. August continuation
GRPO (`response_grpo`, precomputed old-policy logprobs) is a separate path; its
inactive state normalization setting does not replace this preset. The latest
value-only critic experiments also do not change the delta scalar contract.

## Masks, normalization and minimal use

`critic_targets` returns a target vector with `critic_target_mask` and a separate
`critic_signal_mask`. A selected state with a true zero delta is still signal.
For `loss_mask=response`, zero targets at unselected response tokens participate
in critic supervision. `selected_state` and the legacy `selected_delta` alias
both supervise selected positions.

`state_advantages` returns `state_advantage_mask`, independently of
`policy_token_mask`. Fit policy normalization on all train-split active state
tokens **after broadcasting, before padding/truncation and before applying the
policy mask**. Longer segments therefore contribute more observations. Fit
critic statistics separately over the train-split supervised targets, also
before collation/truncation. Both are population standard deviations; empty
populations and std below `1e-8` fail as in the source. `NormalizationStats`
records which population it belongs to and prevents mixing the two.

Source critic normalization transforms even masked background targets. Policy
normalization leaves inactive state tokens at zero. Prediction denormalization
uses only critic target statistics. Apply the same fitted train statistics to
evaluation; never refit per batch or independently on evaluation data.

```python
from examples.delta_critic.batch_adapter import pad_response_rows
from examples.delta_critic.legacy_adapter import adapt_legacy_rows
from examples.delta_critic.target_ops import (
    fit_normalization, normalize, state_advantages, value_model_final_grpo,
)

config = value_model_final_grpo(
    label_mode="paired_next_state",
    loss_mask="selected_state",
    target_normalization="standardize",
)
examples, diagnostics = adapt_legacy_rows(train_rollout_rows, train_mc_rows)
# Predictions below have already been denormalized with critic target statistics.
raw_vectors = [
    state_advantages(example, config, raw_predictions=predictions[example.rollout.rollout_id])
    for example in examples
]
stats = fit_normalization(raw_vectors, population="state_advantages")
advantages = [
    normalize(vector, mode=config.advantage_normalization, population="state_advantages", stats=stats)
    for vector in raw_vectors
]
train_batch = pad_response_rows(examples, config, advantages, width=1024)
# Later state_only PG/KL reductions use each row's policy_loss_mask.
# Evaluation uses the SAME train stats (no fit on eval):
eval_examples, _ = adapt_legacy_rows(eval_rollout_rows, eval_mc_rows)
eval_advantages = [
    normalize(
        state_advantages(example, config, raw_predictions=eval_predictions[example.rollout.rollout_id]),
        mode=config.advantage_normalization, population="state_advantages", stats=stats,
    )
    for example in eval_examples
]
```

`pad_response_rows` creates padded response vectors with separate
`response_valid_mask`, `critic_target_mask`, `critic_signal_mask`,
`state_advantage_mask`, and `policy_token_mask`. Its `policy_loss_mask` is the
intersection of state, policy and validity masks. For the `state_only` path,
subsequent PG/KL losses and their denominators must use this final mask; merely
setting an advantage to zero is insufficient. The adapter returns raw critic
targets and explicitly supplied advantages, and never estimates statistics.
Positions for padding are `-1`; gather only positions enabled by validity masks.

## Compatibility and remaining decisions

- The schema is version 1. Original rows are copied into metadata, preserving
  train/eval split, old logprobs, actor/tokenizer identifiers and MC provenance
  when present. Missing versions, state IDs and sample counts remain unknown.
  The adapter does not manufacture them or recompute a split.
- Duplicate rollouts/states, missing required fields, nonfinite numbers,
  out-of-range positions and conflicting provided prefixes are rejected.
  A stored delta inconsistent with `v_next-v_prefix` generates a diagnostic;
  paired labels retain the original stored-delta precedence.
- The adapter deliberately does not import `delta_last_chunk` as a model
  prediction: source files can use that field for labels or later overwrite it
  with predictions. The caller must identify and pass model outputs explicitly.
  Missing selected predictions fail instead of silently falling back to labels.
- `terminal_after_last_token` exposes both choices explicitly. Source behavior
  is `treat_length_truncation_as_terminal=True` and
  `assume_legacy_last_token_terminal=False`; stop/eos are terminal, other
  non-null reasons are not. Existing label imports need no terminal shortcut
  and can retain an unknown finish reason.
- Token masks in this contract are binary. Fractional sample weights, value
  targets, centering, distributional losses and hybrid terminal-composition
  auxiliary targets are outside step 1.
- **Window policy remains undecided.** Helpers provide untruncated mappings and
  padding only. A too-small padding width is not a truncation mode. The source's
  whole-rollout training tail window differs from its selected-prefix inference
  tail window, and the `verl-agent` full-action approach differs too. No new
  production max length or rejection policy has been chosen here.
- **Checkpoint acceptance and normalization metadata formats remain undecided.**
  No checkpoint reader/worker validator is added. Numerical `none` normalization
  is supported explicitly, but source `mean/std=None` metadata has not been
  converted to the other repository's format.
- `verl-agent` reads the final token of a whole action and uses action/batch
  normalization. The approved path instead retains multiple selected states,
  source segment broadcast, and the completed critic-driven run's train-token
  standardization. No worker or core trainer code is changed.

## New rollout and MC collection

`export_v1_rollouts` accepts one unpadded V1 output row at a time (the fields
from `AgentLoopOutput.as_dict()` plus `uid`, `prompt_id`, and `global_steps`, or an
explicit unique `rollout_id`). It requires an
explicit actor version, generation sampling config, and new-experiment prompt
split seed. It preserves `rollout_log_probs` as `behavior_logprobs`, rather than
substituting recomputed old-policy logprobs. Existing legacy rows continue to
use `adapt_legacy_rows` without recomputing their split.

```python
from functools import partial

from examples.delta_critic.collection import export_v1_rollouts
from examples.delta_critic.mc_labeling import MCConfig, label_mc_states_async
from examples.delta_critic.state_selection import select_states
from examples.delta_critic.v1_sampler import sample_v1_continuations

rollouts = export_v1_rollouts(
    # Supply explicit finish_reason on rows if paired MC may select the last token.
    v1_unpadded_rows, actor_version="actor-step-7",
    sampling_config={"temperature": 0.7, "top_p": 0.9},
    eval_fraction=0.1, split_seed="experiment-1",
)
states = select_states(
    rollouts, strategy="indices", states_per_response=2,
    indices_by_rollout=selected_token_indices,
)
mc_config = MCConfig(
    mode="paired_next_state", continuations_per_state=32,
    sampling_config={"temperature": 0.8, "top_p": 0.9, "max_tokens": 256},
    actor_version="actor-step-7", continuation_skip_special_tokens=True,
)
sampler = partial(
    sample_v1_continuations, server_client=active_actor_server_client,
    tokenizer=actor_tokenizer,
)
mc_rows = await label_mc_states_async(
    rollouts, states, mc_config, sample=sampler,
    decode_prefix=lambda ids: actor_tokenizer.decode(ids, skip_special_tokens=False),
    score=lambda response, rollout: math_reward(response, rollout["gold_answer"]),
)
examples, diagnostics = adapt_legacy_rows(rollouts, mc_rows)
```

The V1 client is the same `LLMServerClient.generate` interface used by the V1
single-turn agent loop. Its `TokenOutput` has token IDs but no completion text;
the caller explicitly chooses `skip_special_tokens` for continuation decoding.
Source uncertainty selection requires per-token top-k candidate rows. V1 output
does not provide those, so new V1 data must supply selected indices explicitly.
MC rejects response masks containing tool/observation tokens. The V1 vLLM and
SGLang backends now preserve raw `finish_reason` in `extra_fields`; older rows
missing it require an explicit terminal policy for paired labels at the last
token. `stop_reason="completed"` is not inferred to mean EOS. The adapter cannot check
remote actor weights, so the caller must ensure the server matches `actor_version`.

The CPU tests exercise this path using a fake V1 server client. For a real
single-GPU interface check, run `python examples/delta_critic/smoke_v1_gpu.py
--model-path /path/to/actor --actor-version snapshot-name` from the repository
root in an environment with compatible Ray, vLLM and CUDA packages. The smoke
reward only checks whether the response contains `2`; it is not a math benchmark.
The cached Qwen3-8B run passed twice with 16 response tokens, two selected
states and two MC samples per state. The migration plan records the exact
snapshot and temporary environment workaround used for that run.

## Source references and verification

Paths below are relative to the `value_model` checkout, under
`delta_value_llm_exp/`:

- `05_train_token_scalar_model_accelerate_local.py`: `DenseTokenScalarDataset`,
  `_target_stats`, `NormalizedTokenScalarDataset`.
- `09_train_policy_grpo_offline.py`: `_critic_build_selected_eval_rows` and
  the state-only policy/KL mask selection.
- `offline_grpo.py`: `state_advantage_vectors`, `state_advantage_stats`,
  `normalize_state_advantages`, `collate_offline_grpo_batch`.
- `03_estimate_values_mc_vllm.py`: `is_terminal_after_last_response_token`.

Run CPU golden tests without models or an external checkout:

```bash
python -m pytest -q tests/examples/delta_critic
```

To also compare against the actual source functions:

```bash
VALUE_MODEL_ROOT=/path/to/value_model python -m pytest -q tests/examples/delta_critic
```

The optional differential test imports the source's dependency-free policy
module and selects the critic dataset/statistics definitions from its AST to
avoid importing the training stack. It checks both label modes, both supervision
masks, learned/oracle propagation, and both statistics populations. Without the
variable, only this external-source comparison is skipped.
