# Delta data contract (migration step 1)

## Batch sampling with verl V1 prompts and server

The default prompt path uses the same `ppo_trainer` data configuration,
`create_rl_dataset`, `StatefulDataLoader`, train sampler, and Continuous Token
builder as V1 PPO/GRPO single-turn rollout. Supply the same `data.*` overrides
used for PPO/GRPO. For example, with a verl-formatted JSONL or Parquet file
containing `prompt` chat messages and `reward_model.ground_truth`:

```bash
python -m examples.delta_critic.batch_sample \
  --config examples/delta_critic/config_sampling_qwen3_8b.yaml \
  --split train \
  --verl-override data.train_files=/path/to/train.parquet \
  --verl-override data.val_files=/path/to/test.parquet \
  --verl-override data.train_batch_size=32 \
  --verl-override data.shuffle=True \
  --verl-override data.seed=42 \
  --verl-override data.max_prompt_length=2048 \
  --verl-override +data.apply_chat_template_kwargs.enable_thinking=False \
  --output-dir /path/to/new/delta_samples
```

Use `data.val_files` with `--split test`; `data.prompt_key`, custom dataset,
filtering, train/validation shuffle, sampler seed, worker count, and batch
sizes also come from the V1 Hydra config. The train loader drops the last
incomplete batch, as V1 does. `--limit N` takes the first N prompts in loader
order. The recorded `raw_prompt` keeps the source chat messages, while
`prompt_token_ids` are the actual token IDs passed to the server after the
same chat template and left cap used by `SingleTurnAgentLoop`. Pass the same
model path with `--model-path` if the PPO/GRPO run uses a local checkpoint.
This delta collector handles text-only, single-turn prompts with
`reward_model.ground_truth`; tool and multimodal requests need their V1
AgentLoop paths to be collected separately.

For the earlier source-style DeepMath text prompts, select that source
explicitly:

```bash
python -m examples.delta_critic.batch_sample \
  --prompt-source value_model \
  --config examples/delta_critic/config_sampling_qwen3_8b.yaml \
  --split train \
  --output-dir /path/to/source_style_samples
```

Run from the repository root in an environment with compatible Ray, vLLM,
`datasets`, `torchdata`, `math-verify`, and CUDA dependencies. The delta config matches
the **effective** sampling settings in
`value_model/delta_value_llm_exp/config_qwen3_8b_densecritic_aligned_sweeps.yaml`:
Qwen3-8B, 2048 DeepMath train prompts in source mode, one rollout each, 2048 maximum response
tokens, temperature 0.7, top-p 0.95, top-20 candidate logprobs, up to 64
selected states spaced by 32 tokens, top-mass 0.8, and 32 MC continuations
with temperature 0.7, top-p 0.95, and 2048 maximum new tokens. The default
MC mode is `prefix_only`, matching the source MC script's default. For the
`ours_local` paired label path, add `--mc-mode paired_next_state`. To sample
the test split, use `--split test`; `--split rank_eval` applies only to source
mode. `--prompts-jsonl` applies only to source mode; `--limit` works with both.

The command writes `rollouts_<split>_regen.jsonl`,
`states_<split>.jsonl`, `mc_labels_<split>.jsonl`, and
`continuations_<split>.jsonl`. The continuation file keeps one row per MC
sample with the state and continuation IDs, token IDs, decoded text, and reward;
the MC label also keeps the 32 individual rewards. An existing final
output file causes an error to prevent accidental replacement. Incomplete
temporary files are not promoted if a run fails. Rows retain the source's
identifiers, token fields, labels, and split, with added actor/version and
behavior-logprob provenance. The V1 backend returns top-k candidates in
`TokenOutput.extra_fields["delta_top_logprobs"]` only when this sampler
requests them; normal rollout responses keep their existing payload.

The state selection and MC formulas match the source, but the default V1
prompt source uses PPO/GRPO chat messages, dataset sampling, and chat-template
tokenization. Its prompt strings and token IDs therefore differ from the
source's raw DeepMath text prompt path. The source calls synchronous vLLM with
string prompts and requests all 32 MC samples in one `SamplingParams(n=32)` call.
This sampler sends token IDs to Ray-managed V1 vLLM. Its default native mode
uses vLLM's `n` sampling and returns every completion through
`TokenOutput.extra_fields["delta_mc_token_ids"]`. `--mc-concurrency` limits the
number of MC sequences in flight (default 16), so 32 samples per state use two
`n=16` requests by default; set it to 32 for one `n=32` request per state.
Set `--mc-native-batch-size 8 --mc-concurrency 32` to admit four state requests
at once while keeping at most 32 MC sequences in flight.
`--mc-sampling-mode requests` keeps the separate-request path for comparison.
Rollout generation uses
`--rollout-concurrency` workers (default 8); finished rollouts enter a bounded
MC queue immediately, and `math_verify` scoring overlaps GPU generation. JSONL
rows may be written in completion order. The source MC path
decodes and retokenizes prefixes; this path sends the original token IDs.
It decodes V1 outputs locally because `TokenOutput` does not carry vLLM
`completion.text`. RNG order, tokenization at prefix boundaries, and backend
version can therefore change exact samples. The source YAML declares rollout
`repetition_penalty` and `stop`, but its rollout script does not pass them
to `SamplingParams`; the equivalent config uses the effective parameters.
The sampler uses the V1 dataloader ordering and server but does not enqueue
data into TransferQueue, apply PPO/GRPO group filtering, or train a critic.

This package establishes the `value_model` token-delta semantics. It imports
existing token IDs and MC rows and now accepts unpadded V1 AgentLoop/TransferQueue
output rows. It includes standalone scalar-delta training, checkpoint import,
and frozen scoring paths. It does not change the default V1 PPO/GRPO policy-loss
path or core trainer loop.

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
- Source `.pt` imports accept only scalar `target_type=delta`, `output_dim=1`,
  `local_td0` or `hybrid_terminal_composition`, MSE or balanced MSE,
  response/selected-state masks, and the supported delta label modes.
  Distributional heads, value centering, target networks, and other objectives
  fail validation. The importer preserves source metadata, step, backbone and
  tokenizer identity, value-layer index, source path, and SHA256. Legacy `.pt`
  files initialize weights only; they do not contain enough state to resume.
- Disabled target normalization keeps source `mean/std=None` metadata and scores
  numerically as `mean=0, std=1`. Enabled statistics must be finite with positive
  population std. Converted artifacts also record a tokenizer fingerprint and
  checksum of the portable model weights.
- Window behavior is explicit. Imported source checkpoints use `legacy_tail`,
  keeping the rightmost checkpoint-sized window of each selected prefix and
  recording its start index. New training artifacts use their stored `full`
  context cap and reject selected prefixes beyond it. `--scoring-max-length`
  may expand that cap; score metadata records the change.
- `FrozenDeltaWorker` reads unpadded TransferQueue `prompts`, `responses`, and
  selected response indices. It writes normalized and raw delta vectors plus
  the selected-signal mask, preserving each row key and jagged response length.
  Indices may be supplied directly or stored in `selected_token_indices`. The
  V1 engine path requires a forward-only `TrainingWorker` initialized from the
  same artifact and without an optimizer; the standalone torch path supports
  parity checks and CLI scoring. Neither path changes the default V1 PPO/GRPO
  policy loss.
- `verl-agent` reads the final token of a whole action and uses action/batch
  normalization. The approved path instead retains multiple selected states,
  source segment broadcast, and train-token standardization.

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

## Critic checkpoint import and frozen scoring

Convert a source scalar `.pt` file into the portable artifact format before
scoring it. Import validates the model/head shape, training semantics,
normalization metadata, backbone/tokenizer identity, hidden-state index, and
training context length.

```bash
python -m examples.delta_critic.checkpoint /path/to/critic.pt /path/to/critic-artifact
python -m examples.delta_critic.score \
  --checkpoint /path/to/critic-artifact \
  --rollouts /path/to/rollouts.jsonl \
  --selected /path/to/selected-states.jsonl \
  --output /path/to/scored.jsonl \
  --device cuda --microbatch 8
```

The selected file needs `rollout_id` and `token_index`; scoring does not read MC
values. For TransferQueue inference, construct `FrozenDeltaWorker` from the same
portable artifact and call `score_transfer_queue(meta)`. To use the V1 engine
path, pass an initialized forward-only `TrainingWorker` created with that
artifact as its initial weights. Both paths read each selected response prefix
through `P+t` and restore scores to original row and token order.

## Scalar critic training with V1/FSDP2

`examples.delta_critic.train` uses a dedicated `model_type=delta_scalar`
TrainingWorker and FSDP2 engine. The backbone follows `dtype` (BF16 by default)
while the biased scalar head remains FP32. `local_td0` and
`hybrid_terminal_composition` are supported. The hybrid objective fits terminal
statistics on training continuations alone and requires full eligible
continuation coverage unless a continuation budget is explicitly configured.

The 4B budgeted hybrid recipe uses `terminal_normalization: td`: terminal
predictions are first restored with the TD mean/std, then the sum residual is
divided by the TD std (or 1 when target normalization is disabled). Thus both
objectives measure errors in the same delta units. The sum includes one TD mean
offset per selected position; terminal predictions are summed, never averaged.
`terminal_normalization: independent` retains the terminal target std used by
older artifacts. Sharing the smaller TD std can substantially increase terminal
loss and gradients, so `terminal_weight` controls its relative strength. Output
units remain shared under either residual scale: both paths always restore each
scalar with the TD mean/std before terminal composition.

The corrected 4B recipes use `terminal_weight: 0.003` and
`terminal_length_weighting: inverse_count`. The latter multiplies each terminal
row's loss by `1 / K`, where `K` is its active suffix count; it never changes the
predicted sum or its target. These are conservative starting weights, not a
performance guarantee. `none` preserves legacy weighting. Held-out terminal
raw MSE/RMSE is reported independently of the optimization weighting so changing
the coefficient or length weights cannot masquerade as a better critic.

`continuation_selection: uncertainty` uses the policy's `1 - top1_prob`
ranking, greedy minimum gap and candidate budget. The continuation's first token
is retained as the anchor boundary, within the total state budget; nearby
uncertainty positions are excluded to maintain the minimum gap. This initial
boundary is required by the unchanged `reward - v_prefix` target. The legacy
`uniform` selector is still the default for old configs/artifacts. Uncertainty
selection requires token-aligned `delta_top_logprobs` from the collection actor
and fails if they are absent. New offline/online exports preserve those rows.

Older continuation files can be enriched without changing their tokens/rewards:

```bash
uv run --no-project --python /path/to/training/python \
  python -m examples.delta_critic.rescore_continuations \
  --model Qwen/Qwen3-4B-Base --actor-version Qwen/Qwen3-4B-Base \
  --labels /data/mc_labels_train.jsonl \
  --continuations /data/continuations_train.jsonl \
  --output /data/continuations_with_probs.jsonl --device cuda:0
```

Use the actual collection actor checkpoint/version; the tool loads cached Qwen3
weights, teacher-forces stored sequences, chunks vocabulary logits and refuses
to overwrite its destination. It reproduces the collector's default vLLM
`raw_logprobs` before sampling temperature and top-p/top-k filtering.
It does not sample new continuations or train.
`config_train_hybrid_qwen3_4b_base.yaml` retains full native context and all stored
branches; the `_2048` recipe is for shorter policy-expansion collections.
To traverse the terminal training pool once, choose at least
`ceil(N_terminal / (global_batch - td_rows_per_batch))` updates. The repeated TD
epoch count alone does not describe terminal sample coverage.

With `hybrid_reduction: separate_samples`, each distributed update minimizes
`td_weight * mean_TD(L_TD) + terminal_weight * mean_terminal(w_K * L_terminal)`.
Here `w_K = 1 / K` for `inverse_count`, and `1` for `none`;
`L_terminal = 0.5 * ((sum_k(TD_std * z_k + TD_mean) - target) / residual_std)^2`.
TD rows have a nonempty TD target mask; terminal rows have a valid terminal target.
Padding contributes to neither count. Counts cover all ranks and accumulation
microbatches, and evaluation uses the same reductions. A missing objective in an
update contributes zero. The sampler still samples the combined dataset, so this
does not ensure that every update contains both objectives. Use `--td-rows-per-batch`
to draw a fixed number of original TD rows and fill the rest with terminal rows;
the two independently shuffled streams persist their positions in checkpoints.
The legacy default
`all_samples` divides both sums by the total number of real rows. Old configs and
artifacts default to `independent` and `all_samples`.

New configs leave `max_length: null`; the trainer resolves it from the loaded
backbone's native `max_position_embeddings` (Qwen3-8B: 40960). Inputs over the
resolved cap are rejected.

Collate right-pads the global batch to its longest row, but each micro-batch is
then trimmed to **its own** longest row before the forward, so padding columns
are only computed when a micro-batch really contains rows of different lengths.
At `--microbatch 1` that means no padding at all, which is what the source
trainer's `batch_size=1` collate produced; running every micro-batch at the
global batch width cost about 1.7x extra forward and backward work on this data.
Sequence parallel size stays 1 and the LM fused-forward kernels stay disabled.

The attention backend comes from `attention_implementation` and can be
overridden per run without editing a config:

```bash
torchrun --standalone --nproc_per_node=1 -m examples.delta_critic.train \
  --config examples/delta_critic/config_train_qwen3_8b.yaml \
  --attention-implementation flash_attention_2 ...
```

`sdpa` is the default and is what the source critic runs used. `flash_attention_2`
additionally requires the `flash-attn` package, which is not installed by
default; without it the run fails at model load with an explicit ImportError.
Do not interpret the old 2048-token source-run length as the default for new runs.

The data split must map every rollout ID to exactly one of `train` or `eval`.
`--global-batch` and `--max-steps` are required so the update count and linear
learning-rate schedule are explicit. Every real row has equal weight in an
optimizer update, including rows with zero supervised tokens; gradient
accumulation and DP partitioning do not change that logical row mean. Synthetic
padding rows are excluded. Start from the repository root, for example:

```bash
torchrun --standalone --nproc_per_node=1 -m examples.delta_critic.train \
  --config examples/delta_critic/config_train_qwen3_8b.yaml \
  --rollouts /data/rollouts.jsonl --labels /data/labels.jsonl \
  --split /data/split.json --output /runs/delta-local \
  --global-batch 1 --max-steps 100 --microbatch 1
```

For hybrid training, use `config_train_hybrid_qwen3_8b.yaml` and add
`--continuations /data/continuations.jsonl`. A portable source `.pt` import or
completed training artifact can be supplied with `--initialize` to load weights
and start a fresh optimizer/data stream. Initializing a hybrid run from a
`local_td0` critic is supported when the local target contract and TD
normalization are identical; changed mean/std is rejected to avoid reinterpreting
the initialized head. The objective transition and initialization identity are
recorded in the artifact. Exact resume instead uses `--resume`
with a completed training checkpoint; it restores model, optimizer, scheduler,
step, row-order cursor, and RNG state, and checks that data, statistics,
configuration, effective batch, and DP topology match. Checkpoints include a
portable scoring artifact plus V1 engine state. Evaluation reports TD and
terminal losses, real/supervised row and token coverage, and data provenance.

For the original 4B TD collection, the weight-only transition is explicit:

```bash
torchrun --standalone --nproc_per_node=4 -m examples.delta_critic.train \
  --config examples/delta_critic/config_train_hybrid_qwen3_4b_base.yaml \
  --rollouts /data/rollouts_train_regen.jsonl --labels /data/mc_labels_train.jsonl \
  --split /data/split.json --continuations /data/continuations_with_probs.jsonl \
  --initialize /runs/td/step_00000080 --output /runs/hybrid-corrected \
  --global-batch 256 --td-rows-per-batch 128 --microbatch 1 \
  --max-steps 1000 --save-every 20
```

Keep the initialized TD artifact's exact TD data and split for this transition.
Choose the update budget from the actual terminal pool; the example does not
start a job or select devices for an existing run.

The small acceptance harness exercises a real tiny Qwen3 through the V1 worker,
including save/resume next-step equivalence, microbatch/DP scaling, and
TransferQueue scoring:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/delta_critic/smoke_training.py \
  --output /tmp/delta-critic-smoke
```

It is a plumbing test, not an 8B training or throughput benchmark. Dynamic
remove-padding, sequence parallelism, fused-forward optimization, actor update,
and actor/critic alternation remain follow-up work.

## Delta critic evaluation against MC labels

`examples.delta_critic.evaluate` ports the delta path of the source
`06_eval_token_delta_unified.py`. It scores every selected MC state at its own
`P+t` position and reports the source's four metric blocks: raw
`delta_regression`, the per-response and per-query z-scored
`normalized_delta_regression`, the three-class `discrete` block, and
`delta_sign_diagnostics`. Predictions are denormalized with the checkpoint's own
target statistics, so every reported number is in raw reward units.

```bash
python -m examples.delta_critic.evaluate \
  --checkpoint /runs/delta-local/step_00001120 \
  --rollouts /data/rollouts_rank_eval_regen.jsonl \
  --labels /data/mc_labels_rank_eval.jsonl \
  --output /runs/delta-local/eval_rank_eval.json \
  --rows-output /runs/delta-local/eval_rows.jsonl \
  --split rank_eval --microbatch 8
```

`--class-margin` defaults to the source's `1/32`; deltas inside the margin count
as `neutral` for both ground truth and predictions, and a delta target is
anchored to the rollout's `terminal_reward` when the state is the last selected
one. The source's value, distributional, anchored-value and policy-gradient
blocks are out of migration scope and are not reproduced.

The reported `gt_delta` is always
`V(next selected state, or terminal reward) - V(current selected state)`. That
matches the source, whose selected-state evaluation has no label-mode branch in
either eval path — so for a checkpoint trained with `paired_next_state` this is
a *different quantity* from the training target, and only `selected_segment`
models are evaluated against the target they were trained on.

## Consistent math rewards for policy and MC

Use `examples/delta_critic/reward.py:compute_score` for new policy runs. The
trainer wrapper and MC scorer both retain the existing extraction rule by
default. To require an explicit final answer, enable it in both places:

```yaml
reward:
  custom_reward_function:
    path: examples/delta_critic/reward.py
    name: compute_score
    reward_kwargs:
      math_verify_require_final_answer: true
algorithm:
  delta_policy:
    mc:
      reward:
        method: math_verify
        fallback_on_import_error: false
        math_verify_require_final_answer: true
```

With this option, scoring requires the latest explicit `\boxed{...}` or anchored
`Answer:` / `Final answer:` field. Fields may put their answer on the next line
or in a following LaTeX math block. An empty or malformed last field scores
zero; arbitrary numbers in unfinished reasoning are not final answers. The
wrapper and `score_text` helper both default this flag to `false`, so set it
explicitly for offline collection as well when adopting this stricter rule.

Threaded reward calls and async MC use the same spawned CPU scorer, preserving
Math-Verify's signal timeouts. Bare gold expressions are parsed as complete math
expressions, and fast numeric comparisons require complete decimal literals
(with the existing absolute tolerance of `1e-6`). Regrade saved responses and
MC labels consistently when adopting this rule; changing source does not
repair rewards or model updates from a running or historical experiment.

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
python -m pytest -q tests/special_standalone/delta_critic
```

To also compare against the actual source functions:

```bash
VALUE_MODEL_ROOT=/path/to/value_model python -m pytest -q tests/special_standalone/delta_critic
```

The optional differential test imports the source's dependency-free policy
module and selects the critic dataset/statistics definitions from its AST to
avoid importing the training stack. It checks both label modes, both supervision
masks, learned/oracle propagation, and both statistics populations. Without the
variable, only this external-source comparison is skipped.

## Value regression and a matched delta control

```bash
bash examples/delta_critic/run_qwen3_4b_value_policy_fsdp2.sh \
  --work-dir "$PWD/outputs/value_difference_80step" --gpus 0 1 2 3
```

This starts a detached, sequential pipeline: value training (80 optimizer
steps, validation every 5) → policy training (10 steps) → matched direct-delta
training → matched policy training. It uses the existing compatible Python
environment via `uv`; override `PYTHON_BIN` if needed. No continuous monitoring
is required. Each failed stage stops the pipeline and records a nonzero exit
code. An existing output directory is never overwritten. `prepare` only
freezes and checks inputs; `run --work-dir ...` runs a prepared experiment in
the foreground. The same GPU leases are held for the complete sequence.

Inputs must match the actual TD80 fingerprints: the P95-filtered collection
with 438 train / 48 heldout trajectories. No MC labels are regenerated or
rescored. Both arms use the identical sealed October 6 TD sources, Qwen3-4B-Base
weights, fresh FP32 scalar head, seed42, deterministic row iterator, DP4,
global batch128, microbatch16, AdamW LR1e-5, clipping1, zero weight decay,
linear decay over 80 steps, raw per-row MSE, and supervision boundaries. The
pipeline compares the actual initial parameter bytes and optimizer batch
schedule between arms. Only the regression target and its prescribed inference
mapping differ. Each arm retains just its best **heldout point-weighted raw
MSE** checkpoint under `critic_{arm}/best`; ties keep the earliest step.

For response tokens `y[0:n]`, boundary `i` means the state
`prompt + y[:i]`, before generating `y[i]`. Its target is the stored
`v_prefix`, the mean final reward of the collector's MC continuations from
that prefix. A causal scalar head therefore reads sequence position
`prompt_length + i - 1`; boundary0 reads the prompt's last token. Boundary
`n` represents the stopped/capped full answer and has the stored final reward
as its value label. It is **not** an absorbing state after reward payment.
Unsampled intermediate values have no training labels.

At selected boundaries `i[k]`, the original selected-segment label is
`V(i[k+1]) - V(i[k])`, with `V(n) = terminal_reward` for the final segment.
The value model predicts each boundary value and subtracts its predictions,
including its **predicted** `V(n)` for the tail. This is not the paired-next-token
label `V(i+1)-V(i)` unless those happen to be adjacent selected boundaries.
The matched direct-delta arm fits those same differences directly at the
same pre-token positions. Both arms supervise the terminal boundary, with
value target `terminal_reward` and direct-delta target0 (no following segment).

These matched controls are not a target-only comparison against the old
TD80 artifact: historical TD80 reads position `prompt_length+i`, uses
response-wide zero background targets, balanced MSE, and delta standardization.
Applying that mask to prefix values would fabricate labels. The common changes
are recorded in `alignment.json`. Value differencing also reads the later
prefix, whereas a single direct-delta output uses only the current prefix;
equal information at each scalar readout cannot be claimed. Policy settings
otherwise match the repaired historical TD recipe, including horizon100,
loop stop10, the expansion budget, and the <=100-token realized-reward override
in both arms. Consequently, short-answer overrides are shared exceptions to
the critic's prediction rule. GPU sampling may diverge even with identical
seeds, and later policy trajectories need not match.

Inspect `status.json`, `alignment.json`, `critic_{arm}/selection.json`, stage
logs, and `policy_{arm}/td/metrics.jsonl` after completion. `exit_code=0` means
all four stages completed; failure details go to `status.json`. Source and
input hashes are checked before every training stage. The driver additionally
checks the source loaded by Ray and the ordered policy prompt schedule.

```bash
uv run --offline --no-project --python "$PYTHON_BIN" python -m pytest -q \
  tests/special_standalone/delta_critic/test_value_difference.py
```
