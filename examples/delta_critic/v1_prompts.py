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
"""Read and tokenize text prompts using the V1 PPO/GRPO data path."""


def load_v1_prompts(config, *, tokenizer, processor, hf_model_type, split, limit=None):
    """Return one V1 dataloader pass with the single-turn AgentLoop prompt IDs.

    The sampler, batch size, dataset factory, and collator intentionally mirror
    PPOTrainer._init_dataloader. The caller may stop early via ``limit``.
    """
    from torchdata.stateful_dataloader import StatefulDataLoader

    from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    data = config.data
    is_train = split == "train"
    paths = data.train_files if is_train else data.val_files
    if is_train:
        filter_groups = config.algorithm.get("filter_groups", None)
        exact_refill = (
            config.trainer.v1.trainer_mode != "sync"
            or bool(filter_groups is not None and filter_groups.get("enable", False))
            or bool(config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
        )
    else:
        exact_refill = False
    dataset = create_rl_dataset(
        paths, data, tokenizer, processor, is_train=is_train,
        max_samples=data.get("train_max_samples" if is_train else "val_max_samples", -1),
    )
    if is_train:
        batch_size = 1 if exact_refill else (data.get("gen_batch_size") or data.train_batch_size)
    else:
        batch_size = data.val_batch_size or len(dataset)
    if batch_size <= 0:
        raise ValueError("V1 prompt dataset is empty")
    loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=data.dataloader_num_workers,
        drop_last=is_train,
        collate_fn=collate_fn,
        **({"sampler": create_rl_sampler(data, dataset)} if is_train else
           {"shuffle": data.get("validation_shuffle", True)}),
    )
    builder = create_continuous_token_builder(
        tokenizer,
        hf_model_type=hf_model_type,
        chat_template_kwargs=data.get("apply_chat_template_kwargs", {}),
        mm_processor_kwargs=data.get("mm_processor_kwargs", {}),
        processor=processor,
    )
    prompt_length = int(config.actor_rollout_ref.rollout.prompt_length)
    rows = []
    for batch in loader:
        for index, messages in enumerate(batch["raw_prompt"]):
            messages = list(messages)
            if any(not isinstance(message.get("content"), str) for message in messages):
                raise ValueError("Delta batch sampling currently supports text-only V1 prompts")
            if batch.get("tools_kwargs") is not None and batch["tools_kwargs"][index]:
                raise ValueError("Delta batch sampling currently supports single-turn prompts without tools")
            if "agent_name" in batch and batch["agent_name"][index] != "single_turn_agent":
                raise ValueError("V1 prompt row selects a custom agent loop")
            prompt_ids = list(builder.build_initial_tokens(messages))
            # AgentLoopBase._cap_text_prompt_length left-truncates after rendering.
            if len(prompt_ids) > prompt_length:
                prompt_ids = prompt_ids[-prompt_length:]
            if not prompt_ids:
                raise ValueError("V1 prompt tokenization produced no tokens")
            ground_truth = batch.get("reward_model")
            reward = ground_truth[index] if ground_truth is not None else None
            answer = reward.get("ground_truth") if isinstance(reward, dict) else None
            if answer is None:
                raise ValueError("V1 prompt needs reward_model.ground_truth for delta MC labels")
            rows.append({
                "prompt_id": str(len(rows)),
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
                "raw_prompt": messages,
                "prompt_token_ids": prompt_ids,
                "gold_answer": str(answer),
                "data_source": str(batch["data_source"][index]) if "data_source" in batch else None,
            })
            if limit is not None and len(rows) >= limit:
                return rows
    return rows
