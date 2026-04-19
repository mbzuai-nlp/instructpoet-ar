#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Supervised fine-tuning script for Arabic Poetry tasks.
Supports curriculum learning and random training modes.

Tasks:
- analysis: Analyze poetry (extract keywords, themes, etc.)
- continuation: Continue a poem
- generation: Generate poetry from themes/keywords
- restoration: Restore/correct corrupted poetry
"""

import logging
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
import wandb

from alignment import ScriptArguments, SFTConfig, get_model, get_tokenizer
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    get_peft_config,
    setup_chat_format,
)

from poetry_data import (
    get_poetry_datasets,
    get_tokenized_poetry_datasets,
    CURRICULUM_ORDER,
)

# Environment setup
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)


@dataclass
class PoetryScriptArguments(ScriptArguments):
    """Extended script arguments for poetry tasks."""

    training_mode: Optional[str] = field(
        default="random",
        metadata={
            "help": "Training mode: 'curriculum' for sequential task training or 'random' for shuffled"
        },
    )

    data_base_path: Optional[str] = field(
        default="/path/to/data/dialectical_IFT_DATA/tsv",
        metadata={"help": "Base path for the poetry TSV data files"},
    )

    tasks: Optional[str] = field(
        default="analysis,continuation,generation,corruption",
        metadata={"help": "Comma-separated list of tasks to include in training"},
    )

    max_samples_per_task: Optional[int] = field(
        default=None, metadata={"help": "Maximum samples per task (None for all)"}
    )

    prompt_type: Optional[str] = field(
        default="instruction",
        metadata={"help": "Prompt type: 'instruction' or 'chat'"},
    )

    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of workers for preprocessing"},
    )

    wandb_project: Optional[str] = field(
        default="arabic_poetry",
        metadata={"help": "Weights and Biases project name"},
    )

    use_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to cache processed datasets for faster loading"},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory for dataset cache (default: ~/.cache/poetry_datasets or POETRY_CACHE_DIR env var)"
        },
    )

    def __post_init__(self):
        # Set a dummy dataset_name to bypass the parent's validation
        # We load our own poetry datasets, not from the hub
        if self.dataset_name is None and self.dataset_mixture is None:
            self.dataset_name = "local_poetry_data"
        super().__post_init__()


def main(script_args, training_args, model_args):
    # HuggingFace token is set via HF_TOKEN environment variable
    # No need to call login() - the token is automatically used

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Parse tasks list
    tasks = [t.strip() for t in script_args.tasks.split(",")]
    training_mode = script_args.training_mode

    # Set output directory structure:
    # Base: /path/to/arab_poetry/
    #   ├── checkpoints/{model}/{mode}/{tasks}/  <- model checkpoints
    #   └── dataset_cache/                        <- cached datasets
    model_name = model_args.model_name_or_path.split("/")[-1]
    tasks_str = "_".join(tasks) if len(tasks) <= 2 else f"{len(tasks)}tasks"

    # Get the base path from output_dir (should be /path/to/arab_poetry)
    base_path = training_args.output_dir

    # Set checkpoint directory
    training_args.output_dir = (
        f"{base_path}/checkpoints/{model_name}/{training_mode}/{tasks_str}"
    )

    # Set cache directory
    if script_args.cache_dir:
        cache_dir = script_args.cache_dir
    else:
        cache_dir = os.path.join(base_path, "dataset_cache")

    # Setup wandb
    WANDB_RUN_NAME = f"{model_name}_{training_mode}_{tasks_str}"
    if training_args.local_rank == 0:
        wandb.init(
            project=script_args.wandb_project,
            name=WANDB_RUN_NAME,
        )

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Training mode: {training_mode}")
    logger.info(f"Tasks: {tasks}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    logger.info("*** Loading Poetry Datasets ***")
    logger.info(f"Dataset cache directory: {cache_dir}")

    ################
    # Load tokenizer first (needed for tokenization caching)
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Setup chat format if needed (before tokenizing)
    # We need a temporary model load just to check, or we do this after model load
    # For now, just set up ChatML if no chat template
    if tokenizer.chat_template is None:
        logger.info(
            "No chat template provided, setting up ChatML template for tokenizer."
        )
        # Set the ChatML template directly on the tokenizer
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # For distributed training, only main process should do the initial tokenization/caching
    # Other processes will load from cache
    local_rank = training_args.local_rank

    if local_rank > 0:
        # Wait for main process to finish tokenizing and caching
        import torch.distributed as dist

        logger.info(
            f"Rank {local_rank}: Waiting for main process to prepare dataset..."
        )
        dist.barrier()

    # Get tokenized datasets with caching
    # Cache key: training_mode + prompt_type + model_name
    processed_datasets = get_tokenized_poetry_datasets(
        base_path=script_args.data_base_path,
        tasks=tasks,
        training_mode=training_mode,
        prompt_type=script_args.prompt_type,
        tokenizer=tokenizer,
        model_name=model_name,
        max_seq_length=(
            training_args.max_seq_length
            if hasattr(training_args, "max_seq_length")
            else 2048
        ),
        max_samples_per_task=script_args.max_samples_per_task,
        shuffle=(training_mode == "random"),
        seed=training_args.seed,
        num_proc=script_args.preprocessing_num_workers or 4,
        use_cache=script_args.use_cache,
        cache_dir=cache_dir,
    )

    if local_rank == 0:
        # Signal other processes that dataset is ready
        import torch.distributed as dist

        if dist.is_initialized():
            logger.info("Main process: Dataset ready, signaling other processes...")
            dist.barrier()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    logger.info(
        f"Training on the following datasets: train={len(train_dataset)}, test={len(eval_dataset)}"
    )
    logger.info(f"Dataset columns: {train_dataset.column_names}")

    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Setup chat format if tokenizer doesn't have one
    # This must happen BEFORE DDP wrapping and must be consistent across all ranks
    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    # Ensure model embedding size matches tokenizer vocab size (important for DDP)
    if len(tokenizer) != model.config.vocab_size:
        logger.info(
            f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}"
        )
        model.resize_token_embeddings(len(tokenizer))

    # Enable input gradients for gradient checkpointing with LoRA/PEFT
    # This is critical for DDP to work correctly with adapters
    if training_args.gradient_checkpointing:
        logger.info("Enabling input gradients for gradient checkpointing")
        model.enable_input_require_grads()

        # For models that support it, set gradient checkpointing use_reentrant
        if hasattr(model, "gradient_checkpointing_enable"):
            use_reentrant = training_args.gradient_checkpointing_kwargs.get(
                "use_reentrant", False
            )
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )

    # Log model parameter count for DDP verification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Rank {local_rank}: Model has {total_params:,} total parameters, {trainable_params:,} trainable"
    )

    # Synchronize all processes before proceeding to training
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        logger.info(f"Rank {local_rank}: Model loaded and synchronized")

    # Log sample tokenized data (only on main process to avoid file conflicts)
    if training_args.local_rank <= 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(
            os.path.join(training_args.output_dir, "sample_tokenized.txt"), "w"
        ) as f:
            for index in random.sample(
                range(len(train_dataset)), min(3, len(train_dataset))
            ):
                sample = train_dataset[index]
                input_ids = sample.get("input_ids", [])
                labels = sample.get("labels", [])

                # Decode to show the text
                decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                # Count how many tokens are masked (labels = -100)
                masked_count = sum(1 for l in labels if l == -100)

                logger.info(
                    f"Sample {index}: {len(input_ids)} tokens, {masked_count} masked (prompt)"
                )
                f.write(
                    f"=== Sample {index} ===\n"
                    f"Token count: {len(input_ids)}\n"
                    f"Masked tokens (prompt): {masked_count}\n"
                    f"Decoded text:\n{decoded_text[:1000]}...\n\n"
                )

    ############################
    # Initialize the SFT Trainer
    # Dataset is pre-tokenized with input_ids, attention_mask, labels
    # SFTTrainer will detect input_ids and skip tokenization
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save metadata
    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": f"arabic_poetry_{tasks_str}",
        "tags": ["arabic-poetry", "alignment-handbook"],
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # Push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")

    # Finish wandb run
    if trainer.accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parser = TrlParser((PoetryScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
