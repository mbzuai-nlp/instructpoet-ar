# coding=utf-8
# Poetry Data Loading Utilities
# Supports curriculum learning and random mode training
# With caching support for faster subsequent runs
# Includes tokenization caching for even faster loading

import os
import hashlib
import pandas as pd
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
import logging

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Task order for curriculum learning
CURRICULUM_ORDER = ["analysis", "continuation", "generation", "corruption"]

# Map task names to folder names (corruption is the restoration task)
TASK_FOLDER_MAP = {
    "analysis": "analysis",
    "continuation": "continuation",
    "generation": "generation",
    "corruption": "corruption",
}

# Default cache directory
DEFAULT_CACHE_DIR = os.environ.get(
    "POETRY_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "poetry_datasets"),
)


def get_cache_path(
    training_mode: str,
    prompt_type: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Generate a cache path based on training_mode and prompt_type."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Simple naming: training_mode + prompt_type
    cache_name = f"poetry_{training_mode}_{prompt_type}"
    return os.path.join(cache_dir, cache_name)


def load_tsv_as_dataset(file_path: str, max_samples: Optional[int] = None) -> Dataset:
    """
    Load a TSV file and convert it to a HuggingFace Dataset.

    Args:
        file_path: Path to the TSV file
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Dataset object
    """
    logger.info(f"Loading data from {file_path}")

    # Read TSV file
    df = pd.read_csv(file_path, sep="\t", on_bad_lines="skip")

    # Limit samples if specified
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"Sampled {max_samples} examples from {len(df)} total")

    # Convert to dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)

    logger.info(f"Loaded {len(dataset)} examples from {file_path}")
    return dataset


def load_poetry_task_data(
    base_path: str, task: str, split: str = "train", max_samples: Optional[int] = None
) -> Dataset:
    """
    Load data for a specific poetry task.

    Args:
        base_path: Base path to the TSV data directory
        task: Task name (analysis, continuation, generation, restoration/corruption)
        split: Data split (train or test)
        max_samples: Maximum samples to load

    Returns:
        Dataset for the specified task
    """
    # Map task name to folder
    folder_name = TASK_FOLDER_MAP.get(task, task)

    # Construct file path
    file_name = f"{folder_name}_ift.tsv"
    file_path = os.path.join(base_path, folder_name, split, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    dataset = load_tsv_as_dataset(file_path, max_samples)

    # Add task column for tracking
    dataset = dataset.add_column("task", [task] * len(dataset))

    return dataset


def load_poetry_datasets(
    base_path: str,
    tasks: List[str],
    training_mode: str = "random",
    split: str = "train",
    max_samples_per_task: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Load poetry datasets for multiple tasks.

    Args:
        base_path: Base path to the TSV data directory
        tasks: List of tasks to load
        training_mode: 'curriculum' for sequential or 'random' for shuffled
        split: Data split (train or test)
        max_samples_per_task: Maximum samples per task
        shuffle: Whether to shuffle (only applies to random mode)
        seed: Random seed for shuffling

    Returns:
        Combined Dataset
    """
    datasets = []

    for task in tasks:
        try:
            dataset = load_poetry_task_data(
                base_path=base_path,
                task=task,
                split=split,
                max_samples=max_samples_per_task,
            )
            datasets.append(dataset)
            logger.info(f"Loaded {len(dataset)} examples for task: {task}")
        except FileNotFoundError as e:
            logger.warning(f"Skipping task {task}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets were loaded successfully")

    # Combine datasets
    combined = concatenate_datasets(datasets)
    logger.info(f"Combined dataset size: {len(combined)}")

    # For random mode, shuffle the combined dataset
    if training_mode == "random" and shuffle:
        combined = combined.shuffle(seed=seed)
        logger.info("Shuffled combined dataset (random mode)")
    elif training_mode == "curriculum":
        # For curriculum learning, data is already in order (analysis -> continuation -> generation -> restoration)
        logger.info("Keeping curriculum order (curriculum mode)")

    return combined


def get_poetry_datasets(
    base_path: str,
    tasks: List[str],
    training_mode: str = "random",
    max_samples_per_task: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    prompt_type: str = "chat",
    num_proc: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Get train and test datasets for poetry tasks with caching support.

    Args:
        base_path: Base path to the TSV data directory
        tasks: List of tasks to load
        training_mode: 'curriculum' or 'random'
        max_samples_per_task: Maximum samples per task
        shuffle: Whether to shuffle
        seed: Random seed
        prompt_type: 'chat' or 'instruction'
        num_proc: Number of processes for data processing
        use_cache: Whether to use cached processed datasets
        cache_dir: Directory for caching (default: ~/.cache/poetry_datasets)

    Returns:
        DatasetDict with 'train' and 'test' splits (already formatted)
    """
    cache_path = get_cache_path(training_mode, prompt_type, cache_dir)

    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading cached dataset from {cache_path}")
        try:
            cached_dataset = load_from_disk(cache_path)
            logger.info(
                f"Loaded cached dataset: train={len(cached_dataset['train'])}, test={len(cached_dataset['test'])}"
            )
            return cached_dataset
        except Exception as e:
            logger.warning(f"Failed to load cache, rebuilding: {e}")

    # Load raw datasets
    logger.info("Loading and processing datasets (this may take a while)...")

    train_dataset = load_poetry_datasets(
        base_path=base_path,
        tasks=tasks,
        training_mode=training_mode,
        split="train",
        max_samples_per_task=max_samples_per_task,
        shuffle=shuffle,
        seed=seed,
    )

    test_dataset = load_poetry_datasets(
        base_path=base_path,
        tasks=tasks,
        training_mode=training_mode,
        split="test",
        max_samples_per_task=max_samples_per_task if max_samples_per_task else None,
        shuffle=False,  # Don't shuffle test data
        seed=seed,
    )

    # Format the datasets
    logger.info("Formatting train dataset...")
    train_dataset = prepare_poetry_dataset_for_training(
        train_dataset,
        prompt_type=prompt_type,
        task="train",
        num_proc=num_proc,
    )

    logger.info("Formatting test dataset...")
    test_dataset = prepare_poetry_dataset_for_training(
        test_dataset,
        prompt_type=prompt_type,
        task="train",  # Use same format for eval during training
        num_proc=num_proc,
    )

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Save to cache
    if use_cache:
        logger.info(f"Saving processed dataset to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            dataset_dict.save_to_disk(cache_path)
            logger.info("Dataset cached successfully!")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    return dataset_dict


def format_poetry_prompt(
    example: Dict[str, Any], prompt_type: str = "chat", task: str = "train"
) -> Dict[str, Any]:
    """
    Format a poetry example into the appropriate prompt format.

    Args:
        example: Data example with 'input' and 'output' fields
        prompt_type: 'chat' or 'instruction'
        task: 'train' or 'evaluation'

    Returns:
        Formatted example with 'text' or 'messages' field
    """
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if prompt_type == "chat":
        prompt = [{"role": "user", "content": input_text}]
        if task == "train":
            completion = [{"role": "assistant", "content": output_text}]
        else:
            completion = []
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    return {"prompt": prompt, "completion": completion}


def prepare_poetry_dataset_for_training(
    dataset: Dataset,
    prompt_type: str = "instruction",
    task: str = "train",
    num_proc: int = 4,
) -> Dataset:
    """
    Prepare a poetry dataset for training by formatting prompts.

    Args:
        dataset: Raw dataset with 'input' and 'output' columns
        prompt_type: 'chat' or 'instruction'
        task: 'train' or 'evaluation'
        num_proc: Number of processes for mapping

    Returns:
        Processed dataset ready for training
    """
    # Keep only the new fields 'prompt' and 'completion'
    columns_to_remove = [
        col for col in dataset.column_names if col not in ["prompt", "completion"]
    ]

    processed = dataset.map(
        lambda x: format_poetry_prompt(x, prompt_type=prompt_type, task=task),
        num_proc=num_proc,
        remove_columns=columns_to_remove,
        desc=f"Formatting prompts ({prompt_type})",
    )

    return processed


# ============================================================================
# Tokenization Caching Functions
# ============================================================================


def get_tokenized_cache_path(
    training_mode: str,
    prompt_type: str,
    model_name: str,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Generate a cache path for tokenized datasets.

    Cache key includes: training_mode, prompt_type, and model_name
    This ensures different models get their own tokenized cache.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # Clean model name (replace / with _)
    model_name_clean = model_name.replace("/", "_").replace("\\", "_")

    cache_name = f"poetry_{training_mode}_{prompt_type}_{model_name_clean}_tokenized"
    return os.path.join(cache_dir, cache_name)


def tokenize_example(
    example: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_length: int = 2048,
) -> Dict[str, Any]:
    """
    Tokenize a single example with prompt and completion.

    Uses the tokenizer's chat template to format the conversation,
    then creates labels with -100 for prompt tokens (completion-only loss).
    """
    prompt = example["prompt"]
    completion = example["completion"]

    # Combine prompt and completion into full conversation
    messages = prompt + completion

    # Apply chat template to get the full text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize the full conversation
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # We'll pad in the data collator
        return_tensors=None,
    )

    # For completion-only loss, we need to mask the prompt tokens
    # Get the prompt-only text to find where completion starts
    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,  # Include the generation prompt marker
    )

    # Tokenize prompt to get its length
    prompt_tokenized = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    prompt_length = len(prompt_tokenized["input_ids"])

    # Create labels: -100 for prompt tokens, actual token ids for completion
    labels = [-100] * prompt_length + tokenized["input_ids"][prompt_length:]

    # Ensure labels has same length as input_ids
    if len(labels) < len(tokenized["input_ids"]):
        labels = labels + tokenized["input_ids"][len(labels) :]
    elif len(labels) > len(tokenized["input_ids"]):
        labels = labels[: len(tokenized["input_ids"])]

    tokenized["labels"] = labels

    return tokenized


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: "PreTrainedTokenizer",
    max_seq_length: int = 2048,
    num_proc: int = 4,
) -> Dataset:
    """
    Tokenize a dataset with parallel processing.

    Args:
        dataset: Dataset with 'prompt' and 'completion' fields
        tokenizer: The tokenizer to use
        max_seq_length: Maximum sequence length
        num_proc: Number of processes for parallel tokenization

    Returns:
        Tokenized dataset with input_ids, attention_mask, and labels
    """

    def tokenize_fn(examples):
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for i in range(len(examples["prompt"])):
            example = {
                "prompt": examples["prompt"][i],
                "completion": examples["completion"][i],
            }
            tokenized = tokenize_example(example, tokenizer, max_seq_length)
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            results["labels"].append(tokenized["labels"])

        return results

    # Remove old columns and add tokenized columns
    columns_to_remove = dataset.column_names

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def get_tokenized_poetry_datasets(
    base_path: str,
    tasks: List[str],
    training_mode: str,
    prompt_type: str,
    tokenizer: "PreTrainedTokenizer",
    model_name: str,
    max_seq_length: int = 2048,
    max_samples_per_task: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    num_proc: int = 4,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Get tokenized train and test datasets with caching.

    This function:
    1. First checks for tokenized cache (model-specific)
    2. If not found, loads formatted cache or raw data
    3. Tokenizes the data with parallel processing
    4. Caches the tokenized result

    Args:
        base_path: Base path to the TSV data directory
        tasks: List of tasks to load
        training_mode: 'curriculum' or 'random'
        prompt_type: 'chat' or 'instruction'
        tokenizer: The tokenizer to use
        model_name: Model name for cache key
        max_seq_length: Maximum sequence length for tokenization
        max_samples_per_task: Maximum samples per task
        shuffle: Whether to shuffle
        seed: Random seed
        num_proc: Number of processes for tokenization
        use_cache: Whether to use caching
        cache_dir: Directory for caching

    Returns:
        DatasetDict with tokenized 'train' and 'test' splits
    """
    tokenized_cache_path = get_tokenized_cache_path(
        training_mode, prompt_type, model_name, cache_dir
    )

    # Try to load tokenized cache
    if use_cache and os.path.exists(tokenized_cache_path):
        logger.info(f"Loading tokenized cache from {tokenized_cache_path}")
        try:
            cached_dataset = load_from_disk(tokenized_cache_path)
            logger.info(
                f"Loaded tokenized cache: train={len(cached_dataset['train'])}, "
                f"test={len(cached_dataset['test'])}"
            )
            return cached_dataset
        except Exception as e:
            logger.warning(f"Failed to load tokenized cache, rebuilding: {e}")

    # Load formatted datasets (may come from cache)
    logger.info("Loading formatted datasets...")
    formatted_datasets = get_poetry_datasets(
        base_path=base_path,
        tasks=tasks,
        training_mode=training_mode,
        max_samples_per_task=max_samples_per_task,
        shuffle=shuffle,
        seed=seed,
        prompt_type=prompt_type,
        num_proc=num_proc,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    # Tokenize the datasets
    logger.info(f"Tokenizing datasets with {num_proc} workers...")

    train_tokenized = tokenize_dataset(
        formatted_datasets["train"],
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_proc=num_proc,
    )

    test_tokenized = tokenize_dataset(
        formatted_datasets["test"],
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_proc=num_proc,
    )

    tokenized_dict = DatasetDict(
        {
            "train": train_tokenized,
            "test": test_tokenized,
        }
    )

    # Cache the tokenized datasets
    if use_cache:
        logger.info(f"Saving tokenized dataset to cache: {tokenized_cache_path}")
        os.makedirs(os.path.dirname(tokenized_cache_path), exist_ok=True)
        try:
            tokenized_dict.save_to_disk(tokenized_cache_path)
            logger.info("Tokenized dataset cached successfully!")
        except Exception as e:
            logger.warning(f"Failed to save tokenized cache: {e}")

    return tokenized_dict
