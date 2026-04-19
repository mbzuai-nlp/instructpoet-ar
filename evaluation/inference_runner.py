#!/usr/bin/env python
# coding=utf-8
"""
Arabic Poetry Inference Runner using vLLM.
Processes TSV files and adds model generation as a new column.
"""

import os
import sys
import json
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

# Add parent directory for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from safetensors.torch import safe_open, save_file
from trl import setup_chat_format

# Environment setup
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Task folder mapping
TASK_FOLDER_MAP = {
    "analysis": "analysis",
    "continuation": "continuation",
    "generation": "generation",
    "restoration": "corruption",
    "corruption": "corruption",
}


def get_model_max_length(model_path: str) -> int:
    """Get the maximum context length from model config."""
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Try different config attributes for max length
        max_length = getattr(config, "max_position_embeddings", None)
        if max_length is None:
            max_length = getattr(config, "max_sequence_length", None)
        if max_length is None:
            max_length = getattr(config, "n_positions", None)
        if max_length is None:
            max_length = getattr(config, "seq_length", None)

        if max_length is None:
            print(
                "Warning: Could not determine max length from config, using default 4096"
            )
            max_length = 4096

        print(f"Model max context length: {max_length}")
        return max_length
    except Exception as e:
        print(f"Warning: Error loading config: {e}. Using default max length 4096")
        return 4096


def truncate_input_to_fit(
    input_text: str,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
    prompt_type: str = "chat",
) -> str:
    """
    Truncate input text to fit within model's context length,
    accounting for space needed for generation.
    """
    # Reserve space for generation and some buffer for chat formatting
    chat_template_overhead = 100  # Estimated tokens for chat template
    max_input_tokens = max_model_len - max_new_tokens - chat_template_overhead

    if max_input_tokens <= 0:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) is too large for model context length ({max_model_len})"
        )

    # Tokenize the input
    if prompt_type == "chat":
        # For chat format, tokenize just the content
        tokens = tokenizer.encode(input_text, add_special_tokens=False)
    else:
        tokens = tokenizer.encode(input_text, add_special_tokens=False)

    # Check if truncation is needed
    if len(tokens) <= max_input_tokens:
        return input_text

    # Truncate tokens and decode back
    truncated_tokens = tokens[:max_input_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    print(f"  Truncated input from {len(tokens)} to {len(truncated_tokens)} tokens")
    return truncated_text


def format_prompt_for_inference(input_text: str, prompt_type: str = "chat") -> str:
    """Format input text as a prompt."""
    if prompt_type == "chat":
        # Chat format for instruction-tuned models
        messages = [{"role": "user", "content": input_text}]
        return messages
    else:
        # Simple instruction format
        return input_text


def load_tsv_data(tsv_path: str, max_samples: int = None) -> pd.DataFrame:
    """Load TSV data file."""
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", on_bad_lines="skip")

    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)

    print(f"Loaded {len(df)} samples from {tsv_path}")
    return df


def get_tsv_files(data_path: str, task: str) -> list:
    """Get all TSV files for a task."""
    task_folder = TASK_FOLDER_MAP.get(task, task)
    task_path = Path(data_path) / task_folder / "test"

    if not task_path.exists():
        raise FileNotFoundError(f"Task test directory not found: {task_path}")

    # Find all TSV files in the test directory
    tsv_files = list(task_path.glob("*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in: {task_path}")

    return [str(f) for f in tsv_files]


def materialize_vllm_adapter(adapter_dir: str) -> str:
    """
    Create a vLLM-friendly LoRA directory without base weights (lm_head, embeddings).
    vLLM only accepts LoRA tensors; PEFT saves extra full-weight tensors we need to drop.
    Returns the directory path containing adapter_model.safetensors + adapter_config.json.
    """
    base = Path(adapter_dir)
    src = base / "adapter_model.safetensors"
    ready_dir = base / "vllm_ready"
    dst = ready_dir / "adapter_model.safetensors"

    if not ready_dir.exists():
        ready_dir.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        print(f"Filtering LoRA weights from {src}...")
        tensors = {}
        with safe_open(src, framework="pt") as f:
            for key in f.keys():
                if "lora" not in key.lower():
                    print(f"  Skipping non-LoRA tensor: {key}")
                    continue  # drop unsupported full-weight tensors
                tensors[key] = f.get_tensor(key)
        print(f"Saving {len(tensors)} LoRA tensors to {dst}")
        save_file(tensors, dst)

    # Copy adapter_config.json alongside the filtered weights
    cfg_src = base / "adapter_config.json"
    cfg_dst = ready_dir / "adapter_config.json"
    if cfg_src.exists() and not cfg_dst.exists():
        shutil.copy2(cfg_src, cfg_dst)

    return str(ready_dir)


def run_vllm_inference(
    model_path: str,
    data_df: pd.DataFrame,
    sampling_params: SamplingParams,
    adapter_path: str = None,
    prompt_type: str = "chat",
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 16,
    enable_thinking: str = "false",
) -> list:
    """Run inference using vLLM with automatic input truncation."""

    # Get model's maximum context length from config
    print("Loading model configuration...")
    max_model_len = get_model_max_length(model_path)
    max_new_tokens = sampling_params.max_tokens

    # Load tokenizer for preprocessing
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Setup chat template if needed for fine-tuned models without chat template
    # Check if this is a fine-tuned model (adapter_path provided) and tokenizer has no chat template
    if adapter_path is not None and tokenizer.chat_template is None:
        print(
            "Fine-tuned model without chat template detected. Setting up ChatML template..."
        )
        # Note: For vLLM, we only need to set the chat template on the tokenizer
        # vLLM will use the tokenizer's chat template for formatting
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        print("ChatML template applied to tokenizer.")

    elif adapter_path is None and tokenizer.chat_template is None:
        ## base model, don't do chat tempelate
        prompt_type = "default"

    # Preprocess data: truncate inputs that are too long
    print("Preprocessing inputs (truncating if necessary)...")
    truncated_count = 0
    processed_inputs = []

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Preprocessing"):
        input_text = row.get("input", "")

        # Truncate if necessary
        original_input = input_text
        truncated_input = truncate_input_to_fit(
            input_text, tokenizer, max_model_len, max_new_tokens, prompt_type
        )

        if len(truncated_input) < len(original_input):
            truncated_count += 1

        processed_inputs.append(truncated_input)

    if truncated_count > 0:
        print(
            f"Truncated {truncated_count}/{len(data_df)} inputs to fit model context length"
        )
    else:
        print("No truncation needed - all inputs fit within model context length")

    # Initialize vLLM
    print("Initializing vLLM...")
    enable_lora = adapter_path is not None

    # Prepare LoRA adapter path (filter out non-LoRA tensors if needed)
    vllm_adapter_path = None
    if enable_lora:
        vllm_adapter_path = materialize_vllm_adapter(adapter_path)
        print(f"Using filtered LoRA adapter from: {vllm_adapter_path}")

    llm = LLM(
        model_path,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    # Prepare prompts from preprocessed inputs
    print("Formatting prompts...")
    prompts = []
    for input_text in tqdm(processed_inputs, desc="Formatting prompts"):
        prompt = format_prompt_for_inference(input_text, prompt_type=prompt_type)
        prompts.append(prompt)

    print(f"Prepared {len(prompts)} prompts")
    if prompts:
        print(f"Sample prompt:\n{prompts[0]}")

    # Run inference
    print("Running inference...")
    lora_request = None
    if enable_lora:
        lora_request = LoRARequest("poetry_adapter", 1, vllm_adapter_path)
        print(f"Created LoRA request for: {vllm_adapter_path}")

    # Convert enable_thinking string to boolean
    enable_thinking_bool = enable_thinking.lower() == "true"
    print(f"Enable thinking: {enable_thinking_bool}")

    if prompt_type == "chat":
        outputs = llm.chat(
            messages=prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
            lora_request=lora_request,
            chat_template_kwargs={"enable_thinking": enable_thinking_bool},
        )
    else:
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
            lora_request=lora_request,
        )

    # Extract generated text
    print("Extracting generated text...")
    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)

    return generated_texts


def save_tsv_with_predictions(
    input_df: pd.DataFrame,
    predictions: list,
    output_path: str,
    model_column_name: str = "model_generation",
):
    """Save TSV file with model predictions as a new column."""
    # Create a copy of the dataframe
    output_df = input_df.copy()

    # Add model predictions
    output_df[model_column_name] = predictions

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to TSV
    output_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved predictions to: {output_path}")

    return output_df


def main():
    parser = argparse.ArgumentParser(description="Arabic Poetry Inference Runner")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or base model",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base path for the poetry TSV data files",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["analysis", "continuation", "generation", "corruption"],
        help="Task to run inference on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for results",
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="chat",
        choices=["instruction", "chat"],
        help="Prompt format type",
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k for sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 means no penalty)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Block repeating n-grams of this size (0 means no blocking)",
    )

    # vLLM arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=16,
        help="Maximum number of sequences to process in parallel",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable thinking mode in chat template (default: false)",
    )

    args = parser.parse_args()

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        # Note: vLLM doesn't support no_repeat_ngram_size in SamplingParams
        # This parameter is handled differently or not available in vLLM
    )

    # Log warning if no_repeat_ngram_size is specified but can't be used
    if args.no_repeat_ngram_size > 0:
        print(
            f"WARNING: no_repeat_ngram_size={args.no_repeat_ngram_size} specified but not supported by vLLM SamplingParams"
        )

    print("=" * 60)
    print("Arabic Poetry Inference Runner")
    print("=" * 60)
    print(f"Model Path: {args.model_path}")
    print(f"Adapter Path: {args.adapter_path or 'None'}")
    print(f"Data Path: {args.data_path}")
    print(f"Task: {args.task}")
    print(f"Output Path: {args.output_path}")
    print(f"Prompt Type: {args.prompt_type}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-P: {args.top_p}")
    print(f"Top-K: {args.top_k}")
    print(f"Repetition Penalty: {args.repetition_penalty}")
    print(f"No Repeat N-gram Size: {args.no_repeat_ngram_size}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Enable Thinking: {args.enable_thinking}")
    print("=" * 60)

    # Get all TSV files for the task
    tsv_files = get_tsv_files(args.data_path, args.task)
    print(f"Found {len(tsv_files)} TSV file(s) for task '{args.task}':")
    for f in tsv_files:
        print(f"  - {f}")

    # Process each TSV file
    for tsv_file in tsv_files:
        print("\n" + "=" * 60)
        print(f"Processing: {tsv_file}")
        print("=" * 60)

        # Load data
        data_df = load_tsv_data(tsv_file, max_samples=args.max_samples)

        # Run inference
        predictions = run_vllm_inference(
            model_path=args.model_path,
            data_df=data_df,
            sampling_params=sampling_params,
            adapter_path=args.adapter_path,
            prompt_type=args.prompt_type,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=args.max_num_seqs,
            enable_thinking=args.enable_thinking,
        )

        # Generate output filename
        tsv_basename = os.path.basename(tsv_file)
        output_filename = tsv_basename.replace(".tsv", "_with_predictions.tsv")
        output_file = os.path.join(args.output_path, args.task, output_filename)

        # Save results
        save_tsv_with_predictions(
            input_df=data_df,
            predictions=predictions,
            output_path=output_file,
            model_column_name="model_generation",
        )

        # Save some sample outputs
        print("\n" + "-" * 60)
        print("Sample predictions (first 3):")
        print("-" * 60)
        for i in range(min(3, len(predictions))):
            print(f"\nSample {i+1}:")
            print(f"Input: {data_df.iloc[i]['input'][:100]}...")
            print(f"Reference: {data_df.iloc[i].get('output', 'N/A')[:100]}...")
            print(f"Generated: {predictions[i][:100]}...")

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print(f"Results saved to: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
