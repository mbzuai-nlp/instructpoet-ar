#!/usr/bin/env python
# coding=utf-8
"""
Interactive Arabic Poetry Inference Script
Test model generation with improved decoding settings to debug repetition issues.
Press Enter to cycle through examples from the test set.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import shutil

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from safetensors.torch import safe_open, save_file

try:
    from termcolor import colored
except ImportError:
    # Fallback if termcolor is not installed
    def colored(text, color=None, attrs=None):
        return text


# Environment setup
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================================
# Helper functions from inference_runner.py
# ============================================================================


def format_prompt_for_inference(input_text: str, prompt_type: str = "chat") -> str:
    """Format input text as a prompt."""
    if prompt_type == "chat":
        # Chat format for instruction-tuned models
        messages = [{"role": "user", "content": input_text}]
        return messages
    else:
        # Simple instruction format
        return input_text


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


def print_separator(char="=", length=100):
    """Print a separator line."""
    print(colored(char * length, "cyan"))


def print_header(text):
    """Print a colored header."""
    print_separator()
    print(colored(text.center(100), "yellow", attrs=["bold"]))
    print_separator()


def print_section(title, content, color="green"):
    """Print a section with title and content."""
    print(colored(f"\n{title}:", color, attrs=["bold"]))
    print(colored("-" * 100, color))
    print(content)
    print(colored("-" * 100, color))


def load_test_data(file_path: str) -> pd.DataFrame:
    """Load the test data from TSV or JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type by extension
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".json":
        df = pd.read_json(file_path)
        print(f"\n✓ Loaded {len(df)} samples from JSON test set")
    elif file_ext in [".tsv", ".txt"]:
        df = pd.read_csv(file_path, sep="\t", on_bad_lines="skip")
        print(f"\n✓ Loaded {len(df)} samples from TSV test set")
    elif file_ext == ".csv":
        df = pd.read_csv(file_path, on_bad_lines="skip")
        print(f"\n✓ Loaded {len(df)} samples from CSV test set")
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Supported formats: .json, .tsv, .csv"
        )

    # Show available columns
    print(f"✓ Columns: {list(df.columns)}")

    return df


def initialize_model(
    model_path: str,
    adapter_path: str = None,
    tensor_parallel_size: int = 1,
) -> LLM:
    """Initialize the vLLM model."""
    print_header("INITIALIZING MODEL")

    enable_lora = adapter_path is not None
    vllm_adapter_path = None

    if enable_lora:
        print(f"✓ Preparing LoRA adapter from: {adapter_path}")
        vllm_adapter_path = materialize_vllm_adapter(adapter_path)
        print(f"✓ Using filtered LoRA adapter: {vllm_adapter_path}")

    print(f"✓ Loading model: {model_path}")
    print(f"✓ Tensor parallel size: {tensor_parallel_size}")

    llm = LLM(
        model_path,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=1,  # Process one at a time for interactive mode
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    print(colored("✓ Model loaded successfully!\n", "green", attrs=["bold"]))

    return llm, vllm_adapter_path


def generate_text(
    llm: LLM,
    input_text: str,
    sampling_params: SamplingParams,
    lora_request: LoRARequest = None,
    prompt_type: str = "chat",
) -> str:
    """Generate text for a single input."""
    prompt = format_prompt_for_inference(input_text, prompt_type=prompt_type)

    if prompt_type == "chat":
        outputs = llm.chat(
            messages=[prompt],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
    else:
        outputs = llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

    return outputs[0].outputs[0].text.strip()


def interactive_inference(
    llm: LLM,
    test_df: pd.DataFrame,
    sampling_params: SamplingParams,
    adapter_path: str = None,
    prompt_type: str = "chat",
    start_idx: int = 0,
):
    """Run interactive inference - press Enter to see next example."""

    lora_request = None
    if adapter_path:
        lora_request = LoRARequest("poetry_adapter", 1, adapter_path)

    print_header("INTERACTIVE INFERENCE MODE")
    print(colored("\nPress Enter to generate for the next example", "yellow"))
    print(colored("Type 'q' or 'quit' to exit", "yellow"))
    print(colored("Type a number to jump to that example", "yellow"))
    print(colored("Type 'r' to regenerate current example\n", "yellow"))

    idx = start_idx
    current_input = None
    current_output = None

    while idx < len(test_df):
        row = test_df.iloc[idx]
        input_text = row.get("input", row.get("question", ""))
        expected_output = row.get("output", "N/A")

        # Store for regeneration
        current_input = input_text
        current_output = expected_output

        print_separator("=", 100)
        print(colored(f"EXAMPLE {idx + 1} / {len(test_df)}", "cyan", attrs=["bold"]))
        print_separator("=", 100)

        # Print input
        print_section("INPUT", input_text, "green")

        # Print expected output
        print_section("EXPECTED OUTPUT", expected_output, "yellow")

        # Generate
        print(colored("\n⏳ Generating...", "magenta", attrs=["bold"]))
        try:
            generated_text = generate_text(
                llm=llm,
                input_text=input_text,
                sampling_params=sampling_params,
                lora_request=lora_request,
                prompt_type=prompt_type,
            )

            # Print model generation
            print_section("MODEL GENERATION", generated_text, "blue")

            # Check for repetition
            lines = generated_text.split("\n")
            if len(lines) > len(set(lines)) and len(lines) > 5:
                print(
                    colored(
                        "\n⚠️  WARNING: Detected repeated lines in generation!",
                        "red",
                        attrs=["bold"],
                    )
                )

        except Exception as e:
            print(colored(f"\n❌ Error during generation: {e}", "red"))

        # Wait for user input
        print(
            colored(
                "\n[Press Enter for next | Type number to jump | 'r' to regenerate | 'q' to quit]",
                "cyan",
            )
        )
        user_input = input("> ").strip().lower()

        if user_input in ["q", "quit", "exit"]:
            print(colored("\n👋 Exiting...", "yellow"))
            break
        elif user_input == "r":
            # Regenerate current example
            continue
        elif user_input.isdigit():
            # Jump to specific example
            new_idx = int(user_input) - 1
            if 0 <= new_idx < len(test_df):
                idx = new_idx
                continue
            else:
                print(
                    colored(
                        f"❌ Invalid index. Must be between 1 and {len(test_df)}", "red"
                    )
                )
                continue
        else:
            # Move to next example
            idx += 1


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Arabic Poetry Inference - Debug Repetition Issues"
    )

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
        "--test_file",
        type=str,
        required=True,
        help="Path to the test TSV file with predictions",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start from this example index (0-based)",
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="chat",
        choices=["instruction", "chat"],
        help="Prompt format type",
    )

    # Generation arguments - Improved settings to reduce repetition
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling (0.7-1.0 recommended for poetry)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
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
        default=1.1,
        help="Repetition penalty (1.05-1.3 recommended)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="N-gram size for no repeat (3 or 4 recommended)",
    )

    # vLLM arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )

    args = parser.parse_args()

    # Setup sampling parameters with anti-repetition settings
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        # Note: vLLM might not support no_repeat_ngram_size directly
        # If it fails, we'll need to remove it
    )

    print_header("ARABIC POETRY INTERACTIVE INFERENCE")
    print(f"\n📊 Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Adapter: {args.adapter_path or 'None'}")
    print(f"  Test File: {args.test_file}")
    print(f"  Prompt Type: {args.prompt_type}")
    print(f"\n🎛️  Generation Settings (Anti-Repetition):")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Repetition Penalty: {args.repetition_penalty}")
    print(f"  No Repeat N-gram Size: {args.no_repeat_ngram_size}")

    # Load test data
    test_df = load_test_data(args.test_file)

    # Initialize model
    llm, vllm_adapter_path = initialize_model(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Run interactive inference
    interactive_inference(
        llm=llm,
        test_df=test_df,
        sampling_params=sampling_params,
        adapter_path=vllm_adapter_path,
        prompt_type=args.prompt_type,
        start_idx=args.start_idx,
    )

    print_header("SESSION COMPLETE")
    print(colored("Thank you for using the interactive inference tool!", "green"))


if __name__ == "__main__":
    main()
