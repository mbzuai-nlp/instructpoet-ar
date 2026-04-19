#!/usr/bin/env python3
"""
Merge LoRA Adapters with Base Model

This script merges LoRA adapter weights with their base model to create
a standalone model that can be used without PEFT.

Usage:
    python merge_lora.py --adapter_path /path/to/checkpoint --output_path /path/to/output
    
    # Or use default paths:
    python merge_lora.py --checkpoint checkpoint-35000

Example:
    python merge_lora.py \
        --adapter_path /path/to/checkpoints/Qwen3-8B/random/4tasks/checkpoint-35000 \
        --output_path /path/to/merged_models/Qwen3-8B-poetry-merged
"""

import argparse
import os
import json
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM


def load_adapter_config(adapter_path: str) -> dict:
    """Load and return the adapter configuration."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def find_latest_checkpoint(folder_path: str) -> str:
    """Find the latest checkpoint in a folder based on checkpoint number."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    checkpoints = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            # Extract checkpoint number
            match = re.search(r"checkpoint-(\d+)", item)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoints.append((checkpoint_num, item_path))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {folder_path}")

    # Sort by checkpoint number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest = checkpoints[0][1]

    print(f"   Found {len(checkpoints)} checkpoint(s)")
    print(f"   Latest: {os.path.basename(latest)}")

    return latest


def find_all_model_folders(base_path: str) -> list:
    """
    Find all model training folders in the checkpoint directory.
    Returns list of tuples: (model_name, training_mode, tasks, folder_path)
    """
    folders = []

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base checkpoint path does not exist: {base_path}")

    # Iterate through model names (e.g., Qwen3-8B, ALLaM-7B-Instruct-preview)
    for model_name in os.listdir(base_path):
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            continue

        # Iterate through training modes (e.g., curriculum, random)
        for training_mode in os.listdir(model_path):
            mode_path = os.path.join(model_path, training_mode)
            if not os.path.isdir(mode_path):
                continue

            # Iterate through task configurations (e.g., 4tasks, 2tasks)
            for tasks in os.listdir(mode_path):
                tasks_path = os.path.join(mode_path, tasks)
                if not os.path.isdir(tasks_path):
                    continue

                # Check if this folder contains checkpoints
                has_checkpoints = any(
                    item.startswith("checkpoint-")
                    for item in os.listdir(tasks_path)
                    if os.path.isdir(os.path.join(tasks_path, item))
                )

                if has_checkpoints:
                    folders.append((model_name, training_mode, tasks, tasks_path))

    return folders


def merge_and_save(
    adapter_path: str,
    output_path: str,
    base_model_override: str = None,
    device_map: str = "auto",
    max_memory: dict = None,
):
    """
    Merge LoRA adapters with base model and save the result.

    Args:
        adapter_path: Path to the LoRA adapter checkpoint
        output_path: Path where merged model will be saved
        base_model_override: Optional override for base model path/name
        device_map: Device mapping strategy (default: "auto")
        max_memory: Optional max memory per device
    """
    print("=" * 80)
    print("LoRA Adapter Merger")
    print("=" * 80)

    # Load adapter config to get base model
    adapter_config = load_adapter_config(adapter_path)
    base_model_name = base_model_override or adapter_config["base_model_name_or_path"]

    print(f"\n📁 Adapter path: {adapter_path}")
    print(f"🤖 Base model: {base_model_name}")
    print(f"💾 Output path: {output_path}")
    print(f"🎯 LoRA rank (r): {adapter_config['r']}")
    print(f"🎯 LoRA alpha: {adapter_config['lora_alpha']}")
    print(f"🎯 Target modules: {', '.join(adapter_config['target_modules'])}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load base model
    print(f"\n⏳ Loading base model: {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )

    # Load tokenizer from adapter path (which may have been resized)
    print(f"⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # Check if tokenizer was resized during training
    original_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)

    if original_vocab_size != tokenizer_vocab_size:
        print(f"\n⚠️  Vocabulary size mismatch detected!")
        print(f"   Base model vocab size: {original_vocab_size}")
        print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
        print(
            f"   ℹ️  The LoRA adapter contains resized embeddings, but we'll ignore them"
        )
        print(f"   ℹ️  Only merging the actual LoRA weights (q_proj, k_proj, etc.)")

    # Load adapter state dict and filter out embedding/lm_head weights
    print(f"\n⏳ Loading LoRA adapters from: {adapter_path}...")
    adapter_state_dict_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_state_dict_path):
        adapter_state_dict_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(adapter_state_dict_path):
        print(f"   Loading from: {os.path.basename(adapter_state_dict_path)}")
        if adapter_state_dict_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            adapter_state_dict = load_file(adapter_state_dict_path)
        else:
            adapter_state_dict = torch.load(adapter_state_dict_path, map_location="cpu")

        # Filter out embedding and lm_head weights
        original_keys = list(adapter_state_dict.keys())
        filtered_keys = [
            k for k in original_keys if "embed_tokens" not in k and "lm_head" not in k
        ]

        if len(original_keys) != len(filtered_keys):
            removed_keys = set(original_keys) - set(filtered_keys)
            print(f"\n   🔧 Filtering out non-LoRA weights:")
            for key in sorted(removed_keys):
                print(f"      - {key}")

            # Create filtered state dict
            filtered_state_dict = {
                k: v for k, v in adapter_state_dict.items() if k in filtered_keys
            }

            # Save filtered adapter temporarily
            temp_adapter_path = os.path.join(output_path, "temp_filtered_adapter")
            os.makedirs(temp_adapter_path, exist_ok=True)

            # Save filtered weights
            if adapter_state_dict_path.endswith(".safetensors"):
                from safetensors.torch import save_file

                save_file(
                    filtered_state_dict,
                    os.path.join(temp_adapter_path, "adapter_model.safetensors"),
                )
            else:
                torch.save(
                    filtered_state_dict,
                    os.path.join(temp_adapter_path, "adapter_model.bin"),
                )

            # Copy adapter config
            import shutil

            shutil.copy(
                os.path.join(adapter_path, "adapter_config.json"),
                os.path.join(temp_adapter_path, "adapter_config.json"),
            )

            print(f"   ✅ Filtered adapter saved to temporary location")
            adapter_path_to_load = temp_adapter_path
        else:
            adapter_path_to_load = adapter_path
    else:
        adapter_path_to_load = adapter_path

    # Load LoRA model with filtered adapters
    print(f"\n⏳ Loading LoRA adapters into base model...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path_to_load,
        device_map=device_map,
    )

    # Merge adapters into base model
    print(f"⏳ Merging LoRA adapters with base model...")
    merged_model = model.merge_and_unload()

    # Save merged model
    print(f"⏳ Saving merged model to: {output_path}...")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Save tokenizer (use the original base model tokenizer, not the resized one)
    print(f"⏳ Saving tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    base_tokenizer.save_pretrained(output_path)

    # Clean up temporary filtered adapter if created
    temp_adapter_path = os.path.join(output_path, "temp_filtered_adapter")
    if os.path.exists(temp_adapter_path):
        import shutil

        shutil.rmtree(temp_adapter_path)
        print(f"   🧹 Cleaned up temporary files")

    # Save merge info
    merge_info = {
        "adapter_path": adapter_path,
        "base_model": base_model_name,
        "lora_config": adapter_config,
        "vocab_size_original": original_vocab_size,
        "vocab_size_adapter": tokenizer_vocab_size,
        "filtered_embedding_weights": original_vocab_size != tokenizer_vocab_size,
        "merged_at": str(torch.cuda.Event(enable_timing=False)),
    }

    info_path = os.path.join(output_path, "merge_info.json")
    with open(info_path, "w") as f:
        json.dump(merge_info, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Merge completed successfully!")
    print("=" * 80)
    print(f"\n📦 Merged model saved to: {output_path}")
    print(f"📄 Merge info saved to: {info_path}")
    print("\nYou can now use this model like any other Hugging Face model:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{output_path}")')

    if original_vocab_size != tokenizer_vocab_size:
        print(
            f"\n⚠️  Note: The merged model uses the original tokenizer with vocab size {original_vocab_size}"
        )
        print(f"   The resized embeddings from training were ignored during merge.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge latest checkpoint from all model folders
  python merge_lora.py --batch

  # Merge with automatic base model detection
  python merge_lora.py \\
      --adapter_path /path/to/checkpoint-35000 \\
      --output_path /path/to/merged_model

  # Override base model path
  python merge_lora.py \\
      --adapter_path /path/to/checkpoint-35000 \\
      --base_model /path/to/local/base/model \\
      --output_path /path/to/merged_model

  # Using checkpoint shorthand
  python merge_lora.py --checkpoint checkpoint-35000
  
  # Merge latest checkpoint from specific folder
  python merge_lora.py --folder /path/to/task/folder
        """,
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Path to LoRA adapter checkpoint directory",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path where merged model will be saved",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Override base model name/path (optional, will use adapter_config.json by default)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint name (e.g., checkpoint-35000). Uses default paths for Qwen3-8B random 4tasks.",
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing checkpoints. Will merge the latest checkpoint.",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: merge latest checkpoint from all model folders in checkpoint directory",
    )

    parser.add_argument(
        "--checkpoint_base",
        type=str,
        default="/path/to/checkpoints",
        help="Base checkpoint directory (default: /path/to/checkpoints)",
    )

    parser.add_argument(
        "--output_base",
        type=str,
        default="/path/to/merged_models",
        help="Base output directory for merged models (default: /path/to/merged_models)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-8B",
        help="Model name for default paths (default: Qwen3-8B)",
    )

    parser.add_argument(
        "--training_mode",
        type=str,
        default="random",
        help="Training mode for default paths (default: random)",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="4tasks",
        help="Tasks identifier for default paths (default: 4tasks)",
    )

    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map strategy (default: auto)",
    )

    args = parser.parse_args()

    # Batch mode: process all model folders
    if args.batch:
        print("=" * 80)
        print("BATCH MODE: Merging latest checkpoints from all folders")
        print("=" * 80)

        model_folders = find_all_model_folders(args.checkpoint_base)

        if not model_folders:
            print(
                f"\n❌ No model folders with checkpoints found in {args.checkpoint_base}"
            )
            return

        print(f"\nFound {len(model_folders)} model folder(s) to process:\n")
        for i, (model_name, training_mode, tasks, folder_path) in enumerate(
            model_folders, 1
        ):
            print(f"{i}. {model_name}/{training_mode}/{tasks}")

        print("\n" + "=" * 80 + "\n")

        for i, (model_name, training_mode, tasks, folder_path) in enumerate(
            model_folders, 1
        ):
            print(f"\n{'='*80}")
            print(
                f"Processing {i}/{len(model_folders)}: {model_name}/{training_mode}/{tasks}"
            )
            print(f"{'='*80}\n")

            try:
                # Find latest checkpoint in this folder
                print(f"📁 Searching for checkpoints in: {folder_path}")
                adapter_path = find_latest_checkpoint(folder_path)

                # Create output path
                checkpoint_name = os.path.basename(adapter_path)
                output_path = os.path.join(
                    args.output_base,
                    f"{model_name}-{training_mode}-{tasks}-{checkpoint_name}",
                )

                # Check if already merged
                if os.path.exists(output_path) and os.path.exists(
                    os.path.join(output_path, "config.json")
                ):
                    print(f"\n⏭️  Skipping: Already merged at {output_path}\n")
                    continue

                # Merge
                merge_and_save(
                    adapter_path=adapter_path,
                    output_path=output_path,
                    base_model_override=args.base_model,
                    device_map=args.device_map,
                )

                print(f"\n✅ Successfully merged {i}/{len(model_folders)}")

            except Exception as e:
                print(f"\n❌ Error processing {model_name}/{training_mode}/{tasks}:")
                print(f"   {str(e)}")
                print(f"   Continuing with next folder...\n")
                continue

        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        return

    # Folder mode: merge latest checkpoint from specific folder
    if args.folder:
        print(f"📁 Searching for checkpoints in: {args.folder}")
        adapter_path = find_latest_checkpoint(args.folder)

        if not args.output_path:
            # Extract meaningful name from folder path
            parts = Path(args.folder).parts
            if len(parts) >= 3:
                model_name = parts[-3]
                training_mode = parts[-2]
                tasks = parts[-1]
                checkpoint_name = os.path.basename(adapter_path)
                output_path = os.path.join(
                    args.output_base,
                    f"{model_name}-{training_mode}-{tasks}-{checkpoint_name}",
                )
            else:
                checkpoint_name = os.path.basename(adapter_path)
                output_path = os.path.join(
                    args.output_base, checkpoint_name + "-merged"
                )
        else:
            output_path = args.output_path

    # Handle checkpoint shorthand
    elif args.checkpoint:
        adapter_path = os.path.join(
            args.checkpoint_base,
            args.model_name,
            args.training_mode,
            args.tasks,
            args.checkpoint,
        )

        if not args.output_path:
            output_path = os.path.join(
                args.output_base,
                f"{args.model_name}-{args.training_mode}-{args.tasks}-{args.checkpoint}",
            )
        else:
            output_path = args.output_path
    else:
        if not args.adapter_path or not args.output_path:
            parser.error(
                "Either --batch, --folder, --checkpoint, or both --adapter_path and --output_path are required"
            )
        adapter_path = args.adapter_path
        output_path = args.output_path

    # Verify adapter path exists
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    # Run merge
    merge_and_save(
        adapter_path=adapter_path,
        output_path=output_path,
        base_model_override=args.base_model,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
