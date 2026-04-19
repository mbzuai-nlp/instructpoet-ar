#!/usr/bin/env python3
"""
Remove LM Head and Embedding Layers from LoRA Adapters

This script processes LoRA adapter safetensors files by removing the lm_head
and embedding layer weights, then saves the cleaned version with the original
name while backing up the original file.

Usage:
    python remove_lm_head_and_embeddings.py --adapter_dir /path/to/checkpoint
    python remove_lm_head_and_embeddings.py --checkpoint_base /path/to/checkpoints
"""

import argparse
import os
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
from typing import Dict
import torch


def remove_unwanted_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove lm_head and embedding layer weights from state dict.
    
    Args:
        state_dict: Dictionary of tensor weights
        
    Returns:
        Cleaned state dict without lm_head and embedding weights
    """
    cleaned_dict = {}
    removed_keys = []
    
    for key, value in state_dict.items():
        # Skip lm_head and embedding layers
        if any(pattern in key.lower() for pattern in ['lm_head', 'embed_tokens', 'wte', 'wpe', 'word_embeddings']):
            removed_keys.append(key)
            continue
        cleaned_dict[key] = value
    
    print(f"  Removed {len(removed_keys)} weight(s):")
    for key in removed_keys:
        print(f"    - {key}")
    
    return cleaned_dict


def process_adapter(adapter_path: str, backup_suffix: str = ".backup") -> bool:
    """
    Process a single adapter by removing unwanted weights.
    
    Args:
        adapter_path: Path to the adapter_model.safetensors file
        backup_suffix: Suffix to add to backup file
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(adapter_path):
        print(f"  ⚠️  Adapter file not found: {adapter_path}")
        return False
    
    try:
        # Load the safetensors file
        print(f"  Loading {adapter_path}...")
        state_dict = load_file(adapter_path)
        
        print(f"  Original model has {len(state_dict)} weight(s)")
        
        # Remove unwanted weights
        cleaned_dict = remove_unwanted_weights(state_dict)
        
        print(f"  Cleaned model has {len(cleaned_dict)} weight(s)")
        
        # Only proceed if we actually removed something
        if len(cleaned_dict) == len(state_dict):
            print(f"  ℹ️  No weights to remove, skipping...")
            return True
        
        # Backup original file
        backup_path = adapter_path + backup_suffix
        print(f"  Backing up original to {os.path.basename(backup_path)}...")
        shutil.copy2(adapter_path, backup_path)
        
        # Save cleaned version with original name
        print(f"  Saving cleaned version...")
        save_file(cleaned_dict, adapter_path)
        
        print(f"  ✓ Successfully processed adapter")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing adapter: {e}")
        return False


def find_adapter_dirs(checkpoint_base: str) -> list:
    """
    Find all directories containing adapter_model.safetensors files.
    
    Args:
        checkpoint_base: Base directory to search
        
    Returns:
        List of paths to adapter files
    """
    adapter_files = []
    
    for root, dirs, files in os.walk(checkpoint_base):
        if "adapter_model.safetensors" in files:
            adapter_path = os.path.join(root, "adapter_model.safetensors")
            adapter_files.append(adapter_path)
    
    return sorted(adapter_files)


def main():
    parser = argparse.ArgumentParser(
        description="Remove LM head and embedding layers from LoRA adapters"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--adapter_dir",
        type=str,
        help="Path to a specific adapter directory containing adapter_model.safetensors"
    )
    group.add_argument(
        "--checkpoint_base",
        type=str,
        help="Base checkpoint directory to search for all adapters"
    )
    
    parser.add_argument(
        "--backup_suffix",
        type=str,
        default=".backup",
        help="Suffix for backup files (default: .backup)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list what would be processed without making changes"
    )
    
    args = parser.parse_args()
    
    # Collect adapter files to process
    adapter_files = []
    
    if args.adapter_dir:
        adapter_path = os.path.join(args.adapter_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            adapter_files = [adapter_path]
        else:
            print(f"Error: adapter_model.safetensors not found in {args.adapter_dir}")
            return 1
    else:
        print(f"Searching for adapters in {args.checkpoint_base}...")
        adapter_files = find_adapter_dirs(args.checkpoint_base)
    
    if not adapter_files:
        print("No adapter files found.")
        return 1
    
    print(f"\nFound {len(adapter_files)} adapter(s) to process:")
    for adapter_path in adapter_files:
        print(f"  - {adapter_path}")
    
    if args.dry_run:
        print("\nDry run mode - no changes will be made.")
        return 0
    
    # Process each adapter
    print(f"\n{'='*60}")
    print("Processing adapters...")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, adapter_path in enumerate(adapter_files, 1):
        print(f"[{i}/{len(adapter_files)}] Processing {os.path.dirname(adapter_path)}...")
        if process_adapter(adapter_path, args.backup_suffix):
            success_count += 1
        else:
            fail_count += 1
        print()
    
    # Summary
    print(f"{'='*60}")
    print("Summary:")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  Total: {len(adapter_files)}")
    print(f"{'='*60}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
