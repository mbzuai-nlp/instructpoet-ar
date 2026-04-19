#!/bin/bash
#
# Merge all latest checkpoints from checkpoint directory
#
# This script finds all model training folders and merges the latest
# checkpoint from each folder automatically.
#
# Usage:
#   bash merge_all_latest.sh

set -e

echo "======================================"
echo "Batch LoRA Adapter Merge Script"
echo "======================================"
echo ""
echo "This will merge the latest checkpoint from each model folder"
echo "in /path/to/checkpoints"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the merge script in batch mode
python "$SCRIPT_DIR/merge_lora.py" --batch

echo ""
echo "======================================"
echo "All merges completed!"
echo "======================================"
echo ""
echo "Merged models are saved in:"
echo "  /path/to/merged_models"
echo ""
