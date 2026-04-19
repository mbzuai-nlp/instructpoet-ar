#!/bin/bash
# Batch merge script for multiple LoRA checkpoints
# This script makes it easy to merge multiple checkpoints at once

# Usage:
#   bash batch_merge.sh checkpoint-35000
#   bash batch_merge.sh checkpoint-34000 checkpoint-34500 checkpoint-35000
#   bash batch_merge.sh all  # merges all available checkpoints

set -e

# Default configuration
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
TRAINING_MODE="${TRAINING_MODE:-random}"
TASKS="${TASKS:-4tasks}"

CHECKPOINT_BASE="${CHECKPOINT_BASE:-/path/to/checkpoints}"
OUTPUT_BASE="${OUTPUT_BASE:-/path/to/merged_models}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   LoRA Adapter Batch Merger${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Model: $MODEL_NAME"
echo "Training Mode: $TRAINING_MODE"
echo "Tasks: $TASKS"
echo ""

# Function to merge a single checkpoint
merge_checkpoint() {
    local checkpoint=$1
    echo -e "${YELLOW}>>> Merging $checkpoint...${NC}"
    
    python merge_lora.py \
        --checkpoint "$checkpoint" \
        --model_name "$MODEL_NAME" \
        --training_mode "$TRAINING_MODE" \
        --tasks "$TASKS"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully merged $checkpoint${NC}"
        echo ""
    else
        echo -e "${YELLOW}✗ Failed to merge $checkpoint${NC}"
        echo ""
        return 1
    fi
}

# Get list of checkpoints
get_all_checkpoints() {
    local checkpoint_dir="$CHECKPOINT_BASE/$MODEL_NAME/$TRAINING_MODE/$TASKS"
    if [ -d "$checkpoint_dir" ]; then
        ls -d "$checkpoint_dir"/checkpoint-* 2>/dev/null | xargs -n 1 basename
    fi
}

# Main logic
if [ $# -eq 0 ]; then
    echo "Usage: bash batch_merge.sh <checkpoint-name> [checkpoint-name2 ...]"
    echo "       bash batch_merge.sh all"
    echo ""
    echo "Available checkpoints for $MODEL_NAME/$TRAINING_MODE/$TASKS:"
    get_all_checkpoints
    exit 1
fi

# Handle "all" option
if [ "$1" == "all" ]; then
    echo "Merging all available checkpoints..."
    CHECKPOINTS=($(get_all_checkpoints))
    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "No checkpoints found in $CHECKPOINT_BASE/$MODEL_NAME/$TRAINING_MODE/$TASKS"
        exit 1
    fi
else
    CHECKPOINTS=("$@")
fi

echo "Checkpoints to merge: ${CHECKPOINTS[@]}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Merge each checkpoint
SUCCESS_COUNT=0
FAIL_COUNT=0

for checkpoint in "${CHECKPOINTS[@]}"; do
    if merge_checkpoint "$checkpoint"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Batch Merge Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Successfully merged: $SUCCESS_COUNT${NC}"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Failed: $FAIL_COUNT${NC}"
fi
echo ""
echo "Merged models saved to: $OUTPUT_BASE"
