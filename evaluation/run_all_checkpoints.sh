#!/bin/bash
# Script to run inference on all model checkpoints and their base models
# This script iterates through all models in the checkpoints directory,
# finds the last checkpoint for each training run, and runs inference.
#
# SLURM directives
#SBATCH --job-name=poetry_infer_all
#SBATCH --output=./logs/run_all_checkpoints-%j.out
#SBATCH --error=./logs/run_all_checkpoints-%j.err
#SBATCH --account=ifm-miscellaneous-1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

set -e  # Exit on error

# Get the directory where this script is located
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    # Running under SLURM - use submission directory
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    # Running directly with bash
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Paths
CHECKPOINTS_BASE="${CHECKPOINTS_BASE:-/path/to/checkpoints}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-$SCRIPT_DIR/inference.sh}"

# Check if inference script exists
if [ ! -f "$INFERENCE_SCRIPT" ]; then
    echo "Error: Inference script not found at $INFERENCE_SCRIPT"
    exit 1
fi

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_BASE" ]; then
    echo "Error: Checkpoints directory not found at $CHECKPOINTS_BASE"
    exit 1
fi

echo "========================================"
echo "Starting inference on all checkpoints"
echo "========================================"
echo "Checkpoints base: $CHECKPOINTS_BASE"
echo "Inference script: $INFERENCE_SCRIPT"
echo ""

# Iterate through each model directory
for model_dir in "$CHECKPOINTS_BASE"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model_name=$(basename "$model_dir")
    echo "========================================="
    echo "Processing model: $model_name"
    echo "========================================="
    
    # Find all unique parent directories that contain checkpoint-* subdirectories
    # This gives us unique training configurations (e.g., curriculum/4tasks, random/4tasks)
    checkpoint_parents=$(find "$model_dir" -type d -name "checkpoint-*" -exec dirname {} \; | sort -u)
    
    if [ -z "$checkpoint_parents" ]; then
        echo "  No checkpoint directories found in $model_dir"
    else
        # For each parent directory, find the last checkpoint
        while IFS= read -r parent_dir; do
            # Get the last checkpoint (highest number) in this parent directory
            last_checkpoint=$(find "$parent_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
            
            if [ -n "$last_checkpoint" ]; then
                echo ""
                echo "  Training path: ${parent_dir#$CHECKPOINTS_BASE/}"
                echo "  Last checkpoint: $(basename $last_checkpoint)"
                echo "  Full path: $last_checkpoint"
                echo "  Submitting inference job to SLURM..."
                
                # Submit inference job to SLURM
                job_id=$(sbatch --parsable "$INFERENCE_SCRIPT" --model_path "$last_checkpoint")
                
                if [ $? -eq 0 ]; then
                    echo "  ✓ Inference job submitted successfully: Job ID $job_id"
                else
                    echo "  ✗ Failed to submit inference job"
                fi
            fi
        done <<< "$checkpoint_parents"
    fi
    
    # Run inference on the base model (the model directory itself)
    # Only if it contains model files (not just subdirectories)
    if [ -f "$model_dir/config.json" ] || [ -f "$model_dir/adapter_config.json" ]; then
        echo ""
        echo "  Submitting inference job for base model: $model_dir"
        job_id=$(sbatch --parsable "$INFERENCE_SCRIPT" --model_path "$model_dir")
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Inference job submitted successfully: Job ID $job_id"
            echo "     Base model: $model_name"
        else
            echo "  ✗ Failed to submit inference job for base model $model_name"
        fi
    fi
    
    echo ""
done

echo "========================================"
echo "All inference jobs submitted!"
echo "Use 'squeue -u $USER' to check job status"
echo "========================================"
