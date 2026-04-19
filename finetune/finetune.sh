#!/bin/bash
# Arabic Poetry Finetuning Script
# Supports both SLURM and local bash execution

# SLURM directives (only used when submitted via sbatch)
#SBATCH --job-name=poetry_sft
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --account=ifm-miscellaneous-1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4 # Number of GPUs (per node)
#SBATCH -p gpumid
#SBATCH --cpus-per-task=16


# Usage:
#   SLURM:  sbatch finetune.sh --model Qwen/Qwen3-8B --mode random
#   Local:  bash finetune.sh --model Qwen/Qwen3-8B --mode random
#   Local:  bash finetune.sh --model Qwen/Qwen3-8B --mode random --gpus 2

# conda init
# conda activate revutil

if [[ "$*" == *"--help"* ]]; then
    echo "Usage: bash finetune.sh [options]  OR  sbatch finetune.sh [options]"
    echo ""
    echo "Options:"
    echo "  --model MODEL          Model name or path (required)"
    echo "  --mode MODE            Training mode: random or curriculum (default: random)"
    echo "  --config CONFIG        Config file name (default: config_random.yaml)"
    echo "  --accelerator ACC      Accelerator config: ddp or deepspeed_zero3 (default: ddp)"
    echo "  --tasks TASKS          Comma-separated tasks (default: analysis,continuation,generation,corruption)"
    echo "  --prompt_type TYPE     Prompt type: instruction or chat (default: chat)"
    echo "  --use_peft BOOL        Use LoRA adapters (default: true)"
    echo "  --max_samples N        Max samples per task (default: all)"
    echo "  --gpus N               Number of GPUs for local run (default: auto-detect)"
    echo "  --args \"ARGS\"        Additional arguments to pass to run_sft.py"
    echo ""
    echo "Examples:"
    echo "  # Local run with 2 GPUs"
    echo "  bash finetune.sh --model Qwen/Qwen3-8B --mode random --gpus 2"
    echo ""
    echo "  # SLURM submission"
    echo "  sbatch finetune.sh --model Qwen/Qwen3-8B --mode curriculum"
    echo ""
    echo "  # Quick test with limited samples"
    echo "  bash finetune.sh --model Qwen/Qwen3-8B --max_samples 100 --gpus 1"
    exit 0
fi

set -e  # Exit on error (remove -x to avoid duplicate output)

# Uncomment the next line for debugging (shows each command before execution)
# set -x

echo "START TIME: $(date)"
START_TIME=$(date +%s)

# Create logs directory
mkdir -p logs

##################### ENVIRONMENT SETUP ################

# Get hostname and setup paths
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *gpumid* ]] || [[ "$HOSTNAME" == *cscc* ]]; then
    PARENT_PATH="${CLUSTER_HOME:-/path/to/cluster_home}"
    OUTPUTPATH="${OUTPUTPATH:-$PARENT_PATH/arab_poetry}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PARENT_PATH}"
    export HF_HOME="${HF_HOME:-$PARENT_PATH/huggingface}"
else
    OUTPUTPATH="${OUTPUTPATH:-./checkpoints}"
fi

mkdir -p "$OUTPUTPATH"

# NCCL settings
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_LOG_MODEL=false
##################### ARGUMENT PARSING ################

# Default values
MODEL="Qwen/Qwen3-8B"
TRAINING_MODE="random"
CONFIG_FILE="config.yaml"
ACCELERATOR="ddp"
TASKS="analysis,continuation,generation,corruption"
PROMPT_TYPE="chat"
USE_PEFT="true"
MAX_SAMPLES=""
NUM_GPUS=""
OPTIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --accelerator)
            ACCELERATOR="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --use_peft)
            USE_PEFT="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --args)
            OPTIONAL_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    echo "Run with --help for usage information"
    exit 1
fi

##################### CONFIGURATION ################

# Extract gradient accumulation steps from config
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE 2>/dev/null | awk '{print $2}' || echo "4")

# Check for gradient_accumulation_steps in optional args
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
        GRAD_ACC_STEPS="${arg#*=}"
        break
    fi
done

# Distributed configuration
if [[ -n "$SLURM_JOB_ID" && -n "$SLURM_NNODES" ]]; then
    # Running under SLURM with proper variables set
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
    NUM_NODES=${SLURM_NNODES:-1}
    GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
    WORLD_SIZE=$(($NUM_NODES * $GPUS_PER_NODE))
    NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST 2>/dev/null))
    MASTER_ADDR=${NODELIST[0]:-localhost}
    MASTER_PORT=6000
    USE_SRUN=true
else
    # Local run (bash execution)
    echo "Running locally (bash mode)"
    NUM_NODES=1
    MASTER_ADDR="localhost"
    MASTER_PORT=6000
    USE_SRUN=false
    
    # Determine number of GPUs
    if [[ -n "$NUM_GPUS" ]]; then
        # User specified --gpus
        GPUS_PER_NODE=$NUM_GPUS
    elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        # Use CUDA_VISIBLE_DEVICES
        GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
    else
        # Auto-detect available GPUs
        GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
    fi
    WORLD_SIZE=$GPUS_PER_NODE
fi

# # Set output path based on PEFT setting
# MODEL_SHORT=$(basename $MODEL)
# if [[ "$USE_PEFT" == "true" ]]; then
#     MODEL_OUTPUTPATH="$OUTPUTPATH/adapters/$MODEL_SHORT/$TRAINING_MODE"
# else
#     MODEL_OUTPUTPATH="$OUTPUTPATH/full/$MODEL_SHORT/$TRAINING_MODE"
# fi
# mkdir -p $MODEL_OUTPUTPATH

MODEL_OUTPUTPATH="$OUTPUTPATH"

##################### LOGGING ################

echo "============================================"
echo "Arabic Poetry Finetuning"
echo "============================================"
echo "Model: $MODEL"
echo "Training Mode: $TRAINING_MODE"
echo "Config File: $CONFIG_FILE"
echo "Accelerator: $ACCELERATOR"
echo "Tasks: $TASKS"
echo "Prompt Type: $PROMPT_TYPE"
echo "Use PEFT: $USE_PEFT"
echo "Max Samples: ${MAX_SAMPLES:-all}"
echo "Output Path: $MODEL_OUTPUTPATH"
echo "Num Nodes: $NUM_NODES"
echo "GPUs per Node: $GPUS_PER_NODE"
echo "World Size: $WORLD_SIZE"
echo "Gradient Accumulation Steps: $GRAD_ACC_STEPS"
echo "============================================"

##################### BUILD COMMAND ################

# Build the training command
CMD="run_sft.py \
    --config $CONFIG_FILE \
    --output_dir=$MODEL_OUTPUTPATH \
    --model_name_or_path=$MODEL \
    --training_mode=$TRAINING_MODE \
    --tasks=$TASKS \
    --prompt_type=$PROMPT_TYPE \
    --use_peft=$USE_PEFT"

# Add max_samples if specified
if [[ -n "$MAX_SAMPLES" ]]; then
    CMD="$CMD --max_samples_per_task=$MAX_SAMPLES"
fi

# Add optional args
if [[ -n "$OPTIONAL_ARGS" ]]; then
    CMD="$CMD $OPTIONAL_ARGS"
fi

# Build the launcher command
LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file accelerate_configs/$ACCELERATOR.yaml \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank ${SLURM_PROCID:-0}"

##################### LAUNCH TRAINING ################

USE_SRUN=false

if [[ "$USE_SRUN" == "true" ]]; then
    # Running under SLURM with srun
    SRUN_ARGS="--wait=60 --kill-on-bad-exit=1 --nodes=$NUM_NODES --ntasks=$NUM_NODES"
    srun $SRUN_ARGS bash -c "$LAUNCHER $CMD" 2>&1
else
    # Direct run (no SLURM)
    eval "$LAUNCHER $CMD" 2>&1
fi

##################### FINISH ################

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
echo "Training completed for $MODEL"
