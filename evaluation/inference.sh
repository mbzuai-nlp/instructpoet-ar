#!/bin/bash
# Arabic Poetry Inference Script
# Supports both SLURM and local bash execution
#
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
#   SLURM:  sbatch inference.sh --model_path /path/to/checkpoint --data_path /path/to/test/data --tasks analysis,continuation
#   Local:  bash inference.sh --model_path /path/to/checkpoint --data_path /path/to/test/data --tasks analysis,continuation
#   Local:  bash inference.sh --model_path /path/to/checkpoint --data_path /path/to/test/data --tasks all --gpus 2

if [[ "$*" == *"--help"* ]]; then
    echo "Usage: bash inference.sh [options]  OR  sbatch inference.sh [options]"
    echo ""
    echo "Options:"
    echo "  --model_path PATH         Path to model checkpoint (required)"
    echo "  --base_model MODEL        Base model name for LoRA (optional, auto-detected if not provided)"
    echo "  --data_path PATH          Path to test data directory with task subfolders (required)"
    echo "  --output_path PATH        Output directory (default: ./inference_outputs)"
    echo "  --tasks TASKS             Comma-separated tasks or 'all' (default: all)"
    echo "  --prompt_type TYPE        Prompt type: instruction or chat (default: chat)"
    echo "  --max_new_tokens N        Max tokens to generate (default: 512)"
    echo "  --temperature TEMP        Temperature for sampling (default: 0.0)"
    echo "  --top_p PROB              Nucleus sampling probability (default: 0.9)"
    echo "  --top_k K                 Top-k sampling (default: 50)"
    echo "  --repetition_penalty P    Repetition penalty (default: 1.1)"
    echo "  --no_repeat_ngram_size N  Block repeating n-grams (default: 10)"
    echo "  --max_num_seqs N          Max parallel sequences for vLLM (default: 16)"
    echo "  --gpus N                  Number of GPUs for local run (default: auto-detect)"
    echo "  --max_samples N           Max samples per task (default: all)"
    echo "  --args \"ARGS\"            Additional arguments to pass to vllm_inference.py"
    echo ""
    echo "Examples:"
    echo "  # Run inference on trained Qwen model with all tasks"
    echo "  bash inference.sh --model_path /path/to/checkpoints/Qwen3-8B/.../checkpoint-36000 \\"
    echo "                    --data_path /path/to/dialectical_IFT_DATA/tsv \\"
    echo "                    --tasks all"
    echo ""
    echo "  # Run on specific tasks with 2 GPUs"
    echo "  bash inference.sh --model_path /path/to/checkpoints/ALLaM-7B/.../checkpoint-32000 \\"
    echo "                    --data_path /path/to/dialectical_IFT_DATA/tsv \\"
    echo "                    --tasks analysis,continuation --gpus 2"
    echo ""
    echo "  # SLURM submission"
    echo "  sbatch inference.sh --model_path /path/to/checkpoints/checkpoint-36000 \\"
    echo "                      --data_path /path/to/dialectical_IFT_DATA/tsv"
    exit 0
fi

set -e  # Exit on error

echo "START TIME: $(date)"
START_TIME=$(date +%s)

# Get the directory where this script is located
# Handle both direct bash execution and SLURM submission
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    # Running under SLURM - use submission directory
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    # Running directly with bash
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

##################### ENVIRONMENT SETUP ################

# Get hostname and setup paths
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *gpumid* ]] || [[ "$HOSTNAME" == *cscc* ]]; then
    PARENT_PATH="${CLUSTER_HOME:-/path/to/cluster_home}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PARENT_PATH}"
    export HF_HOME="${HF_HOME:-$PARENT_PATH/huggingface}"
else
    PARENT_PATH="."
fi

# vLLM settings
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1

##################### ARGUMENT PARSING ################

# Pre-defined model paths (optional shortcuts)
ALLAM="${ALLAM_CHECKPOINT:-/path/to/checkpoints/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000}"
QWEN_CUR="${QWEN_CUR_CHECKPOINT:-/path/to/checkpoints/Qwen3-8B/curriculum/4tasks/checkpoint-36000}"
QWEN_RANDOM="${QWEN_RANDOM_CHECKPOINT:-/path/to/checkpoints/Qwen3-8B/random/4tasks/checkpoint-35000}"

# Default values
# MAX_SAMPLES=1000
MODEL_PATH=""
BASE_MODEL=""
DATA_PATH="${DATA_PATH:-/path/to/dialectical_IFT_DATA/tsv}"
OUTPUT_PATH="${OUTPUT_PATH:-/path/to/outputs}"
TASKS="generation,continuation,corruption"
# TASKS="generation"
PROMPT_TYPE="chat"
MAX_NEW_TOKENS=1024
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.15
NO_REPEAT_NGRAM_SIZE=0  # vLLM doesn't support this parameter
MAX_NUM_SEQS=64
NUM_GPUS="${NUM_GPUS:-4}"
ENABLE_THINKING="false"
OPTIONAL_ARGS=""

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
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
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --repetition_penalty)
            REPETITION_PENALTY="$2"
            shift 2
            ;;
        --no_repeat_ngram_size)
            NO_REPEAT_NGRAM_SIZE="$2"
            shift 2
            ;;
        --max_num_seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --enable_thinking)
            ENABLE_THINKING="$2"
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
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    echo "Run with --help for usage information"
    exit 1
fi

if [[ -z "$DATA_PATH" ]]; then
    echo "Error: --data_path is required"
    echo "Run with --help for usage information"
    exit 1
fi

##################### MODEL DETECTION ################

# Check if model path exists (allow HuggingFace model identifiers)
# HuggingFace identifiers contain exactly one '/' and don't start with '/' or '.'
if [[ "$MODEL_PATH" =~ ^[^/\.][^/]+/[^/]+$ ]]; then
    # Looks like a HuggingFace model identifier (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
    echo "Detected HuggingFace model identifier: $MODEL_PATH"
    MODEL_TYPE="huggingface"
    BASE_MODEL="$MODEL_PATH"
elif [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Check if data path exists
if [[ ! -d "$DATA_PATH" ]]; then
    echo "Error: Data path does not exist: $DATA_PATH"
    exit 1
fi

# Detect model type (LoRA adapters or full model) - only if not already set to huggingface
if [[ "$MODEL_TYPE" != "huggingface" ]] && [[ -f "$MODEL_PATH/adapter_config.json" ]]; then
    MODEL_TYPE="adapters"
    echo "Detected LoRA adapter checkpoint"
    
    # Auto-detect base model if not provided
    if [[ -z "$BASE_MODEL" ]]; then
        if [[ "$MODEL_PATH" == *"Qwen3-8B"* ]]; then
            BASE_MODEL="Qwen/Qwen3-8B"
            echo "Auto-detected base model for Qwen3-8B: $BASE_MODEL"
        elif [[ "$MODEL_PATH" == *"ALLaM-7B"* ]]; then
            BASE_MODEL="humain-ai/ALLaM-7B-Instruct-preview"
            echo "Auto-detected base model for ALLaM-7B: $BASE_MODEL"
        elif [[ "$MODEL_PATH" == *"gemma-3-12b-it"* ]] || [[ "$MODEL_PATH" == *"Gemma-3-12b-it"* ]]; then
            BASE_MODEL="google/gemma-3-12b-it"
            echo "Auto-detected base model for Gemma-3-12B: $BASE_MODEL"
        elif [[ "$MODEL_PATH" == *"Meta-Llama-3"* ]]; then
            BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
            echo "Auto-detected base model for Meta-Llama-3: $BASE_MODEL"
        elif [[ "$MODEL_PATH" == *"Fanar"* ]]; then
            BASE_MODEL="QCRI/Fanar-1-9B"
            echo "Auto-detected base model for Fanar: $BASE_MODEL"
        else
            echo "Error: Cannot auto-detect base model. Please specify --base_model"
            exit 1
        fi
    fi
elif [[ "$MODEL_TYPE" != "huggingface" ]] && [[ -f "$MODEL_PATH/config.json" ]]; then
    MODEL_TYPE="full"
    BASE_MODEL="$MODEL_PATH"
    echo "Detected full finetuned model"
elif [[ "$MODEL_TYPE" != "huggingface" ]]; then
    echo "Error: Invalid model path. No config.json or adapter_config.json found"
    exit 1
fi

# Extract model name from path for output organization
if [[ "$MODEL_TYPE" == "huggingface" ]]; then
    # For HuggingFace models, extract the model name from the identifier
    MODEL_NAME=$(basename "$MODEL_PATH")
else
    MODEL_NAME=$(echo "$MODEL_PATH" | grep -oP '(Qwen3-8B|ALLaM-7B-Instruct-preview|gemma-3-12b|Gemma-3-12B|Qwen[^/]+|ALLaM[^/]+|[Gg]emma[^/]+)' | head -1)
    if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME=$(basename $(dirname $(dirname "$MODEL_PATH")))
    fi
fi

# Extract training info from path if available
if [[ "$MODEL_TYPE" == "huggingface" ]]; then
    TRAINING_MODE="base"
    CHECKPOINT="base-model"
else
    TRAINING_MODE=$(echo "$MODEL_PATH" | grep -oP '(curriculum|random)' | head -1)
    if [[ -z "$TRAINING_MODE" ]]; then
        TRAINING_MODE="unknown"
    fi
    CHECKPOINT=$(basename "$MODEL_PATH")
fi

##################### GPU CONFIGURATION ################

# Determine number of GPUs
if [[ -n "$SLURM_JOB_ID" ]]; then
    # Running under SLURM
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
    GPUS_COUNT=${SLURM_GPUS_ON_NODE:-4}
else
    # Local run
    echo "Running locally (bash mode)"
    if [[ -n "$NUM_GPUS" ]]; then
        GPUS_COUNT=$NUM_GPUS
    elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        GPUS_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
    else
        GPUS_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
    fi
fi

##################### TASK CONFIGURATION ################

# Expand tasks
if [[ "$TASKS" == "all" ]]; then
    TASK_LIST=("analysis" "continuation" "generation" "corruption")
else
    IFS=',' read -ra TASK_LIST <<< "$TASKS"
fi

# Create output directory structure
OUTPUT_DIR="$OUTPUT_PATH/$MODEL_NAME/$TRAINING_MODE/$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"

##################### LOGGING ################

echo "============================================"
echo "Arabic Poetry Inference"
echo "============================================"
echo "Model Path: $MODEL_PATH"
echo "Model Type: $MODEL_TYPE"
echo "Base Model: $BASE_MODEL"
echo "Model Name: $MODEL_NAME"
echo "Training Mode: $TRAINING_MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Tasks: ${TASK_LIST[@]}"
echo "Prompt Type: $PROMPT_TYPE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Top-P: $TOP_P"
echo "Top-K: $TOP_K"
echo "Repetition Penalty: $REPETITION_PENALTY"
echo "No Repeat N-gram Size: $NO_REPEAT_NGRAM_SIZE"
echo "Max Parallel Seqs: $MAX_NUM_SEQS"
echo "GPUs: $GPUS_COUNT"
echo "Enable Thinking: $ENABLE_THINKING"
echo "Max Samples: ${MAX_SAMPLES:-all}"
echo "============================================"

##################### RUN INFERENCE ################

# Save metadata
cat > "$OUTPUT_DIR/inference_metadata.json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "model_type": "$MODEL_TYPE",
  "base_model": "$BASE_MODEL",
  "model_name": "$MODEL_NAME",
  "training_mode": "$TRAINING_MODE",
  "checkpoint": "$CHECKPOINT",
  "data_path": "$DATA_PATH",
  "tasks": "${TASK_LIST[@]}",
  "prompt_type": "$PROMPT_TYPE",
  "max_new_tokens": $MAX_NEW_TOKENS,
  "temperature": $TEMPERATURE,
  "top_p": $TOP_P,
  "top_k": $TOP_K,
  "repetition_penalty": $REPETITION_PENALTY,
  "no_repeat_ngram_size": $NO_REPEAT_NGRAM_SIZE,
  "max_num_seqs": $MAX_NUM_SEQS,
  "gpus": $GPUS_COUNT,
  "enable_thinking": $ENABLE_THINKING,
  "max_samples": "${MAX_SAMPLES:-null}",
  "start_time": "$(date)",
  "hostname": "$HOSTNAME"
}
EOF

# Run inference for each task
for TASK in "${TASK_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running inference on task: $TASK"
    echo "=========================================="
    
    TASK_START=$(date +%s)
    
    # Check if task data exists
    TASK_DATA_PATH="$DATA_PATH/$TASK/test"
    if [[ ! -d "$TASK_DATA_PATH" ]]; then
        echo "Warning: Task data not found at $TASK_DATA_PATH, skipping..."
        continue
    fi
    
    # Build command based on model type
    if [[ "$MODEL_TYPE" == "adapters" ]]; then
        CMD="python $SCRIPT_DIR/inference_runner.py \
            --model_path $BASE_MODEL \
            --adapter_path $MODEL_PATH \
            --data_path $DATA_PATH \
            --task $TASK \
            --output_path $OUTPUT_DIR \
            --prompt_type $PROMPT_TYPE \
            --max_new_tokens $MAX_NEW_TOKENS \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --top_k $TOP_K \
            --repetition_penalty $REPETITION_PENALTY \
            --no_repeat_ngram_size $NO_REPEAT_NGRAM_SIZE \
            --max_num_seqs $MAX_NUM_SEQS \
            --tensor_parallel_size $GPUS_COUNT"
    else
        CMD="python $SCRIPT_DIR/inference_runner.py \
            --model_path $MODEL_PATH \
            --data_path $DATA_PATH \
            --task $TASK \
            --output_path $OUTPUT_DIR \
            --prompt_type $PROMPT_TYPE \
            --max_new_tokens $MAX_NEW_TOKENS \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --top_k $TOP_K \
            --repetition_penalty $REPETITION_PENALTY \
            --no_repeat_ngram_size $NO_REPEAT_NGRAM_SIZE \
            --max_num_seqs $MAX_NUM_SEQS \
            --tensor_parallel_size $GPUS_COUNT"
    fi
    
    # Add enable_thinking parameter
    CMD="$CMD --enable_thinking $ENABLE_THINKING"
    
    # Add max_samples if specified
    if [[ -n "$MAX_SAMPLES" ]]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
    
    # Add optional args
    if [[ -n "$OPTIONAL_ARGS" ]]; then
        CMD="$CMD $OPTIONAL_ARGS"
    fi
    
    # Execute
    echo "Command: $CMD"
    eval $CMD
    
    TASK_END=$(date +%s)
    TASK_ELAPSED=$((TASK_END - TASK_START))
    echo "Task $TASK completed in ${TASK_ELAPSED}s"
done

##################### FINISH ################

END_TIME=$(date +%s)
echo ""
echo "============================================"
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
echo "Inference completed for $MODEL_NAME"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"
