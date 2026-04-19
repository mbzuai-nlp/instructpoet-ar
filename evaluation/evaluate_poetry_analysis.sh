#!/bin/bash
# SLURM directives (only used when submitted via sbatch)
#SBATCH --job-name=poetry_analysis_eval
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --account=ifm-miscellaneous-1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4 # Number of GPUs (per node)
#SBATCH -p gpumid
#SBATCH --cpus-per-task=16

################################################################################
# Poetry Analysis Evaluation using lm-eval-harness
# 
# Features:
#   - Supports both base models and LoRA adapters
#   - Works with vLLM (faster) or HuggingFace (more compatible) backends
#   - Automatic base model detection for LoRA adapters
#   - Sequential evaluation with full GPU allocation per model
#
# Usage:
#   SLURM:  sbatch evaluate_poetry_analysis.sh
#   Local:  bash evaluate_poetry_analysis.sh
#
# Configuration:
#   - Set USE_VLLM=true for vLLM backend (recommended for speed)
#   - Set USE_VLLM=false for HuggingFace/PEFT backend
#   - Modify LORA_CHECKPOINTS, BASE_MODELS, BASELINE_MODELS arrays as needed
#   - Adjust BENCHMARK to evaluate different tasks
#
# LoRA Support:
#   - vLLM: Uses enable_lora=True with lora_local_path parameter
#   - HF: Uses peft= parameter in model_args
################################################################################

set -e  # Exit on error

echo "=============================="
echo "Poetry Analysis Evaluation"
echo "Starting at: $(date)"
echo "=============================="

##################### ENVIRONMENT SETUP ################

# Get the directory where this script is located
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Environment paths
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
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_LAUNCH_BLOCKING=1

##################### CONFIGURATION ################

# Results and tasks paths
RESULTS_PATH="$SCRIPT_DIR/evaluation_results_dialect_continuation"
TASKS_PATH="${TASKS_PATH:-/path/to/lm-evaluation-harness/lm_eval/tasks}"

# Evaluation settings
# BENCHMARK="poetry_analysis"
BENCHMARK="dialectical_poetry_analysis_continuation"

OVERWRITE=false
SEED=42
BATCH_SIZE="auto"
USE_VLLM=false
ENABLE_THINKING=false
MAX_LENGTH=8192  # Max model length

##################### MODEL PATHS ################

# Base model checkpoint path
CKPT_PATH="${CKPT_PATH:-/path/to/checkpoints}"

# LoRA adapter checkpoints (trained models)
LORA_CHECKPOINTS=(
    # # # Gemma models
    # "$CKPT_PATH/gemma-3-12b-it/curriculum/4tasks/checkpoint-41500/"
    # "$CKPT_PATH/gemma-3-12b-it/random/4tasks/checkpoint-41500/"


    # # # ALLaM models
    # "$CKPT_PATH/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000"
    # "$CKPT_PATH/ALLaM-7B-Instruct-preview/random/4tasks/checkpoint-42000"

    
    # # # Qwen models
    # "$CKPT_PATH/Qwen3-8B/curriculum/4tasks/checkpoint-36000"
    # "$CKPT_PATH/Qwen3-8B/random/4tasks/checkpoint-35000"

    # Llama models
    "$CKPT_PATH/Meta-Llama-3-8B-Instruct/curriculum/4tasks/checkpoint-42216"
    "$CKPT_PATH/Meta-Llama-3-8B-Instruct/random/4tasks/checkpoint-42216"

    # Fanar LoRA adapters
    # "$CKPT_PATH/Fanar-1-9B/curriculum/4tasks/checkpoint-42216"
    # "$CKPT_PATH/Fanar-1-9B/random/4tasks/checkpoint-42216"


    
)

# Base models (without LoRA - for comparison)
BASE_MODELS=(
    # Merged LoRA models
    # "/path/to/merged_models/ALLaM-7B-Instruct-preview-curriculum-4tasks-checkpoint-32000"
    # "/path/to/merged_models/ALLaM-7B-Instruct-preview-random-4tasks-checkpoint-25000"
    # "/path/to/merged_models/gemma-3-12b-it-curriculum-4tasks-checkpoint-34500"
    # "/path/to/merged_models/gemma-3-12b-it-random-4tasks-checkpoint-34500"
    # "/path/to/merged_models/Qwen3-8B-curriculum-4tasks-checkpoint-36000"
    # "/path/to/merged_models/Qwen3-8B-random-4tasks-checkpoint-35000-merged"
    
    # Original base models for comparison
    # "Qwen/Qwen3-8B"
    # "humain-ai/ALLaM-7B-Instruct-preview"
    # "google/gemma-3-12b-it"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "QCRI/Fanar-1-9B"

)

# Other instruction-tuned models for comparison
BASELINE_MODELS=(
    # "humain-ai/ALLaM-7B-Instruct-preview"
    # "Qwen/Qwen3-8B"
)

##################### BASE MODEL MAPPING ################

# Function to auto-detect base model for LoRA adapters
get_base_model() {
    local adapter_path="$1"
    
    if [[ "$adapter_path" == *"Qwen3-8B"* ]]; then
        echo "Qwen/Qwen3-8B"
    elif [[ "$adapter_path" == *"ALLaM-7B"* ]]; then
        echo "humain-ai/ALLaM-7B-Instruct-preview"
    elif [[ "$adapter_path" == *"gemma-3-12b-it"* ]] || [[ "$adapter_path" == *"Gemma-3-12b-it"* ]]; then
        echo "google/gemma-3-12b-it"
    elif [[ "$adapter_path" == *"Meta-Llama-3"* ]]; then
        echo "meta-llama/Meta-Llama-3-8B-Instruct"
    elif [[ "$adapter_path" == *"Fanar"* ]]; then
        echo "QCRI/Fanar-1-9B"
    else
        echo ""
    fi
}

##################### GPU CONFIGURATION ################

# Determine available GPUs
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
    TOTAL_GPUS=${SLURM_GPUS_ON_NODE:-4}
elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    TOTAL_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
else
    TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
fi

echo "Total GPUs available: $TOTAL_GPUS"

# GPU allocation strategy
# For parallel evaluation, we can run multiple models simultaneously
# Allocate GPUs per model based on model size
available_gpu_pairs=("0,1,2,3")  # Adjust based on your needs
gpu_pair_count=${#available_gpu_pairs[@]}

##################### HELPER FUNCTIONS ################




# Function to evaluate a model with lm-eval-harness
evaluate_model() {
    local model_path="$1"
    local model_type="$2"  # "base", "lora", or "baseline"
    local base_model="$3"  # Only needed for LoRA
    local gpu_devices="$4"
    
    # Determine model name for output
    if [[ "$model_type" == "lora" ]]; then
        # Extract meaningful name from LoRA path
        local model_name=$(echo "$model_path" | sed 's|.*/\([^/]*/[^/]*/[^/]*/[^/]*\)$|\1|' | sed 's|/|_|g')
        model_name="LORA_${model_name}"
    else
        model_name=$(basename "${model_path}")
    fi
    
    echo "=============================="
    echo "Evaluating: $model_name"
    echo "Type: $model_type"
    echo "Path: $model_path"
    if [[ "$model_type" == "lora" ]]; then
        echo "Base Model: $base_model"
    fi
    echo "GPUs: $gpu_devices"
    echo "=============================="
    
    OUT_PATH="${RESULTS_PATH}/${BENCHMARK}/${model_name}"
    mkdir -p "${OUT_PATH}"
    
    # Skip if results already exist
    if [ "$OVERWRITE" = false ] && find "${OUT_PATH}" -maxdepth 1 -type f -name "results*" | grep -q .; then
        echo "Skipping (results already exist)"
        return
    fi
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Calculate tensor parallel size
    local tp_size=$(echo $gpu_devices | awk -F',' '{print NF}')


    # # Apply chat template for instruction-tuned models
    # APPLY_CHAT_TEMPLATE=""
    # if [[ "$model_type" == "baseline" ]] || [[ "$model_type" == "base" && "$model_path" == *"Instruct"* ]]; then
    #     APPLY_CHAT_TEMPLATE="--apply_chat_template"
    # fi
    
    # # For LoRA models, we need to apply chat template
    # if [[ "$model_type" == "lora" ]]; then
    #     APPLY_CHAT_TEMPLATE="--apply_chat_template"
    # fi



    
    APPLY_CHAT_TEMPLATE="--apply_chat_template"

    # Override chat template for Fanar base models (not finetuned)
    if [[ "$model_type" == "base" ]] && [[ "$model_path" == *"Fanar"* ]]; then
        APPLY_CHAT_TEMPLATE=""
    fi
    
    # For fine-tuned Fanar models (LoRA adapters), explicitly provide ChatML template
    CHAT_TEMPLATE_ARG=""
    if [[ "$model_path" == *"Fanar"* ]]; then
        # ChatML template used during fine-tuning
        APPLY_CHAT_TEMPLATE=""
    fi
    
    # Build the evaluation command
    if [ "$USE_VLLM" = true ]; then
        echo "Using vLLM backend..."
        
        # Build JSON model_args based on whether it's a LoRA adapter or base model
        if [[ "$model_type" == "lora" ]]; then
            MODEL_ARGS=$(printf '{
                "pretrained": "%s",
                "gpu_memory_utilization": 0.80,
                "tensor_parallel_size": %d,
                "dtype": "bfloat16",
                "trust_remote_code": true,
                "enable_lora": true,
                "max_lora_rank": 64,
                "max_model_len": 8192,
                "enable_prefix_caching": true,
                "lora_local_path": "%s",
                "enable_thinking": %s,
                "max_model_len": %d
            }' "$base_model" "$tp_size" "$model_path" "$ENABLE_THINKING" "$MAX_LENGTH")
        else
            MODEL_ARGS=$(printf '{
                "pretrained": "%s",
                "gpu_memory_utilization": 0.80,
                "tensor_parallel_size": %d,
                "dtype": "bfloat16",
                "trust_remote_code": true,
                "max_model_len": 8192,
                "enable_prefix_caching": true,
                "enable_thinking": %s,
                "max_model_len": %d
            }' "$model_path" "$tp_size" "$ENABLE_THINKING" "$MAX_LENGTH")
        fi
        
        EVAL_CMD="lm-eval \
            --model vllm \
            --model_args '$MODEL_ARGS' \
            --tasks ${BENCHMARK} \
            --log_samples \
            --batch_size ${BATCH_SIZE} \
            ${APPLY_CHAT_TEMPLATE} \
            --output_path \"${OUT_PATH}/results.json\" \
            --seed ${SEED} \
            --verbosity DEBUG \
            --use_cache \"${OUT_PATH}/cache\""
    else
        echo "Using HuggingFace transformers backend..."
        
        # Nested dict for chat_template_args as JSON
        # Convert boolean to string for HF backend
        CHAT_ARGS=$(printf '{"enable_thinking": "%s"}' "$ENABLE_THINKING")


        # Build JSON model_args for HF backend
        if [[ "$model_type" == "lora" ]]; then
            MODEL_ARGS=$(printf \
            'pretrained=%s,parallelize=True,peft=%s,trust_remote_code=True,enable_thinking=%s,max_length=%d,' \
            "$base_model" \
            "$model_path" \
            "$ENABLE_THINKING" \
            "$MAX_LENGTH")

        else
            MODEL_ARGS=$(printf \
            'pretrained=%s,parallelize=True,trust_remote_code=True,enable_thinking=%s,max_length=%d' \
            "$model_path" \
            "$ENABLE_THINKING" \
            "$MAX_LENGTH")
        fi
        EVAL_CMD="lm-eval \
            --model hf \
            --model_args '$MODEL_ARGS' \
            --tasks ${BENCHMARK} \
            --log_samples \
            --batch_size ${BATCH_SIZE} \
            ${APPLY_CHAT_TEMPLATE} \
            --output_path \"${OUT_PATH}/results.json\" \
            --use_cache \"${OUT_PATH}/cache\" \
            --seed ${SEED} \
            --trust_remote_code \
            --verbosity DEBUG"
    fi
    
    echo ""
    echo "Running command:"
    echo "$EVAL_CMD"
    echo ""
    
    eval "$EVAL_CMD"
    
    echo "✅ Completed: $model_name"
    echo ""
}







##################### MAIN EXECUTION ################

# Kill all child processes on exit
trap "echo 'Stopping...'; kill 0" SIGINT

echo ""
echo "=============================="
echo "Configuration Summary"
echo "=============================="
echo "Benchmark: $BENCHMARK"
echo "Results Path: $RESULTS_PATH"
echo "Overwrite: $OVERWRITE"
echo "Seed: $SEED"
echo "Batch Size: $BATCH_SIZE"
echo "Use vLLM: $USE_VLLM"
echo "Total GPUs: $TOTAL_GPUS"
echo ""
echo "Models to evaluate:"
echo "  - ${#LORA_CHECKPOINTS[@]} LoRA checkpoints"
echo "  - ${#BASE_MODELS[@]} base models"
echo "  - ${#BASELINE_MODELS[@]} baseline models"
echo "=============================="
echo ""

# Combine all models into a single list with their types
declare -a ALL_MODELS
declare -a MODEL_TYPES
declare -a BASE_MODEL_REFS

# Add LoRA checkpoints
for ckpt in "${LORA_CHECKPOINTS[@]}"; do
    ALL_MODELS+=("$ckpt")
    MODEL_TYPES+=("lora")
    base_model=$(get_base_model "$ckpt")
    if [[ -z "$base_model" ]]; then
        echo "Error: Cannot determine base model for $ckpt"
        echo "Please update the get_base_model() function"
        exit 1
    fi
    BASE_MODEL_REFS+=("$base_model")
done

# Add base models
for model in "${BASE_MODELS[@]}"; do
    ALL_MODELS+=("$model")
    MODEL_TYPES+=("base")
    BASE_MODEL_REFS+=("")
done

# Add baseline models
for model in "${BASELINE_MODELS[@]}"; do
    ALL_MODELS+=("$model")
    MODEL_TYPES+=("baseline")
    BASE_MODEL_REFS+=("")
done

# Sequential evaluation (recommended for LoRA to avoid conflicts)
echo "Starting sequential evaluation..."
echo ""

for i in "${!ALL_MODELS[@]}"; do
    model="${ALL_MODELS[$i]}"
    model_type="${MODEL_TYPES[$i]}"
    base_model="${BASE_MODEL_REFS[$i]}"
    
    # Use all available GPUs for each model
    GPU_DEVICES=$(seq -s, 0 $((TOTAL_GPUS-1)))
    
    evaluate_model "$model" "$model_type" "$base_model" "$GPU_DEVICES"
done

echo ""
echo "=============================="
echo "All evaluations completed!"
echo "Results saved to: $RESULTS_PATH"
echo "Finished at: $(date)"
echo "=============================="
