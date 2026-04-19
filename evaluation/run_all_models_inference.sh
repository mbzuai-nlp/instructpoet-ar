#!/bin/bash
# Run inference on all 5 models with curriculum, random, and base model checkpoints
# Each model runs as ONE job that performs all 3 inferences sequentially
# Total: 5 jobs (one per model)

# Base paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/path/to/checkpoints}"
DATA_PATH="${DATA_PATH:-/path/to/dialectical_IFT_DATA/tsv}"
OUTPUT_BASE="${OUTPUT_BASE:-/path/to/outputs}"
CLUSTER_HOME="${CLUSTER_HOME:-/path/to/cluster_home}"

# Common parameters
TASKS="generation,continuation,corruption"
MAX_NEW_TOKENS=1024
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.15
MAX_NUM_SEQS=64
NUM_GPUS=4

# Model configurations: MODEL_NAME|BASE_MODEL|CURRICULUM_CHECKPOINT|RANDOM_CHECKPOINT
MODELS=(
    "ALLaM-7B-Instruct-preview|humain-ai/ALLaM-7B-Instruct-preview|checkpoint-32000|checkpoint-42216"
    "Fanar-1-9B|QCRI/Fanar-1-9B|checkpoint-42216|checkpoint-42216"
    "gemma-3-12b-it|google/gemma-3-12b-it|checkpoint-42216|checkpoint-42216"
    "Meta-Llama-3-8B-Instruct|meta-llama/Meta-Llama-3-8B-Instruct|checkpoint-38500|checkpoint-38500"
    "Qwen3-8B|Qwen/Qwen3-8B|checkpoint-36000|checkpoint-35000"
)

echo "=========================================="
echo "Starting inference on all models"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Create a temporary directory for job scripts
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"
echo ""

# Counter for tracking progress
TOTAL_JOBS=5
CURRENT_JOB=0

# Loop through each model and create a job script
for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME BASE_MODEL CURRICULUM_CKPT RANDOM_CKPT <<< "$model_config"
    
    CURRENT_JOB=$((CURRENT_JOB + 1))
    echo "=========================================="
    echo "[$CURRENT_JOB/$TOTAL_JOBS] Creating job for: $MODEL_NAME"
    echo "Base Model: $BASE_MODEL"
    echo "=========================================="
    echo ""
    
    # Create a job script for this model
    JOB_SCRIPT="$TEMP_DIR/inference_${MODEL_NAME}.sh"
    
    cat > "$JOB_SCRIPT" << 'EOF_OUTER'
#!/bin/bash
#SBATCH --job-name=MODEL_NAME_PLACEHOLDER_inference
#SBATCH --output=./logs/MODEL_NAME_PLACEHOLDER_inference-%j.out
#SBATCH --error=./logs/MODEL_NAME_PLACEHOLDER_inference-%j.err
#SBATCH --account=ifm-miscellaneous-1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --cpus-per-task=16

set -e  # Exit on error

# Model-specific variables
MODEL_NAME="MODEL_NAME_PLACEHOLDER"
BASE_MODEL="BASE_MODEL_PLACEHOLDER"
CURRICULUM_CKPT="CURRICULUM_CKPT_PLACEHOLDER"
RANDOM_CKPT="RANDOM_CKPT_PLACEHOLDER"

# Paths and parameters
CHECKPOINT_BASE="CHECKPOINT_BASE_PLACEHOLDER"
DATA_PATH="DATA_PATH_PLACEHOLDER"
OUTPUT_BASE="OUTPUT_BASE_PLACEHOLDER"
TASKS="TASKS_PLACEHOLDER"
MAX_NEW_TOKENS=MAX_NEW_TOKENS_PLACEHOLDER
TEMPERATURE=TEMPERATURE_PLACEHOLDER
TOP_P=TOP_P_PLACEHOLDER
TOP_K=TOP_K_PLACEHOLDER
REPETITION_PENALTY=REPETITION_PENALTY_PLACEHOLDER
MAX_NUM_SEQS=MAX_NUM_SEQS_PLACEHOLDER
NUM_GPUS=NUM_GPUS_PLACEHOLDER

echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Get the directory where the original script is located
SCRIPT_DIR="SCRIPT_DIR_PLACEHOLDER"
cd "$SCRIPT_DIR"

# Setup environment
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *gpumid* ]] || [[ "$HOSTNAME" == *cscc* ]]; then
    PARENT_PATH="CLUSTER_HOME_PLACEHOLDER"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PARENT_PATH}"
    export HF_HOME="${HF_HOME:-$PARENT_PATH/huggingface}"
fi

export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Run inference on all 3 checkpoints sequentially
echo ""

# Helper function: run inference only if output doesn't already exist or is empty
run_checkpoint() {
    local DESC="$1"
    local MODEL_PATH="$2"
    local OUTPUT_PATH="$3"

    echo ""
    echo "=========================================="
    echo "$DESC"
    echo "Model Path: $MODEL_PATH"
    echo "Output: $OUTPUT_PATH"
    echo "=========================================="
    echo ""

    # If output directory exists and is non-empty, assume inference already done
    if [ -d "$OUTPUT_PATH" ] && [ "$(ls -A "$OUTPUT_PATH" 2>/dev/null)" ]; then
        echo "Skipping inference: output already exists at $OUTPUT_PATH"
        return 0
    fi

    mkdir -p "$OUTPUT_PATH"

    bash inference.sh \
        --model_path "$MODEL_PATH" \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --output_path "$OUTPUT_PATH" \
        --tasks "$TASKS" \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --top_k $TOP_K \
        --repetition_penalty $REPETITION_PENALTY \
        --max_num_seqs $MAX_NUM_SEQS \
        --gpus $NUM_GPUS
}

# Curriculum checkpoint
run_checkpoint "[1/3] Running inference on CURRICULUM checkpoint..." \
    "$CHECKPOINT_BASE/$MODEL_NAME/curriculum/4tasks/$CURRICULUM_CKPT" \
    "$OUTPUT_BASE/${MODEL_NAME}_curriculum_${CURRICULUM_CKPT}"

echo ""
echo "Completed curriculum checkpoint inference (or skipped if already done)"
echo ""

# Random checkpoint
run_checkpoint "[2/3] Running inference on RANDOM checkpoint..." \
    "$CHECKPOINT_BASE/$MODEL_NAME/random/4tasks/$RANDOM_CKPT" \
    "$OUTPUT_BASE/${MODEL_NAME}_random_${RANDOM_CKPT}"

echo ""
echo "Completed random checkpoint inference (or skipped if already done)"
echo ""

# Base model
# Use the base model identifier as the model_path and also pass it via --base_model
run_checkpoint "[3/3] Running inference on BASE MODEL..." \
    "$BASE_MODEL" \
    "$OUTPUT_BASE/${MODEL_NAME}_base"

echo ""
echo "Completed base model inference (or skipped if already done)"
echo ""

echo "=========================================="
echo "All 3 inferences completed for $MODEL_NAME"
echo "End Time: $(date)"
echo "=========================================="
EOF_OUTER

    # Replace placeholders with actual values
    sed -i "s|MODEL_NAME_PLACEHOLDER|$MODEL_NAME|g" "$JOB_SCRIPT"
    sed -i "s|BASE_MODEL_PLACEHOLDER|$BASE_MODEL|g" "$JOB_SCRIPT"
    sed -i "s|CURRICULUM_CKPT_PLACEHOLDER|$CURRICULUM_CKPT|g" "$JOB_SCRIPT"
    sed -i "s|RANDOM_CKPT_PLACEHOLDER|$RANDOM_CKPT|g" "$JOB_SCRIPT"
    sed -i "s|CHECKPOINT_BASE_PLACEHOLDER|$CHECKPOINT_BASE|g" "$JOB_SCRIPT"
    sed -i "s|DATA_PATH_PLACEHOLDER|$DATA_PATH|g" "$JOB_SCRIPT"
    sed -i "s|OUTPUT_BASE_PLACEHOLDER|$OUTPUT_BASE|g" "$JOB_SCRIPT"
    sed -i "s|SCRIPT_DIR_PLACEHOLDER|$SCRIPT_DIR|g" "$JOB_SCRIPT"
    sed -i "s|CLUSTER_HOME_PLACEHOLDER|$CLUSTER_HOME|g" "$JOB_SCRIPT"
    sed -i "s|TASKS_PLACEHOLDER|$TASKS|g" "$JOB_SCRIPT"
    sed -i "s|MAX_NEW_TOKENS_PLACEHOLDER|$MAX_NEW_TOKENS|g" "$JOB_SCRIPT"
    sed -i "s|TEMPERATURE_PLACEHOLDER|$TEMPERATURE|g" "$JOB_SCRIPT"
    sed -i "s|TOP_P_PLACEHOLDER|$TOP_P|g" "$JOB_SCRIPT"
    sed -i "s|TOP_K_PLACEHOLDER|$TOP_K|g" "$JOB_SCRIPT"
    sed -i "s|REPETITION_PENALTY_PLACEHOLDER|$REPETITION_PENALTY|g" "$JOB_SCRIPT"
    sed -i "s|MAX_NUM_SEQS_PLACEHOLDER|$MAX_NUM_SEQS|g" "$JOB_SCRIPT"
    sed -i "s|NUM_GPUS_PLACEHOLDER|$NUM_GPUS|g" "$JOB_SCRIPT"

    chmod +x "$JOB_SCRIPT"
    
    # Submit the job
    echo "Submitting job for $MODEL_NAME..."
    sbatch "$JOB_SCRIPT"
    echo "Job submitted!"
    echo ""
    sleep 2  # Brief delay between submissions
done

# Cleanup
echo ""
echo "=========================================="
echo "All $TOTAL_JOBS jobs submitted!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Job scripts created in: $TEMP_DIR"
echo "To check job status: squeue -u \$USER"
echo "To check logs: ls -lth logs/"
echo ""
echo "Note: Temporary job scripts will be automatically deleted on reboot"
echo "      or can be manually removed: rm -rf $TEMP_DIR"
