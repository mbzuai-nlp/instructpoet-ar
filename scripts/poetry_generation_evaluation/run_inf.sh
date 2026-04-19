#!/bin/bash
set -e

# ===== Fill these variables =====
MODELS=("ALLaM-AI/ALLaM-7B-Instruct-preview" "model_2_name")   # e.g. "meta-llama/Llama-2-7b-chat-hf"
INPUT_FILE="path/to/input.csv"
OUTPUT_FILE="path/to/output.csv"
MAX_TOKENS=512
TEMPERATURE=0.7
# =================================

#!/bin/bash
set -e

# ===== Define your models =====
BASE_MODELS=(

)

INSTRUCT_MODELS=(
 "ALLaM-AI/ALLaM-7B-Instruct-preview"
)
# ==============================

INPUT_FILE="/mnt/data/users/abdelrahman.sadallah/ONEDRIVE/poetry/generation/test/generation_ift.tsv"
OUTPUT_DIR="results"
MAX_TOKENS=512
TEMPERATURE=0.7
BATCH_SIZE=16

mkdir -p "$OUTPUT_DIR"
# Set Hugging Face cache folder
export HF_HOME="/mnt/data/users/abdelrahman.sadallah/huggingface"
# Combine both model lists
ALL_MODELS=("${BASE_MODELS[@]}" "${INSTRUCT_MODELS[@]}")

for model in "${ALL_MODELS[@]}"; do
  echo ">>> Running model: $model"
  OUTPUT_FILE="$OUTPUT_DIR/$(basename $model)_output.csv"

    APPLY_CHAT_TEMPLATE=""
    # Check if model is in INSTRUCT_MODELS
    if [[ " ${INSTRUCT_MODELS[@]} " =~ " ${model} " ]]; then
    echo "Applying chat template for instruct model: $model"
    APPLY_CHAT_TEMPLATE="--apply_chat_template"
    fi

    python vllm_model_infer.py \
        --model "$model" \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --batch_size $BATCH_SIZE \
        ${APPLY_CHAT_TEMPLATE}

done
