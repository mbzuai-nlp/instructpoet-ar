#!/bin/bash
# Interactive inference script to debug repetition issues

# Configuration
CHECKPOINT_PATH="/path/to/checkpoints/gemma-3-12b-it/curriculum/4tasks/checkpoint-42216/"
TEST_FILE="/path/to/data/dialectical_poetry_analysis_lm_harness_format/analysis_poem_text__poet_name___meter.json"

# Detect if this is a LoRA adapter or full model
if [[ -f "$CHECKPOINT_PATH/adapter_config.json" ]]; then
    echo "Detected LoRA adapter checkpoint"
    ADAPTER_PATH="$CHECKPOINT_PATH"
    
    # Auto-detect base model
    if [[ "$CHECKPOINT_PATH" == *"Qwen3-8B"* ]]; then
        BASE_MODEL="Qwen/Qwen3-8B"
        echo "Using base model: $BASE_MODEL"
    elif [[ "$CHECKPOINT_PATH" == *"ALLaM-7B"* ]]; then
        BASE_MODEL="humain-ai/ALLaM-7B-Instruct-preview"
        echo "Using base model: $BASE_MODEL"
    elif [[ "$CHECKPOINT_PATH" == *"gemma-3-12b-it"* ]]; then
        BASE_MODEL="google/gemma-3-12b-it"
        echo "Using base model: $BASE_MODEL"
    else
        echo "Error: Cannot auto-detect base model"
        exit 1
    fi
    
    MODEL_ARG="--model_path $BASE_MODEL --adapter_path $ADAPTER_PATH"
elif [[ -f "$CHECKPOINT_PATH/config.json" ]]; then
    echo "Detected full finetuned model"
    BASE_MODEL="$CHECKPOINT_PATH"
    MODEL_ARG="--model_path $BASE_MODEL"
else
    echo "Error: Invalid checkpoint path. No config.json or adapter_config.json found"
    exit 1
fi

# Generation settings to reduce repetition
MAX_NEW_TOKENS=512
TEMPERATURE=0.0      # Higher temperature for diversity
TOP_P=0.9            # Nucleus sampling
TOP_K=50             # Top-k sampling
REPETITION_PENALTY=1.1  # Penalize repetitions
NO_REPEAT_NGRAM_SIZE=10  # Block repeating 3-grams

# Run interactive inference
python interactive_inference.py \
    $MODEL_ARG \
    --test_file "$TEST_FILE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE" \
    --prompt_type chat \
    --tensor_parallel_size 1 \
    --start_idx 0

# To use with different settings, you can modify the parameters above or call directly:
# python interactive_inference.py --model_path <path> --test_file <path> --temperature 0.9 --repetition_penalty 1.2
