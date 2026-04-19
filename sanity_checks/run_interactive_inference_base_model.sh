#!/bin/bash
# Interactive inference script for BASE MODEL ONLY (no adapter)
# Use this to compare base model behavior with fine-tuned version

# Configuration
BASE_MODEL="google/gemma-3-12b-it"
TEST_FILE="/path/to/outputs/Qwen3-8B/curriculum/checkpoint-36000/generation/generation_ift_with_predictions.tsv"

echo "====================================="
echo "Testing BASE MODEL (NO ADAPTER)"
echo "Model: $BASE_MODEL"
echo "====================================="

# Generation settings to reduce repetition
MAX_NEW_TOKENS=1024
TEMPERATURE=0.7      # Higher temperature for diversity
TOP_P=0.9            # Nucleus sampling
TOP_K=50             # Top-k sampling
REPETITION_PENALTY=1.1  # Penalize repetitions
NO_REPEAT_NGRAM_SIZE=10  # Block repeating 10-grams

# Run interactive inference (NO adapter)
python interactive_inference.py \
    --model_path "$BASE_MODEL" \
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
# python interactive_inference.py --model_path Qwen/Qwen3-8B --test_file <path> --temperature 0.9
