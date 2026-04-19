#!/bin/bash
# Shell script to run decoding hyperparameter grid search
# Usage: bash run_grid_search.sh --model_path /path/to/model

set -e

echo "Arabic Poetry Decoding Grid Search"
echo "===================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
MODEL_PATH="/path/to/checkpoints/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000"
DATA_PATH="/path/to/data/dialectical_IFT_DATA/tsv"
OUTPUT_PATH="$SCRIPT_DIR/grid_search_results"
BEST_CONFIG_DIR="$SCRIPT_DIR/best_decoding_parameters"
TASKS="generation continuation corruption"
REPETITION_PENALTIES="1.0 1.10 1.15"
TEMPERATURES="0.0 0.7"
N_SAMPLES=100
FORCE_FLAG=""

# Help message
if [[ "$*" == *"--help"* ]]; then
    echo "Usage: bash run_grid_search.sh [options]"
    echo ""
    echo "Options:"
    echo "  --model_path PATH              Path to model checkpoint (required)"
    echo "  --data_path PATH               Base path to data directory"
    echo "  --output_path PATH             Base path for outputs"
    echo "  --best_config_dir PATH         Directory to save best configurations"
    echo "  --tasks TASKS                  Space-separated tasks (default: generation continuation corruption)"
    echo "  --repetition_penalties VALS    Space-separated penalties (default: 1.0 1.10 1.15)"
    echo "  --temperatures VALS            Space-separated temperatures (default: 0.0 0.7)"
    echo "  --n_samples N                  Number of samples per task (default: 100)"
    echo "  --force                        Force re-run inference even if outputs exist"
    echo ""
    echo "Example:"
    echo "  bash run_grid_search.sh --model_path /path/to/checkpoint-36000"
    echo ""
    echo "  bash run_grid_search.sh --model_path /path/to/model --n_samples 50 --tasks generation continuation"
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
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
        --best_config_dir)
            BEST_CONFIG_DIR="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --repetition_penalties)
            REPETITION_PENALTIES="$2"
            shift 2
            ;;
        --temperatures)
            TEMPERATURES="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --force)
            FORCE_FLAG="--force"
            shift 1
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
    echo "Use --help for usage information"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_PATH"
mkdir -p "$BEST_CONFIG_DIR"

echo ""
echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Data Path: $DATA_PATH"
echo "  Output Path: $OUTPUT_PATH"
echo "  Best Config Dir: $BEST_CONFIG_DIR"
echo "  Tasks: $TASKS"
echo "  Repetition Penalties: $REPETITION_PENALTIES"
echo "  Temperatures: $TEMPERATURES"
echo "  Samples per Task: $N_SAMPLES"
echo ""

# Run the Python script
python "$SCRIPT_DIR/decoding_grid_search.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --best_config_dir "$BEST_CONFIG_DIR" \
    --tasks $TASKS \
    --repetition_penalties $REPETITION_PENALTIES \
    --temperatures $TEMPERATURES \
    --n_samples $N_SAMPLES \
    $FORCE_FLAG

echo ""
echo "Grid search completed!"
echo "Results saved to: $BEST_CONFIG_DIR"
