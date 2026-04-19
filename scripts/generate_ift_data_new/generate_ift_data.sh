#!/bin/bash

# Example bash command to print a message
echo "Generating IFT data..."

# Define the path to the Python script
SCRIPT_PATH="generate_IFT_data_new.py"

# for DATA_TYPE in "test" ; do
for DATA_TYPE in "test" "train"  ; do
    for TASK in "corruption" "generation" "continuation" "analysis"; do
    # for TASK in "continuation"; do

        RAW_DATA="/path/to/data/${DATA_TYPE}_data.csv"
        # RAW_DATA="/path/to/data/de_dupped_train.csv"
        # OUTPUT_DIR="../../data/IFT_FannOrFlop_60verses"
        OUTPUT_DIR="../../data/dialectical_IFT_DATA"

        CREATE_MCQ_BENCHMARK=""
        
        if [ "$TASK" == "analysis" ]; then
            CREATE_MCQ_BENCHMARK="--create_mcq_benchmark"
        fi

        python "$SCRIPT_PATH" \
        --raw_data "$RAW_DATA" \
        --templates /path/to/data/poetry_tempelates_msa+dialectical.xlsx \
        --output_dir "$OUTPUT_DIR" \
        --task "$TASK" \
        --total_num_samples -1 \
        --min_num_verses 2 \
        --max_poem_verses 0 \
        --preferred_dialect "random" \
        $CREATE_MCQ_BENCHMARK
    done
done