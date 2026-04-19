#!/bin/bash

python gemini_fast_infer.py \
  --input /mnt/data/users/abdelrahman.sadallah/ONEDRIVE/poetry/final_train_data.csv \
  --output ../../data/outputs/final_train_data.csv \
  --min_verses 4 \
  --max_verses 15 \
  --max_rows 100_000 \
  --num_of_workers 64 \
  --task corruption \
  --corruption_template corruption_template.tsv \
  --gemini_key_env_var GEMINI_KEY_ANWAR



# python gemini_fast_infer.py \
#   --input /mnt/data/users/abdelrahman.sadallah/ONEDRIVE/poetry/final_test_data.csv \
#   --output ../../data/outputs/final_test_data.csv \
#   --min_verses 4 \
#   --max_verses 15 \
#   --max_rows 100_000 \
#   --num_of_workers 64 \
#   --task corruption \
#   --corruption_template corruption_template.tsv \
#   --gemini_key_env_var GEMINI_KEY_ANWAR
