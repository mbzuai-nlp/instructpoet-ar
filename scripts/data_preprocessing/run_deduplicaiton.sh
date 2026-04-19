#!/bin/bash

DATA_DIR="../../data"

python dedup_data.py \
  --train_path /path/to/data/final_train_data.csv \
  --test_path /path/to/data/test_data_v2.csv \
  --out_train_path $DATA_DIR/de_dupped_train.tsv \
  --out_test_path $DATA_DIR/de_dupped_test.tsv \
  --threshold 0.8 \
  --n_jobs 16 \
  --matched_output_path $DATA_DIR/matched_duplicates.tsv
