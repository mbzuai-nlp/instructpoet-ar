import os
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def process_tsv_directory(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    split_name = input_dir.name  # 'train' or 'test'

    # Prepare file handles
    writers = {
        "LongContext": defaultdict(lambda: None),
        "ShortContext": defaultdict(lambda: None),
    }

    tsv_files = list(input_dir.glob("*.tsv"))
    print(f"Found {len(tsv_files)} TSV files.")
    # Skip the template_stats.tsv file
    tsv_files = [file for file in tsv_files if file.name != "template_stats.tsv"]

    for file_path in tqdm(tsv_files, desc="Processing files"):
        df = pd.read_csv(file_path, sep="\t", encoding="utf-8")

        # if not {"input", "output", "num_tokens", "instruction"}.issubset(df.columns):
        #     print(f"[WARN] Skipping {file_path.name} — missing required columns.")
        #     continue

        task_name = file_path.stem  # filename without extension (e.g., "poem_genre")

        for _, row in df.iterrows():
            try:
                tokens = int(row["num_tokens"])
                context_type = "LongContext" if tokens >= 8000 else "ShortContext"

                item = {
                    "instruction": row["instruction"] if split_name == "test" else row['input'],
                    "input": "",
                    "output": row["output"]
                }

                # Initialize file handle if not yet created
                if writers[context_type][task_name] is None:
                    out_path = output_dir / context_type / split_name
                    out_path.mkdir(parents=True, exist_ok=True)
                    file_handle = open(out_path / f"{task_name}.jsonl", "w", encoding="utf-8")
                    writers[context_type][task_name] = file_handle

                json.dump(item, writers[context_type][task_name], ensure_ascii=False)
                writers[context_type][task_name].write("\n")
            except Exception as e:
                print(f"[ERROR] Skipping row in {file_path.name} due to: {e}")

    # Close all open files
    for context_type in writers:
        for task_name, handle in writers[context_type].items():
            if handle:
                handle.close()
                print(f"✅ Saved {context_type}/{split_name}/{task_name}.jsonl")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine TSV files into JSONL based on token length.")
    parser.add_argument("--input_dir", default="/path/to/IFT_DATA/analysis/test", help="Directory containing TSV files.")
    parser.add_argument("--output_dir", default='/path/to/IFT_DATA/analysis/final/', help="Directory to save JSONL files.")

    args = parser.parse_args()
    process_tsv_directory(args.input_dir, args.output_dir)