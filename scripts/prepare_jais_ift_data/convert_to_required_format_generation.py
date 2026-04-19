import os
import pandas as pd
import json
import random
from pathlib import Path
from tqdm import tqdm

def strip_lang_prefix(filename: str):
    if filename.startswith("ar_") or filename.startswith("en_"):
        return filename[3:]  # Remove first 3 chars (e.g., 'ar_' or 'en_')
    return filename

def process_multilang_rows(ar_dir: str, en_dir: str, output_dir: str, seed: int = 42):
    ar_dir = Path(ar_dir)
    en_dir = Path(en_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    long_context_path = output_dir / "long_context.jsonl"
    short_context_path = output_dir / "short_context.jsonl"

    long_context_file = open(long_context_path, "w", encoding="utf-8")
    short_context_file = open(short_context_path, "w", encoding="utf-8")

    total_tokens = 0
    num_long = 0
    num_short = 0

    # Build mappings from core template name -> file path
    ar_files = {strip_lang_prefix(f.name): f for f in ar_dir.glob("*.tsv")}
    en_files = {strip_lang_prefix(f.name): f for f in en_dir.glob("*.tsv")}

    common_templates = sorted(set(ar_files.keys()) & set(en_files.keys()))
    print(f"📁 Found {len(common_templates)} matching template files after stripping prefixes.")

    for core_name in tqdm(common_templates, desc="Row-wise multilingual sampling"):
        ar_file = ar_files[core_name]
        en_file = en_files[core_name]

        ar_df = pd.read_csv(ar_file, sep="\t", encoding="utf-8")
        en_df = pd.read_csv(en_file, sep="\t", encoding="utf-8")

        required_columns = {"input", "output", "num_tokens"}
        if not (required_columns.issubset(ar_df.columns) and required_columns.issubset(en_df.columns)):
            print(f"[WARN] Skipping {core_name} — missing required columns.")
            continue

        if len(ar_df) != len(en_df):
            print(f"[WARN] Skipping {core_name} — mismatched row counts ({len(ar_df)} vs {len(en_df)}).")
            continue

        for i in range(len(ar_df)):
            row = ar_df.iloc[i] if random.random() < 0.5 else en_df.iloc[i]

            try:
                tokens = int(row["num_tokens"])
                total_tokens += tokens

                item = {
                    "instruction": row["input"],
                    "input": "",
                    "output": row["output"]
                }

                if tokens >= 8000:
                    json.dump(item, long_context_file, ensure_ascii=False)
                    long_context_file.write("\n")
                    num_long += 1
                else:
                    json.dump(item, short_context_file, ensure_ascii=False)
                    short_context_file.write("\n")
                    num_short += 1
            except Exception as e:
                print(f"[ERROR] Skipping row {i} in {core_name} due to error: {e}")

    long_context_file.close()
    short_context_file.close()

    print(f"✅ Saved output to {long_context_path} and {short_context_path}")
    print(f"📊 Total tokens: {total_tokens}")
    print(f"📦 Long context samples: {num_long}")
    print(f"📦 Short context samples: {num_short}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Row-wise random sampling between ar/en TSV files.")
    parser.add_argument("--ar_dir", default="data/IFT_DATA/ar", help="Directory containing Arabic TSV files.")
    parser.add_argument("--en_dir", default="data/IFT_DATA/en", help="Directory containing English TSV files.")
    parser.add_argument("--output_dir", default="data/IFT_DATA/final", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    process_multilang_rows(args.ar_dir, args.en_dir, args.output_dir, args.seed)
