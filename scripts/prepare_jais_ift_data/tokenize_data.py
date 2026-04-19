import os
import sys
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tokenization import JaisPlusTokenizer, tokenize_batch

# Load tokenizer
tokenizer = JaisPlusTokenizer("../tokenizer_src")

# Input-output field pairs
io_pairs = [
    ("poem_text", "poet_name"),
    ("poem_text", "poem_title"),
    ("poem_text", "meter"),
    ("poem_text", "poet_era"),
    ("poem_text", "genre"),
    ("poem_text", "location"),
    ("poem_text", "overall_explanation"),
    ("poem_text", "verses_explanation"),
    ("poet_name", "poet_description"),
    ("poet_name", "location"),
    ("poet_name", "poet_era"),
]

# Load data
train_df = pd.read_csv("data/cleaned_train.tsv", sep="\t")
test_df = pd.read_csv("data/cleaned_test.tsv", sep="\t")

# Prepare token cache directory
cache_dir = "data/tokenization"
os.makedirs(cache_dir, exist_ok=True)

# Tokenize a specific column and save to cache if not already cached
def tokenize_column(df, column, split_name):
    cache_path = os.path.join(cache_dir, f"{split_name}_{column}_lens.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"Loaded token cache for {column} [{split_name}]")
            return pickle.load(f)

    print(f"Tokenizing {column} [{split_name}]...")
    texts = df[column].astype(str).fillna("").tolist()
    tokenized = tokenize_batch(tokenizer, texts)
    token_lens = {(column, idx): len(tokens) for idx, tokens in enumerate(tokenized)}

    with open(cache_path, "wb") as f:
        pickle.dump(token_lens, f)

    return token_lens

# Collect only needed columns from io_pairs
def prepare_token_lens(df, split_name):
    needed_cols = set()
    for inp, out in io_pairs:
        needed_cols.add(inp)
        if (inp, out) == ("poem_text", "poet_name"):
            needed_cols.add("poet_description")
        else:
            needed_cols.add(out)

    token_lens = {}
    for col in needed_cols:
        if col in df.columns:
            token_lens.update(tokenize_column(df, col, split_name))
        else:
            print(f"Warning: Column '{col}' not found in {split_name} data.")

    return token_lens

# Tokenize all necessary columns
train_token_lens = prepare_token_lens(train_df, "train")
test_token_lens = prepare_token_lens(test_df, "test")

# Process input-output pair
def process_io_pair(input_col, output_col, df, token_lens, split_name):
    pair_name = f"{input_col}__{output_col}"
    out_dir = f"data/processed/{pair_name}"
    os.makedirs(out_dir, exist_ok=True)

    if input_col == "poem_text" and output_col == "poet_name":
        if "poet_description" not in df.columns:
            print(f"Skipping {pair_name} [{split_name}] - Missing poet_description")
            return

        subset = df[[input_col, output_col, "poet_description"]].dropna()
        subset = subset[
            subset[input_col].astype(str).str.strip().ne("") &
            subset[output_col].astype(str).str.strip().ne("") &
            subset["poet_description"].astype(str).str.strip().ne("")
        ].copy()

        if subset.empty:
            print(f"Skipping {pair_name} [{split_name}] - No valid data")
            return

        subset[output_col] = subset[output_col].astype(str) + " " + subset["poet_description"].astype(str)
        subset[f"{input_col}_num_tokens"] = [token_lens.get((input_col, idx), 0) for idx in subset.index]

        output_texts = subset[output_col].astype(str).tolist()
        output_tokens = tokenize_batch(tokenizer, output_texts)
        subset[f"{output_col}_num_tokens"] = [len(toks) for toks in output_tokens]

        final_df = subset[[input_col, output_col, f"{input_col}_num_tokens", f"{output_col}_num_tokens"]]

    else:
        subset = df[[input_col, output_col]].dropna()
        subset = subset[
            subset[input_col].astype(str).str.strip().ne("") &
            subset[output_col].astype(str).str.strip().ne("")
        ].copy()

        if subset.empty:
            print(f"Skipping {pair_name} [{split_name}] - No valid data")
            return

        subset[f"{input_col}_num_tokens"] = [token_lens.get((input_col, idx), 0) for idx in subset.index]
        subset[f"{output_col}_num_tokens"] = [token_lens.get((output_col, idx), 0) for idx in subset.index]

        final_df = subset[[input_col, output_col, f"{input_col}_num_tokens", f"{output_col}_num_tokens"]]

    out_path = os.path.join(out_dir, f"{split_name}.tsv")
    final_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {pair_name} [{split_name}] -> {out_path}")

# Run in parallel
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=64) as executor:
        for input_col, output_col in io_pairs:
            executor.submit(process_io_pair, input_col, output_col, train_df, train_token_lens, "train")
            executor.submit(process_io_pair, input_col, output_col, test_df, test_token_lens, "test")
