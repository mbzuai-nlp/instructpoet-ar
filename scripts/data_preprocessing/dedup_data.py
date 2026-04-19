import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from joblib import Parallel, delayed
import hashlib
import pickle
import psutil


def profile_resources():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e9
    cpu = psutil.cpu_percent(interval=1)
    return f"[Resource usage] Memory: {mem:.2f} GB | CPU: {cpu:.1f}%"


def normalize_text(text):
    return ' '.join(sorted(text.lower().split()))


def hash_text(text, method='md5', n=6):
    return hashlib.new(method, text.encode()).hexdigest()[:n]


def get_cache_path(dataset_name):
    os.makedirs("data/deduplication_hashes", exist_ok=True)
    return os.path.join("data/deduplication_hashes", f"{dataset_name}_hashes.pkl")


def load_hash_cache(dataset_name):
    path =  (dataset_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            print(f"[Cache] Loaded cached hashes from {path}")
            return pickle.load(f)
    return None


def save_hash_cache(dataset_name, norm_texts, buckets):
    path = get_cache_path(dataset_name)
    with open(path, "wb") as f:
        pickle.dump((norm_texts, buckets), f)
    print(f"[Cache] Saved hashes to {path}")


def compare_block(texts, idxs, threshold, non_missing_counts, match_log):
    local_to_drop = set()
    for i in range(len(texts)):
        if idxs[i] in local_to_drop:
            continue
        for j in range(i + 1, len(texts)):
            if idxs[j] in local_to_drop:
                continue
            score = fuzz.ratio(texts[i], texts[j])
            score = score / 100  # Normalize score to be in the range [0, 1]

            assert 0 <= threshold <= 1, f"Threshold must be between 0 and 1, got {threshold}"
            assert 0 <= score <= 1, f"Score must be between 0 and 1, got {score}"

            if score >= threshold:
                if non_missing_counts[idxs[i]] >= non_missing_counts[idxs[j]]:
                    local_to_drop.add(idxs[j])
                    match_log.append((idxs[i], idxs[j], score))
                else:
                    local_to_drop.add(idxs[i])
                    match_log.append((idxs[j], idxs[i], score))
                    break
    return local_to_drop



def block_and_deduplicate(df, threshold=.95, n_jobs=-1, dataset_name=None, matched_output_path=None):
    df = df.copy()
    df['poem_text_no_diacritics'] = df['poem_text_no_diacritics'].fillna('')

    if dataset_name:
        cached = load_hash_cache(dataset_name)
        if cached:
            df['norm_text'], df['bucket'] = cached
        else:
            df['norm_text'] = df['poem_text_no_diacritics'].apply(normalize_text)
            df['bucket'] = df['norm_text'].apply(lambda x: hash_text(x))
            save_hash_cache(dataset_name, df['norm_text'], df['bucket'])
    else:
        df['norm_text'] = df['poem_text_no_diacritics'].apply(normalize_text)
        df['bucket'] = df['norm_text'].apply(lambda x: hash_text(x))

    non_missing_counts = df.notna().sum(axis=1).values
    grouped = df.groupby('bucket')
    all_blocks = [(group['poem_text_no_diacritics'].tolist(), group.index.tolist()) for _, group in grouped]

    to_drop = set()
    match_log = []

    with tqdm(total=len(all_blocks), desc="Deduplicating") as pbar:
        def wrapper(block):
            return compare_block(*block, threshold, non_missing_counts, match_log)

        for i in range(0, len(all_blocks), 1000):
            batch = all_blocks[i:i+1000]
            results = Parallel(n_jobs=n_jobs)(
                delayed(wrapper)(block) for block in batch
            )
            to_drop.update(*results)
            pbar.update(len(batch))

        if matched_output_path:
            match_df = pd.DataFrame(match_log, columns=["kept_index", "dropped_index", "similarity"])
            match_df.to_csv(matched_output_path, sep="\t", index=False)
            print(f"[Match Log] Saved matched entries to {matched_output_path}")


    return df.drop(index=list(to_drop)).reset_index(drop=True)


def remove_train_dupes_against_test(train_df, test_df, threshold=0.95):
    print("Matching test rows against train set and removing similar entries from train (bucketed)...")

    train_df = train_df.copy()
    train_df['norm_text'] = train_df['poem_text_no_diacritics'].fillna('').apply(normalize_text)
    train_df['bucket'] = train_df['norm_text'].apply(lambda x: hash_text(x))

    test_df = test_df.copy()
    test_df['norm_text'] = test_df['poem_text_no_diacritics'].fillna('').apply(normalize_text)
    test_df['bucket'] = test_df['norm_text'].apply(lambda x: hash_text(x))

    test_buckets = set(test_df['bucket'])

    filtered_train_df = train_df[train_df['bucket'].isin(test_buckets)]

    to_remove = set()
    train_texts = filtered_train_df['poem_text_no_diacritics'].tolist()
    train_index = filtered_train_df.index.tolist()
    test_texts = test_df['poem_text_no_diacritics'].tolist()

    with tqdm(total=len(test_texts), desc="Removing train entries similar to test") as pbar:
        for test_text in test_texts:
            for idx, train_text in zip(train_index, train_texts):
                if idx in to_remove:
                    continue
                score = fuzz.ratio(test_text, train_text)
                score = score / 100  # Normalize score to be in the range [0, 1]
                if score >= threshold:
                    to_remove.add(idx)
            pbar.update(1)

    print(f"Removing {len(to_remove)} similar entries from train set.")
    return train_df.drop(index=list(to_remove)).reset_index(drop=True)




def main(train_path, test_path, out_train_path, out_test_path, threshold, n_jobs, matched_output_path=None):
    print(f"Loading train from {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Train rows before deduplication: {len(train_df)}")

    print(f"Loading test from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test rows before deduplication: {len(test_df)}")

    print("Deduplicating train...")
    train_df = block_and_deduplicate(train_df, threshold, n_jobs, dataset_name="train", matched_output_path=matched_output_path)
    print(f"Train rows after deduplication: {len(train_df)}")
    
    print("Deduplicating test...")
    test_df = block_and_deduplicate(test_df, threshold, n_jobs, dataset_name="test")
    print(f"Test rows after deduplication: {len(test_df)}")

    intermediate_train_path = out_train_path.replace(".tsv", "_intermediate.tsv")
    print(f"Saving intermediate train to {intermediate_train_path}")
    train_df.to_csv(intermediate_train_path, index=False)

    print(f"Saving intermediate test to {out_test_path}")
    test_df.to_csv(out_test_path, index=False)

    print(f"Reloading train from {intermediate_train_path}")
    train_df = pd.read_csv(intermediate_train_path)

    print(f"Reloading test from {out_test_path}")
    test_df = pd.read_csv(out_test_path)

    print("Removing train entries similar to test set...")
    train_df = remove_train_dupes_against_test(train_df, test_df, threshold)
    print(f"Train rows after removing similar entries to test: {len(train_df)}")

    print(f"Saving final cleaned train to {out_train_path}")
    train_df.to_csv(out_train_path, index=False)

    print(f"Saving final cleaned test to {out_test_path}")
    test_df.to_csv(out_test_path, index=False)

    print(profile_resources())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/cleaned_train.csv")
    parser.add_argument("--test_path", type=str, default="data/cleaned_test.csv")
    parser.add_argument("--out_train_path", type=str, default="data/de_dupped_train.csv")
    parser.add_argument("--out_test_path", type=str, default="data/de_dupped_test.csv")
    parser.add_argument("--threshold", type=float, default=.90)
    parser.add_argument("--matched_output_path", type=str, default=None, help="Path to save matched similar entries (optional)")
    parser.add_argument("--n_jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(
        args.train_path,
        args.test_path,
        args.out_train_path,
        args.out_test_path,
        args.threshold,
        args.n_jobs,
        args.matched_output_path
    )
