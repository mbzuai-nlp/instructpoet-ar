import pandas as pd
import yaml
import os
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import defaultdict
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tokenization import JaisPlusTokenizer, tokenize_batch

tokenizer = JaisPlusTokenizer("../tokenizer_src")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_data_from_templates(
    raw_data_path: str,
    template_path: str,
    output_base_dir: str,
    languages: List[str],
    task: str
):
    df = pd.read_csv(raw_data_path, sep='\t')
    templates_yaml = load_yaml(template_path)
    print("[INFO] Data columns:", df.columns)

    # === Task-specific preprocessing ===
    if task == "analysis":
        if "poet_name" in df.columns and "poet_description" in df.columns:
            df["poet_name+poet_description"] = df.apply(
                lambda x: f"{x['poet_name']}\n{x['poet_description']}" 
                if pd.notna(x["poet_name"]) and pd.notna(x["poet_description"]) else pd.NA, axis=1
            )

    sampling_fractions = [0.10, 0.25, 0.50, 1.00]

    for language in languages:
        if language not in templates_yaml:
            print(f"[SKIP] Language '{language}' not found in templates.")
            continue

        templates = templates_yaml[language]
        templates.sort(key=lambda x: len(x['input']), reverse=True)

        output_dir = Path(output_base_dir) / language
        os.makedirs(output_dir, exist_ok=True)

        task_statistics = defaultdict(int)
        used_indices: Set[int] = set()
        collected_data: Dict[str, List[Dict]] = defaultdict(list)

        print(f"[LANGUAGE] Generating data for language: {language}")

        for frac in sampling_fractions:
            print(f"[INFO] Sampling with fraction: {frac}")
            for template in tqdm(templates, desc=f"Sampling fraction {frac}"):
                template_id = template['id']
                input_keys = template['input']
                output_key = template['output']
                paraphrased_texts = template['text']

                if not isinstance(paraphrased_texts, list):
                    print(f"[WARNING] Template {template_id}: 'text' should be a list.")
                    continue

                needed_columns = input_keys + ([output_key] if isinstance(output_key, str) else output_key)
                filtered_df = df[needed_columns].copy()
                filtered_df.replace('None', pd.NA, inplace=True)
                filtered_df.dropna(subset=needed_columns, inplace=True)
                filtered_df = filtered_df.loc[~filtered_df.index.isin(used_indices)]

                if filtered_df.empty:
                    continue

                to_sample = int(frac * len(filtered_df))
                if to_sample <= 0:
                    continue

                sampled_df = filtered_df.sample(n=to_sample, replace=False, random_state=42)
                concat_texts = []
                formatted_batch = []

                for idx, row in sampled_df.iterrows():
                    try:
                        row_data = row[input_keys].to_dict()

                        # === Task-specific logic: analysis ===
                        if task == "analysis" and "poem_text" in row_data:
                            poem_lines = row_data["poem_text"].splitlines()
                            if len(poem_lines) > 1:
                                k = random.randint(1, min(5, len(poem_lines)))
                                start = random.randint(0, len(poem_lines) - k)
                                selected = poem_lines[start:start + k]
                                row_data["poem_text"] = "\n".join(selected)

                        instruction = random.choice(paraphrased_texts)
                        formatted_input = instruction.format(**row_data)

                        if isinstance(output_key, str):
                            formatted_output = row[output_key]
                        else:
                            formatted_output = "\n".join([str(row[k]) for k in output_key])

                        concat_texts.append(formatted_input + "\n" + formatted_output)
                        formatted_batch.append({
                            'input': formatted_input,
                            'output': formatted_output
                        })

                        used_indices.add(idx)
                        task_statistics[template_id] += 1
                    except Exception as e:
                        print(f"[ERROR] Template {template_id}: formatting error → {e}")

                if formatted_batch:
                    tokenized = tokenize_batch(tokenizer, concat_texts)
                    token_lengths = [len(tokens) for tokens in tokenized]
                    for i, length in enumerate(token_lengths):
                        formatted_batch[i]['num_tokens'] = length
                    collected_data[template_id].extend(formatted_batch)

        # Save per-template files
        for template_id, samples in collected_data.items():
            output_file = Path(output_base_dir) / language / f"{template_id}.tsv"
            pd.DataFrame(samples).to_csv(output_file, sep='\t', index=False)
            print(f"[DONE] {template_id}: {len(samples)} samples saved → {output_file}")

        # Save statistics
        stats_df = pd.DataFrame(
            [{"template_id": tid, "num_samples": count} for tid, count in task_statistics.items()]
        )
        stats_file = Path(output_base_dir) / language / "template_stats.tsv"
        stats_df.to_csv(stats_file, sep='\t', index=False)
        print(f"[STATS] Saved statistics → {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instruction fine-tuning data from templates.")
    parser.add_argument("--raw_data", default='data/de_dupped_train.tsv', help="Path to the raw data file (TSV).")
    parser.add_argument("--templates", default='/path/to/templates/poetry_generation.yaml', help="Path to the YAML file with templates.")
    parser.add_argument("--output_dir", default='data/IFT_DATA', help="Directory to save generated samples.")
    parser.add_argument("--language", default='ar', help="Comma-separated list of languages (e.g., ar,en).")
    parser.add_argument("--task", default='generation', help="Task name: generation | analysis")

    args = parser.parse_args()
    languages = [lang.strip() for lang in args.language.split(',') if lang.strip()]

    generate_data_from_templates(
        raw_data_path=args.raw_data,
        template_path=args.templates,
        output_base_dir=args.output_dir,
        languages=languages,
        task=args.task
    )
