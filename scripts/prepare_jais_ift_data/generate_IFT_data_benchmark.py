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

    if task == "analysis":
        if "poet_name" in df.columns and "poet_description" in df.columns:
            df["poet_name+poet_description"] = df.apply(
                lambda x: f"{x['poet_name']}\n{x['poet_description']}"
                if pd.notna(x["poet_name"]) and pd.notna(x["poet_description"]) else pd.NA, axis=1
            )

    for language in languages:
        if language not in templates_yaml:
            print(f"[SKIP] Language '{language}' not found in templates.")
            continue

        templates = templates_yaml[language]
        templates.sort(key=lambda x: len(x['input']), reverse=True)

        output_dir = Path(output_base_dir)
        os.makedirs(output_dir, exist_ok=True)

        task_statistics = defaultdict(int)

        print(f"[LANGUAGE] Generating data for language: {language}")

        for template in tqdm(templates, desc=f"Generating data"):
            template_id = template['id']
            input_keys = template['input']
            output_key = template['output']
            paraphrased_texts = template['text']

            if not isinstance(paraphrased_texts, list) or len(paraphrased_texts) == 0:
                print(f"[WARNING] Template {template_id} has no usable paraphrased text.")
                continue

            instruction_text = paraphrased_texts[0]  # Only use the first variation

            needed_columns = input_keys + ([output_key] if isinstance(output_key, str) else output_key)
            print(f"[INFO] Template {template_id} requires columns: {needed_columns}")
            filtered_df = df[needed_columns].copy()
            filtered_df.replace('None', pd.NA, inplace=True)
            filtered_df.dropna(subset=needed_columns, inplace=True)

            if filtered_df.empty:
                print(f"[SKIP] Template {template_id} has no valid rows after filtering.")
                continue

            output_values = filtered_df[output_key].dropna().unique().tolist() if isinstance(output_key, str) else []
            formatted_batch = []
            concat_texts = []
            for idx, row in filtered_df.iterrows():
                try:
                    row_data = row[input_keys].to_dict()
                    filled_instruction = instruction_text.format(**row_data)

                    if isinstance(output_key, str):
                        correct_output = row[output_key]
                    else:
                        correct_output = "\n".join([str(row[k]) for k in output_key])

                    # Sample 4 incorrect choices, combine with correct one, then shuffle
                    n_choices = 5
                    incorrect_pool = [x for x in output_values if x != correct_output]
                    if len(incorrect_pool) < n_choices - 1:
                        print(f"[WARNING] Not enough incorrect choices for template {template_id}. Skipping this row.")
                        continue

                    incorrect_choices = random.sample(incorrect_pool, k=n_choices - 1)
                    all_choices = incorrect_choices + [correct_output]
                    random.shuffle(all_choices)

                    choice_letters = ['أ', 'ب', 'ج', 'د', 'و']
                    letter_map = dict(zip(choice_letters, all_choices))
                    correct_letter = [k for k, v in letter_map.items() if v == correct_output][0]

                    # Create instruction: format with choices
                    joined_choices = "\n".join([f"{k}. {v}" for k, v in letter_map.items()])
                    final_instruction = f"{filled_instruction.strip()}\n{joined_choices}"

                    concat_texts.append(final_instruction)
                    formatted_batch.append({
                        'instruction': final_instruction,
                        'input': "",
                        'output': correct_letter
                    })
                    task_statistics[template_id] += 1

                except Exception as e:
                    print(f"[ERROR] Template {template_id}: formatting error → {e}")

            if formatted_batch:
                print(f"[INFO] Template {template_id}: {len(formatted_batch)} samples generated.")
                tokenized = tokenize_batch(tokenizer, concat_texts)
                token_lengths = [len(tokens) for tokens in tokenized]
                for i, length in enumerate(token_lengths):
                    formatted_batch[i]['num_tokens'] = length

                output_file = Path(output_base_dir)  / f"{template_id}.tsv"
                pd.DataFrame(formatted_batch).to_csv(output_file, sep='\t', index=False)
                print(f"[DONE] {template_id}: {len(formatted_batch)} samples saved → {output_file}")

        # Save statistics
        stats_df = pd.DataFrame(
            [{"template_id": tid, "num_samples": count} for tid, count in task_statistics.items()]
        )
        stats_file = Path(output_base_dir)  / "template_stats.tsv"
        stats_df.to_csv(stats_file, sep='\t', index=False)
        print(f"[STATS] Saved statistics → {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instruction fine-tuning data from templates.")
    parser.add_argument("--raw_data", default='data/de_dupped_test.tsv', help="Path to the raw data file (TSV).")
    parser.add_argument("--templates", default='/path/to/templates/poetry_analysis.yaml', help="Path to the YAML file with templates.")
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