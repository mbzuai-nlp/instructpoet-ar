import pandas as pd
import os
import argparse
import random
import ast
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
from tqdm import tqdm
import sys


import ast
import json
from typing import Any
import re


def naturalize_list_field(value: Any) -> str:
    """Convert list-like strings into natural Arabic phrases."""
    if not isinstance(value, str):
        return str(value)

    # Normalize malformed double double-quotes: "" -> "
    normalized = value.replace('""', '"').strip()

    parsed = None
    try:
        # First try JSON
        parsed = json.loads(normalized)
    except Exception:
        try:
            # Fall back to Python literal
            parsed = ast.literal_eval(normalized)
        except Exception:
            return value  # not a list, just return original

    if isinstance(parsed, list):
        parsed = [str(x).strip() for x in parsed if x]
        if not parsed:
            return ""
        if len(parsed) == 1:
            return parsed[0]
        return "، ".join(parsed[:-1]) + "، و" + parsed[-1]

    return value


def load_templates_excel(
    path: str, task: str, preferred_dialect: str = "random"
) -> List[Dict[str, Any]]:
    """Load and group templates from Excel file, selecting the right sheet for the task."""
    sheet_map = {
        "generation": "Poetry Generation",
        "analysis": "Poetry Analysis",
        "continuation": "Poetry Continuation",
        "corruption": "Poetry Corruption",
    }
    sheet_name = sheet_map[task]
    df = pd.read_excel(path, sheet_name=sheet_name)

    # Lowercase column names
    df.columns = df.columns.str.lower()

    grouped_templates: Dict[str, Dict[str, Any]] = {}

    # Strip trailing spaces from column names
    df.columns = df.columns.str.strip()

    print(df.columns)

    # Define available dialect columns (MSA + 4 dialects)
    dialect_columns = ["msa", "nile valley", "north africa", "gulf", "levant"]

    # For corruption task, use different column structure
    if task == "corruption":
        # Check which dialect columns are available in the corruption task
        available_dialect_columns = [
            col for col in dialect_columns if col in df.columns
        ]
        if not available_dialect_columns:
            # Fallback to MSA if no dialect columns found
            available_dialect_columns = ["msa"]

        required_columns = [
            available_dialect_columns[0],
            "corruption_type",
            "placeholder",
        ]
        # Drop rows with NaN values in restoration prompt and template
        df.dropna(subset=required_columns, inplace=True)

        return df

    else:

        # Original logic for other tasks
        # Check which dialect columns are available
        available_dialect_columns = [
            col for col in dialect_columns if col in df.columns
        ]
        if not available_dialect_columns:
            # Fallback to MSA if no dialect columns found
            available_dialect_columns = ["msa"]

        # Ensure required columns exist before dropping rows with NaN values
        required_columns = ["placeholder", "output"] + available_dialect_columns
        existing_columns = [col for col in required_columns if col in df.columns]

        # Drop rows with NaN values only in the existing required columns
        if existing_columns:
            df.dropna(subset=existing_columns, inplace=True)

        for _, row in df.iterrows():
            # Strip trailing spaces from column names and values
            row = {
                k.strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }

            placeholders = [p.strip() for p in str(row["placeholder"]).split(",")]

            # Select dialect based on preference
            if preferred_dialect == "random":
                selected_dialect = random.choice(available_dialect_columns)
            elif preferred_dialect.lower() in available_dialect_columns:
                selected_dialect = preferred_dialect.lower()
            else:
                print(
                    f"[WARNING] Preferred dialect '{preferred_dialect}' not found. Using random selection."
                )
                selected_dialect = random.choice(available_dialect_columns)

            text = row[selected_dialect].strip()

            # Insert a newline before "{existing_verses}" if it exists in the text
            if "{existing_verses}" in text:
                text = text.replace("{existing_verses}", "\n{existing_verses}")

            if task == "continuation":
                output_field = None
            else:
                output_field = row.get("output", "poem_text")

            key = (tuple(placeholders), output_field)
            if key not in grouped_templates:
                grouped_templates[key] = {
                    "input": placeholders,
                    "output": output_field,
                    "text": [],
                }
            # store both text and which dialect it came from so we can track stats later
            grouped_templates[key]["text"].append(
                {"text": text, "dialect": selected_dialect}
            )

        return list(grouped_templates.values())


def remove_arabic_elongation(text: str) -> str:
    """
    Removes Arabic elongation characters (tatweel) and extra spaces.
    """
    # Remove tatweel
    text = text.replace("ـ", "")
    # Remove extra spaces
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def generate_data_from_templates(
    raw_data_path: str,
    template_path: str,
    output_base_dir: str,
    task: str,
    total_num_samples: int = -1,
    min_num_verses: int = 1,
    create_mcq_benchmark: bool = False,
    max_poem_verses: int = -1,
    preferred_dialect: str = "random",
):
    if raw_data_path.endswith(".tsv"):
        df = pd.read_csv(raw_data_path, sep="\t")
    else:
        df = pd.read_csv(raw_data_path)
    print("[INFO] Data columns:", df.columns)

    # Determine dataset type from raw_data_path early to use throughout the function
    is_training = "train" in raw_data_path.lower()
    dataset_type = "train" if is_training else "test"

    # Filter by min number of verses
    df = df[df["poem_verses"] >= min_num_verses]

    templates = load_templates_excel(template_path, task, preferred_dialect)

    output_dir = Path(output_base_dir)
    os.makedirs(output_dir, exist_ok=True)

    task_statistics = defaultdict(int)
    # Track how many generated samples use each dialect
    dialect_statistics = defaultdict(int)
    used_indices: Set[int] = set()
    collected_data: List[Dict] = []

    sampling_fractions = [0.10, 0.25, 0.50, 1.00]

    # Special handling for corruption task
    if task == "corruption":
        # templates is now a DataFrame with corruption templates
        template_df = templates

        # Filter for rows that have corruption data
        available_columns = [
            "corrupted_poem",
            "poem_text",
            "corruption_type",
            "corruption_assigned_template",
            "corruption_metadata",
        ]
        # filtered_df = df[available_columns].copy()
        filtered_df = df.copy()
        filtered_df.replace("None", pd.NA, inplace=True)
        # Only keep rows that have both corrupted_poem and corruption_assigned_template
        filtered_df.dropna(
            subset=["corrupted_poem", "corruption_assigned_template"], inplace=True
        )

        print(
            f"[INFO] Processing all {len(filtered_df)} corruption samples (no sampling)"
        )

        for idx, row in tqdm(
            filtered_df.iterrows(),
            desc="Processing corruption samples",
            total=len(filtered_df),
        ):
            try:
                # Get the assigned template index
                assigned_template_idx = row["corruption_assigned_template"]

                # Find the template in the template DataFrame by index
                template_row = template_df.iloc[int(assigned_template_idx)]

                # Define available dialect columns (MSA + 4 dialects) and randomly sample one
                dialect_columns = [
                    "msa",
                    "nile valley",
                    "north africa",
                    "gulf",
                    "levant",
                ]
                available_dialect_columns = [
                    col for col in dialect_columns if col in template_df.columns
                ]
                if not available_dialect_columns:
                    available_dialect_columns = ["msa"]  # Fallback to MSA

                # Select dialect based on preference for corruption task
                if preferred_dialect == "random":
                    selected_dialect = random.choice(available_dialect_columns)
                elif preferred_dialect.lower() in available_dialect_columns:
                    selected_dialect = preferred_dialect.lower()
                else:
                    print(
                        f"[WARNING] Preferred dialect '{preferred_dialect}' not found for corruption. Using random selection."
                    )
                    selected_dialect = random.choice(available_dialect_columns)

                restoration_prompt = str(template_row[selected_dialect]).strip()

                corrupted_poem = str(row["corrupted_poem"]).strip()
                original_poem = str(row["poem_text"]).strip()
                corruption_type = str(row["corruption_type"]).strip()
                corruption_metadata = str(row["corruption_metadata"]).strip()

                dict_row_data = {
                    "poem_text": corrupted_poem,
                }
                for k in corruption_metadata.split(","):
                    if k.strip() == "poem_text":
                        continue
                    dict_row_data[k.strip()] = row.get(k.strip())

                # Add quotes around poem_title if it exists
                if "poem_title" in dict_row_data and dict_row_data["poem_title"]:
                    dict_row_data["poem_title"] = f'"{dict_row_data["poem_title"]}"'

                formatted_input = restoration_prompt.format(**dict_row_data)

                formatted_input = remove_arabic_elongation(formatted_input).strip()

                # print(f"[DEBUG] Input: {formatted_input}")

                # Create dictionaries with field keys and their values
                template_output_dict = {"poem_text": original_poem}

                # Parse corruption_metadata as input fields dictionary
                template_input_dict = {}
                try:
                    if corruption_metadata and corruption_metadata != "None":
                        # Handle different formats of corruption_metadata
                        if isinstance(corruption_metadata, str):
                            # If it's a string representation of a dict, try to parse it
                            if corruption_metadata.startswith(
                                "{"
                            ) and corruption_metadata.endswith("}"):
                                template_input_dict = ast.literal_eval(
                                    corruption_metadata
                                )
                            else:
                                # If it's just field names, use the values from dict_row_data
                                for field in corruption_metadata.split(", "):
                                    field = field.strip()
                                    if field in dict_row_data:
                                        template_input_dict[field] = dict_row_data[
                                            field
                                        ]
                        elif isinstance(corruption_metadata, dict):
                            template_input_dict = corruption_metadata
                except:
                    # Fallback: use all available fields from dict_row_data
                    template_input_dict = {
                        k: v for k, v in dict_row_data.items() if k != "poem_text"
                    }

                sample_entry = {
                    "input": formatted_input,
                    "output": original_poem,
                    "template_output_field": template_output_dict,
                    "template_input_fields": template_input_dict,
                    "corruption_type": corruption_type,
                    "dialect": selected_dialect,
                }

                collected_data.append(sample_entry)
                task_statistics[f"corruption_template_{assigned_template_idx}"] += 1
                # count dialect usage for stats
                try:
                    dialect_statistics[selected_dialect] += 1
                except Exception:
                    dialect_statistics["msa"] += 1

            except Exception as e:
                print(f"[ERROR] formatting error for row {idx} → {e}")
                continue

    else:
        # Original logic for other tasks
        # Pre-compute all unique values for MCQ generation if needed (optimization)
        mcq_unique_values_cache = {}
        if task == "analysis" and create_mcq_benchmark:
            print("[INFO] Pre-computing unique values for MCQ generation...")
            for template in templates:
                output_key = template["output"]
                if output_key and output_key not in mcq_unique_values_cache:
                    all_values = df[output_key].dropna().unique().tolist()
                    all_values = [
                        str(x).strip() for x in all_values if str(x).strip() != ""
                    ]
                    mcq_unique_values_cache[output_key] = all_values
                    print(
                        f"[INFO] Cached {len(all_values)} unique values for {output_key}"
                    )

        for frac in sampling_fractions:
            print(f"[INFO] Sampling with fraction: {frac}")
            for template in tqdm(templates, desc=f"Sampling fraction {frac}"):
                input_keys = template["input"]
                output_key = template["output"]

                # template["text"] now contains dicts: {"text": <str>, "dialect": <str>}
                # Clean up text and preserve dialect info
                paraphrased_texts = []
                for tdict in template["text"]:
                    raw_text = tdict.get("text", "")
                    cleaned = (
                        raw_text.replace("-", "")
                        .replace("{existing_verses}", "\n{existing_verses}\n")
                        .replace("{poem_text}", "\n{poem_text}\n")
                    )
                    paraphrased_texts.append(
                        {"text": cleaned, "dialect": tdict.get("dialect", "msa")}
                    )

                needed_columns = input_keys.copy()
                if task != "continuation" and output_key:
                    needed_columns.append(output_key)

                ## we remove this as it's not orignally in the df, but we create this down from the poem verses
                if "existing_verses" in needed_columns:
                    needed_columns.remove("existing_verses")

                filtered_df = df[needed_columns + ["poem_text"]].copy()
                filtered_df.replace("None", pd.NA, inplace=True)
                filtered_df.dropna(subset=needed_columns, inplace=True)
                filtered_df = filtered_df.loc[~filtered_df.index.isin(used_indices)]

                if filtered_df.empty:
                    continue

                to_sample = int(frac * len(filtered_df))
                if total_num_samples != -1:
                    to_sample = min(to_sample, total_num_samples - len(collected_data))
                    if to_sample <= 0:
                        continue

                sampled_df = filtered_df.sample(
                    n=to_sample, replace=False, random_state=42
                )

                for idx, row in sampled_df.iterrows():
                    try:
                        row_data = row.to_dict() if input_keys else {}
                        # Convert list-like fields to natural strings
                        for k, v in row_data.items():
                            row_data[k] = naturalize_list_field(v)

                        # Add quotes around poem_title if it exists
                        if "poem_title" in row_data and row_data["poem_title"]:
                            row_data["poem_title"] = f'"{row_data["poem_title"]}"'

                        # === Limit poem verses based on max_poem_verses parameter ===
                        if max_poem_verses > 0 and "poem_text" in row_data:
                            poem_lines = str(row_data["poem_text"]).split("\n")
                            if len(poem_lines) > max_poem_verses:
                                row_data["poem_text"] = "\n".join(
                                    poem_lines[:max_poem_verses]
                                )
                                print(
                                    f"[INFO] Limited poem to {max_poem_verses} verses (was {len(poem_lines)} verses)"
                                )
                        # ===================================================================================
                        # === Continuation task logic ===
                        if task == "continuation":
                            poem_lines = [
                                line
                                for line in row["poem_text"].split("\n")
                                if line.strip()
                            ]
                            if len(poem_lines) < 2:
                                continue  # Skip poems that are too short (need at least 2 verses)
                            percentage = random.choice(
                                range(10, 90, 10)
                            )  # 10%,20%,...,80%
                            cut = max(1, int(len(poem_lines) * (percentage / 100)))
                            cut = min(
                                cut, len(poem_lines) - 1
                            )  # Ensure at least 1 verse for output
                            cut = max(
                                1, cut
                            )  # Ensure at least 1 verse for existing_verses
                            row_data["existing_verses"] = "\n".join(poem_lines[:cut])
                            output = "\n".join(poem_lines[cut:])

                            assert (
                                output.strip() != ""
                            ), "[ERROR] Output for continuation is empty."
                            assert (
                                row_data["existing_verses"].strip() != ""
                            ), "[ERROR] Existing verses for continuation is empty."

                            formatted_output = output
                            # print(f"[DEBUG] Existing verses: {row_data['existing_verses']}")
                        else:
                            formatted_output = str(df.at[idx, output_key]).strip()

                        instruction = random.choice(paraphrased_texts)
                        instruction_text = instruction["text"]
                        instruction_dialect = instruction.get("dialect", "msa")
                        formatted_input = instruction_text.format(**row_data)

                        # print(f"[DEBUG] Input: {formatted_input}")
                        # print(f"[DEBUG] Output: {formatted_output}")
                        # break

                        formatted_input = remove_arabic_elongation(formatted_input)
                        formatted_output = remove_arabic_elongation(formatted_output)

                        # Create dictionaries with field keys and their values
                        template_output_dict = {
                            (
                                output_key if output_key else "poem_continuation"
                            ): formatted_output
                        }

                        template_input_dict = {}
                        if input_keys:
                            for key in input_keys:
                                if key in row_data:
                                    template_input_dict[key] = row_data[key]

                        sample_entry = {
                            "input": formatted_input,
                            "output": formatted_output,
                            "template_output_field": template_output_dict,
                            "template_input_fields": template_input_dict,
                            "dialect": instruction_dialect,
                        }

                        # === Benchmark multiple-choice logic (only for analysis + flag) ===
                        # print(f"[DEBUG] Task: {task}, Create MCQ: {create_mcq_benchmark}")
                        if task == "analysis" and create_mcq_benchmark:
                            # print("[INFO] Creating MCQ benchmark for analysis task")
                            # Use pre-computed unique values from cache
                            all_values = mcq_unique_values_cache.get(output_key, [])
                            correct_answer = formatted_output

                            # Get up to 4 distractors not equal to correct answer
                            available_distractors = [
                                val for val in all_values if val != correct_answer
                            ]
                            if len(available_distractors) >= 4:
                                distractors = random.sample(available_distractors, k=4)
                            else:
                                distractors = available_distractors

                            # Combine and shuffle
                            choices = distractors + [correct_answer]
                            random.shuffle(choices)

                            # Use Arabic choice letters
                            arabic_letters = ["أ", "ب", "ج", "د", "هـ"]
                            choices_with_labels = {
                                letter: choice
                                for letter, choice in zip(arabic_letters, choices)
                            }

                            # Find correct choice letter
                            correct_letter = [
                                letter
                                for letter, choice in choices_with_labels.items()
                                if choice == correct_answer
                            ][0]

                            # Add to sample entry
                            sample_entry.update(
                                {
                                    "choices": json.dumps(
                                        choices_with_labels, ensure_ascii=False
                                    ),
                                    "correct_answer": correct_answer,
                                    "correct_choice": correct_letter,
                                }
                            )

                            # print(f"[DEBUG] Sample Entry: {sample_entry}")

                        collected_data.append(sample_entry)
                        # increment dialect stats for this generated sample
                        dialect_statistics[instruction_dialect] += 1

                        used_indices.add(idx)
                        key_str = f"{tuple(input_keys)} -> {output_key}"
                        task_statistics[key_str] += 1
                    except Exception as e:
                        print(f"[ERROR] formatting error → {e}")
                        break

    # Assert that we have generated some data
    assert (
        len(collected_data) > 0
    ), f"[ERROR] No data was generated for task '{task}'. Check your templates and input data."

    # Create new output structure with separate folders for TSV, JSONL, and stats
    base_output_path = Path(output_base_dir)

    # Create the three main folders
    tsv_dir = base_output_path / "tsv" / task
    jsonl_dir = base_output_path / "jsonl" / task
    stats_dir = base_output_path / "stats" / task

    # Create subdirectories for train/test
    tsv_task_dir = tsv_dir / dataset_type
    jsonl_task_dir = jsonl_dir / dataset_type
    stats_task_dir = stats_dir / dataset_type

    # Create all directories
    os.makedirs(tsv_task_dir, exist_ok=True)
    os.makedirs(jsonl_task_dir, exist_ok=True)
    os.makedirs(stats_task_dir, exist_ok=True)

    # Save TSV results
    tsv_file = tsv_task_dir / f"{task}_ift.tsv"
    pd.DataFrame(collected_data).to_csv(tsv_file, sep="\t", index=False)
    print(f"[DONE] Saved {len(collected_data)} TSV samples → {tsv_file}")

    # Save JSONL results with simplified format
    jsonl_file = jsonl_task_dir / f"{task}_ift.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in collected_data:
            # Create simplified JSONL entry with only instruction, input, and output
            jsonl_entry = {
                "instruction": item["input"],  # Current input becomes instruction
                "input": "",  # Input is always empty
                "output": item["output"],  # Output remains the same
            }
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
    print(f"[DONE] Saved {len(collected_data)} JSONL samples → {jsonl_file}")

    # Save stats
    stats_file = stats_task_dir / f"{task}_stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("Template Statistics\n")
        f.write("===================\n")
        for k, v in task_statistics.items():
            f.write(f"{k}: {v}\n")
        # Write dialect statistics
        f.write("\nDialect Statistics\n")
        f.write("===================\n")
        for k, v in dialect_statistics.items():
            f.write(f"{k}: {v}\n")
    print(f"[DONE] Stats saved → {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instruction fine-tuning data from Excel templates."
    )
    parser.add_argument(
        "--raw_data",
        default="data/de_dupped_train.tsv",
        help="Path to the raw data file (TSV).",
    )
    parser.add_argument(
        "--templates",
        default="templates.xlsx",
        help="Path to the Excel file with templates.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/IFT_DATA",
        help="Directory to save generated samples.",
    )
    parser.add_argument(
        "--task",
        default="generation",
        help="Task name: generation | analysis | continuation | corruption",
    )
    parser.add_argument(
        "--total_num_samples",
        type=int,
        default=-1,
        help="Total number of samples to generate (-1 for full dataset).",
    )
    parser.add_argument(
        "--min_num_verses",
        type=int,
        default=1,
        help="Minimum number of verses to consider.",
    )
    parser.add_argument(
        "--create_mcq_benchmark",
        action="store_true",
        help="If set, create MCQ benchmark for analysis task",
    )
    parser.add_argument(
        "--max_poem_verses",
        type=int,
        default=-1,
        help="Maximum number of verses to include in poems. Use 0 or -1 for no limit (default: -1).",
    )
    parser.add_argument(
        "--preferred_dialect",
        type=str,
        default="random",
        choices=["random", "msa", "nile valley", "north africa", "gulf", "levant"],
        help="Preferred dialect for templates. Use 'random' to sample randomly from all available dialects (default: random).",
    )

    args = parser.parse_args()

    # Use the output_dir as the base for all three folder structures
    base_output_dir = Path(args.output_dir)

    generate_data_from_templates(
        raw_data_path=args.raw_data,
        template_path=args.templates,
        output_base_dir=base_output_dir,
        task=args.task,
        total_num_samples=args.total_num_samples,
        min_num_verses=args.min_num_verses,
        create_mcq_benchmark=args.create_mcq_benchmark,
        max_poem_verses=args.max_poem_verses,
        preferred_dialect=args.preferred_dialect,
    )
