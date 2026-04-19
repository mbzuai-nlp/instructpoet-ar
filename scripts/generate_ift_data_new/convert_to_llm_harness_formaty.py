#!/usr/bin/env python3
"""
Convert TSV files to LM Eval Harness format for Arabic poetry analysis tasks.
"""

import os
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict


def main():
    # Input and output directories
    input_dir = (
        "/path/to/data/dialectical_IFT_DATA/tsv"
    )
    output_dir = "/path/to/data/dialectical_poetry_analysis_lm_harness_format"

    mkdirs = Path(output_dir)
    mkdirs.mkdir(parents=True, exist_ok=True)

    # Focus only on analysis task for LM Eval Harness format
    task = "analysis"

    print(f"Processing task: {task}")

    # Read TSV file from test data
    tsv_path = os.path.join(input_dir, task, "test", f"{task}_ift.tsv")

    if not os.path.exists(tsv_path):
        print(f"TSV file not found: {tsv_path}")
        return

    print(f"Reading {tsv_path}")

    # Read TSV with proper handling of quotes and tabs
    try:
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            encoding="utf-8",
            quotechar='"',
            on_bad_lines="skip",
            low_memory=False,
        )
    except Exception as e:
        print(f"Error reading {tsv_path}: {e}")
        return

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Filter rows that have choices (multiple choice questions)
    df_with_choices = df[df["choices"].notna() & (df["choices"] != "")]
    print(f"Found {len(df_with_choices)} rows with choices")

    # Group by template fields for different subtasks
    if (
        "template_input_fields" in df_with_choices.columns
        and "template_output_field" in df_with_choices.columns
    ):
        # Create a combined key for grouping by extracting dictionary keys
        def extract_dict_keys(row):
            try:
                # Parse input fields
                input_fields_data = row["template_input_fields"]
                output_field_data = row["template_output_field"]

                # Handle NaN values
                if pd.isna(input_fields_data) or pd.isna(output_field_data):
                    return "default"

                # Handle different data types - could be dict, string representation of dict, or JSON string
                if isinstance(input_fields_data, dict):
                    input_fields = input_fields_data
                elif isinstance(input_fields_data, str):
                    # Try to parse as JSON, handling single quotes
                    try:
                        # Replace single quotes with double quotes for valid JSON
                        input_fields_str = input_fields_data.replace("'", '"')
                        input_fields = json.loads(input_fields_str)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, use eval as fallback (be careful!)
                        input_fields = eval(input_fields_data)
                else:
                    input_fields = {"default": str(input_fields_data)}

                if isinstance(output_field_data, dict):
                    output_field = output_field_data
                elif isinstance(output_field_data, str):
                    # Try to parse as JSON, handling single quotes
                    try:
                        # Replace single quotes with double quotes for valid JSON
                        output_field_str = output_field_data.replace("'", '"')
                        output_field = json.loads(output_field_str)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, use eval as fallback (be careful!)
                        output_field = eval(output_field_data)
                else:
                    output_field = {"default": str(output_field_data)}

                # Extract keys and sort them for consistency
                input_keys = (
                    sorted(list(input_fields.keys()))
                    if isinstance(input_fields, dict)
                    else [str(input_fields)]
                )
                output_keys = (
                    sorted(list(output_field.keys()))
                    if isinstance(output_field, dict)
                    else [str(output_field)]
                )

                # Create template key by concatenating sorted keys
                input_key_str = "__".join(input_keys)
                output_key_str = "__".join(output_keys)

                return f"{input_key_str}___{output_key_str}"

            except Exception as e:
                print(f"Error parsing template fields: {e}")
                print(f"Input: {row['template_input_fields']}")
                print(f"Output: {row['template_output_field']}")
                return "default"

        df_with_choices["template_key"] = df_with_choices.apply(
            extract_dict_keys, axis=1
        )
        grouped = df_with_choices.groupby("template_key")

        print(f"Found {len(grouped)} unique template combinations")

        for template_key, group in grouped:
            # Clean template key for filename
            if template_key == "nan___nan" or template_key == "___":
                template_key = "default"

            # Create meaningful filename based on template
            clean_key = (
                str(template_key).replace(" ", "_").replace(",", "_").replace("/", "_")
            )
            filename = f"analysis_{clean_key}.json"

            convert_group_to_lm_harness_format(
                group, os.path.join(output_dir, filename)
            )
    else:
        print("No template fields found, creating single file")
        filename = "analysis_default.json"
        convert_group_to_lm_harness_format(
            df_with_choices, os.path.join(output_dir, filename)
        )


def convert_group_to_lm_harness_format(df_group, output_path):
    """Convert a group of data to LM Eval Harness format"""

    lm_harness_data = []

    for _, row in df_group.iterrows():
        try:
            # Parse the choices JSON string
            choices_str = row["choices"]
            if pd.isna(choices_str) or choices_str == "":
                continue

            choices_dict = json.loads(choices_str)

            # Get the input question
            input_text = str(row["input"]) if pd.notna(row["input"]) else ""

            # Create the question by concatenating input with choices
            question_parts = [input_text]

            # Sort choices by Arabic letter order (أ، ب، ج، د، هـ، و)
            arabic_order = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي"]
            sorted_choices = []
            for letter in arabic_order:
                if letter in choices_dict:
                    choice_text = f"{letter}. {choices_dict[letter]}"
                    question_parts.append(choice_text)
                    sorted_choices.append(letter)

            # Join question and choices
            question = "\n".join(question_parts)

            # Get correct answer
            correct_choice = (
                str(row["correct_choice"]).strip()
                if pd.notna(row["correct_choice"])
                else ""
            )

            # Create the LM harness entry
            entry = {
                "question": question,
                "options": sorted_choices,
                "correct_answer": correct_choice,
            }

            lm_harness_data.append(entry)

        except json.JSONDecodeError as e:
            print(f"Error parsing choices JSON: {e}")
            print(f"Choices string: {choices_str}")
            continue
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    # Write to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lm_harness_data, f, ensure_ascii=False, indent=2)

    print(f"Created {output_path} with {len(lm_harness_data)} entries")


if __name__ == "__main__":
    main()
