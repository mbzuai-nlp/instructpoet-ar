#!/usr/bin/env python3
"""
Convert TSV files to JSONL format for Arabic poetry tasks.
"""

import os
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import ast


def parse_template_field(field_str):
    """
    Safely parse template field strings that can be in various formats:
    - Simple strings: 'poet_name' or "poet_name"
    - Unquoted strings: poet_name
    - Dictionary strings: {'key': 'value'}
    - JSON strings: {"key": "value"}
    - Lists: ['item1', 'item2']
    """
    if not field_str or pd.isna(field_str):
        return {"default": "unknown"}

    field_str = str(field_str).strip()

    # If it's a simple quoted string, extract the content
    if (field_str.startswith("'") and field_str.endswith("'")) or (
        field_str.startswith('"') and field_str.endswith('"')
    ):
        # Remove quotes and return as single key
        content = field_str[1:-1]
        return {content: content}

    # Try to parse as JSON first
    try:
        # Replace single quotes with double quotes for valid JSON
        json_str = field_str.replace("'", '"')
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list):
            # Convert list to dict with items as both key and value
            return {str(item): str(item) for item in parsed}
        else:
            return {str(parsed): str(parsed)}
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to parse as Python literal (safer than eval)
    try:
        parsed = ast.literal_eval(field_str)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list):
            # Convert list to dict with items as both key and value
            return {str(item): str(item) for item in parsed}
        else:
            return {str(parsed): str(parsed)}
    except (ValueError, SyntaxError):
        pass

    # If all parsing fails, treat as unquoted string
    # This handles cases like: poet_name, genre, etc.
    return {field_str: field_str}


def main():
    # Input and output directories
    input_dir = (
        "/path/to/data/continuation_IFT_DATA/tsv"
    )
    output_dir = "/path/to/data/poetry_share_crebras/ShortContext/train"

    mkdirs = Path(output_dir)
    mkdirs.mkdir(parents=True, exist_ok=True)

    # Task folders
    # tasks = ["continuation"]
    tasks = ["analysis", "continuation", "corruption", "generation"]

    for task in tasks:
        print(f"\nProcessing task: {task}")

        # Read TSV file
        tsv_path = os.path.join(input_dir, task, "train", f"{task}_ift.tsv")

        if not os.path.exists(tsv_path):
            print(f"TSV file not found: {tsv_path}")
            continue

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
            continue

        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")

        # Group by template fields for subtasks
        if task == "corruption":
            # For corruption task, group by corruption_type
            if "corruption_type" in df.columns:
                grouped = df.groupby("corruption_type")
                print(f"Found corruption types: {list(grouped.groups.keys())}")

                for corruption_type, group in grouped:
                    if pd.isna(corruption_type) or corruption_type == "":
                        corruption_type = "unknown"

                    # Clean corruption type for filename
                    clean_type = (
                        str(corruption_type)
                        .replace(" ", "_")
                        .replace(",", "_")
                        .replace("/", "_")
                    )
                    filename = f"corruption_{clean_type}.jsonl"

                    convert_group_to_jsonl(group, os.path.join(output_dir, filename))
            else:
                print("No corruption_type column found, creating single file")
                filename = f"{task}.jsonl"
                convert_group_to_jsonl(df, os.path.join(output_dir, filename))
        else:
            # For other tasks, group by template fields
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
                        input_fields = parse_template_field(input_fields_data)
                    else:
                        input_fields = {"default": str(input_fields_data)}

                    if isinstance(output_field_data, dict):
                        output_field = output_field_data
                    elif isinstance(output_field_data, str):
                        output_field = parse_template_field(output_field_data)
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

            if (
                "template_input_fields" in df.columns
                and "template_output_field" in df.columns
            ):
                # Create a combined key for grouping
                df["template_key"] = df.apply(extract_dict_keys, axis=1)

                grouped = df.groupby("template_key")

                print(f"Found {len(grouped)} unique template combinations")

                for template_key, group in grouped:
                    # Clean template key for filename
                    if template_key == "nan___nan" or template_key == "___":
                        template_key = "default"

                    # Limit filename length and clean characters
                    clean_key = (
                        str(template_key)
                        .replace(" ", "_")
                        .replace(",", "_")
                        .replace("/", "_")
                    )
                    # clean_key = clean_key[:50] if len(clean_key) > 50 else clean_key

                    filename = f"{task}__{clean_key}.jsonl"

                    convert_group_to_jsonl(group, os.path.join(output_dir, filename))
            else:
                print("No template fields found, creating single file")
                filename = f"{task}.jsonl"
                convert_group_to_jsonl(df, os.path.join(output_dir, filename))


def convert_group_to_jsonl(df_group, output_path):
    """Convert a group of data to JSONL format"""

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df_group.iterrows():
            # Create JSONL entry with required format
            entry = {
                "instruction": str(row["input"]) if pd.notna(row["input"]) else "",
                "output": str(row["output"]) if pd.notna(row["output"]) else "",
                "input": "",
            }

            # Write to file
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Created {output_path} with {count} entries")


if __name__ == "__main__":
    main()
