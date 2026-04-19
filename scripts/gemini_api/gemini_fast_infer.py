import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
import re


import time
import yaml
import json
import logging
from tqdm import tqdm
from google import genai
from pathlib import Path
from threading import Lock
from pydantic import BaseModel
from google.genai.errors import ServerError
from tqdm.contrib.concurrent import thread_map

# add path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
# Load environment variables
load_dotenv()


def load_yaml(yaml_filepath: Path) -> dict:
    if not yaml_filepath.exists():
        raise FileNotFoundError(f"Credentials file {yaml_filepath} does not exist.")
    with open(yaml_filepath, "r") as f:
        return yaml.safe_load(f)


# setup logger
log_filepath = (
    Path(__file__).resolve().parent / "logs" / "gemini_key_phrases_generation.log"
)
log_filepath.parent.mkdir(parents=True, exist_ok=True)
log_format = (
    "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s) - %(message)s"
)
# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Capture all log levels
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(log_filepath, mode="w", encoding="utf-8"),  # Logs to file
    ],
)
logger = logging.getLogger(__name__)


def load_gemini(api_key: str) -> genai.Client:
    while True:
        try:
            # Attempt to create a Gemini client
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client loaded successfully.")
            return client
        except ServerError as e:
            logger.debug(
                f"Failed to load Gemini client: {e}. Retrying in five seconds..."
            )
            time.sleep(5)
        except Exception as e:
            raise e


def gen_gemini(client, schema: BaseModel, prompt, max_retries=3, retry_delay=5):
    """Generate content using Gemini."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            return response
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise RuntimeError("Failed to generate content after multiple attempts.")


def process_entry_wrapper(args):
    entry, client, schema, id_col, input_query, max_retries, retry_delay = args
    for attempt in range(max_retries):
        try:
            response = gen_gemini(client, schema, input_query)
            if response.parsed:
                parsed_output = response.parsed.model_dump()
            elif response.text:
                logger.warning(
                    f"Couldn't get the response in JSON for entry: {entry[id_col]}!"
                )
                parsed_output = {"text": response.text}
            else:
                logger.warning(f"Couldn't get an output from entry: {entry[id_col]}!")
                return None, 0
            parsed_output[id_col] = entry[id_col]
            return parsed_output, response.usage_metadata.total_token_count
        except Exception as e:
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for entry {entry[id_col]} failed: {e}"
            )
            time.sleep(retry_delay)

    logger.warning(f"Skipping entry {entry[id_col]} after {max_retries} retries.")
    return None, 0


def process_data_with_gemini(
    client,
    schema: BaseModel,
    data: list[dict],
    id_col: str,
    input_col: str,
    out_jsonl_filepath: Path,
    max_retries=3,
    retry_delay=5,
    max_workers=10,
):
    """Generate content using Gemini in parallel using thread_map with retries."""
    if not out_jsonl_filepath.parent.exists():
        out_jsonl_filepath.parent.mkdir(parents=True, exist_ok=True)
    written_ids = set()
    if out_jsonl_filepath.exists():
        with open(out_jsonl_filepath, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    written_ids.add(obj[id_col])
                except json.JSONDecodeError:
                    continue

    entries_to_process = [entry for entry in data if entry[id_col] not in written_ids]
    logger.info(
        f"Skipping {len(written_ids)} rows. "
        + f"Processing {len(entries_to_process)} new entries."
    )

    # Prepare arguments for thread_map
    args_list = [
        (entry, client, schema, id_col, entry[input_col], max_retries, retry_delay)
        for entry in entries_to_process
    ]
    total_tokens = 0
    lock = Lock()
    with open(out_jsonl_filepath, "a", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(args_list), max_workers), desc="Processing"):
            batch_args = args_list[i : i + max_workers]
            results = thread_map(
                process_entry_wrapper,
                batch_args,
                max_workers=max_workers,
                desc=f"Batch {i//max_workers+1}",
            )
            # write out the results
            for result, tokens_used in results:
                if result:
                    with lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        total_tokens += tokens_used

            fout.flush()
            logger.info(f"Tokens used so far: {total_tokens}")

    logger.info(f"Finished processing. Output saved to {out_jsonl_filepath}")


KEYWORD_EXTRACTION_PROMPT_TEMPLATE = """
You will be given an Arabic poem. Your task is to analyze its content and return:

- 3 keywords that best represent the core themes or concepts of the poem.
- 3 key phrases that are meaningful or characteristic expressions from the poem.

All output must be in Arabic. 
Return the result strictly in JSON format with the following structure:
{{ "keywords": [], "key_phrases": [] }}

Poem:
{poem_text}
"""


def extract_json_string(llm_output: str) -> str:
    """
    Extract the JSON string from a response that may include optional
    markdown formatting like ```json or triple quotes.

    Args:
        llm_output (str): The raw string output from the LLM.

    Returns:
        str: A valid JSON string that can be passed to json.loads().

    Raises:
        ValueError: If no JSON block is found.
    """
    # Try fenced code block with language (```json ... ```)
    match = re.search(r"```json\s*(.*?)\s*```", llm_output, re.DOTALL)
    if match:
        return match.group(1)

    # Try generic fenced block (``` ... ```)
    match = re.search(r"```\s*(.*?)\s*```", llm_output, re.DOTALL)
    if match:
        return match.group(1)

    # Try triple-quoted block (""" ... """)
    match = re.search(r'"""\s*(\{.*?\})\s*"""', llm_output, re.DOTALL)
    if match:
        return match.group(1)

    # Try to match raw JSON directly (starts with { and ends with })
    match = re.search(r"(\{.*\})", llm_output, re.DOTALL)
    if match:
        return match.group(1)

    raise ValueError("No JSON content found in the input.")


import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_dataset(args, full_load=False) -> pd.DataFrame:
    # Load dataset
    df = (
        pd.read_csv(args.input, sep="\t")
        if args.input.endswith(".tsv")
        else pd.read_csv(args.input)
    )

    if full_load:
        return df

    if args.task == "keywords_extraction":
        # Ensure target columns exist
        for col in ["keywords", "key_phrases"]:
            if col not in df.columns:
                df[col] = None

        # Filter by verse count
        df_filtered = df[
            df["poem_verses"].between(args.min_verses, args.max_verses)
            & (df["keywords"].isnull() | df["key_phrases"].isnull())
        ]

    elif args.task == "corruption":
        templates = pd.read_csv(args.corruption_template, sep="\t")

        # Initialize assigned_template column if it doesn't exist
        if "assigned_template" not in df.columns:
            df["assigned_template"] = None

        # Count already assigned templates from input file
        input_template_counts = {}
        already_assigned_mask = df["assigned_template"].notnull()
        if already_assigned_mask.any():
            input_assigned = df[already_assigned_mask]
            for template_id in input_assigned["assigned_template"]:
                input_template_counts[template_id] = (
                    input_template_counts.get(template_id, 0) + 1
                )
            print(f"Found {len(input_assigned)} already assigned poems in input file.")
            print(f"Input template counts: {input_template_counts}")

        # Filter rows by verse count and exclude already assigned
        df_filtered = df[
            df["poem_verses"].between(args.min_verses, args.max_verses)
            & df["assigned_template"].isnull()
        ].copy()

        args.max_rows = min(args.max_rows, len(df_filtered)) if args.max_rows else None

        # Rows per template
        n_templates = len(templates)
        base_rows_per_template = (
            args.max_rows // n_templates
            if args.max_rows
            else len(df_filtered) // n_templates
        )

        print(
            f"Distributing {len(df_filtered)} remaining rows across {n_templates} templates, "
            f"base ~{base_rows_per_template} rows per template."
        )

        for t_idx, row in reversed(list(templates.iterrows())):
            placeholders = [col.strip() for col in str(row["Placeholder"]).split(",")]

            # Calculate remaining rows needed for this template
            already_assigned_for_template = input_template_counts.get(t_idx, 0)
            remaining_rows_needed = max(
                0, base_rows_per_template - already_assigned_for_template
            )

            print(
                f"Template {t_idx}: already assigned {already_assigned_for_template}, "
                f"need {remaining_rows_needed} more rows"
            )

            if remaining_rows_needed == 0:
                continue  # This template is already complete

            # Consider only rows not yet assigned
            eligible = df_filtered[df_filtered["assigned_template"].isnull()].copy()

            # Keep rows that satisfy *this template's* placeholders
            for col in placeholders:
                if col in eligible.columns:
                    eligible = eligible[eligible[col].notnull()]

            if len(eligible) == 0:
                print(f"No eligible rows for template {t_idx}.")
                continue  # no rows for this template

            # Pick rows
            if len(eligible) > remaining_rows_needed:
                assigned = eligible.sample(remaining_rows_needed, random_state=42)
            else:
                assigned = eligible

            # Assign
            df_filtered.loc[assigned.index, "assigned_template"] = t_idx

        # Keep only rows actually assigned
        df_filtered = df_filtered[df_filtered["assigned_template"].notnull()]

    # Global cap
    if args.max_rows:
        print(f"⚠️ Limiting to {args.max_rows} rows for processing.")
        df_filtered = df_filtered.head(args.max_rows)

    print(f"🚀 Starting processing of {len(df_filtered)} poems...")
    logger.info(f"Loaded {len(df)} entries from {args.input}")

    print(df_filtered["assigned_template"].value_counts())

    return df_filtered


import random
import logging

logger = logging.getLogger(__name__)


def prepare_prompt(df: pd.DataFrame, templates: pd.DataFrame, args) -> list[dict]:
    """
    Prepare prompts for each entry in the dataset.
    Handles both keywords_extraction and corruption tasks.
    """

    if args.task == "keywords_extraction":
        # create the prompt column
        df["prompt"] = df.apply(
            lambda row: KEYWORD_EXTRACTION_PROMPT_TEMPLATE.format(
                poem_text=row["poem_text_no_diacritics"]
            ),
            axis=1,
        )

    elif args.task == "corruption":
        # Precompute unique values for all target_* columns
        target_values = {}
        pool = load_dataset(args, full_load=True)
        for col in pool.columns:
            if col in ["meter", "rhyme", "poet_era"]:
                values = pool[col].dropna().unique().tolist()
                if len(values) > 1:  # only meaningful if >1 option
                    target_values[col.replace("poet_", "")] = values

        def pick_alternative(value, choices):
            """Pick a random alternative different from the current value."""
            alternatives = [v for v in choices if v != value]
            if alternatives:
                return random.choice(alternatives)
            return value  # fallback: keep the same if no alternative

        prompts = []
        for idx, row in df.iterrows():

            print(f"Preparing prompt {idx+1}/{len(df)}", end="\r")
            # Get assigned template row
            template_row = templates.iloc[int(row["assigned_template"])]

            # Copy template text
            prompt_template = template_row[
                "Prompt to alter original poetry (for Gemini 2.5)"
            ]

            # Replace placeholders
            placeholders = [
                p.strip() for p in str(template_row["Placeholder"]).split(",")
            ]
            values = {}

            # print(target_values)

            for col in placeholders:
                original_col_name = col.replace("target_", "")
                if col.startswith("target") and original_col_name in target_values:

                    values[col] = pick_alternative(
                        row[original_col_name],
                        target_values[original_col_name],
                    )
                else:

                    values[col] = row[col]

            # Format prompt
            prompt = prompt_template
            for k, v in values.items():
                prompt = prompt.replace("{" + k + "}", str(v))

            prompts.append(
                {
                    "poem_id": row["poem_id"],
                    "assigned_template": int(row["assigned_template"]),
                    "prompt": prompt,
                    "corruption_type": template_row["corruption_type"],
                    "Placeholders": template_row["Placeholder"],
                }
            )

        df = pd.DataFrame(prompts)

    logger.info(f"Prepared prompts for {len(df)} entries.")
    return df.to_dict(orient="records")


class DataEntry(BaseModel):
    keywords: list[str]
    key_phrases: list[str]


class CorruptionEntry(BaseModel):
    corrupted_poem: str


def postprocess_output(args, corruption_data=None) -> list[DataEntry]:
    """Post-process the output data into a list of DataEntry objects."""
    # load input dataset
    in_df = load_dataset(args, full_load=True)

    corruption_data = (
        pd.DataFrame(corruption_data) if corruption_data is not None else None
    )
    # intermediate_df = load_dataset(args, full_load=False)

    # convert the output to JSON format
    out_jsonl_filepath = Path(args.output).with_suffix(".jsonl")
    out_df = pd.read_json(out_jsonl_filepath, lines=True)

    if args.task == "corruption":
        ## merge the two dataframes using poem_id. don't lose any columns. even in the output dataframe
        merged_df = in_df.merge(
            out_df,
            on="poem_id",
            how="left",
            suffixes=("", "_new"),
        )
        # add the assigned_template and corruption_type columns from intermediate_df to merged_df
        merged_df = merged_df.merge(
            corruption_data[
                ["poem_id", "assigned_template", "corruption_type", "Placeholders"]
            ],
            on="poem_id",
            how="left",
        )

    elif args.task == "keywords_extraction":
        merged_df = in_df.merge(
            out_df[["poem_id", "keywords", "key_phrases"]],
            on="poem_id",
            how="left",
            suffixes=("", "_new"),
        )
        # If the merged columns have values, overwrite the originals
        merged_df["keywords"] = merged_df["keywords_new"].combine_first(
            merged_df["keywords"]
        )
        merged_df["key_phrases"] = merged_df["key_phrases_new"].combine_first(
            merged_df["key_phrases"]
        )
        # Drop the temporary columns
        merged_df.drop(columns=["keywords_new", "key_phrases_new"], inplace=True)

    # print(merged_df)
    # Save the merged DataFrame as a CSV file
    output_csv_filepath = Path(args.output).with_suffix(".csv")
    merged_df.to_csv(output_csv_filepath, index=False, encoding="utf-8")
    logger.info(f"Merged DataFrame saved to {output_csv_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate keywords and key phrases from Arabic poetry using Gemini."
    )
    parser.add_argument("--input", required=True, help="Path to input TSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--min_verses",
        type=int,
        default=2,
        help="Minimum number of verses to consider.",
    )
    parser.add_argument(
        "--max_verses",
        type=int,
        default=20,
        help="Maximum number of verses to consider.",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Maximum number of rows to process."
    )
    parser.add_argument(
        "--num_of_workers",
        type=int,
        default=None,
        help="Maximum number of rows to process.",
    )
    parser.add_argument("--task", type=str, default="", help="Task to perform.")
    parser.add_argument(
        "--corruption_template", type=str, default="", help="Path to the template file."
    )
    parser.add_argument(
        "--gemini_key_env_var",
        type=str,
        default="GEMINI_KEY_ANWAR",
        help="Environment variable name for Gemini API key.",
    )
    args = parser.parse_args()

    gemini_key = os.getenv(args.gemini_key_env_var)
    assert gemini_key is not None, "Gemini API key not found in environment variables."

    # load Gemini client
    gemini = load_gemini(gemini_key)
    # load the dataset
    df = load_dataset(args)

    # print(f"Total rows to process: {len(df)}")
    # print(df.head())
    # prepare the prompts

    if args.corruption_template:
        template = pd.read_csv(args.corruption_template, sep="\t")

    data = prepare_prompt(df, args=args, templates=template)
    out_jsonl_filepath = Path(args.output)
    # if out_jsonl_filepath.suffix != ".jsonl":
    #     out_jsonl_filepath = out_jsonl_filepath.with_suffix(".jsonl")
    # # generate content using Gemini
    # process_data_with_gemini(
    #     client=gemini,
    #     schema=DataEntry if args.task == "keywords_extraction" else CorruptionEntry,
    #     data=data,
    #     id_col="poem_id",
    #     input_col="prompt",
    #     out_jsonl_filepath=out_jsonl_filepath,
    #     max_workers=args.num_of_workers,  # NOTE: Due to limitation to my subscription, this is the maximum concurrent calls allowed by Gemini.
    # )
    # # check if postprocess_output() function is imported
    # # postprocess the output
    # postprocess_output(args, corruption_data=data)
    # logger.info("Done postprocessed the output data!")
