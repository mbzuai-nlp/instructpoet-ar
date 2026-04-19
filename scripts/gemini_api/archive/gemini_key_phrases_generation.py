import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
import re


# Load environment variables
load_dotenv()
gemini_key = os.getenv("KEY")
assert gemini_key is not None, "Gemini API key not found in environment variables."

# Initialize Gemini client
client = genai.Client(api_key=gemini_key)

PROMPT_TEMPLATE = """
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
    match = re.search(r'(\{.*\})', llm_output, re.DOTALL)
    if match:
        return match.group(1)

    raise ValueError("No JSON content found in the input.")

def generate_keywords_and_phrases(poem_text):
    prompt = PROMPT_TEMPLATE.format(poem_text=poem_text.strip())
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = response.text.strip()
        data = extract_json_string(raw)
        data = json.loads(data)


        return data.get("keywords", []), data.get("key_phrases", [])
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], []

def main(args):
    # Load dataset
    df = pd.read_csv(args.input, sep="\t")

    # Ensure target columns exist
    for col in ["keywords", "key_phrases"]:
        if col not in df.columns:
            df[col] = None

    # Filter by verse count
    df_filtered = df[
        df["poem_verses"].between(args.min_verses, args.max_verses) &
        (df["keywords"].isnull() | df["key_phrases"].isnull())
    ]

    if args.max_rows:
        df_filtered = df_filtered.head(args.max_rows)

    print(f"🚀 Starting processing of {len(df_filtered)} poems...")

    success_count = 0
    fail_count = 0

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing"):
        poem_id = row.get("poem_id")
        poem_text = row.get("poem_text_no_diacritics", "")

        if not isinstance(poem_text, str) or poem_text.strip() == "":
            print(f"⚠️ Skipping empty poem at index {idx}")
            continue

        print(f"\n📝 Processing poem ID {poem_id}...")
        keywords, key_phrases = generate_keywords_and_phrases(poem_text)

        if keywords or key_phrases:
            df.loc[df["poem_id"] == poem_id, "keywords"] = json.dumps(keywords, ensure_ascii=False)
            df.loc[df["poem_id"] == poem_id, "key_phrases"] = json.dumps(key_phrases, ensure_ascii=False)
            print(f"✅ Done: {len(keywords)} keywords, {len(key_phrases)} key phrases.")
            success_count += 1
        else:
            print(f"❌ Failed to extract data for poem ID {poem_id}.")
            fail_count += 1



    # Save the updated DataFrame
    df.to_csv(args.output, index=False)
    print(f"\n📦 Saved results to: {args.output}")
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keywords and key phrases from Arabic poetry using Gemini.")
    parser.add_argument("--input", required=True, help="Path to input TSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--min_verses", type=int, default=2, help="Minimum number of verses to consider.")
    parser.add_argument("--max_verses", type=int, default=20, help="Maximum number of verses to consider.")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process.")
    args = parser.parse_args()
    main(args)
