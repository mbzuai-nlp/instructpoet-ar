import os
import yaml
import openai
from copy import deepcopy
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# Get the OpenAI API key from the environment
API_KEY = os.getenv("GPT_KEY_ANWAR")
client = OpenAI(api_key=API_KEY)

def generate_paraphrases(prompt, n=25, model="gpt-4o-2024-11-20"):
    # system_prompt = (
    #     "You are a helpful assistant. Your task is to generate multiple neutral and natural-sounding "
    #     "paraphrases of a user instruction for a language model that generates poetry. "
    #     "Do not add extra explanation. Just return a list of {} paraphrased instructions. Write each instruction in a new line. Only ouput the instructions".format(n)
    # )
    system_prompt = (
        "You are a helpful assistant. Your task is to generate multiple neutral and natural-sounding "
        "paraphrases of a user instruction for a language model that analyzes poetry. "
        "Do not add extra explanation. Just return a list of {} paraphrased instructions. Write each instruction in a new line. Only ouput the instructions".format(n)
    )
    user_prompt = f"Instruction: {prompt}\n\nGenerate {n} paraphrased versions."
    
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    

    # print(response.output_text)
    # Try parsing the list of paraphrases
    text = response.output_text
    lines = [line.strip("-• ").strip() for line in text.strip().split('\n') if line.strip()]
    return lines[:n]

def process_yaml(input_path, output_path, n_paraphrases=25):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    updated_data = deepcopy(data)

    for lang in updated_data:
        for i, item in enumerate(tqdm(updated_data[lang], desc=f"Processing {lang}")):
            original_text = item['text']
            paraphrases = generate_paraphrases(original_text, n=n_paraphrases)
            item['text'] = paraphrases

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(updated_data, f, allow_unicode=True, sort_keys=False)

    print(f"Saved updated YAML to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/path/to/templates/poetry_analysis.yaml", help="Path to input YAML template")
    parser.add_argument("--output", type=str, default="/path/to/templates/poetry_analysis_all.yaml", help="Path to save output YAML")
    parser.add_argument("--num", type=int, default=25, help="Number of paraphrases to generate per instruction")
    args = parser.parse_args()

    process_yaml(args.input, args.output, n_paraphrases=args.num)
