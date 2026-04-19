import argparse
import pandas as pd
import random
import openai
from tqdm import tqdm

# ----------------------
# Placeholder for evaluation prompt
# ----------------------
EVAL_PROMPT = """
You are an expert judge in Arabic poetry. Your task is to evaluate a given Arabic poem based on multiple dimensions and provide a score on a 1–5 Likert scale (1 = very poor, 5 = excellent) for each dimension. Be precise and provide a short reasoning for each score. The evaluation dimensions are:

Constraint Satisfaction: Does the poem adhere to the given instructions, constraints, or specifications (e.g., meter, rhyme, theme, length)?

Fluency: Is the poem grammatically correct, smooth, and readable in Modern Standard Arabic or the specified dialect?

Coherence: Do the ideas in the poem flow logically and maintain consistency throughout?

Rhetoric / Poeticness: Does the poem employ literary devices effectively (e.g., metaphor, imagery, alliteration) and convey aesthetic poetic qualities?

Originality: Is the poem creative, novel, and distinctive, or does it feel generic or derivative?

Instructions for the LLM:

Read the poem carefully.

Evaluate each dimension independently on a 1–5 Likert scale.

For each score, provide a brief explanation (1–2 sentences) justifying your rating.

Output the results in JSON format, like this:
{
  "constraint_satisfaction": {"score": 4, "reason": ""},
  "fluency": {"score": 5, "reason": ""},
  "coherence": {"score": 3, "reason": ""},
  "poeticness": {"score": 4, "reason": ""},
  "originality": {"score": 5, "reason": ""}
}

Poem:
{poem_text}
"""


# ----------------------
# Function to query ChatGPT
# ----------------------
def get_judge_scores(input_text: str, model_output: str) -> dict:
    prompt = EVAL_PROMPT.format(input_text=input_text, model_output=model_output)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # replace with the judge model you want
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    # Extract JSON response from the model
    try:
        scores = eval(response["choices"][0]["message"]["content"].strip())
    except Exception:
        scores = {"Fluency": None, "Relevance": None, "Creativity": None}

    return scores


# ----------------------
# Main evaluation script
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs using LLM-as-a-judge"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save evaluation results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--input_col", type=str, default="input", help="Column name for input text"
    )
    parser.add_argument(
        "--output_col", type=str, default="output", help="Column name for model output"
    )

    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.input)

    # Sample subset
    if args.num_samples < len(df):
        df = df.sample(args.num_samples, random_state=42)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row[args.input_col]
        model_output = row[args.output_col]

        scores = get_judge_scores(input_text, model_output)
        results.append({**row.to_dict(), **scores})

    results_df = pd.DataFrame(results)

    # Compute averages
    aspects = [
        c for c in results_df.columns if c in ["Fluency", "Relevance", "Creativity"]
    ]
    averages = results_df[aspects].mean().to_dict()
    averages["Overall_Average"] = results_df[aspects].mean(axis=1).mean()

    # Save results
    results_df.to_csv(args.output, index=False)

    print("Evaluation completed.")
    print("Aspect-wise averages:", averages)


if __name__ == "__main__":
    main()
