import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def run_inference(
    model_name,
    input_file,
    output_file,
    max_tokens=512,
    temperature=0.7,
    batch_size=16,
    apply_chat_template=False,
):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    # Load CSV with instructions
    df = pd.read_csv(input_file, sep="\t" if input_file.endswith(".tsv") else ",")
    if "input" not in df.columns:
        raise ValueError("CSV must contain a column named 'input'")

    # Prepare inputs
    inputs = df["input"].tolist()

    if apply_chat_template:
        print(f"Applying chat template for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        conversations = [[{"role": "user", "content": text}] for text in inputs]
        inputs = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations
        ]

    # Load LLM
    print(f"Running inference with {model_name}...")
    llm = LLM(model=model_name)

    outputs = llm.generate(
        prompts=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    # Combine outputs with original data
    results = []
    for idx, out in enumerate(outputs):
        generated_text = out.outputs[0].text.strip() if out.outputs else ""
        row = df.iloc[idx].to_dict()
        row["generated_text"] = generated_text
        results.append(row)

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a model using vLLM"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name to load")
    parser.add_argument(
        "--input", type=str, required=True, help="CSV file with 'input' column"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template for instruct models",
    )

    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        input_file=args.input,
        output_file=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        apply_chat_template=args.apply_chat_template,
    )


if __name__ == "__main__":
    main()
