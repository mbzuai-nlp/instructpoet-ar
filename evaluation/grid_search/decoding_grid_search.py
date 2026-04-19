#!/usr/bin/env python
# coding=utf-8
"""
Decoding Hyperparameter Grid Search for Arabic Poetry Generation
This script performs a grid search over decoding hyperparameters to find the best
configuration that minimizes repetition while maintaining verse quality.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import pandas as pd
import re
from tqdm import tqdm
import numpy as np

# Add parent directory for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def extract_model_name(model_path: str) -> str:
    """
    Extract a meaningful model name from the checkpoint path.

    Examples:
        /path/to/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000 -> allam-curriculum
        /path/to/Qwen3-8B/random/4tasks/checkpoint-35000 -> qwen3-random
        /path/to/Qwen2.5-7B-Instruct/curriculum/checkpoint-36000 -> qwen2.5-curriculum

    Args:
        model_path: Full path to the model checkpoint

    Returns:
        Formatted model name with training mode
    """
    path_parts = Path(model_path).parts

    # Find the model name (e.g., ALLaM-7B-Instruct-preview, Qwen3-8B)
    model_name = None
    training_mode = None

    for i, part in enumerate(path_parts):
        # Look for common model name patterns
        if any(
            keyword in part for keyword in ["ALLaM", "Qwen", "Llama", "jais", "JAIS"]
        ):
            model_name = part
            # Look for training mode in subsequent parts
            if i + 1 < len(path_parts):
                next_part = path_parts[i + 1]
                if next_part in ["curriculum", "random", "mixed"]:
                    training_mode = next_part
            break

    if not model_name:
        # Fallback to checkpoint name
        model_name = Path(model_path).name

    # Clean up model name
    model_name = model_name.lower()
    model_name = model_name.replace("-instruct-preview", "").replace("-instruct", "")
    model_name = model_name.replace("-7b", "").replace("-8b", "")

    # Combine model name with training mode
    if training_mode:
        return f"{model_name}-{training_mode}"
    else:
        return model_name


def calculate_repetition_score(text: str) -> Dict[str, float]:
    """
    Calculate repetition metrics for generated text.

    Args:
        text: Generated poem text

    Returns:
        Dictionary with repetition metrics:
        - word_repetition_rate: Percentage of repeated words
        - unique_word_ratio: Ratio of unique words to total words
        - avg_word_frequency: Average frequency of words
        - max_word_frequency: Maximum frequency of any word
    """
    if not text or not text.strip():
        return {
            "word_repetition_rate": 0.0,
            "unique_word_ratio": 0.0,
            "avg_word_frequency": 0.0,
            "max_word_frequency": 0,
            "total_words": 0,
            "unique_words": 0,
        }

    # Tokenize by Arabic words (remove punctuation)
    # Keep only Arabic characters and spaces
    cleaned_text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    words = cleaned_text.split()

    if not words:
        return {
            "word_repetition_rate": 0.0,
            "unique_word_ratio": 0.0,
            "avg_word_frequency": 0.0,
            "max_word_frequency": 0,
            "total_words": 0,
            "unique_words": 0,
        }

    # Count word frequencies
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)

    # Calculate metrics
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    word_repetition_rate = (
        (repeated_words / unique_words * 100) if unique_words > 0 else 0.0
    )
    unique_word_ratio = (unique_words / total_words) if total_words > 0 else 0.0
    avg_word_frequency = total_words / unique_words if unique_words > 0 else 0.0
    max_word_frequency = max(word_counts.values()) if word_counts else 0

    return {
        "word_repetition_rate": word_repetition_rate,
        "unique_word_ratio": unique_word_ratio,
        "avg_word_frequency": avg_word_frequency,
        "max_word_frequency": max_word_frequency,
        "total_words": total_words,
        "unique_words": unique_words,
    }


def count_verses(text: str) -> int:
    """
    Count the number of verses (lines) in a poem.

    Args:
        text: Poem text

    Returns:
        Number of verses
    """
    if not text or not text.strip():
        return 0

    # Split by newlines and count non-empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return len(lines)


def sample_data(data_path: str, task: str, n_samples: int = 100) -> pd.DataFrame:
    """
    Sample random examples from a task dataset.

    Args:
        data_path: Base path to data directory
        task: Task name (generation, continuation, corruption)
        n_samples: Number of samples to draw

    Returns:
        DataFrame with sampled data
    """
    # Task folder mapping
    task_folder_map = {
        "generation": "generation",
        "continuation": "continuation",
        "corruption": "corruption",
        "restoration": "corruption",
    }

    folder_name = task_folder_map.get(task, task)
    file_path = os.path.join(data_path, folder_name, "test", f"{folder_name}_ift.tsv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load data
    df = pd.read_csv(file_path, sep="\t", on_bad_lines="skip")

    # Sample n_samples randomly
    if len(df) > n_samples:
        df_sampled = df.sample(n=n_samples, random_state=42)
    else:
        df_sampled = df

    print(f"Sampled {len(df_sampled)} examples from {task} (total: {len(df)})")
    return df_sampled


def create_sampled_data_file(
    data_path: str, tasks: List[str], n_samples: int, output_dir: str
):
    """
    Create sampled data files for each task.

    Args:
        data_path: Base path to data directory
        tasks: List of tasks to sample
        n_samples: Number of samples per task
        output_dir: Directory to save sampled data
    """
    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        df_sampled = sample_data(data_path, task, n_samples)

        # Save to temporary directory
        task_output_dir = os.path.join(output_dir, task, "test")
        os.makedirs(task_output_dir, exist_ok=True)

        output_file = os.path.join(task_output_dir, f"{task}_ift.tsv")
        df_sampled.to_csv(output_file, sep="\t", index=False)
        print(f"Saved sampled data to {output_file}")


def run_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    task: str,
    temperature: float,
    repetition_penalty: float,
    max_samples: int = None,
) -> str:
    """
    Run inference using the inference.sh script.

    Args:
        model_path: Path to the model
        data_path: Path to sampled data
        output_path: Path to save outputs
        task: Task name
        temperature: Temperature for sampling
        repetition_penalty: Repetition penalty
        max_samples: Maximum number of samples

    Returns:
        Path to output directory
    """
    # Get the parent directory (evaluation) where inference.sh is located
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(script_dir, "inference.sh")

    # Build command
    cmd = [
        "bash",
        script_path,
        "--model_path",
        model_path,
        "--data_path",
        data_path,
        "--output_path",
        output_path,
        "--tasks",
        task,
        "--temperature",
        str(temperature),
        "--repetition_penalty",
        str(repetition_penalty),
        "--prompt_type",
        "chat",
        "--max_new_tokens",
        "1024",
    ]

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    print(f"\n{'='*80}")
    print(
        f"Running inference for {task} with temp={temperature}, rep_penalty={repetition_penalty}"
    )
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")

    # Run inference
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running inference: {result.stderr}")
        raise RuntimeError(f"Inference failed with return code {result.returncode}")

    return output_path


def check_outputs_exist(output_dir: str, task: str) -> bool:
    """
    Check if inference outputs already exist.

    Args:
        output_dir: Directory that would contain inference outputs
        task: Task name

    Returns:
        True if TSV file with predictions exists, False otherwise
    """
    # Search for TSV file with predictions in the output directory
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_with_predictions.tsv") and task in root:
                output_file = os.path.join(root, file)
                # Check if file is not empty
                if os.path.getsize(output_file) > 0:
                    print(f"Found existing outputs: {output_file}")
                    return True
    return False


def analyze_outputs(output_dir: str, task: str) -> Dict[str, Any]:
    """
    Analyze inference outputs and calculate metrics.

    Args:
        output_dir: Directory containing inference outputs
        task: Task name

    Returns:
        Dictionary with aggregated metrics
    """
    # Find the output TSV file
    # The structure is: output_dir/<model_name>/<training_mode>/<checkpoint>/<task>/<task>_ift_with_predictions.tsv
    # We need to search for the TSV files with predictions

    output_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_with_predictions.tsv") and task in root:
                output_files.append(os.path.join(root, file))

    if not output_files:
        raise FileNotFoundError(
            f"No *_with_predictions.tsv found in {output_dir} for task {task}"
        )

    # Use the most recent one
    output_file = sorted(output_files)[-1]
    print(f"Analyzing outputs from: {output_file}")

    # Load outputs from TSV
    df = pd.read_csv(output_file, sep="\t", on_bad_lines="skip")

    if "model_generation" not in df.columns:
        raise ValueError(f"Column 'model_generation' not found in {output_file}")

    # Calculate metrics for each output
    repetition_scores = []
    verse_counts = []

    for _, row in df.iterrows():
        generated_text = row.get("model_generation", "")

        # Calculate repetition
        rep_score = calculate_repetition_score(generated_text)
        repetition_scores.append(rep_score)

        # Count verses
        n_verses = count_verses(generated_text)
        verse_counts.append(n_verses)

    # Aggregate metrics
    metrics = {
        "n_samples": len(df),
        "avg_word_repetition_rate": np.mean(
            [s["word_repetition_rate"] for s in repetition_scores]
        ),
        "avg_unique_word_ratio": np.mean(
            [s["unique_word_ratio"] for s in repetition_scores]
        ),
        "avg_word_frequency": np.mean(
            [s["avg_word_frequency"] for s in repetition_scores]
        ),
        "max_word_frequency": np.mean(
            [s["max_word_frequency"] for s in repetition_scores]
        ),
        "avg_total_words": np.mean([s["total_words"] for s in repetition_scores]),
        "avg_unique_words": np.mean([s["unique_words"] for s in repetition_scores]),
        "avg_verses": np.mean(verse_counts),
        "std_verses": np.std(verse_counts),
        "min_verses": np.min(verse_counts),
        "max_verses": np.max(verse_counts),
    }

    return metrics


def run_grid_search(
    model_path: str,
    model_name: str,
    data_base_path: str,
    output_base_path: str,
    tasks: List[str] = None,
    repetition_penalties: List[float] = None,
    temperatures: List[float] = None,
    n_samples: int = 100,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run grid search over decoding hyperparameters.

    Args:
        model_path: Path to the model
        model_name: Name of the model for organizing outputs
        data_base_path: Base path to data directory
        output_base_path: Base path for outputs
        tasks: List of tasks to evaluate
        repetition_penalties: List of repetition penalties to try
        temperatures: List of temperatures to try
        n_samples: Number of samples per task
        force: Force re-run inference even if outputs exist

    Returns:
        Dictionary with all results
    """
    # Default parameters
    if tasks is None:
        tasks = ["generation", "continuation", "corruption"]

    if repetition_penalties is None:
        repetition_penalties = [1.0, 1.10, 1.15]

    if temperatures is None:
        temperatures = [0.0, 0.7]

    # Create model-specific output directories
    model_output_base = os.path.join(output_base_path, model_name)

    # Create sampled data directory (shared across all models)
    sampled_data_dir = os.path.join(output_base_path, "sampled_data")
    print(f"\nCreating sampled data with {n_samples} examples per task...")
    create_sampled_data_file(data_base_path, tasks, n_samples, sampled_data_dir)

    # Grid search
    all_results = {}

    for task in tasks:
        task_results = {}

        for temp in temperatures:
            for rep_penalty in repetition_penalties:
                config_name = f"temp_{temp}_rp_{rep_penalty}"
                print(f"\n{'#'*80}")
                print(f"Task: {task} | Configuration: {config_name}")
                print(f"{'#'*80}")

                # Create output directory for this configuration (organized by model)
                config_output_dir = os.path.join(
                    model_output_base, "grid_search_outputs", task, config_name
                )
                os.makedirs(config_output_dir, exist_ok=True)

                try:
                    # Check if outputs already exist
                    if not force and check_outputs_exist(config_output_dir, task):
                        print(
                            f"⏭️  Skipping inference - outputs already exist (use --force to re-run)"
                        )
                    else:
                        if force and check_outputs_exist(config_output_dir, task):
                            print(f"🔄 Forcing re-run of inference...")
                        else:
                            print(f"🚀 Running inference...")
                        run_inference(
                            model_path=model_path,
                            data_path=sampled_data_dir,
                            output_path=config_output_dir,
                            task=task,
                            temperature=temp,
                            repetition_penalty=rep_penalty,
                            max_samples=n_samples,
                        )

                    # Analyze outputs
                    print(f"📊 Analyzing outputs...")
                    metrics = analyze_outputs(config_output_dir, task)

                    # Store results
                    task_results[config_name] = {
                        "temperature": temp,
                        "repetition_penalty": rep_penalty,
                        "metrics": metrics,
                    }

                    print(f"\nResults for {config_name}:")
                    print(
                        f"  Repetition Rate: {metrics['avg_word_repetition_rate']:.2f}%"
                    )
                    print(
                        f"  Unique Word Ratio: {metrics['avg_unique_word_ratio']:.4f}"
                    )
                    print(f"  Avg Word Frequency: {metrics['avg_word_frequency']:.2f}")
                    print(
                        f"  Avg Verses: {metrics['avg_verses']:.2f} ± {metrics['std_verses']:.2f}"
                    )
                    print(f"  Avg Total Words: {metrics['avg_total_words']:.2f}")

                except Exception as e:
                    print(f"Error processing {config_name}: {e}")
                    task_results[config_name] = {
                        "temperature": temp,
                        "repetition_penalty": rep_penalty,
                        "error": str(e),
                    }

        all_results[task] = task_results

    return all_results


def find_best_hyperparameters(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the best hyperparameters based on repetition score and verse count.

    The best configuration should:
    1. Minimize repetition rate
    2. Have reasonable verse count (not too short, not too long)
    3. Maximize unique word ratio

    Args:
        results: Dictionary with all grid search results

    Returns:
        Dictionary with best hyperparameters per task
    """
    best_configs = {}

    for task, task_results in results.items():
        if not task_results:
            continue

        # Calculate composite score for each configuration
        # Lower is better for repetition, we want reasonable verse count
        scores = {}

        for config_name, config_data in task_results.items():
            if "error" in config_data:
                continue

            metrics = config_data["metrics"]

            # Composite score: prioritize low repetition and good unique word ratio
            # Normalize metrics to [0, 1] range
            repetition_penalty_factor = metrics["avg_word_repetition_rate"] / 100.0
            unique_word_bonus = 1.0 - metrics["avg_unique_word_ratio"]
            avg_freq_penalty = (
                metrics["avg_word_frequency"] - 1.0
            ) / 10.0  # Penalize high frequencies

            # Lower score is better
            composite_score = (
                0.5 * repetition_penalty_factor  # 50% weight on repetition rate
                + 0.3 * unique_word_bonus  # 30% weight on unique words
                + 0.2 * avg_freq_penalty  # 20% weight on word frequency
            )

            scores[config_name] = {
                "composite_score": composite_score,
                "temperature": config_data["temperature"],
                "repetition_penalty": config_data["repetition_penalty"],
                "metrics": metrics,
            }

        # Find best configuration
        if scores:
            best_config_name = min(
                scores.keys(), key=lambda k: scores[k]["composite_score"]
            )
            best_configs[task] = {
                "config_name": best_config_name,
                **scores[best_config_name],
            }

    return best_configs


def _make_serializable(obj):
    """
    Recursively convert numpy / pandas types to native Python types so JSON
    dumping doesn't fail (e.g., numpy.int64).
    """
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if _pd is not None and isinstance(obj, _pd.Series):
        return _make_serializable(obj.tolist())
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    if _np is not None and isinstance(obj, _np.ndarray):
        return _make_serializable(obj.tolist())
    return obj


def save_results(
    results: Dict[str, Any],
    best_configs: Dict[str, Any],
    output_dir: str,
    model_name: str,
):
    """
    Save grid search results and best configurations.

    Args:
        results: All grid search results
        best_configs: Best configurations per task
        output_dir: Directory to save results
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save all results (convert numpy/pandas types to native python types first)
    results_file = os.path.join(output_dir, f"{model_name}_grid_search_results.json")
    serializable_results = _make_serializable(results)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all results to: {results_file}")

    # Save best configurations
    best_config_file = os.path.join(output_dir, f"{model_name}_best_config.json")
    serializable_best = _make_serializable(best_configs)
    with open(best_config_file, "w", encoding="utf-8") as f:
        json.dump(serializable_best, f, indent=2, ensure_ascii=False)
    print(f"Saved best configurations to: {best_config_file}")

    # Create summary report
    summary_file = os.path.join(output_dir, f"{model_name}_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Grid Search Summary for {model_name}\n")
        f.write("=" * 80 + "\n\n")

        for task, config in best_configs.items():
            f.write(f"\nTask: {task}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Configuration: {config['config_name']}\n")
            f.write(f"  Temperature: {config['temperature']}\n")
            f.write(f"  Repetition Penalty: {config['repetition_penalty']}\n")
            f.write(f"  Composite Score: {config['composite_score']:.4f}\n")
            f.write(f"\nMetrics:\n")
            for metric, value in config["metrics"].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

    print(f"Saved summary report to: {summary_file}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("GRID SEARCH SUMMARY")
    print("=" * 80)

    for task, config in best_configs.items():
        print(f"\n{task.upper()}:")
        print(f"  Best Config: {config['config_name']}")
        print(f"    Temperature: {config['temperature']}")
        print(f"    Repetition Penalty: {config['repetition_penalty']}")
        print(
            f"    Repetition Rate: {config['metrics']['avg_word_repetition_rate']:.2f}%"
        )
        print(
            f"    Unique Word Ratio: {config['metrics']['avg_unique_word_ratio']:.4f}"
        )
        print(f"    Avg Verses: {config['metrics']['avg_verses']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for decoding hyperparameters in Arabic poetry generation"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (placeholder supported)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/dialectical_IFT_DATA/tsv",
        help="Base path to data directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "grid_search_results"),
        help="Base path for outputs",
    )
    parser.add_argument(
        "--best_config_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "best_decoding_parameters"),
        help="Directory to save best configurations",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["generation", "continuation", "corruption"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--repetition_penalties",
        type=float,
        nargs="+",
        default=[1.0, 1.10, 1.15],
        help="Repetition penalties to try",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.0, 0.7],
        help="Temperatures to try",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples per task"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run inference even if outputs already exist",
    )

    args = parser.parse_args()

    # Extract model name from path
    model_name = extract_model_name(args.model_path)
    print(f"\n{'='*80}")
    print(f"Starting Grid Search for Model: {model_name}")
    print(f"Full model path: {args.model_path}")
    print(f"{'='*80}")
    print(f"Tasks: {args.tasks}")
    print(f"Repetition Penalties: {args.repetition_penalties}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Samples per task: {args.n_samples}")
    print(f"Force re-run: {args.force}")
    print(f"{'='*80}\n")

    # Run grid search
    results = run_grid_search(
        model_path=args.model_path,
        model_name=model_name,
        data_base_path=args.data_path,
        output_base_path=args.output_path,
        tasks=args.tasks,
        repetition_penalties=args.repetition_penalties,
        temperatures=args.temperatures,
        n_samples=args.n_samples,
        force=args.force,
    )

    # Find best hyperparameters
    best_configs = find_best_hyperparameters(results)

    # Save results
    save_results(
        results=results,
        best_configs=best_configs,
        output_dir=args.best_config_dir,
        model_name=model_name,
    )

    print("\n" + "=" * 80)
    print("Grid search completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
