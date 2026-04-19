#!/usr/bin/env python
# coding=utf-8
"""
Utility script to load and use best decoding parameters from grid search results.
"""

import os
import json
import argparse
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_DIR = os.path.join(SCRIPT_DIR, "best_decoding_parameters")


def load_best_config(model_name: str, config_dir: str = None) -> dict:
    """
    Load the best configuration for a model.

    Args:
        model_name: Name of the model
        config_dir: Directory containing the best config files

    Returns:
        Dictionary with best configurations per task
    """
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR

    config_file = Path(config_dir) / f"{model_name}_best_config.json"

    if not config_file.exists():
        raise FileNotFoundError(f"Best config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def get_task_params(model_name: str, task: str, config_dir: str = None) -> dict:
    """
    Get the best parameters for a specific task.

    Args:
        model_name: Name of the model
        task: Task name (generation, continuation, corruption)
        config_dir: Directory containing the best config files

    Returns:
        Dictionary with temperature and repetition_penalty
    """
    config = load_best_config(model_name, config_dir)

    if task not in config:
        raise ValueError(
            f"Task '{task}' not found in config. Available tasks: {list(config.keys())}"
        )

    task_config = config[task]

    return {
        "temperature": task_config["temperature"],
        "repetition_penalty": task_config["repetition_penalty"],
        "metrics": task_config["metrics"],
    }


def print_best_configs(model_name: str, config_dir: str = None):
    """
    Print best configurations for all tasks.

    Args:
        model_name: Name of the model
        config_dir: Directory containing the best config files
    """
    config = load_best_config(model_name, config_dir)

    print("=" * 80)
    print(f"Best Decoding Parameters for {model_name}")
    print("=" * 80)

    for task, task_config in config.items():
        print(f"\n{task.upper()}:")
        print(f"  Configuration: {task_config['config_name']}")
        print(f"  Temperature: {task_config['temperature']}")
        print(f"  Repetition Penalty: {task_config['repetition_penalty']}")
        print(f"  Composite Score: {task_config['composite_score']:.4f}")
        print(f"\n  Key Metrics:")
        print(
            f"    Repetition Rate: {task_config['metrics']['avg_word_repetition_rate']:.2f}%"
        )
        print(
            f"    Unique Word Ratio: {task_config['metrics']['avg_unique_word_ratio']:.4f}"
        )
        print(
            f"    Avg Word Frequency: {task_config['metrics']['avg_word_frequency']:.2f}"
        )
        print(
            f"    Avg Verses: {task_config['metrics']['avg_verses']:.2f} ± {task_config['metrics']['std_verses']:.2f}"
        )


def generate_inference_command(
    model_name: str,
    model_path: str,
    task: str,
    config_dir: str = None,
    data_path: str = None,
    output_path: str = None,
) -> str:
    """
    Generate an inference command using the best parameters.

    Args:
        model_name: Name of the model
        model_path: Path to the model
        task: Task name
        config_dir: Directory containing the best config files
        data_path: Path to data (optional)
        output_path: Path to outputs (optional)

    Returns:
        String with the inference command
    """
    params = get_task_params(model_name, task, config_dir)

    cmd = f"bash inference.sh \\\n"
    cmd += f"    --model_path {model_path} \\\n"
    cmd += f"    --tasks {task} \\\n"
    cmd += f"    --temperature {params['temperature']} \\\n"
    cmd += f"    --repetition_penalty {params['repetition_penalty']}"

    if data_path:
        cmd += f" \\\n    --data_path {data_path}"

    if output_path:
        cmd += f" \\\n    --output_path {output_path}"

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Load and display best decoding parameters from grid search"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model (e.g., checkpoint-36000)",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Directory containing the best config files (default: ./best_decoding_parameters in script dir)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["generation", "continuation", "corruption"],
        help="Get parameters for specific task only",
    )
    parser.add_argument(
        "--generate_command",
        action="store_true",
        help="Generate inference command with best parameters",
    )
    parser.add_argument(
        "--model_path", type=str, help="Model path for generating inference command"
    )

    args = parser.parse_args()

    if args.generate_command:
        if not args.model_path:
            print("Error: --model_path required when using --generate_command")
            return

        if not args.task:
            print("Error: --task required when using --generate_command")
            return

        cmd = generate_inference_command(
            model_name=args.model_name,
            model_path=args.model_path,
            task=args.task,
            config_dir=args.config_dir,
        )

        print("\nGenerated Inference Command:")
        print("=" * 80)
        print(cmd)
        print("=" * 80)

    elif args.task:
        params = get_task_params(args.model_name, args.task, args.config_dir)
        print(f"\nBest parameters for {args.task}:")
        print(f"  Temperature: {params['temperature']}")
        print(f"  Repetition Penalty: {params['repetition_penalty']}")
        print(f"\nMetrics:")
        for key, value in params["metrics"].items():
            print(f"  {key}: {value:.4f}")

    else:
        print_best_configs(args.model_name, args.config_dir)


if __name__ == "__main__":
    main()
