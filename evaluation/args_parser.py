import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Arabic Poetry Inference Arguments")
    add_args(parser)
    args, unknown = parser.parse_known_args()
    return args


def add_args(parser: argparse.ArgumentParser):
    # Model arguments
    parser.add_argument(
        "--full_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--finetuning_type",
        type=str,
        default="baseline",
        choices=["baseline", "adapters", "full"],
        help="Type of model: baseline (no finetuning), adapters (LoRA), or full (full finetuning)",
    )
    parser.add_argument(
        "--checkpoint_parent_path",
        type=str,
        default=None,
        help="Parent path for the checkpoints",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="0",
        help="Checkpoint step to load (0 for final checkpoint)",
    )

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="analysis",
        choices=["analysis", "continuation", "generation", "corruption", "all"],
        help="Task to evaluate",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="random",
        choices=["random", "curriculum"],
        help="Training mode used for the model",
    )

    # Data arguments
    parser.add_argument(
        "--data_base_path",
        type=str,
        default="/path/to/dialectical_IFT_DATA/tsv",
        help="Base path for the poetry TSV data files",
    )
    parser.add_argument(
        "--data_split", type=str, default="test", help="Data split to evaluate on"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="instruction",
        choices=["instruction", "chat"],
        help="Prompt format type",
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 for greedy)",
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p for nucleus sampling"
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling")

    # vLLM arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=16,
        help="Maximum number of sequences to process in parallel",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=4096, help="Maximum model context length"
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_outputs",
        help="Path to save evaluation outputs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for non-vLLM inference"
    )
