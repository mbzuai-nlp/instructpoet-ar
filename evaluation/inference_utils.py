# coding=utf-8
# Inference utilities for Arabic Poetry evaluation

import re
import json
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean generated text by removing extra whitespace and normalizing."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_poetry_output(generated_text: str, task: str) -> Dict[str, Any]:
    """
    Extract the relevant output from generated text based on the task.

    Args:
        generated_text: Raw generated text from the model
        task: The poetry task (analysis, continuation, generation, corruption)

    Returns:
        Dictionary with extracted fields
    """
    result = {
        "raw_output": generated_text,
        "cleaned_output": clean_text(generated_text),
        "task": task,
    }

    if task == "analysis":
        # Analysis might contain keywords, themes, etc.
        result["analysis_text"] = result["cleaned_output"]

    elif task == "continuation":
        # Continuation should be poetry verses
        result["continuation_text"] = result["cleaned_output"]

    elif task == "generation":
        # Generation produces new poetry
        result["generated_poem"] = result["cleaned_output"]

    elif task in ["corruption", "restoration"]:
        # Restoration fixes corrupted poetry
        result["restored_text"] = result["cleaned_output"]

    return result


def format_prompt_for_inference(
    input_text: str, prompt_type: str = "instruction", task: str = "analysis"
) -> str:
    """
    Format input text into the appropriate prompt format for inference.

    Args:
        input_text: The input text from the dataset
        prompt_type: 'instruction' or 'chat'
        task: The poetry task

    Returns:
        Formatted prompt string
    """
    if prompt_type == "instruction":
        return f"###Input:\n{input_text}\n\n###Output:\n"
    elif prompt_type == "chat":
        # For chat, return the message structure
        return input_text
    else:
        return input_text


def format_chat_messages(input_text: str) -> List[Dict[str, str]]:
    """
    Format input text into chat message format.

    Args:
        input_text: The input text from the dataset

    Returns:
        List of message dictionaries
    """
    return [{"role": "user", "content": input_text}]


def save_outputs(outputs: List[Dict[str, Any]], output_path: str):
    """Save inference outputs to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(outputs)} outputs to {output_path}")


def load_outputs(output_path: str) -> List[Dict[str, Any]]:
    """Load inference outputs from a JSONL file."""
    outputs = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            outputs.append(json.loads(line))
    return outputs


def compute_basic_metrics(
    predictions: List[str], references: List[str], task: str
) -> Dict[str, float]:
    """
    Compute basic evaluation metrics.

    Args:
        predictions: List of predicted outputs
        references: List of reference outputs
        task: The poetry task

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic statistics
    metrics["num_samples"] = len(predictions)
    metrics["avg_pred_length"] = (
        sum(len(p) for p in predictions) / len(predictions) if predictions else 0
    )
    metrics["avg_ref_length"] = (
        sum(len(r) for r in references) / len(references) if references else 0
    )

    # Empty output rate
    empty_count = sum(1 for p in predictions if not p.strip())
    metrics["empty_rate"] = empty_count / len(predictions) if predictions else 0

    # Exact match (for analysis tasks)
    if task == "analysis":
        exact_matches = sum(
            1 for p, r in zip(predictions, references) if p.strip() == r.strip()
        )
        metrics["exact_match"] = exact_matches / len(predictions) if predictions else 0

    return metrics


DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"""
