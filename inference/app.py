#!/usr/bin/env python
"""Lightweight Gradio playground for the Arabic poetry adapters using vLLM."""

import os
import sys
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
from dotenv import load_dotenv
from safetensors.torch import safe_open, save_file
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Make evaluation utilities importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "evaluation"))
from inference_utils import (  # type: ignore
    DEFAULT_CHAT_TEMPLATE,
    format_prompt_for_inference,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

# Adapter and base model configuration
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "Qwen3-8B adapters (step-36000)": {
        "base_model": os.getenv("QWEN3_BASE_MODEL", "Qwen/Qwen3-8B"),
        "adapter_path": os.getenv(
            "QWEN3_ADAPTER_PATH",
            "/path/to/checkpoints/Qwen3-8B/curriculum/4tasks/checkpoint-36000",
        ),
        "adapter_name": "qwen3_poetry_lora",
        "note": "Curriculum, 4 tasks, checkpoint-36000",
    },
    "ALLaM-7B adapters (step-32000)": {
        "base_model": os.getenv(
            "ALLAM_BASE_MODEL", "humain-ai/ALLaM-7B-Instruct-preview"
        ),
        "adapter_path": os.getenv(
            "ALLAM_ADAPTER_PATH",
            "/path/to/checkpoints/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000",
        ),
        "adapter_name": "allam_poetry_lora",
        "note": "Curriculum, 4 tasks, checkpoint-32000",
    },
}

TASK_HINTS = {
    "analysis": "Analyze the verse: meter, rhyme, style, and themes.",
    "continuation": "Continue the poem while keeping meter and rhyme.",
    "generation": "Generate new Arabic poetry based on the cues.",
    "corruption": "Restore or correct the corrupted poem.",
}

DEFAULT_SYSTEM = (
    "You are an Arabic poetry expert. Be concise and keep the output only in Arabic."
)

DEFAULT_TP = int(os.getenv("VLLM_TENSOR_PARALLEL", "1"))
DEFAULT_GPU_UTIL = float(os.getenv("VLLM_GPU_UTIL", "0.9"))
DEFAULT_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
DEFAULT_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "16"))


def _materialize_vllm_adapter(adapter_dir: str) -> str:
    """
    Create a vLLM-friendly LoRA directory without base weights (lm_head, embeddings).
    vLLM only accepts LoRA tensors; PEFT saves extra full-weight tensors we need to drop.
    Returns the directory path containing adapter_model.safetensors + adapter_config.json.
    """
    base = Path(adapter_dir)
    src = base / "adapter_model.safetensors"
    ready_dir = base / "vllm_ready"
    dst = ready_dir / "adapter_model.safetensors"

    if not ready_dir.exists():
        ready_dir.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tensors = {}
        with safe_open(src, framework="pt") as f:
            for key in f.keys():
                if "lora" not in key:
                    continue  # drop unsupported full-weight tensors
                tensors[key] = f.get_tensor(key)
        save_file(tensors, dst)

    # Copy adapter_config.json alongside the filtered weights
    cfg_src = base / "adapter_config.json"
    cfg_dst = ready_dir / "adapter_config.json"
    if cfg_src.exists() and not cfg_dst.exists():
        shutil.copy2(cfg_src, cfg_dst)

    return str(ready_dir)


@lru_cache(maxsize=4)
def get_llm(model_key: str) -> Tuple[LLM, LoRARequest]:
    """Create (or reuse) a vLLM engine and its LoRA request."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")

    cfg = MODEL_CONFIGS[model_key]
    adapter_weights_dir = _materialize_vllm_adapter(cfg["adapter_path"])
    llm = LLM(
        cfg["base_model"],
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        tensor_parallel_size=DEFAULT_TP,
        gpu_memory_utilization=DEFAULT_GPU_UTIL,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        max_num_seqs=DEFAULT_MAX_NUM_SEQS,
    )
    lora_request = LoRARequest(cfg["adapter_name"], 1, adapter_weights_dir)
    return llm, lora_request


def build_prompt(
    task: str, user_text: str, prompt_type: str, system_prompt: str
) -> Dict[str, object]:
    """Return either an instruction prompt or chat messages."""
    if prompt_type == "chat":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return {"chat": messages}

    if system_prompt:
        user_text = f"{system_prompt}\n\n{user_text}"
    prompt = format_prompt_for_inference(user_text, prompt_type="instruction", task=task)
    return {"prompt": prompt}


def run_generation(
    model_key: str,
    task: str,
    prompt_type: str,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    system_prompt: str,
):
    """Execute one generation call through vLLM."""
    user_text = (user_text or "").strip()
    if not user_text:
        return "Please provide an input.", ""

    system_prompt = system_prompt.strip() if system_prompt else DEFAULT_SYSTEM
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )

    llm, lora_request = get_llm(model_key)
    prompt_payload = build_prompt(task, user_text, prompt_type, system_prompt)

    try:
        if "chat" in prompt_payload:
            outputs = llm.chat(
                messages=[prompt_payload["chat"]],
                chat_template=DEFAULT_CHAT_TEMPLATE,
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
        else:
            outputs = llm.generate(
                prompts=[prompt_payload["prompt"]],
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
    except Exception as exc:  # pragma: no cover - surfaced in UI
        return f"Generation failed: {exc}", ""

    if not outputs or not outputs[0].outputs:
        return "No text returned.", ""

    text = outputs[0].outputs[0].text.strip()
    info = (
        f"Model: {MODEL_CONFIGS[model_key]['base_model']} "
        f"+ adapter {MODEL_CONFIGS[model_key]['adapter_path']}\n"
        f"Task: {task} | Prompt type: {prompt_type}"
    )
    return text, info


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Arabic Poetry vLLM Playground") as demo:
        gr.Markdown(
            "# Arabic Poetry vLLM Playground\n"
            "Switch between the Qwen3-8B and ALLaM-7B adapters (curriculum, 4 tasks). "
            "Base models load via vLLM; LoRA adapters are applied on-the-fly."
        )

        with gr.Row():
            model_choice = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=list(MODEL_CONFIGS.keys())[0],
                label="Model",
            )
            task_choice = gr.Dropdown(
                choices=list(TASK_HINTS.keys()),
                value="generation",
                label="Task",
                info="Sets the prompt hint only; no extra constraints are enforced.",
            )
            prompt_type = gr.Radio(
                choices=["instruction", "chat"],
                value="instruction",
                label="Prompt format",
            )

        with gr.Row():
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM,
                lines=3,
                label="System prompt (optional)",
                placeholder="Add guidance or leave as-is.",
            )

        with gr.Row():
            max_new_tokens = gr.Slider(
                minimum=16,
                maximum=1024,
                step=16,
                value=256,
                label="Max new tokens",
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.7,
                step=0.05,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
                step=0.01,
                label="Top-p",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=200,
                value=50,
                step=1,
                label="Top-k",
            )

        user_text = gr.Textbox(
            lines=8,
            label="Input",
            placeholder="Paste your poem, analysis request, or corruption to fix.",
        )
        default_hint = f"Hint: {TASK_HINTS['generation']}"
        hint_box = gr.Markdown(value=default_hint)

        with gr.Row():
            submit_btn = gr.Button("Generate", variant="primary")
            clear_btn = gr.Button("Clear")

        output_text = gr.Textbox(lines=10, label="Model output")
        run_info = gr.Markdown()

        def update_hint(task):
            return f"Hint: {TASK_HINTS.get(task, '')}"

        submit_btn.click(
            fn=run_generation,
            inputs=[
                model_choice,
                task_choice,
                prompt_type,
                user_text,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                system_prompt,
            ],
            outputs=[output_text, run_info],
        )
        clear_btn.click(
            lambda: ("", "", ""), None, [user_text, output_text, run_info], queue=False
        )
        task_choice.change(update_hint, inputs=task_choice, outputs=hint_box)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share_flag = os.getenv("GRADIO_SHARE", "true").lower() == "true"
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=share_flag,
    )
