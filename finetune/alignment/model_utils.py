# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from .configs import SFTConfig

# Import Gemma3 specific class
try:
    from transformers import Gemma3ForCausalLM

    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(
    model_args: ModelConfig, training_args: SFTConfig
) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    # Check if model is Gemma3
    model_name_lower = model_args.model_name_or_path.lower()
    is_gemma3 = "gemma3" in model_name_lower or "gemma-3" in model_name_lower

    # For DDP training, we should not use device_map - let DDP handle device placement
    # Only use device_map for quantization in single-GPU scenarios
    import torch.distributed as dist

    is_distributed = dist.is_initialized()

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
    )

    # Only use device_map if not in distributed mode and using quantization
    if quantization_config is not None and not is_distributed:
        model_kwargs["device_map"] = get_kbit_device_map()

    # Add use_cache for non-Gemma3 models
    if not is_gemma3:
        model_kwargs["use_cache"] = (
            False if training_args.gradient_checkpointing else True
        )

    # Use Gemma3ForCausalLM for Gemma3 models to avoid vocab_size issues
    if is_gemma3 and GEMMA3_AVAILABLE:
        model = Gemma3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        # Set use_cache on the config after initialization for Gemma3
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = (
                False if training_args.gradient_checkpointing else True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

    return model
