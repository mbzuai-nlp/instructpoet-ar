# Arabic Poetry Model Inference

This directory contains scripts for running inference on trained Arabic poetry models. The inference system processes TSV test data and generates predictions, saving them as new columns in the output TSV files.

## 🚀 Quick Start

### Running Inference on Your Trained Models

```bash
# For Qwen3-8B model (SLURM)
sbatch inference.sh \
    --model_path /path/to/checkpoints/Qwen3-8B/curriculum/4tasks/checkpoint-36000 \
    --data_path /path/to/data/dialectical_IFT_DATA/tsv \
    --tasks all

# For ALLaM-7B model (SLURM)
sbatch inference.sh \
    --model_path /path/to/checkpoints/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000 \
    --data_path /path/to/data/dialectical_IFT_DATA/tsv \
    --tasks all
```

### Local Testing (with limited samples)

```bash
bash inference.sh \
    --model_path /path/to/checkpoint \
    --data_path /path/to/test/data \
    --tasks analysis,continuation \
    --gpus 2 \
    --max_samples 100
```

## 📁 File Structure

```
evaluation/
├── inference.sh              # Main bash script for orchestrating inference
├── inference_runner.py       # Python script that runs vLLM inference
├── run_inference_examples.sh # Example commands for your models
├── vllm_inference.py         # Original inference script (kept for reference)
├── args_parser.py            # Argument parsing utilities
├── inference_utils.py        # Utility functions
└── README_INFERENCE.md       # This file
```

## 🔧 Main Components

### 1. `inference.sh` - Orchestration Script

The main bash script that handles:
- **Automatic model type detection** (LoRA adapters vs. full finetuned models)
- **Base model auto-detection** for LoRA checkpoints
- **Multi-task execution** (processes all specified tasks sequentially)
- **GPU configuration** (automatic detection or manual specification)
- **SLURM and local execution** support

**Key Features:**
- Inspired by your finetuning script structure
- Automatically detects if checkpoint is LoRA or full model
- Creates organized output directory structure
- Saves metadata for reproducibility
- Comprehensive logging

### 2. `inference_runner.py` - Core Inference Engine

The Python script that:
- Loads TSV test data files
- Runs vLLM inference with tensor parallelism
- Adds model generations as a new column to TSV files
- Preserves all original columns
- Handles both LoRA and full finetuned models

**Output Format:**
The script creates new TSV files with suffix `_with_predictions.tsv` containing:
- All original columns from input TSV
- New column: `model_generation` with model predictions

## 📋 Usage Guide

### Command-Line Arguments

#### `inference.sh` Options:

```bash
--model_path PATH         # Path to model checkpoint (required)
--base_model MODEL        # Base model for LoRA (auto-detected if not provided)
--data_path PATH          # Path to test data directory (required)
--output_path PATH        # Output directory (default: ./inference_outputs)
--tasks TASKS             # Comma-separated tasks or 'all' (default: all)
--prompt_type TYPE        # instruction or chat (default: chat)
--max_new_tokens N        # Max tokens to generate (default: 512)
--temperature TEMP        # Temperature for sampling (default: 0.0)
--max_num_seqs N          # Max parallel sequences (default: 16)
--gpus N                  # Number of GPUs for local run (auto-detect)
--max_samples N           # Max samples per task (default: all)
--args "ARGS"             # Additional arguments
```

#### `inference_runner.py` Options:

```bash
--model_path PATH         # Path to model or base model
--adapter_path PATH       # Path to LoRA adapter (optional)
--data_path PATH          # Base path for TSV data
--task TASK               # Single task: analysis, continuation, generation, corruption
--output_path PATH        # Output directory
--prompt_type TYPE        # instruction or chat
--max_new_tokens N        # Max tokens to generate
--temperature FLOAT       # Temperature
--top_p FLOAT             # Top-p sampling
--top_k INT               # Top-k sampling
--tensor_parallel_size N  # Number of GPUs for tensor parallelism
--max_num_seqs N          # Max parallel sequences
--max_samples N           # Max samples to process
```

## 📊 Data Structure

### Expected Input Structure

```
data_path/
├── analysis/
│   └── test/
│       └── analysis_ift.tsv
├── continuation/
│   └── test/
│       └── continuation_ift.tsv
├── generation/
│   └── test/
│       └── generation_ift.tsv
└── corruption/
    └── test/
        └── corruption_ift.tsv
```

### TSV Input Format

Each TSV file should have at least these columns:
- `input`: The input prompt/text
- `output`: The reference/expected output (optional for pure inference)
- Other task-specific columns

### Output Structure

```
inference_outputs/
└── Qwen3-8B/                    # Model name
    └── curriculum/               # Training mode
        └── checkpoint-36000/     # Checkpoint
            ├── inference_metadata.json
            ├── analysis/
            │   └── analysis_ift_with_predictions.tsv
            ├── continuation/
            │   └── continuation_ift_with_predictions.tsv
            ├── generation/
            │   └── generation_ift_with_predictions.tsv
            └── corruption/
                └── corruption_ift_with_predictions.tsv
```

## 🎯 Example Workflows

### 1. Full Evaluation on All Tasks (SLURM)

```bash
sbatch inference.sh \
    --model_path /path/to/checkpoint-36000 \
    --data_path /path/to/dialectical_IFT_DATA/tsv \
    --tasks all \
    --output_path ./inference_outputs
```

### 2. Quick Test with Limited Samples (Local)

```bash
bash inference.sh \
    --model_path /path/to/checkpoint-36000 \
    --data_path /path/to/dialectical_IFT_DATA/tsv \
    --tasks analysis \
    --gpus 2 \
    --max_samples 50
```

### 3. Specific Tasks Only

```bash
bash inference.sh \
    --model_path /path/to/checkpoint-36000 \
    --data_path /path/to/dialectical_IFT_DATA/tsv \
    --tasks analysis,continuation,generation \
    --gpus 4
```

### 4. Custom Base Model for LoRA

```bash
sbatch inference.sh \
    --model_path /path/to/checkpoint-36000 \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --data_path /path/to/dialectical_IFT_DATA/tsv \
    --tasks all
```

### 5. With Custom Generation Parameters

```bash
bash inference.sh \
    --model_path /path/to/checkpoint-36000 \
    --data_path /path/to/dialectical_IFT_DATA/tsv \
    --tasks all \
    --temperature 0.7 \
    --max_new_tokens 1024 \
    --max_num_seqs 32
```

## 🔍 Model Type Detection

The script automatically detects:

1. **LoRA Adapters**: If `adapter_config.json` exists in model path
   - Auto-detects base model from path (Qwen, ALLaM)
   - Can override with `--base_model`

2. **Full Finetuned Model**: If `config.json` exists (no adapter_config.json)
   - Uses the checkpoint directly as the model

## 📈 Output Files

### 1. Metadata File (`inference_metadata.json`)

```json
{
  "model_path": "/path/to/checkpoint-36000",
  "model_type": "adapters",
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "model_name": "Qwen3-8B",
  "training_mode": "curriculum",
  "checkpoint": "checkpoint-36000",
  "data_path": "/path/to/tsv",
  "tasks": ["analysis", "continuation", "generation", "corruption"],
  "prompt_type": "chat",
  "max_new_tokens": 512,
  "temperature": 0.0,
  "max_num_seqs": 16,
  "gpus": 4,
  "start_time": "2025-12-05 10:30:00"
}
```

### 2. TSV Files with Predictions

Original columns + `model_generation` column:

```tsv
input	output	template_output_field	...	model_generation
"حدد الكلمات..."	"الكلمات: حب، غرام"	...	"الكلمات المفتاحية..."
```

## 🚨 Troubleshooting

### Issue: "Cannot auto-detect base model"
**Solution**: Specify `--base_model` explicitly
```bash
--base_model "Qwen/Qwen2.5-7B-Instruct"
```

### Issue: Out of GPU memory
**Solutions**:
- Reduce `--max_num_seqs`
- Reduce `--tensor_parallel_size` and use fewer GPUs
- Reduce `--max_new_tokens`
- Enable `gpu_memory_utilization=0.8` (edit script)

### Issue: Task data not found
**Solution**: Verify data path structure matches expected format
```bash
ls $DATA_PATH/analysis/test/
```

### Issue: vLLM import errors
**Solution**: Ensure vLLM is installed
```bash
pip install vllm
```

## 🔗 Integration with Existing Scripts

This inference system integrates with:
- **Finetuning**: Uses checkpoints from `finetune/finetune.sh`
- **Evaluation**: Output TSV files can be used with evaluation metrics
- **Data Pipeline**: Works with TSV format from your data processing

## 📝 Notes

1. **Model Detection**: The script looks for "Qwen" or "ALLaM" in the path to auto-detect base models
2. **Task Mapping**: `corruption` and `restoration` map to the same folder
3. **Prompt Format**: Default is `chat` (recommended for instruction-tuned models)
4. **GPU Settings**: Script auto-detects available GPUs in local mode
5. **SLURM Integration**: Same SLURM directives as your finetuning script

## 🎓 Best Practices

1. **Always start with a small test**:
   ```bash
   bash inference.sh --max_samples 10 --gpus 1
   ```

2. **Check logs** before processing full dataset:
   ```bash
   tail -f logs/poetry_inference-*.out
   ```

3. **Verify outputs** after first task completes:
   ```bash
   head inference_outputs/.../analysis/*_with_predictions.tsv
   ```

4. **Monitor GPU usage** during inference:
   ```bash
   watch -n 1 nvidia-smi
   ```

## � Grid Search for Decoding Hyperparameters

If you're experiencing repetition issues in generated text, use the grid search tool to find optimal decoding parameters:

```bash
# Navigate to grid_search directory
cd grid_search

# Run grid search
bash run_grid_search.sh --model_path /path/to/your/model

# View results
python load_best_config.py --model_name checkpoint-36000
```

The grid search will:
- Test different combinations of `temperature` and `repetition_penalty`
- Measure repetition rates and text quality
- Automatically identify the best hyperparameters
- Save results in `best_decoding_parameters/`

**See [`grid_search/README_GRID_SEARCH.md`](grid_search/README_GRID_SEARCH.md) for detailed documentation.**

## �📚 Additional Resources

- See `run_inference_examples.sh` for ready-to-run examples
- Check `vllm_inference.py` for the original implementation
- Refer to `../finetune/README.md` for finetuning documentation
- Use `grid_search/` for optimizing decoding parameters

---

**Need Help?** Check the help message:
```bash
bash inference.sh --help
```
