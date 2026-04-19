# Inference System - Quick Reference

## 🎯 What Was Created

A complete inference system for running your trained Arabic poetry models on test data, inspired by your finetuning script structure.

## 📦 New Files

1. **`inference.sh`** - Main orchestration script (bash)
   - Auto-detects model type (LoRA vs full)
   - Handles multiple tasks and GPUs
   - Works with SLURM and local execution

2. **`inference_runner.py`** - Core inference engine (Python)
   - Runs vLLM inference
   - Outputs TSV files with model predictions added
   - Handles both LoRA adapters and full models

3. **`run_inference_examples.sh`** - Ready-to-run examples
   - Pre-configured for your specific model checkpoints
   - Various usage patterns demonstrated

4. **`README_INFERENCE.md`** - Complete documentation
   - Detailed usage guide
   - Troubleshooting tips
   - Best practices

## 🚀 How to Use

### For Your Qwen3-8B Model:

```bash
sbatch inference.sh \
    --model_path /path/to/checkpoints/Qwen3-8B/curriculum/4tasks/checkpoint-36000 \
    --data_path /path/to/data/dialectical_IFT_DATA/tsv \
    --tasks all
```

### For Your ALLaM-7B Model:

```bash
sbatch inference.sh \
    --model_path /path/to/checkpoints/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000 \
    --data_path /path/to/data/dialectical_IFT_DATA/tsv \
    --tasks all
```

### Quick Local Test:

```bash
bash inference.sh \
    --model_path /path/to/checkpoint-36000 \
    --data_path /path/to/dialectical_IFT_DATA/tsv \
    --tasks analysis \
    --gpus 2 \
    --max_samples 50
```

## 📊 Output Structure

Your outputs will be organized as:

```
inference_outputs/
├── Qwen3-8B/
│   └── curriculum/
│       └── checkpoint-36000/
│           ├── inference_metadata.json
│           ├── analysis/
│           │   └── analysis_ift_with_predictions.tsv
│           ├── continuation/
│           │   └── continuation_ift_with_predictions.tsv
│           ├── generation/
│           │   └── generation_ift_with_predictions.tsv
│           └── corruption/
│               └── corruption_ift_with_predictions.tsv
└── ALLaM-7B-Instruct-preview/
    └── curriculum/
        └── checkpoint-32000/
            └── ... (same structure)
```

## 🎁 Key Features

✅ **Auto-Detection**
- Automatically detects if checkpoint is LoRA or full model
- Auto-detects base model for LoRA (Qwen, ALLaM)

✅ **Input Preservation**
- Original TSV files remain unchanged
- New TSV files have `_with_predictions.tsv` suffix
- All original columns preserved + new `model_generation` column

✅ **Multi-Task Support**
- Run on all tasks: `--tasks all`
- Or specific tasks: `--tasks analysis,continuation`

✅ **Flexible Execution**
- SLURM job submission
- Local bash execution
- GPU auto-detection

✅ **Inspired by Your Workflow**
- Similar argument structure to `finetune.sh`
- Familiar configuration patterns
- Comprehensive logging

## 📋 What the Output TSV Contains

Original columns from input TSV + new column:
- `input` (original)
- `output` (original)
- `template_output_field` (original)
- ... (all other original columns)
- **`model_generation`** ← NEW! Your model's prediction

## 🔧 Common Options

```bash
# All tasks, default settings
--tasks all

# Specific tasks only
--tasks analysis,continuation

# Limit samples for testing
--max_samples 100

# Use specific number of GPUs locally
--gpus 2

# Adjust generation parameters
--temperature 0.7 --max_new_tokens 1024

# Specify base model for LoRA
--base_model "Qwen/Qwen2.5-7B-Instruct"
```

## 📝 Next Steps

1. **Test with small dataset first**:
   ```bash
   bash inference.sh --model_path /path/to/checkpoint-36000 \
       --data_path /path/to/tsv --tasks analysis --max_samples 10 --gpus 1
   ```

2. **Check the output**:
   ```bash
   head inference_outputs/Qwen3-8B/curriculum/checkpoint-36000/analysis/*_with_predictions.tsv
   ```

3. **Run full inference** when satisfied:
   ```bash
   sbatch inference.sh --model_path /path/to/checkpoint-36000 \
       --data_path /path/to/tsv --tasks all
   ```

4. **Use outputs for evaluation**:
   - The TSV files with predictions can be used for computing metrics
   - Compare `model_generation` vs `output` columns

## 🆘 Quick Help

```bash
# See all options
bash inference.sh --help

# Check example runs
cat run_inference_examples.sh

# Read full documentation
cat README_INFERENCE.md
```

## 🔗 File Locations

All files are in: `<PROJECT_ROOT>/evaluation/`

- Main script: `inference.sh`
- Python runner: `inference_runner.py`
- Examples: `run_inference_examples.sh`
- Full docs: `README_INFERENCE.md`
- This summary: `INFERENCE_SUMMARY.md`

---

**Ready to Go!** 🚀

Start with the examples in `run_inference_examples.sh` or run your own commands following the patterns above.
