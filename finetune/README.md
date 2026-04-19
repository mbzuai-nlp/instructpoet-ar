# Arabic Poetry Finetuning and Evaluation

This directory contains scripts for finetuning and evaluating language models on Arabic poetry tasks.

## Requirements

Follow these steps to set up the environment:

1. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Flash Attention:
   1. Got to this Website: https://flashattn.dev/#finder
   2. Select Flash Attention `2.8.3`
   3. Set your Python, CUDA, and PyTorch versions
   4. Follow the given command to install the package (here is mine: `pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl`)


## Tasks

The system supports four poetry tasks:

1. **Analysis**: Analyze poetry (extract keywords, themes, meter, etc.)
2. **Continuation**: Continue a poem given a starting portion
3. **Generation**: Generate poetry from themes/keywords
4. **Restoration** (corruption): Restore/correct corrupted poetry

## Training Modes

### Random Mode
All task data is concatenated and shuffled together. The model learns all tasks simultaneously without any specific order.

```bash
--training_mode=random
```

### Curriculum Learning Mode
Data is presented in a specific order: analysis → continuation → generation → restoration. This follows a pedagogical approach where:
- First, the model learns to understand/analyze poetry
- Then, it learns to continue existing poems
- Next, it learns to generate new poetry
- Finally, it learns to restore/correct corrupted poetry

```bash
--training_mode=curriculum
```

## Directory Structure

```
finetune/
├── run_sft.py              # Main training script
├── poetry_data.py          # Data loading utilities
├── config_random.yaml      # Config for random training
├── config_curriculum.yaml  # Config for curriculum training
├── finetune.sh            # Training shell script
├── deepspeed_zero3.yaml   # DeepSpeed configuration
└── alignment/             # HuggingFace alignment utilities

evaluation/
├── vllm_inference.py      # vLLM-based inference script
├── args_parser.py         # Argument parser
├── inference_utils.py     # Inference utilities
└── evaluate.sh            # Evaluation shell script
```

## Data Format

The training data is expected in TSV format with the following structure:
- Located at: `/path/to/data/dialectical_IFT_DATA/tsv`
- Structure:
  ```
  tsv/
  ├── analysis/
  │   ├── train/analysis_ift.tsv
  │   └── test/analysis_ift.tsv
  ├── continuation/
  │   ├── train/continuation_ift.tsv
  │   └── test/continuation_ift.tsv
  ├── generation/
  │   ├── train/generation_ift.tsv
  │   └── test/generation_ift.tsv
  └── corruption/
      ├── train/corruption_ift.tsv
      └── test/corruption_ift.tsv
  ```

Each TSV file should have at least `input` and `output` columns.

## Usage

### Training

1. **Configure your training** by editing `config_random.yaml` or `config_curriculum.yaml`

2. **Run training with random mode:**
   ```bash
   cd finetune
   bash finetune.sh
   ```

3. **Run training with curriculum learning:**
   Edit `finetune.sh` and set `TRAINING_MODE="curriculum"`

4. **Custom training command:**
   ```bash
   ACCELERATE_LOG_LEVEL=info accelerate launch \
       --config_file deepspeed_zero3.yaml \
       --num_processes=4 \
       run_sft.py \
       config_random.yaml \
       --output_dir=/path/to/output \
       --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
       --training_mode=curriculum \
       --tasks=analysis,continuation,generation,corruption \
       --use_peft=true
   ```

### Evaluation

1. **Configure evaluation** by editing `evaluate.sh`

2. **Run evaluation:**
   ```bash
   cd evaluation
   bash evaluate.sh
   ```

3. **Custom evaluation command:**
   ```bash
   python vllm_inference.py \
       --full_model_name meta-llama/Llama-3.1-8B-Instruct \
       --finetuning_type adapters \
       --checkpoint_parent_path /path/to/checkpoints \
       --training_mode random \
       --task all \
       --prompt_type instruction \
       --output_path evaluation_outputs
   ```

## Configuration Options

### Training Configuration (YAML)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `training_mode` | `random` or `curriculum` | `random` |
| `tasks` | Comma-separated task list | `analysis,continuation,generation,corruption` |
| `data_base_path` | Path to TSV data | See config |
| `max_samples_per_task` | Limit samples per task | `null` (all) |
| `use_peft` | Use LoRA adapters | `true` |
| `prompt_type` | `instruction` or `chat` | `instruction` |
| `learning_rate` | Learning rate | `2e-5` |
| `num_train_epochs` | Training epochs | `1` |

### Evaluation Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `finetuning_type` | `baseline`, `adapters`, or `full` | `adapters` |
| `training_mode` | Mode used during training | `random` |
| `task` | Task to evaluate or `all` | `analysis` |
| `max_new_tokens` | Max generation tokens | `512` |
| `temperature` | Sampling temperature | `0` |



## Environment Variables

Create a `.env` file with:
```
HF_TOKEN=your_huggingface_token
```

## Tips

1. **For curriculum learning**: Ensure tasks are ordered correctly in the config
2. **For large datasets**: Use `max_samples_per_task` for testing
3. **Memory issues**: Reduce `per_device_train_batch_size` or enable gradient checkpointing
4. **Multi-GPU training**: Adjust `num_processes` in accelerate launch command
