# Instruction-Guided Poetry Generation in Arabic and Its Dialects

Official repository for paper: Instruction-Guided Poetry Generation in Arabic and Its Dialects.

**Authors:** Abdelrahman Sadallah, Kareem Elozeiri, Mervat Abassy, Rania Elbadry, Mohamed Anwar, Abed Alhakim Freihat, Preslav Nakov, Fajri Koto

**Resources**
- Instruction templates / dataset release: [MBZUAI/instructpoet-ar](https://huggingface.co/datasets/MBZUAI/instructpoet-ar)

## Overview

Arabic poetry remains a culturally central and structurally demanding form of writing, yet it has been underrepresented in controllable generation research. This repository accompanies our work on instruction-guided Arabic poetry modeling, where we build a large-scale instruction-tuning pipeline for poetry generation and understanding across five language varieties:

- Modern Standard Arabic (MSA)
- Gulf
- Levantine
- Nile Valley
- North African

The project covers four core task families:

1. `generation`: write new poems under user constraints
2. `continuation`: continue unfinished poems while preserving structure
3. `corruption` / `revision`: restore corrupted poetic text
4. `analysis`: answer poetry-analysis questions in MCQ format

The full workflow in this repository spans data collection, preprocessing, instruction-template generation, supervised fine-tuning, inference, decoding search, automatic evaluation, and human evaluation.

## Highlights

- A large Arabic poetry instruction-tuning setup with `4` task families and `54` subtasks.
- `3,220` manually designed instruction templates across MSA and four major Arabic dialect groups.
- `1.35M` training pairs and `24.8K` test pairs reported in the paper.
- Fine-tuning experiments on Arabic-centric and multilingual LLMs, including ALLaM, Fanar, Qwen3, and Llama-family models.
- Evaluation via automatic metrics, LLM-as-a-judge, MCQ analysis accuracy, and blind human evaluation with Arabic-speaking annotators.

## Repository At A Glance

```text
arab_poetry/
├── evaluation/              # Batch inference, scoring, grid search, and human evaluation
├── finetune/                # SFT training code, configs, and launch scripts
├── inference/               # Gradio + vLLM interactive demo
├── merge_adapters/          # Utilities to merge LoRA adapters into standalone checkpoints
├── sanity_checks/           # Interactive inspection scripts for qualitative debugging
├── scrappers/               # Data collection scripts for public Arabic poetry sources
└── scripts/                 # Preprocessing, IFT generation, corruption, and evaluation helpers
```

## Dataset And Templates

The public Hugging Face release is available at:

- [https://huggingface.co/datasets/MBZUAI/instructpoet-ar](https://huggingface.co/datasets/MBZUAI/instructpoet-ar)

At a high level, the data pipeline follows the paper:

1. Aggregate Arabic poetry from multiple public sources.
2. Normalize metadata and verse structure.
3. Deduplicate train/test data and reduce leakage.
4. Generate instruction-following examples from curated templates.
5. Export task-specific TSV files for fine-tuning and evaluation.

The repository contains the scripts used for these stages, including:

- `scrappers/` for source collection
- `scripts/data_preprocessing/` for cleaning and deduplication
- `scripts/generate_ift_data_new/` for generating instruction-tuning TSVs
- `scripts/poetry_corruption/` and `scripts/gemini_api/` for corruption-based revision data creation

## Supported Tasks

| Task | Description | Typical Inputs | Typical Outputs |
|------|-------------|----------------|-----------------|
| `generation` | Create a poem from constraints such as theme, era, meter, rhyme, or dialect | themes, keywords, style constraints, metadata | a newly generated poem |
| `continuation` | Complete an unfinished poem while preserving poetic consistency | existing verses, rhyme or meter hints | continuation verses |
| `corruption` | Restore noisy or corrupted poetry | corrupted verses or incomplete text | corrected poem |
| `analysis` | Answer poetry-analysis questions framed as multiple choice | poem text and optional metadata | target label such as meter, era, genre, poet, or rhyme |

## Installation

This repository assumes a Linux environment with CUDA-enabled GPUs for training and most inference workflows.

### 1. Install Python dependencies

```bash
cd arab_poetry
pip install -r requirements.txt
```

### 2. Install Flash Attention if needed

Training config defaults to `flash_attention_2`, so you may need to install a Flash Attention build matching your Python, CUDA, and PyTorch versions.

### 3. Set authentication tokens

Some workflows require Hugging Face access:

```bash
export HF_TOKEN=your_huggingface_token
```

The demo and several inference scripts also honor `HUGGINGFACE_TOKEN`.

## Reproducing The Pipeline

### 1. Data Collection And Preprocessing

Source collection and extraction utilities live under `scrappers/`. Preprocessing and leakage control scripts live under `scripts/data_preprocessing/`.

Example deduplication command:

```bash
python scripts/data_preprocessing/dedup_data.py \
  --train_path /path/to/cleaned_train.csv \
  --test_path /path/to/cleaned_test.csv \
  --out_train_path /path/to/de_dupped_train.csv \
  --out_test_path /path/to/de_dupped_test.csv \
  --threshold 0.90
```

### 2. Generate Instruction-Tuning Data

The main generator is `scripts/generate_ift_data_new/generate_IFT_data_new.py`. It creates task-specific train/test TSV files from cleaned poetry data plus the instruction templates.

Example:

```bash
cd scripts/generate_ift_data_new
python generate_IFT_data_new.py \
  --raw_data /path/to/train_or_test.csv \
  --templates /path/to/poetry_templates.xlsx \
  --output_dir /path/to/dialectical_IFT_DATA \
  --task generation \
  --total_num_samples -1 \
  --min_num_verses 2 \
  --max_poem_verses 0 \
  --preferred_dialect random
```

For analysis benchmarking, enable MCQ export:

```bash
python generate_IFT_data_new.py \
  --raw_data /path/to/test.csv \
  --templates /path/to/poetry_templates.xlsx \
  --output_dir /path/to/dialectical_IFT_DATA \
  --task analysis \
  --create_mcq_benchmark
```

### 3. Fine-Tune Models

The training entry point is `finetune/finetune.sh`, which supports both local execution and SLURM submission. The main config file is `finetune/config.yaml`.

Example local run:

```bash
cd finetune
bash finetune.sh \
  --model Qwen/Qwen3-8B \
  --mode curriculum \
  --tasks analysis,continuation,generation,corruption \
  --prompt_type chat \
  --use_peft true \
  --gpus 4
```

Example SLURM run:

```bash
cd finetune
sbatch finetune.sh \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --mode random \
  --tasks analysis,continuation,generation,corruption
```

Two training regimes are supported:

- `random`: mix all task examples together
- `curriculum`: present tasks in the sequence `analysis -> continuation -> generation -> corruption`

### 4. Run Batch Inference

Inference is handled from `evaluation/` and supports LoRA adapters, merged checkpoints, and base models.

Example:

```bash
cd evaluation
bash inference.sh \
  --model_path /path/to/checkpoint_or_adapter \
  --data_path /path/to/dialectical_IFT_DATA/tsv \
  --tasks all \
  --prompt_type chat \
  --gpus 4
```

The script auto-detects common base models for LoRA adapters and writes task-wise TSV files with a new `model_generation` column.

### 5. Tune Decoding Hyperparameters

Grid search utilities for decoding settings are in `evaluation/grid_search/`.

Example:

```bash
cd evaluation/grid_search
bash run_grid_search.sh --model_path /path/to/checkpoint_or_adapter
```

### 6. Prepare Human Evaluation

Human-evaluation utilities are in `evaluation/human_evaluation/`. The main preparation script creates blind evaluation sheets and model-ID mappings from generation outputs.

Example:

```bash
cd evaluation/human_evaluation
python prepare_human_evaluation.py \
  --base_dir /path/to/model_outputs \
  --output_dir ./ \
  --task generation \
  --n_samples 100 \
  --seed 42
```

## Interactive Demo

The repository includes a lightweight Gradio + vLLM interface in `inference/app.py` for trying LoRA adapters interactively.

```bash
cd inference
python app.py
```

By default, the app listens on `0.0.0.0:7860`. It expects accessible base models and adapter checkpoints, and it can load Hugging Face credentials from the environment.

Useful environment overrides:

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`
- `QWEN3_BASE_MODEL`
- `ALLAM_BASE_MODEL`
- `VLLM_TENSOR_PARALLEL`
- `VLLM_GPU_UTIL`
- `VLLM_MAX_MODEL_LEN`
- `VLLM_MAX_NUM_SEQS`
- `PORT`

## Evaluation In The Paper

The paper reports a multi-part evaluation setup:

- `generation`, `continuation`, and `corruption` are evaluated with task-aware LLM-as-a-judge prompts and automatic metrics.
- `analysis` is evaluated as multiple-choice prediction accuracy.
- Human evaluation is conducted on `100` prompts with `4` model variants and `2` Arabic-speaking annotators using blind scoring on compliance, fluency, coherence, and poetic quality.

Key takeaways reported in the paper:

- Fine-tuning consistently improves performance over base models across nearly all settings.
- Random and curriculum training are both effective, with neither strategy universally dominating all tasks and dialects.
- Fine-tuned ALLaM-7B achieved the best overall human-evaluation score (`3.99/5.0`), improving from `3.02`.
- Fine-tuned Qwen3-8B improved from `2.24` to `3.66/5.0`, showing a large relative gain after poetry-focused tuning.

## Important Notes For Reproducibility

- Many scripts in this repository currently contain cluster-specific absolute paths used during the original experiments. Replace these with local paths before running.
- Large raw corpora and trained checkpoints are not bundled directly in this repository snapshot.
- Main training and inference scripts support both SLURM and direct `bash` execution.
- Several evaluation and demo workflows assume multi-GPU inference with vLLM.
- LoRA checkpoints may need to be materialized into vLLM-friendly adapter directories; helper code for this is already included in the repo.

## Main Components

### `finetune/`

- `run_sft.py`: core supervised fine-tuning entry point
- `poetry_data.py`: dataset loading and task mixing logic
- `config.yaml`: training defaults
- `accelerate_configs/`: distributed launch configs for DDP and DeepSpeed
- `finetune.sh`: SLURM/local training launcher

### `evaluation/`

- `inference.sh`: orchestration wrapper for local or SLURM inference
- `inference_runner.py`: vLLM inference engine for TSV datasets
- `grid_search/`: decoding-parameter search utilities
- `human_evaluation/`: blind human evaluation preparation and analysis
- `evaluation_results*/`: stored experiment outputs and summaries

### `inference/`

- `app.py`: Gradio playground for interactive generation with vLLM + LoRA

### `merge_adapters/`

- utilities for merging or cleaning adapter checkpoints before standalone use

### `sanity_checks/`

- quick qualitative inspection and interactive inference helpers

## Acknowledgments

This repository builds on publicly available Arabic poetry resources, Hugging Face tooling, TRL/Transformers, PEFT, and vLLM.
