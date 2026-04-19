# Inference Commands for All Models

## Overview
This document contains commands to run inference on **5 models** with **3 configurations each** (curriculum, random, and base model).

**Job Structure**: Each model runs as **ONE job** that performs all 3 inferences sequentially.
- **Total Jobs**: 5 (one per model)
- **Total Inferences**: 15 (5 models × 3 configurations each)

## Models and Checkpoints

| Model | Base Model | Curriculum Checkpoint | Random Checkpoint |
|-------|-----------|----------------------|-------------------|
| ALLaM-7B-Instruct-preview | humain-ai/ALLaM-7B-Instruct-preview | checkpoint-32000 | checkpoint-42216 |
| Fanar-1-9B | QCRI/Fanar-1-9B | checkpoint-42216 | checkpoint-42216 |
| gemma-3-12b-it | google/gemma-3-12b-it | checkpoint-42216 | checkpoint-42216 |
| Meta-Llama-3-8B-Instruct | meta-llama/Meta-Llama-3-8B-Instruct | checkpoint-38500 | checkpoint-38500 |
| Qwen3-8B | Qwen/Qwen3-8B | checkpoint-36000 | checkpoint-35000 |

## Job Execution Flow

Each job performs the following steps **sequentially**:
1. **Curriculum Checkpoint** inference
2. **Random Checkpoint** inference  
3. **Base Model** inference (baseline)

This ensures all three configurations for a model are completed before moving to the next model.

## Configuration Details

### Common Parameters
- **Tasks**: `generation,continuation,corruption`
- **Max New Tokens**: 1024
- **Temperature**: 0.7
- **Top P**: 0.9
- **Top K**: 50
- **Repetition Penalty**: 1.15
- **Max Num Seqs**: 64
- **GPUs**: 4 (per job)

### Paths
- **Checkpoint Base**: `/path/to/checkpoints`
- **Data Path**: `/path/to/data/dialectical_IFT_DATA/tsv`
- **Output Base**: `/path/to/outputs`

## Usage Options

### Option 1: Run All Jobs Automatically (Recommended)
Execute the batch script to submit all 5 jobs:

```bash
cd <PROJECT_ROOT>/evaluation
bash run_all_models_inference.sh
```

This will:
- Create 5 SLURM job scripts (one per model)
- Submit all 5 jobs to the queue
- Each job runs all 3 inferences sequentially for that model
- Create organized output directories for each configuration

### Option 2: Run Individual Commands
If you prefer to run specific models or configurations manually, use the commands in `inference_commands.txt`.

## Advantages of This Approach

1. **Better Resource Management**: Only 5 jobs in the queue instead of 15
2. **Sequential Execution**: Each model's inferences run one after another, avoiding conflicts
3. **Cleaner Logs**: One log file per model (containing all 3 inferences)
4. **Easier Tracking**: Monitor 5 jobs instead of 15

## Monitoring Jobs

### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# Expected output: 5 jobs (one per model)
# JOBID   PARTITION     NAME               USER    ST  TIME  NODES
# 123456  gpumid        ALLaM-7B...        user    R   10:30  1
# 123457  gpumid        Fanar-1-9B...      user    R   8:15   1
# ...
```

### Check Job Queue (detailed)
```bash
squeue -u $USER -l
```

### Check Specific Job
```bash
squeue -j <job_id>
```

### View Logs (Real-time)
Each model has its own log file:
```bash
# List all logs (newest first)
ls -lth logs/

# View a specific model's log
tail -f logs/ALLaM-7B-Instruct-preview_inference-<job_id>.out
tail -f logs/Qwen3-8B_inference-<job_id>.out

# View errors
tail -f logs/ALLaM-7B-Instruct-preview_inference-<job_id>.err
```

### Cancel a Job
```bash
# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Output Structure

Each inference run will create output in the following structure:

```
/path/to/outputs/
├── ALLaM-7B_curriculum_checkpoint-32000/
├── ALLaM-7B_random_checkpoint-42216/
├── ALLaM-7B_base/
├── Fanar-1-9B_curriculum_checkpoint-42216/
├── Fanar-1-9B_random_checkpoint-42216/
├── Fanar-1-9B_base/
├── gemma-3-12b-it_curriculum_checkpoint-42216/
├── gemma-3-12b-it_random_checkpoint-42216/
├── gemma-3-12b-it_base/
├── Meta-Llama-3-8B_curriculum_checkpoint-38500/
├── Meta-Llama-3-8B_random_checkpoint-38500/
├── Meta-Llama-3-8B_base/
├── Qwen3-8B_curriculum_checkpoint-36000/
├── Qwen3-8B_random_checkpoint-35000/
└── Qwen3-8B_base/
```

## Files Created

1. **run_all_models_inference.sh** - Automated batch script to submit all 5 jobs
2. **inference_commands.txt** - Individual commands for manual execution
3. **INFERENCE_GUIDE.md** - This guide

## Troubleshooting

### Job Failed to Start
- Check GPU availability: `sinfo -p gpumid`
- Check your account: `sacctmgr show assoc user=$USER`
- Check queue limits: `squeue -p gpumid`

### Out of Memory
- The script automatically handles this per inference
- If needed, edit the generated job script to reduce:
  - `--max_num_seqs` (try 32 or 16)
  - `--max_new_tokens` (try 512)

### Job Stopped After First Inference
- Check the log file for errors
- The script uses `set -e`, so any error will stop the job
- Fix the issue and resubmit

### Model Not Found
- Verify checkpoint paths exist:
  ```bash
  ls /path/to/checkpoints/<model>/curriculum/4tasks/
  ls /path/to/checkpoints/<model>/random/4tasks/
  ```

### Data Path Issues
- Verify data path:
  ```bash
  ls /path/to/data/dialectical_IFT_DATA/tsv/
  ```

### Check Job Progress
Each log file will show progress through the 3 stages:
```bash
tail -f logs/<model>_inference-<job_id>.out
# Look for:
# [1/3] Running inference on CURRICULUM checkpoint...
# [2/3] Running inference on RANDOM checkpoint...
# [3/3] Running inference on BASE MODEL...
```

## Estimated Runtime

Each model's job will take approximately:
- **Per inference**: 30-60 minutes (depends on model size and tasks)
- **Total per model**: 1.5-3 hours (3 inferences)
- **All models**: Jobs run in parallel, so ~1.5-3 hours total (if resources available)

## Notes

- Each model is tested on 3 configurations to compare:
  1. **Curriculum learning** approach
  2. **Random shuffling** approach  
  3. **Base model** (no fine-tuning) as baseline
  
- The base model inference helps establish a baseline performance before fine-tuning

- Jobs are submitted with a 2-second delay to avoid overwhelming the SLURM scheduler

- All jobs use the same hyperparameters for fair comparison

- The script creates temporary job files in `/tmp/` that are automatically cleaned up on reboot
