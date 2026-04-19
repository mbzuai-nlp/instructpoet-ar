# Decoding Hyperparameter Grid Search

This directory contains tools for performing grid search over decoding hyperparameters to optimize Arabic poetry generation and minimize text repetition.

## Overview

The grid search script tests different combinations of decoding hyperparameters to find the optimal configuration that:
1. **Minimizes repetition** in generated text
2. **Maintains quality** with appropriate verse counts
3. **Maximizes diversity** through unique word usage

## Features

- **Automatic Sampling**: Randomly samples 100 examples from each task (generation, continuation, corruption)
- **Comprehensive Metrics**: Calculates repetition scores, unique word ratios, and verse counts
- **Batch Processing**: Tests multiple hyperparameter combinations automatically
- **Best Config Selection**: Automatically identifies and saves the best hyperparameters
- **Detailed Reports**: Generates JSON results and human-readable summaries

## Hyperparameters Tested

### Default Grid Search Space:

- **Repetition Penalty**: `1.0`, `1.10`, `1.15`
- **Temperature**: `0.0`, `0.7`
- **Tasks**: `generation`, `continuation`, `corruption`
- **Samples per Task**: `100`

## Repetition Metrics

The script calculates the following metrics for each generated text:

1. **Word Repetition Rate**: Percentage of words that appear more than once
2. **Unique Word Ratio**: Ratio of unique words to total words
3. **Average Word Frequency**: Average number of times each word appears
4. **Max Word Frequency**: Maximum number of times any single word appears
5. **Average Verses**: Mean number of verses (lines) in generated poems

## Usage

### Quick Start

```bash
# Navigate to the grid_search directory
cd evaluation/grid_search

# Run with default parameters
bash run_grid_search.sh --model_path /path/to/your/model

# Example with specific model
bash run_grid_search.sh --model_path /path/to/checkpoints/Qwen3-8B/curriculum/4tasks/checkpoint-36000
```

### Custom Configuration

```bash
# From the grid_search directory
cd evaluation/grid_search

# Customize the grid search
bash run_grid_search.sh \
    --model_path /path/to/model \
    --n_samples 50 \
    --tasks "generation continuation" \
    --repetition_penalties "1.0 1.05 1.10 1.15" \
    --temperatures "0.0 0.5 0.7 1.0"
```

### Python Script Direct Usage

```bash
# From the grid_search directory
cd evaluation/grid_search

python decoding_grid_search.py \
    --model_path /path/to/model \
    --data_path ../../data/dialectical_IFT_DATA/tsv \
    --output_path ./grid_search_results \
    --best_config_dir ./best_decoding_parameters \
    --tasks generation continuation corruption \
    --repetition_penalties 1.0 1.10 1.15 \
    --temperatures 0.0 0.7 \
    --n_samples 100
```

## Command Line Arguments

### Shell Script (`run_grid_search.sh`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to model checkpoint (required) | - |
| `--data_path` | Base path to data directory | `../../data/dialectical_IFT_DATA/tsv` |
| `--output_path` | Base path for outputs | `./grid_search_results` |
| `--best_config_dir` | Directory to save best configurations | `./best_decoding_parameters` |
| `--tasks` | Space-separated tasks | `generation continuation corruption` |
| `--repetition_penalties` | Space-separated penalties | `1.0 1.10 1.15` |
| `--temperatures` | Space-separated temperatures | `0.0 0.7` |
| `--n_samples` | Number of samples per task | `100` |

### Python Script (`decoding_grid_search.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to model checkpoint (required) | - |
| `--data_path` | Base path to data directory | `../../data/dialectical_IFT_DATA/tsv` |
| `--output_path` | Base path for outputs | `./grid_search_results` |
| `--best_config_dir` | Directory to save best configurations | `./best_decoding_parameters` |
| `--tasks` | List of tasks | `[generation, continuation, corruption]` |
| `--repetition_penalties` | List of penalties | `[1.0, 1.10, 1.15]` |
| `--temperatures` | List of temperatures | `[0.0, 0.7]` |
| `--n_samples` | Number of samples per task | `100` |

## Output Structure

### Grid Search Results
```
grid_search_results/
├── sampled_data/                      # Randomly sampled data for testing
│   ├── generation/
│   │   └── test/
│   │       └── generation_ift.tsv
│   ├── continuation/
│   └── corruption/
└── grid_search_outputs/               # Raw inference outputs
    ├── generation/
    │   ├── temp_0.0_rp_1.0/
    │   ├── temp_0.0_rp_1.10/
    │   ├── temp_0.0_rp_1.15/
    │   ├── temp_0.7_rp_1.0/
    │   ├── temp_0.7_rp_1.10/
    │   └── temp_0.7_rp_1.15/
    ├── continuation/
    └── corruption/
```

### Best Configuration Results
```
best_decoding_parameters/
├── <model_name>_grid_search_results.json    # All results in JSON
├── <model_name>_best_config.json            # Best configurations per task
└── <model_name>_summary.txt                 # Human-readable summary
```

## Output Files

### 1. `<model_name>_grid_search_results.json`
Complete results for all configurations tested, including:
- Hyperparameter values
- All calculated metrics
- Error messages (if any)

### 2. `<model_name>_best_config.json`
Best hyperparameter configuration for each task, including:
- Best temperature
- Best repetition penalty
- Composite score
- All metrics for the best configuration

### 3. `<model_name>_summary.txt`
Human-readable summary with:
- Best configuration per task
- Key metrics
- Comparison across tasks

## How Best Configuration is Selected

The script uses a **composite score** to rank configurations:

```
Composite Score = 0.5 * (Repetition Rate / 100) 
                + 0.3 * (1 - Unique Word Ratio)
                + 0.2 * ((Avg Word Frequency - 1) / 10)
```

**Weights:**
- 50% - Repetition Rate (lower is better)
- 30% - Unique Word Ratio (higher is better)
- 20% - Word Frequency (lower is better)

The configuration with the **lowest composite score** is selected as the best.

## Example Output

### Console Output
```
================================================================================
GRID SEARCH SUMMARY
================================================================================

GENERATION:
  Best Config: temp_0.7_rp_1.15
    Temperature: 0.7
    Repetition Penalty: 1.15
    Repetition Rate: 23.45%
    Unique Word Ratio: 0.8234
    Avg Verses: 14.32

CONTINUATION:
  Best Config: temp_0.0_rp_1.10
    Temperature: 0.0
    Repetition Penalty: 1.10
    Repetition Rate: 18.76%
    Unique Word Ratio: 0.8567
    Avg Verses: 8.45
```

### Best Config JSON
```json
{
  "generation": {
    "config_name": "temp_0.7_rp_1.15",
    "composite_score": 0.1523,
    "temperature": 0.7,
    "repetition_penalty": 1.15,
    "metrics": {
      "n_samples": 100,
      "avg_word_repetition_rate": 23.45,
      "avg_unique_word_ratio": 0.8234,
      "avg_word_frequency": 1.21,
      "avg_verses": 14.32,
      "std_verses": 3.45,
      "avg_total_words": 145.23,
      "avg_unique_words": 119.56
    }
  },
  "continuation": {
    ...
  }
}
```

## Workflow

1. **Sample Data**: Randomly selects 100 examples from each task
2. **Generate Responses**: Uses `inference.sh` to generate model outputs for each hyperparameter combination
3. **Calculate Metrics**: Analyzes generated text for repetition and quality
4. **Rank Configurations**: Computes composite scores for all configurations
5. **Save Results**: Saves detailed results and best configurations

## Tips for Interpretation

### Repetition Rate
- **< 20%**: Excellent diversity
- **20-30%**: Good diversity
- **30-40%**: Moderate repetition
- **> 40%**: High repetition (problematic)

### Unique Word Ratio
- **> 0.85**: Excellent diversity
- **0.75-0.85**: Good diversity
- **0.65-0.75**: Moderate diversity
- **< 0.65**: Poor diversity

### Temperature vs Repetition Penalty
- **Low Temperature (0.0)**: More deterministic, may need higher repetition penalty
- **Higher Temperature (0.7)**: More diverse naturally, may work with lower repetition penalty
- **Repetition Penalty**: Start with 1.0 (no penalty), increase if repetition is high

## Troubleshooting

### Issue: "No outputs.jsonl found"
**Solution**: Check that inference.sh ran successfully. Verify the output path structure.

### Issue: High repetition across all configurations
**Solution**: Try higher repetition penalties (1.2, 1.3) or different temperature values.

### Issue: Very short or very long poems
**Solution**: Adjust `max_new_tokens` in the inference parameters.

### Issue: Script fails during inference
**Solution**: Check GPU availability and model path. Verify data files exist.

## Dependencies

- Python 3.8+
- pandas
- numpy
- tqdm
- vLLM (for inference)
- transformers

## Related Files

- `inference.sh` - Main inference script
- `vllm_inference.py` - vLLM-based inference
- `inference_utils.py` - Utility functions
- `args_parser.py` - Argument parsing

## Notes

- The script uses the `inference.sh` wrapper to ensure consistent inference setup
- All inference is done using vLLM for efficiency
- Results are automatically saved with model name for easy tracking
- The script handles interruptions gracefully and logs errors

## Future Enhancements

Potential improvements:
- Add support for top_p and top_k in grid search
- Include perplexity metrics
- Add human evaluation integration
- Support for beam search parameters
- Multi-GPU parallelization of different configurations

## Contact

For issues or questions, please refer to the main project README or contact the development team.
