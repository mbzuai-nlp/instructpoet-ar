# Grid Search for Decoding Hyperparameters

This folder contains tools for optimizing decoding hyperparameters to reduce repetition in Arabic poetry generation.

## Quick Start

```bash
# Run grid search on your model
bash run_grid_search.sh --model_path /path/to/your/model

# View the best configurations found
python load_best_config.py --model_name your-model-name
```

## What This Does

The grid search automatically:
1. **Samples 100 random examples** from generation, continuation, and corruption tasks
2. **Tests different parameter combinations**:
   - Repetition penalties: 1.0, 1.10, 1.15
   - Temperatures: 0.0, 0.7
3. **Measures text quality**:
   - Word repetition rates
   - Unique word ratios
   - Average verse counts
4. **Saves best configurations** for each task

## Output

Results are saved in:
- `best_decoding_parameters/<model>_best_config.json` - Best parameters per task
- `best_decoding_parameters/<model>_summary.txt` - Human-readable summary
- `grid_search_results/` - Full inference outputs

## Example Usage

```bash
# Run with custom parameters
bash run_grid_search.sh \
    --model_path /path/to/checkpoint-36000 \
    --n_samples 50 \
    --tasks "generation continuation"

# Load and view best config
python load_best_config.py --model_name checkpoint-36000

# Generate inference command with best params
python load_best_config.py \
    --model_name checkpoint-36000 \
    --task generation \
    --model_path /path/to/model \
    --generate_command
```

## Files

- `decoding_grid_search.py` - Main grid search script
- `run_grid_search.sh` - Shell wrapper for easy execution
- `load_best_config.py` - Utility to view/use best configs
- `README_GRID_SEARCH.md` - **Detailed documentation**
- `best_decoding_parameters/` - Saved results directory

## Full Documentation

**See [README_GRID_SEARCH.md](README_GRID_SEARCH.md) for complete documentation**, including:
- Detailed explanation of metrics
- How best configurations are selected
- Troubleshooting guide
- Advanced usage examples

## Quick Tips

- **Start with defaults**: The default parameters work well for most cases
- **Check repetition**: If you see high repetition (>30%), increase repetition_penalty
- **Balance quality**: Higher repetition penalties reduce repetition but may affect fluency
- **Temperature matters**: Use 0.0 for deterministic, 0.7 for more diverse outputs
