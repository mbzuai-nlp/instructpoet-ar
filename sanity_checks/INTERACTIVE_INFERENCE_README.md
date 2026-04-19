# Interactive Inference Tool - Debug Repetition Issues

This tool allows you to interactively test your model's generation quality and debug repetition issues.

## Features

- **Interactive Mode**: Press Enter to cycle through test examples
- **Side-by-side Comparison**: See input, expected output, and model generation together
- **Improved Decoding Settings**: Uses anti-repetition parameters to reduce repeated text
- **Repetition Detection**: Automatically warns when repeated lines are detected
- **Flexible Navigation**: Jump to specific examples or regenerate current one

## Quick Start

### Option 1: Using the Shell Script (Easiest)

```bash
./run_interactive_inference.sh
```

This uses the default configuration for your checkpoint.

### Option 2: Custom Parameters

```bash
python interactive_inference.py \
    --model_path /path/to/model \
    --test_file /path/to/test.tsv \
    --temperature 0.8 \
    --repetition_penalty 1.1 \
    --top_p 0.9
```

## Anti-Repetition Parameters

The tool uses these improved decoding settings to reduce repetition:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.8 | Higher = more diverse (0.7-1.0 recommended) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Top-k sampling |
| `repetition_penalty` | 1.1 | Penalize repetitions (1.05-1.3 range) |
| `no_repeat_ngram_size` | 3 | Block repeating n-grams |
| `max_new_tokens` | 80 | Maximum generation length |

## Interactive Commands

While running the tool:

- **Press Enter**: Generate for the next example
- **Type a number**: Jump to that example (e.g., `42` for example 42)
- **Type `r`**: Regenerate the current example
- **Type `q`** or **`quit`**: Exit the tool

## Testing Different Settings

### If repetition persists, try:

1. **Increase temperature**:
```bash
python interactive_inference.py ... --temperature 0.9
```

2. **Increase repetition penalty**:
```bash
python interactive_inference.py ... --repetition_penalty 1.2
```

3. **Use both**:
```bash
python interactive_inference.py ... --temperature 0.9 --repetition_penalty 1.3
```

4. **Adjust top_p for more diversity**:
```bash
python interactive_inference.py ... --top_p 0.85
```

## Arguments

### Required:
- `--model_path`: Path to your model checkpoint
- `--test_file`: Path to the TSV file with test data

### Optional:
- `--adapter_path`: Path to LoRA adapter (if using)
- `--start_idx`: Start from specific example (default: 0)
- `--prompt_type`: `chat` or `instruction` (default: chat)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_p`: Nucleus sampling (default: 0.9)
- `--top_k`: Top-k sampling (default: 50)
- `--repetition_penalty`: Penalty for repetitions (default: 1.1)
- `--no_repeat_ngram_size`: N-gram blocking (default: 3)
- `--max_new_tokens`: Max tokens to generate (default: 80)
- `--tensor_parallel_size`: Number of GPUs (default: 1)

## Example Session

```
================================================================================
EXAMPLE 5 / 100
================================================================================

INPUT:
--------------------------------------------------------------------------------
[Arabic poetry input text here]
--------------------------------------------------------------------------------

EXPECTED OUTPUT:
--------------------------------------------------------------------------------
[Expected Arabic poetry output here]
--------------------------------------------------------------------------------

⏳ Generating...

MODEL GENERATION:
--------------------------------------------------------------------------------
[Model's generated text here]
--------------------------------------------------------------------------------

⚠️  WARNING: Detected repeated lines in generation!

[Press Enter for next | Type number to jump | 'r' to regenerate | 'q' to quit]
> 
```

## Troubleshooting

### If you see "No module named 'termcolor'":
```bash
pip install termcolor
```

### If vLLM complains about parameters:
Some parameters like `no_repeat_ngram_size` might not be supported in all vLLM versions. You can comment it out in the script.

### If you want to test specific examples:
```bash
python interactive_inference.py ... --start_idx 50
```

Then press Enter to go through examples starting from #50.

## Tips

1. **Compare with baseline**: Run once with default settings, then adjust parameters
2. **Test on diverse examples**: Use the jump feature to check different types of inputs
3. **Look for patterns**: If certain inputs always cause repetition, note their characteristics
4. **Experiment**: Try different combinations of temperature and repetition_penalty
5. **Document what works**: Keep track of which settings produce the best results

## Output Display

- 🟢 **Green**: Input text
- 🟡 **Yellow**: Expected output (ground truth)
- 🔵 **Blue**: Model generation
- 🔴 **Red**: Warnings/errors
