# Arabic Poetry vLLM Gradio App

Quick Gradio playground to try the Arabic poetry adapters on top of the base models using vLLM.

## Prerequisites
- GPU with enough memory for `Qwen/Qwen3-8B` or `humain-ai/ALLaM-7B-Instruct-preview`
- Python deps: `vllm`, `gradio`, `python-dotenv`, `transformers` (already used elsewhere in this repo)
- HF auth token available in the environment (`HF_TOKEN` or `HUGGINGFACE_TOKEN`)

Optional environment overrides:
- `QWEN3_BASE_MODEL` / `ALLAM_BASE_MODEL` to point to local base model copies
- `VLLM_TENSOR_PARALLEL`, `VLLM_GPU_UTIL`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS` to tune vLLM loading
- `GRADIO_SHARE` to toggle public sharing (defaults to `true`)

## Run
```bash
cd inference
python app.py
```
The UI listens on `0.0.0.0:7860` by default (override with `PORT`).
Set `GRADIO_SHARE=true` (default) to expose a public Gradio share link if the host allows it.

## Notes
- Adapters are hard-wired to the curriculum 4-task checkpoints:
  - `/path/to/checkpoints/Qwen3-8B/curriculum/4tasks/checkpoint-36000`
  - `/path/to/checkpoints/ALLaM-7B-Instruct-preview/curriculum/4tasks/checkpoint-32000`
- Prompts default to the `###Input/###Output` style for instruction mode and the bundled chat template for chat mode.
- The app auto-strips unsupported base weights from the adapter file into `adapter_model.vllm.safetensors` so vLLM LoRA loading succeeds.
