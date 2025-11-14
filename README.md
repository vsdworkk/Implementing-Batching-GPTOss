# Resume Review Batch Pipeline

This repository runs batch inference with `openai/gpt-oss-20b` to classify resume “about me” sections as `good` or `bad`. It follows the harmony chat template so that the model outputs both reasoning (analysis channel) and a structured verdict (final channel).

## 1. Requirements
- **Hardware**: Single NVIDIA H100 (80 GB) or any Hopper GPU that supports MXFP4. Expect ~20 GB VRAM usage in MXFP4 and ~48 GB in bfloat16.
- **OS / Drivers**: Linux or macOS with CUDA 12.4+ drivers (for Linux) and matching cuDNN; verify `nvidia-smi` shows the H100.
- **Python**: 3.10 or newer.
- **Dependencies**: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `pandas`, `tqdm`, `numpy`, `python-dotenv`. Installable via `pip install -r requirements.txt`.

## 2. Environment Setup
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Hugging Face authentication
1. Create/read a Hugging Face access token with **read** scope.
2. Store it locally (never commit secrets):
   ```bash
   export HUGGINGFACE_TOKEN=hf_xxx   # or place inside .env.local
   ```
3. (Optional) Login once via CLI: `huggingface-cli login`.

## 3. Data & Prompting
- Input CSV: `test_dataset.csv` with columns `about_me` and `Human_flag`.
- Prompting logic lives in `prompt.py` and builds harmony-compliant messages:
  - **System**: declares reasoning level, valid channels.
  - **Developer**: instructions + JSON response schema.
  - **User**: injects each resume summary.

## 4. Running Inference
`main.py` orchestrates the full pipeline (load data → batch prompts → generate → parse → export).

Common flags:
```
python main.py \
  --dataset-path test_dataset.csv \
  --output-path resume_inference_results.csv \
  --model-name openai/gpt-oss-20b \
  --batch-size 5 \
  --max-new-tokens 256 \
  --temperature 0.1 \
  --top-p 0.9 \
  --repetition-penalty 1.05 \
  --seed 42
```

Smoke test two samples before the full batch:
```
python main.py --limit 2 --batch-size 2 --log-level DEBUG
```

Useful flags:
- `--hf-token`: override the `HUGGINGFACE_TOKEN` env var.
- `--use-8bit`: load via bitsandbytes if VRAM is tight.
- `--dtype float16`: switch precision (default is bfloat16 for H100).

## 5. Outputs & Metrics
- Results CSV columns:
  - `about_me`, `Human_flag`
  - `raw_output`: full decoded assistant output (analysis + final)
  - `final_channel`: extracted JSON payload from `<|channel|>final<|message|>`
  - `model_prediction`, `is_correct`
  - `prompt_tokens`, `completion_tokens`, `latency_seconds`
- Console logs include accuracy, token totals, and warnings if parsing fails.

## 6. Operational Notes
- **Batching**: Fixed batch size of 5; the runner falls back to sequential sampling if CUDA OOM occurs.
- **Parsing**: `extract_final_channel` grabs everything between `<|channel|>final<|message|>` and the next stop token. `parse_classification` expects a JSON payload but gracefully degrades to keyword matching.
- **Reproducibility**: Seeded torch/random/NumPy; log config printed at startup.
- **Logging**: Uses Python logging + tqdm progress bar. Increase verbosity with `--log-level DEBUG`.
- **Testing**: Unit-style hooks are in pure Python (prompt construction, parser). For rapid validation, keep `--limit` low and `--max-new-tokens` ~64.

## 7. Next Steps Checklist
1. Provision the H100 box and install CUDA 12.4 drivers.
2. Create/activate the Python environment & install requirements.
3. Export `HUGGINGFACE_TOKEN`.
4. Run a two-sample smoke test (`--limit 2`) to validate parsing/output.
5. Execute the full batch and inspect `resume_inference_results.csv`.
6. Review accuracy logs and spot-check `raw_output` vs. `final_channel`.

