from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from prompt import build_messages
from tqdm import tqdm
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
)


@dataclass
class GenerationParams:
  max_new_tokens: int = 256
  temperature: float = 0.1
  top_p: float = 0.9
  repetition_penalty: float = 1.05


@dataclass
class PipelineConfig:
  model_name: str = "openai/gpt-oss-20b"
  dataset_path: Path = Path("test_dataset.csv")
  output_path: Path = Path("resume_inference_results.csv")
  batch_size: int = 5
  seed: int = 42
  limit: Optional[int] = None
  dtype: str = "bfloat16"
  use_8bit: bool = False
  generation: GenerationParams = GenerationParams()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Batch resume review pipeline powered by GPT-OSS-20B."
  )
  parser.add_argument("--dataset-path", type=Path, default=Path("test_dataset.csv"))
  parser.add_argument("--output-path", type=Path, default=Path("resume_inference_results.csv"))
  parser.add_argument("--model-name", type=str, default="openai/gpt-oss-20b")
  parser.add_argument("--batch-size", type=int, default=5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (useful for smoke tests).")
  parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
  parser.add_argument("--use-8bit", action="store_true", help="Load the model with 8-bit quantization via bitsandbytes.")
  parser.add_argument("--max-new-tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=0.1)
  parser.add_argument("--top-p", type=float, default=0.9)
  parser.add_argument("--repetition-penalty", type=float, default=1.05)
  parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token. Falls back to HUGGINGFACE_TOKEN env var.")
  parser.add_argument("--log-level", type=str, default="INFO")
  return parser.parse_args()


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path: Path, limit: Optional[int]) -> pd.DataFrame:
  if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

  df = pd.read_csv(dataset_path)
  required_columns = {"about_me", "Human_flag"}
  missing = required_columns - set(df.columns)
  if missing:
    raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

  df = df.dropna(subset=["about_me", "Human_flag"])
  if limit:
    df = df.head(limit)
  return df.reset_index(drop=True)


def prepare_model_and_tokenizer(
  model_name: str,
  dtype: str,
  use_8bit: bool,
  hf_token: Optional[str],
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
  torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

  tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    use_fast=True,
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"

  quantization_config = None
  if use_8bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    logging.info("Loading model with 8-bit quantization.")

  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch_dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
  )

  return tokenizer, model


def chunk_records(records: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
  for idx in range(0, len(records), batch_size):
    yield records[idx : idx + batch_size]


def render_prompts(tokenizer: AutoTokenizer, about_mes: Sequence[str]) -> List[str]:
  rendered: List[str] = []
  for about_me in about_mes:
    messages = build_messages(about_me)
    rendered_prompt = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )
    rendered.append(rendered_prompt)
  return rendered


def safe_generate(
  model: AutoModelForCausalLM,
  generation_config: GenerationConfig,
  model_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
  batch_size = model_inputs["input_ids"].shape[0]
  try:
    return model.generate(
      **model_inputs,
      generation_config=generation_config,
    )
  except torch.cuda.OutOfMemoryError:
    if batch_size == 1:
      logging.error("Out of memory even for a single sample.")
      raise
    logging.warning("CUDA OOM for batch size %s. Retrying sequentially.", batch_size)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    outputs = []
    for idx in range(batch_size):
      single_inputs = {k: v[idx : idx + 1].clone() for k, v in model_inputs.items()}
      out = safe_generate(model, generation_config, single_inputs)
      outputs.append(out)
    return torch.cat(outputs, dim=0)


FINAL_MARKER = "<|channel|>final<|message|>"
STOP_TOKENS = ["<|end|>", "<|return|>", "<|call|>"]


def extract_final_channel(text: str) -> str:
  idx = text.rfind(FINAL_MARKER)
  if idx == -1:
    return ""
  start = idx + len(FINAL_MARKER)
  end = len(text)
  for token in STOP_TOKENS:
    token_idx = text.find(token, start)
    if token_idx != -1:
      end = min(end, token_idx)
  return text[start:end].strip()


def parse_classification(final_text: str) -> str:
  if not final_text:
    return "unknown"

  lowered = final_text.lower()
  try:
    payload = json.loads(final_text)
    label = payload.get("classification")
    if label:
      return label.strip().lower()
  except json.JSONDecodeError:
    pass

  for label in ("good", "bad"):
    if re.search(rf"\\b{label}\\b", lowered):
      return label
  return "unknown"


def main() -> None:
  args = parse_args()
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
  )

  config = PipelineConfig(
    model_name=args.model_name,
    dataset_path=args.dataset_path,
    output_path=args.output_path,
    batch_size=args.batch_size,
    seed=args.seed,
    limit=args.limit,
    dtype=args.dtype,
    use_8bit=args.use_8bit,
    generation=GenerationParams(
      max_new_tokens=args.max_new_tokens,
      temperature=args.temperature,
      top_p=args.top_p,
      repetition_penalty=args.repetition_penalty,
    ),
  )

  set_seed(config.seed)

  df = load_dataset(config.dataset_path, config.limit)
  if df.empty:
    raise ValueError("Dataset is empty after filtering.")

  logging.info("Loaded %d samples from %s", len(df), config.dataset_path)

  hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
  if hf_token:
    logging.info("Using Hugging Face token from CLI/env for gated model access.")
  else:
    logging.warning("No Hugging Face token provided. Ensure the model is public or cached locally.")

  tokenizer, model = prepare_model_and_tokenizer(
    config.model_name,
    config.dtype,
    config.use_8bit,
    hf_token,
  )

  generation_config = GenerationConfig(
    max_new_tokens=config.generation.max_new_tokens,
    temperature=config.generation.temperature,
    top_p=config.generation.top_p,
    repetition_penalty=config.generation.repetition_penalty,
    do_sample=config.generation.temperature > 0,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
  )

  records = df.to_dict("records")
  total_prompt_tokens = 0
  total_completion_tokens = 0
  results: List[Dict[str, Any]] = []

  for chunk in tqdm(
    list(chunk_records(records, config.batch_size)),
    desc="Running inference",
  ):
    about_mes = [row["about_me"] for row in chunk]
    rendered_prompts = render_prompts(tokenizer, about_mes)

    model_inputs = tokenizer(
      rendered_prompts,
      return_tensors="pt",
      padding=True,
      truncation=True,
    )
    prompt_lengths = model_inputs["attention_mask"].sum(dim=1)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    batch_start = time.perf_counter()
    outputs = safe_generate(model, generation_config, model_inputs)
    batch_latency = time.perf_counter() - batch_start

    for idx, row in enumerate(chunk):
      prompt_tokens = int(prompt_lengths[idx].item())
      generated_ids = outputs[idx][prompt_tokens:]
      completion_tokens = generated_ids.shape[0]

      raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
      final_channel = extract_final_channel(raw_output)
      prediction = parse_classification(final_channel)
      human_label = str(row["Human_flag"]).strip().lower()

      total_prompt_tokens += prompt_tokens
      total_completion_tokens += completion_tokens

      result_row = {
        "about_me": row["about_me"],
        "Human_flag": row["Human_flag"],
        "raw_output": raw_output,
        "final_channel": final_channel,
        "model_prediction": prediction,
        "is_correct": prediction == human_label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_seconds": batch_latency / max(1, len(chunk)),
      }
      results.append(result_row)

  accuracy = sum(1 for row in results if row["is_correct"]) / len(results)
  logging.info("Accuracy: %.2f%%", accuracy * 100)
  logging.info(
    "Token usage — prompt: %d | completion: %d | total samples: %d",
    total_prompt_tokens,
    total_completion_tokens,
    len(results),
  )

  output_path = config.output_path
  output_path.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(results).to_csv(output_path, index=False)
  logging.info("Saved detailed results to %s", output_path.resolve())


if __name__ == "__main__":
  main()

