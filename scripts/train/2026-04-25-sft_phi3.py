"""SFT warmup for Phi-3-Mini on CoT math data.

Designed for H100 80 GB (bf16 + LoRA, no 4-bit). A100-40GB fallback is possible
by uncommenting the 4-bit bitsandbytes block.

Expected input: a JSONL file where each line has at least `prompt` and
`completion` fields. `scripts/data/2026-04-25-prepare_sft_data.py` produces
this format from OpenMathInstruct-2.

Usage:
  # Sanity check config + data pipeline on CPU (no training, no model load):
  python scripts/train/2026-04-25-sft_phi3.py --dry-run \
      --train-file data/sft/phi3_sft.jsonl

  # Real training on the GPU host:
  python scripts/train/2026-04-25-sft_phi3.py \
      --train-file data/sft/phi3_sft.jsonl \
      --output-dir outputs/sft/phi3-mini-ft \
      --num-epochs 1 --per-device-batch-size 4 --grad-accum 4 \
      --learning-rate 2e-5

This script intentionally avoids importing torch / transformers at module import
time so that --dry-run can be exercised on a Mac without CUDA libraries present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft/phi3-mini-ft"


@dataclass
class SFTConfig:
    model_name: str
    train_file: Path
    eval_file: Path | None
    output_dir: Path
    num_epochs: float
    per_device_batch_size: int
    grad_accum: int
    learning_rate: float
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    bf16: bool
    report_to: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train-file", required=True, type=Path)
    p.add_argument("--eval-file", default=None, type=Path)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + data only; do not load model or train.")
    return p.parse_args()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def format_example(row: dict) -> dict[str, str]:
    """Convert a {prompt, completion} row into a single chat-formatted string.

    The completion is expected to already contain the <think>...</think> block
    plus the final answer. We do NOT re-wrap it here; prepare_sft_data handles
    formatting.
    """
    prompt = row["prompt"]
    completion = row["completion"]
    return {"text": prompt + completion}


def dry_run(cfg: SFTConfig) -> int:
    print("=== SFT dry-run ===")
    print(f"model: {cfg.model_name}")
    print(f"train_file: {cfg.train_file}")
    if not cfg.train_file.exists():
        print(f"  [WARN] train file does not exist yet; run prepare_sft_data first")
        return 0
    rows = load_jsonl(cfg.train_file, limit=5)
    print(f"  loaded {len(rows)} sample rows")
    for i, row in enumerate(rows):
        missing = [k for k in ("prompt", "completion") if k not in row]
        if missing:
            print(f"  [FAIL] row {i} missing keys: {missing}")
            return 1
        joined = format_example(row)["text"]
        print(f"  row[{i}] joined_len={len(joined)} chars")
    print(f"output_dir: {cfg.output_dir}")
    print(f"epochs={cfg.num_epochs} bs={cfg.per_device_batch_size} accum={cfg.grad_accum} "
          f"lr={cfg.learning_rate} lora(r={cfg.lora_r}, a={cfg.lora_alpha})")
    print("Dry-run OK. No model loaded, no training executed.")
    return 0


def real_run(cfg: SFTConfig) -> int:
    # Heavy imports live here so dry-run does not require torch/transformers.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {cfg.model_name} (bf16={cfg.bf16})")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    train_rows = load_jsonl(cfg.train_file)
    train_ds = Dataset.from_list([format_example(r) for r in train_rows])

    eval_ds = None
    if cfg.eval_file and cfg.eval_file.exists():
        eval_rows = load_jsonl(cfg.eval_file)
        eval_ds = Dataset.from_list([format_example(r) for r in eval_rows])

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TRLSFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        bf16=cfg.bf16,
        seed=cfg.seed,
        gradient_checkpointing=True,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to=cfg.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    print(f"Saved SFT LoRA adapter to {cfg.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    cfg = SFTConfig(
        model_name=args.model,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        report_to=args.report_to,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return dry_run(cfg)
    return real_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
