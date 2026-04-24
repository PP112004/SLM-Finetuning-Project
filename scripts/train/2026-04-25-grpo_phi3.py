"""GRPO training for Phi-3-Mini on GSM8K (+ MATH optional) with rule-based rewards.

Reward = accuracy_reward + 0.5 * format_reward, where:
  - accuracy_reward = 1 if extracted number matches GSM8K gold, else 0
  - format_reward   = 0.5 if response contains a <think>...</think> block

Starting point can be either the HF base model or the SFT LoRA adapter from
scripts/train/2026-04-25-sft_phi3.py. Target: H100 80 GB, bf16 + LoRA.

Usage:
  # Dry-run (CPU; validates config + reward wiring on a few GSM8K rows):
  python scripts/train/2026-04-25-grpo_phi3.py --dry-run

  # Real training (GPU):
  python scripts/train/2026-04-25-grpo_phi3.py \
      --base-model microsoft/Phi-3-mini-4k-instruct \
      --sft-adapter outputs/sft/phi3-mini-ft \
      --output-dir outputs/grpo/phi3-mini-rl \
      --num-generations 4 --per-device-batch-size 2 \
      --learning-rate 1e-5 --num-train-epochs 2

Requires trl >= 0.12 (GRPOTrainer).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.ps06.data import load_gsm8k_split, make_reasoning_prompt
from src.ps06.rewards import has_think_format, extract_final_number, extract_gsm8k_reference


DEFAULT_BASE = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_OUTPUT = "outputs/grpo/phi3-mini-rl"


@dataclass
class GRPOConfig:
    base_model: str
    sft_adapter: Path | None
    output_dir: Path
    num_train_epochs: float
    per_device_batch_size: int
    grad_accum: int
    learning_rate: float
    kl_beta: float
    temperature: float
    top_p: float
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    logging_steps: int
    save_steps: int
    report_to: str
    train_limit: int | None


# --- Reward functions -------------------------------------------------------
# TRL's GRPOTrainer expects reward_funcs to be callables returning list[float]
# given `completions` and any dataset columns passed through (here: "answer").

def reward_accuracy(completions, answer, **kwargs):  # noqa: ARG001
    """1.0 when the extracted numeric answer matches the GSM8K gold."""
    rewards: list[float] = []
    for comp, ref in zip(completions, answer):
        # TRL may pass chat-format list of messages per completion; normalize to str.
        if isinstance(comp, list):
            text = "".join(m.get("content", "") for m in comp if isinstance(m, dict))
        else:
            text = str(comp)
        pred = extract_final_number(text)
        gold = extract_gsm8k_reference(ref)
        rewards.append(1.0 if pred is not None and gold is not None and pred == gold else 0.0)
    return rewards


def reward_format(completions, **kwargs):  # noqa: ARG001
    """0.5 when the response contains a <think>...</think> block."""
    rewards: list[float] = []
    for comp in completions:
        if isinstance(comp, list):
            text = "".join(m.get("content", "") for m in comp if isinstance(m, dict))
        else:
            text = str(comp)
        rewards.append(0.5 if has_think_format(text) else 0.0)
    return rewards


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--sft-adapter", default=None, type=Path,
                   help="Optional path to an SFT LoRA adapter to warm-start from.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT, type=Path)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--train-limit", type=int, default=None,
                   help="Cap training examples (for smoke / debug runs).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_dataset(train_limit: int | None):
    """Load GSM8K train split and project to {prompt, answer}."""
    from datasets import Dataset  # local import so dry-run stays light

    raw = load_gsm8k_split(split="train", limit=train_limit)
    records = []
    for ex in raw:
        records.append({
            "prompt": make_reasoning_prompt(ex["question"]),
            "answer": ex["answer"],
        })
    return Dataset.from_list(records)


def dry_run(cfg: GRPOConfig) -> int:
    print("=== GRPO dry-run ===")
    print(f"base_model: {cfg.base_model}")
    print(f"sft_adapter: {cfg.sft_adapter}")
    print(f"epochs={cfg.num_train_epochs} bs={cfg.per_device_batch_size} "
          f"accum={cfg.grad_accum} lr={cfg.learning_rate} kl_beta={cfg.kl_beta}")
    print(f"num_generations={cfg.num_generations} temp={cfg.temperature} top_p={cfg.top_p}")
    print(f"max_prompt={cfg.max_prompt_length} max_completion={cfg.max_completion_length}")
    print(f"lora(r={cfg.lora_r}, a={cfg.lora_alpha})")

    ds = build_dataset(train_limit=3)
    print(f"dataset columns: {ds.column_names}")
    print(f"first example prompt (head): {ds[0]['prompt'][:120]!r}...")

    # Exercise reward functions with a mix of correct/incorrect/malformed.
    fake_completions = [
        "<think>9*2</think>\nAnswer: 18",
        "No reasoning, just 18.",
        "<think>nope</think> Answer: 42",
    ]
    gold = ["#### 18", "#### 18", "#### 18"]
    acc = reward_accuracy(fake_completions, gold)
    fmt = reward_format(fake_completions)
    print(f"reward_accuracy = {acc}  (expect [1.0, 1.0, 0.0])")
    print(f"reward_format   = {fmt}  (expect [0.5, 0.0, 0.5])")

    expected_acc = [1.0, 1.0, 0.0]
    expected_fmt = [0.5, 0.0, 0.5]
    if acc != expected_acc or fmt != expected_fmt:
        print("  [FAIL] reward outputs do not match expectations")
        return 1
    print("Dry-run OK. No model loaded, no training executed.")
    return 0


def real_run(cfg: GRPOConfig) -> int:
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

    print(f"Loading tokenizer: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model bf16: {cfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if cfg.sft_adapter is not None and Path(cfg.sft_adapter).exists():
        print(f"Attaching SFT LoRA adapter: {cfg.sft_adapter}")
        model = PeftModel.from_pretrained(model, str(cfg.sft_adapter), is_trainable=True)

    train_ds = build_dataset(train_limit=cfg.train_limit)
    print(f"GRPO train examples: {len(train_ds)}")

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TRLGRPOConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=True,
        seed=cfg.seed,
        gradient_checkpointing=True,
        beta=cfg.kl_beta,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        report_to=cfg.report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        reward_funcs=[reward_accuracy, reward_format],
        peft_config=lora_cfg if cfg.sft_adapter is None else None,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    print(f"Saved GRPO LoRA adapter to {cfg.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    cfg = GRPOConfig(
        base_model=args.base_model,
        sft_adapter=args.sft_adapter,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        temperature=args.temperature,
        top_p=args.top_p,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        train_limit=args.train_limit,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return dry_run(cfg)
    return real_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
