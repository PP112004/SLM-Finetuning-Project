from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.ps06.data import gsm8k_prompt_record, load_gsm8k_split
from src.ps06.rewards import score_math_response


DEFAULT_MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"


def format_chat_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {
                "role": "system",
                "content": "You are a careful math reasoning assistant. Return only one final numeric answer after your reasoning.",
            },
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny GSM8K baseline with MLX-LM.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX model repo or local model path")
    parser.add_argument("--split", default="test", help="GSM8K split")
    parser.add_argument("--limit", type=int, default=3, help="Number of examples to evaluate")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generated tokens per problem")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument(
        "--output",
        default="outputs/baselines/2026-04-25-gsm8k-baseline-mlx.jsonl",
        help="JSONL output path",
    )
    args = parser.parse_args()

    print(f"Loading GSM8K/{args.split} with limit={args.limit}")
    dataset = load_gsm8k_split(split=args.split, limit=args.limit)

    print(f"Loading model: {args.model}")
    print("First run can take several minutes because the model is downloaded once.")
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temperature)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    rows = []
    for idx, example in enumerate(dataset, start=1):
        record = gsm8k_prompt_record(example)
        prompt = format_chat_prompt(tokenizer, record["prompt"])

        print(f"\n[{idx}/{len(dataset)}] Generating...")
        prediction = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        reward = score_math_response(prediction, record["answer"])

        row = {
            "idx": idx - 1,
            "model": args.model,
            "question": record["question"],
            "reference_answer": record["answer"],
            "prediction": prediction,
            "extracted_prediction": reward.extracted_prediction,
            "extracted_reference": reward.extracted_reference,
            "accuracy_reward": reward.accuracy_reward,
            "format_reward": reward.format_reward,
            "total_reward": reward.total,
        }
        rows.append(row)

        print(f"prediction: {reward.extracted_prediction}")
        print(f"reference:  {reward.extracted_reference}")
        print(f"correct:    {bool(reward.accuracy_reward)}")

        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    correct = sum(row["accuracy_reward"] for row in rows)
    accuracy = correct / len(rows) if rows else 0.0
    print()
    print(f"Evaluated: {len(rows)}")
    print(f"Correct: {correct:.0f}/{len(rows)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
