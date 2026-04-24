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


def build_dummy_prediction(reference_answer: str) -> str:
    final_answer = reference_answer.rsplit("####", maxsplit=1)[-1].strip()
    return f"<think>Using the provided reference for smoke testing.</think>\n{final_answer}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test GSM8K loading and reward scoring.")
    parser.add_argument("--split", default="test", help="GSM8K split to load")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to process")
    parser.add_argument(
        "--output",
        default="outputs/smoke/2026-04-25-gsm8k-reward-smoke.jsonl",
        help="JSONL output path",
    )
    args = parser.parse_args()

    dataset = load_gsm8k_split(split=args.split, limit=args.limit)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, example in enumerate(dataset):
        record = gsm8k_prompt_record(example)
        prediction = build_dummy_prediction(record["answer"])
        reward = score_math_response(prediction, record["answer"])
        rows.append(
            {
                "idx": idx,
                "question": record["question"],
                "prompt": record["prompt"],
                "reference_answer": record["answer"],
                "prediction": prediction,
                "extracted_prediction": reward.extracted_prediction,
                "extracted_reference": reward.extracted_reference,
                "accuracy_reward": reward.accuracy_reward,
                "format_reward": reward.format_reward,
                "total_reward": reward.total,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    correct = sum(row["accuracy_reward"] for row in rows)
    print(f"Loaded {len(rows)} examples from GSM8K/{args.split}")
    print(f"Accuracy reward sum: {correct:.0f}/{len(rows)}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
