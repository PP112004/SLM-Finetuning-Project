from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a GSM8K prediction JSONL file.")
    parser.add_argument("path", help="Prediction JSONL path")
    parser.add_argument("--show-errors", type=int, default=5, help="Number of incorrect examples to show")
    args = parser.parse_args()

    path = Path(args.path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    total = len(rows)
    correct_rows = [row for row in rows if row.get("accuracy_reward") == 1.0]
    wrong_rows = [row for row in rows if row.get("accuracy_reward") != 1.0]
    accuracy = len(correct_rows) / total if total else 0.0

    print(f"File: {path}")
    print(f"Evaluated: {total}")
    print(f"Correct: {len(correct_rows)}/{total}")
    print(f"Accuracy: {accuracy:.2%}")

    if wrong_rows and args.show_errors:
        print()
        print(f"First {min(args.show_errors, len(wrong_rows))} errors:")
        for row in wrong_rows[: args.show_errors]:
            print("-" * 80)
            print(f"idx: {row['idx']}")
            print(f"question: {row['question']}")
            print(f"prediction: {row['extracted_prediction']}")
            print(f"reference:  {row['extracted_reference']}")
            generated = row.get("prediction", "").strip().replace("\n", " ")
            print(f"generated:  {generated[:500]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
