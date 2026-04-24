"""Prepare a small, high-quality SFT dataset from OpenMathInstruct-2.

OpenMathInstruct-2 (nvidia/OpenMathInstruct-2) has ~14M rows. We stream it and
keep ~5-10K examples after filtering for:
  - non-empty problem + generated_solution + expected_answer
  - length within [min_len, max_len] characters
  - solution contains at least one digit and ends with a non-punctuation char
  - dedup on the problem text (exact match)

The output JSONL has the format expected by scripts/train/2026-04-25-sft_phi3.py:
  {"prompt": "...Problem: {q}...Answer:", "completion": "<think>{sol}</think>\\nAnswer: {a}"}

Usage:
  # Streamed prep — only reads what it needs:
  python scripts/data/2026-04-25-prepare_sft_data.py \
      --output data/sft/phi3_sft.jsonl \
      --num-keep 5000 \
      --max-scan 150000

  # Tiny smoke (no network hammering):
  python scripts/data/2026-04-25-prepare_sft_data.py \
      --output data/sft/phi3_sft_smoke.jsonl \
      --num-keep 100 \
      --max-scan 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.ps06.data import make_reasoning_prompt


DATASET_NAME = "nvidia/OpenMathInstruct-2"
DATASET_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num-keep", type=int, default=5000,
                   help="Target number of filtered examples to keep.")
    p.add_argument("--max-scan", type=int, default=150_000,
                   help="Max rows to stream from the dataset (hard cap).")
    p.add_argument("--min-len", type=int, default=40,
                   help="Min total (problem+solution) characters.")
    p.add_argument("--max-len", type=int, default=4000,
                   help="Max total characters (keeps things <2048 tokens after tmpl).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sources", nargs="*", default=None,
                   help="If given, only keep rows whose problem_source matches one "
                        "of these (e.g. 'math' 'gsm8k').")
    p.add_argument("--report-every", type=int, default=5000)
    return p.parse_args()


def good_row(row: dict, min_len: int, max_len: int, seen_problems: set[str],
             sources: list[str] | None) -> bool:
    problem = (row.get("problem") or "").strip()
    solution = (row.get("generated_solution") or "").strip()
    answer = str(row.get("expected_answer") or "").strip()
    source = row.get("problem_source") or ""

    if not (problem and solution and answer):
        return False
    if sources and source not in sources:
        return False
    total_len = len(problem) + len(solution)
    if total_len < min_len or total_len > max_len:
        return False
    if not any(ch.isdigit() for ch in solution):
        return False
    if problem in seen_problems:
        return False
    seen_problems.add(problem)
    return True


def format_record(row: dict) -> dict[str, str]:
    problem = row["problem"].strip()
    solution = row["generated_solution"].strip()
    answer = str(row["expected_answer"]).strip()
    prompt = make_reasoning_prompt(problem)
    completion = f"\n<think>\n{solution}\n</think>\n\nAnswer: {answer}"
    return {"prompt": prompt, "completion": completion,
            "source": row.get("problem_source", "")}


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Streaming avoids downloading the full multi-GB dataset.
    from datasets import load_dataset

    print(f"Streaming {DATASET_NAME} / {DATASET_SPLIT} (max_scan={args.max_scan:,})")
    stream = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    seen: set[str] = set()
    kept = 0
    scanned = 0

    with args.output.open("w", encoding="utf-8") as fh:
        for row in stream:
            scanned += 1
            if scanned > args.max_scan:
                break
            if kept >= args.num_keep:
                break
            if not good_row(row, args.min_len, args.max_len, seen, args.sources):
                continue
            rec = format_record(row)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if kept % args.report_every == 0:
                print(f"  kept={kept:,}  scanned={scanned:,}")

    print(f"\nDone. scanned={scanned:,}  kept={kept:,}  -> {args.output}")
    if kept < args.num_keep:
        print(f"  [warn] kept < num_keep={args.num_keep:,}; raise --max-scan to get more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
