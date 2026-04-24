"""Smoke test for MMLU and StrategyQA loaders.

Downloads small slices and prints a couple of formatted prompt records.
First run downloads datasets: MMLU ~160 MB, StrategyQA ~few MB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.ps06.data import (
    load_mmlu_split,
    mmlu_prompt_record,
    load_strategyqa_split,
    strategyqa_prompt_record,
)


def smoke_mmlu(limit: int = 3) -> None:
    print(f"\n=== MMLU test[:{limit}] ===")
    ds = load_mmlu_split(split="test", limit=limit)
    for i, ex in enumerate(ds):
        rec = mmlu_prompt_record(ex)
        print(f"\n--- MMLU[{i}] subject={rec['subject']} gold={rec['answer_letter']} ---")
        print(rec["prompt"])


def smoke_strategyqa(limit: int = 3) -> None:
    # StrategyQA mirrors on HF differ; try a few fallbacks.
    print(f"\n=== StrategyQA test[:{limit}] ===")
    from datasets import load_dataset

    candidates = [
        ("ChilleD/StrategyQA", "test"),
        ("ChilleD/StrategyQA", "train"),
        ("voidful/StrategyQA", "test"),
        ("wics/strategy-qa", "test"),
    ]
    ds = None
    last_err = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            print(f"Loaded {name} split={split}  n={len(ds)}  cols={ds.column_names}")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  [skip] {name}:{split} -> {type(exc).__name__}")
    if ds is None:
        raise RuntimeError(f"No StrategyQA variant loaded. Last error: {last_err}")

    # Take the first `limit` rows safely.
    ds = ds.select(range(min(limit, len(ds))))
    for i, ex in enumerate(ds):
        rec = strategyqa_prompt_record(ex)
        print(f"\n--- SQA[{i}] gold={rec['answer_bool']} ---")
        print(rec["prompt"])


def main() -> int:
    try:
        smoke_mmlu(limit=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[MMLU] FAILED: {type(exc).__name__}: {exc}")
    try:
        smoke_strategyqa(limit=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[StrategyQA] FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
