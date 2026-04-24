"""Compare lm-eval-harness results across two runs and print a delta table.

Usage:
  python scripts/eval/2026-04-25-compare_lm_eval.py \
      outputs/eval/baseline \
      outputs/eval/grpo

Each directory should contain lm-eval output subdirs (e.g. gsm8k/, mmlu/,
strategyqa/). The tool walks each subdir, finds the most recent
`results_*.json` file, extracts the primary metric, and prints a comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# primary metric we care about per task (falls back to any *_acc* key)
TASK_PRIMARY_METRIC = {
    "gsm8k": "exact_match,strict-match",
    "gsm8k_cot": "exact_match,strict-match",
    "mmlu": "acc,none",
    "strategyqa": "acc,none",
    "bbh_cot_fewshot_strategyqa": "acc,none",
}


def find_results(root: Path) -> dict[str, dict]:
    """Return {task_key: results_dict} for the latest results_*.json in each subdir."""
    out: dict[str, dict] = {}
    if not root.exists():
        print(f"[warn] no such dir: {root}", file=sys.stderr)
        return out
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        # also search recursively (lm-eval sometimes nests deeper).
        candidates = sorted(subdir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            continue
        latest = candidates[-1]
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to parse {latest}: {exc}", file=sys.stderr)
            continue
        out[subdir.name] = data
    return out


def pick_metric(results_block: dict) -> tuple[str, float] | None:
    """Given one lm-eval results['results'][task] dict, pull the primary metric."""
    # Prefer the explicit mapping.
    for key, val in results_block.items():
        if not isinstance(val, (int, float)):
            continue
        if "acc" in key or "exact_match" in key:
            return key, float(val)
    return None


def summarize(data: dict) -> dict[str, tuple[str, float]]:
    """Return {task_name: (metric_key, metric_value)} for all tasks in data."""
    summary: dict[str, tuple[str, float]] = {}
    results = data.get("results", {})
    for task_name, block in results.items():
        picked = pick_metric(block)
        if picked is not None:
            summary[task_name] = picked
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline_dir", type=Path)
    p.add_argument("finetuned_dir", type=Path)
    args = p.parse_args()

    base = find_results(args.baseline_dir)
    fine = find_results(args.finetuned_dir)

    print(f"{'task':35s}  {'metric':30s}  {'baseline':>10s}  {'finetuned':>10s}  {'delta':>8s}")
    print("-" * 100)

    all_subdirs = sorted(set(base) | set(fine))
    for sub in all_subdirs:
        base_data = base.get(sub)
        fine_data = fine.get(sub)
        base_summary = summarize(base_data) if base_data else {}
        fine_summary = summarize(fine_data) if fine_data else {}
        tasks = sorted(set(base_summary) | set(fine_summary))
        for task in tasks:
            bk, bv = base_summary.get(task, ("", float("nan")))
            fk, fv = fine_summary.get(task, ("", float("nan")))
            metric = bk or fk
            delta = fv - bv if (bv == bv and fv == fv) else float("nan")
            delta_str = f"{delta:+.2%}" if delta == delta else "   n/a"
            base_str = f"{bv:.2%}" if bv == bv else "  n/a"
            fine_str = f"{fv:.2%}" if fv == fv else "  n/a"
            print(f"{task:35s}  {metric:30s}  {base_str:>10s}  {fine_str:>10s}  {delta_str:>8s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
