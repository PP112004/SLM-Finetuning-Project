# PS06 SLM Reasoning Hackathon

This project is for Ennovate-X Problem Statement 06: improving reasoning in a small language model with SFT and reinforcement learning.

## Current Status

- Local Python environment works.
- GSM8K loading works.
- Rule-based answer and format reward works.
- Local MLX baseline with `mlx-community/Phi-3-mini-4k-instruct-4bit` works.
- Current tiny baseline: 11/20 correct on GSM8K test examples.

## Directory Layout

```text
docs/
  planning/             Project memory and decision context
  problem-statement/    Official hackathon problem statement
  research/             Literature survey and cloud research
  setup/                Local execution and setup notes
requirements/
  local.txt             Mac/local development dependencies
src/ps06/
  data.py               Dataset loading and prompt formatting
  rewards.py            Answer extraction and reward scoring
scripts/
  setup/                Environment checks
  smoke/                Small plumbing checks
  baseline/             Baseline model inference scripts
  eval/                 Result summarization and evaluation helpers
outputs/
  smoke/                Smoke-test artifacts
  baselines/            Baseline prediction JSONL files
  eval/                 Evaluation summaries
```

## Common Commands

Activate the environment:

```bash
source .venv/bin/activate
```

Check local environment:

```bash
python scripts/setup/check_env.py
```

Check GSM8K loading and rewards:

```bash
python scripts/smoke/2026-04-25-smoke_gsm8k_rewards.py --limit 5
```

Run a small local baseline:

```bash
python scripts/baseline/2026-04-25-baseline_gsm8k_mlx.py --limit 20 --output outputs/baselines/2026-04-25-gsm8k-baseline-mlx-20.jsonl
```

Summarize baseline errors:

```bash
python scripts/eval/2026-04-25-summarize_gsm8k_predictions.py outputs/baselines/2026-04-25-gsm8k-baseline-mlx-20.jsonl
```

## Next Engineering Step

Prepare the GPU environment files and scripts for larger baseline evaluation, SFT warmup, and GRPO training.
