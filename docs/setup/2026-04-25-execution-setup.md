# 2026-04-25 Execution Setup

This file converts the research plan into execution steps for the PS06 SLM reasoning hackathon.

## Current decision

- Base model for first implementation: `microsoft/Phi-3-mini-4k-instruct`
- Local machine role: dataset prep, reward functions, tiny baseline checks, report assets
- GPU machine role: SFT, GRPO, full benchmark evaluation
- First target benchmark: GSM8K, because answer verification is deterministic

## Step 1: Local environment sanity

Run from the project root:

```bash
source .venv/bin/activate
python scripts/setup/check_env.py
```

Expected result:

- Core packages should show `[ok]`.
- `torch.backends.mps.is_built()` should be `True`.
- `torch.backends.mps.is_available()` may be `False` inside Codex/headless sessions. Re-run from normal macOS Terminal to verify Apple Silicon access.
- `mlx` / `mlx_lm` are optional for the hackathon pipeline. If they fail only in the Codex session, do not block.

## Step 2: Dataset and reward scaffold

Create small, local-first scripts for:

- loading a sample of GSM8K
- extracting final numeric answers
- scoring model outputs with accuracy and `<think>...</think>` format rewards
- writing sample predictions into `outputs/`

## Step 3: Baseline evaluation

Start with a tiny sample before any training:

- 20 GSM8K examples for local dry run
- 100 GSM8K examples for a cheap cloud sanity run
- full GSM8K test set only after the pipeline is stable

Tiny local baseline command:

```bash
python scripts/baseline/2026-04-25-baseline_gsm8k_mlx.py --limit 3
```

This uses `mlx-community/Phi-3-mini-4k-instruct-4bit`. The first run downloads about 2.15 GB of model weights into `.cache/huggingface/`.

Summarize a baseline output file:

```bash
python scripts/eval/2026-04-25-summarize_gsm8k_predictions.py outputs/baselines/2026-04-25-gsm8k-baseline-mlx-20.jsonl
```

## Step 4: GPU training setup

Prepare separate GPU setup files for:

- `requirements-gpu.txt`
- SFT script
- GRPO script
- evaluator command notes

Keep local and GPU requirements separate because Unsloth/bitsandbytes/Linux CUDA packages are not appropriate for the Mac local environment.

## Immediate next file targets

- `requirements/gpu.txt`
- `scripts/setup/2026-04-25-check_gpu_env.py`
- SFT training script
- GRPO training script
- evaluator command notes
