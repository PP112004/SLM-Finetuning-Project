# PS06 SLM Reasoning Hackathon

Ennovate-X Problem Statement 06: improving reasoning in a small language model with SFT + reinforcement learning (GRPO).

See [`docs/planning/project-memory_claude.md`](docs/planning/project-memory_claude.md) for the full continuity / planning document.

## Current Status (2026-04-25, session 2)

- Local environment + GSM8K loading + rule-based reward + MLX baseline working.
- Rewards now prefer `\boxed{…}` → text after `</think>` → `"Answer:"` → last number (prevents picking scratch-work numbers).
- MMLU and StrategyQA dataset loaders added (both verified on local smoke).
- GPU-side artifacts authored (not yet executed — H100 pending):
  - `requirements/gpu.txt`, `scripts/setup/2026-04-25-check_gpu_env.py`
  - `scripts/train/2026-04-25-sft_phi3.py`, `scripts/train/2026-04-25-grpo_phi3.py` (both with `--dry-run` verified)
  - `scripts/eval/2026-04-25-lm_eval_baselines.sh`, `scripts/eval/2026-04-25-lm_eval_finetuned.sh`
  - `scripts/eval/2026-04-25-compare_lm_eval.py`
  - `scripts/data/2026-04-25-prepare_sft_data.py` (streaming OpenMathInstruct-2, smoke verified)

## Directory Layout

```text
docs/
  planning/             Project memory and decision context
  problem-statement/    Official hackathon problem statement
  research/             Literature survey and cloud research
  setup/                Local execution and setup notes
requirements/
  local.txt             Mac local development dependencies
  gpu.txt               GPU host (CUDA 12.1) training dependencies
src/ps06/
  data.py               GSM8K / MMLU / StrategyQA loaders and prompt formatting
  rewards.py            Answer extraction (boxed → answer: → last number) + rewards
scripts/
  setup/                Environment checks (local + GPU)
  smoke/                Small plumbing checks (rewards, loaders)
  baseline/             Baseline model inference scripts (MLX local)
  data/                 SFT dataset prep (OpenMathInstruct-2 streaming)
  train/                SFT + GRPO training scripts (GPU)
  eval/                 lm-eval-harness wrappers + summarizers + comparators
outputs/
  smoke/                Smoke-test artifacts
  baselines/            Baseline prediction JSONL files
  eval/                 lm-eval-harness results (baseline/ vs tag/)
  sft/                  SFT LoRA adapter checkpoints
  grpo/                 GRPO LoRA adapter checkpoints
data/
  sft/                  Prepared SFT JSONL (prompt + completion)
```

## Common Commands

Activate the environment:

```bash
source .venv/bin/activate
```

### Local (Mac) workflow

```bash
# Env check
python scripts/setup/check_env.py

# GSM8K + rewards smoke
python scripts/smoke/2026-04-25-smoke_gsm8k_rewards.py --limit 5

# MMLU + StrategyQA loader smoke
python scripts/smoke/2026-04-25-smoke_mmlu_strategyqa_loaders.py

# Local MLX baseline (200 examples, ~100 min on M4)
python scripts/baseline/2026-04-25-baseline_gsm8k_mlx.py --limit 200 --max-tokens 768 \
    --output outputs/baselines/2026-04-25-gsm8k-baseline-mlx-200.jsonl

# Summarize baseline results
python scripts/eval/2026-04-25-summarize_gsm8k_predictions.py \
    outputs/baselines/2026-04-25-gsm8k-baseline-mlx-200.jsonl

# Prepare SFT data (streaming OpenMathInstruct-2; safe to run locally)
python scripts/data/2026-04-25-prepare_sft_data.py \
    --output data/sft/phi3_sft.jsonl --num-keep 5000 --max-scan 150000
```

### GPU host workflow (H100 recommended)

```bash
# 0. Environment
pip install -r requirements/gpu.txt
python scripts/setup/2026-04-25-check_gpu_env.py

# 1. Baseline (all 3 KPI benchmarks)
bash scripts/eval/2026-04-25-lm_eval_baselines.sh

# 2. SFT warmup
python scripts/train/2026-04-25-sft_phi3.py \
    --train-file data/sft/phi3_sft.jsonl \
    --output-dir outputs/sft/phi3-mini-ft \
    --num-epochs 1 --per-device-batch-size 4 --grad-accum 4 --learning-rate 2e-5

# 3. GRPO
python scripts/train/2026-04-25-grpo_phi3.py \
    --sft-adapter outputs/sft/phi3-mini-ft \
    --output-dir outputs/grpo/phi3-mini-rl \
    --num-train-epochs 2 --per-device-batch-size 2 --num-generations 4 \
    --learning-rate 1e-5 --kl-beta 0.02

# 4. Final eval + compare
bash scripts/eval/2026-04-25-lm_eval_finetuned.sh outputs/grpo/phi3-mini-rl grpo
python scripts/eval/2026-04-25-compare_lm_eval.py outputs/eval/baseline outputs/eval/grpo
```
