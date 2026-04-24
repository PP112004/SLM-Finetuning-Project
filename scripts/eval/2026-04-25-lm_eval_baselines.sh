#!/usr/bin/env bash
# Run baseline lm-evaluation-harness on Phi-3-Mini across the three KPI benchmarks.
#
# Usage (on the GPU host):
#   bash scripts/eval/2026-04-25-lm_eval_baselines.sh [MODEL]
#
# Default MODEL is the HF Phi-3-Mini 4k-instruct (bf16, no quantization).
# Outputs are written to outputs/eval/baseline/<task>/ with full JSON results.
#
# After SFT/GRPO training, re-run with MODEL pointing at the fine-tuned checkpoint
# (or use --peft for a LoRA adapter) and compare against these baseline numbers.

set -euo pipefail

MODEL="${1:-microsoft/Phi-3-mini-4k-instruct}"
OUT_DIR="outputs/eval/baseline"
BATCH_SIZE="${BATCH_SIZE:-8}"      # safe for an H100 80 GB with Phi-3-Mini bf16
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"

MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True"

echo "=== lm-eval baseline ==="
echo "MODEL=${MODEL}"
echo "OUT_DIR=${OUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo ""

# GSM8K — 5-shot chain-of-thought (standard protocol).
echo ">>> GSM8K (gsm8k_cot, 5-shot)"
lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks gsm8k_cot \
    --num_fewshot 5 \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output_path "${OUT_DIR}/gsm8k" \
    --log_samples

# MMLU — 5-shot, aggregate over all 57 subjects.
echo ">>> MMLU (mmlu, 5-shot)"
lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output_path "${OUT_DIR}/mmlu" \
    --log_samples

# StrategyQA — 6-shot CoT is the BIG-Bench-Hard convention.
# lm-eval-harness registers it as bbh_cot_fewshot_strategyqa (BBH branch) or
# strategyqa (standalone). We run the BBH-style task for consistency with papers.
echo ">>> StrategyQA (bbh_cot_fewshot_strategyqa, 3-shot BBH default)"
lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks bbh_cot_fewshot_strategyqa \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output_path "${OUT_DIR}/strategyqa" \
    --log_samples

echo ""
echo "=== DONE. Results under ${OUT_DIR}/ ==="
