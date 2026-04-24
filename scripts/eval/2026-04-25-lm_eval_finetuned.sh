#!/usr/bin/env bash
# Run lm-evaluation-harness on a fine-tuned Phi-3-Mini + LoRA adapter.
#
# Usage:
#   bash scripts/eval/2026-04-25-lm_eval_finetuned.sh ADAPTER_PATH [TAG]
#
# Example:
#   bash scripts/eval/2026-04-25-lm_eval_finetuned.sh \
#       outputs/grpo/phi3-mini-rl grpo
#
# Results land under outputs/eval/<tag>/<task>/ so you can diff against baseline.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 ADAPTER_PATH [TAG]" >&2
    exit 2
fi

ADAPTER="$1"
TAG="${2:-finetuned}"
BASE_MODEL="${BASE_MODEL:-microsoft/Phi-3-mini-4k-instruct}"
OUT_DIR="outputs/eval/${TAG}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"

MODEL_ARGS="pretrained=${BASE_MODEL},peft=${ADAPTER},dtype=bfloat16,trust_remote_code=True"

echo "=== lm-eval fine-tuned ==="
echo "BASE_MODEL=${BASE_MODEL}"
echo "ADAPTER=${ADAPTER}"
echo "TAG=${TAG}"
echo ""

for TASK_SPEC in \
    "gsm8k_cot:5:gsm8k" \
    "mmlu:5:mmlu" \
    "bbh_cot_fewshot_strategyqa:0:strategyqa" ; do
    IFS=':' read -r TASK SHOTS SUBDIR <<< "$TASK_SPEC"
    echo ">>> ${TASK} (${SHOTS}-shot)"
    FEWSHOT_ARG=""
    if [[ "${SHOTS}" -gt 0 ]]; then
        FEWSHOT_ARG="--num_fewshot ${SHOTS}"
    fi
    lm_eval \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASK}" \
        ${FEWSHOT_ARG} \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --output_path "${OUT_DIR}/${SUBDIR}" \
        --log_samples
done

echo ""
echo "=== DONE. Results under ${OUT_DIR}/ ==="
