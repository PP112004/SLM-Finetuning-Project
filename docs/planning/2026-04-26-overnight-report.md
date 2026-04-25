# Overnight Report — 2026-04-26

Short human-readable summary of what changed while you slept, what it means, and what it sets up.

---

## The one-line story

**The "55% on GSM8K" number was wrong.** The real baseline is **79.5%**. The remaining 20 points are the real target for SFT + GRPO to chip away at.

---

## What was broken, and what I fixed

### Bug 1 — Truncation
The baseline script was capped at `max_tokens=256`. Phi-3-Mini loves to write multiple `<think>` blocks, so on harder problems the model was being cut off mid-reasoning, before it ever emitted the final number. Raised to **768**.

### Bug 2 — Answer extraction
The old extractor just grabbed the last number it saw anywhere in the response. That meant when the model was truncated mid-scratch-work, we were grading a random intermediate number as the "answer." Example: model computes `$130,000 cost` then `$70,000 profit`, gets truncated before writing the final "Answer:" line. Old extractor returned `130000` and scored it wrong. Real answer was `70000`.

New extractor priority:
1. `\boxed{...}` (math paper convention)
2. The last number **after** the final `</think>` tag (proper final-answer zone)
3. Explicit `"Answer: N"` or `"answer is N"`
4. Last number overall (fallback)

Unit-tested on 5 tricky cases — all pass. This matters for **GRPO too**: without this fix, the model could reward-hack by writing the right number inside `<think>` and anything after.

---

## The real baseline

| Metric | Before fixes | After fixes |
|---|---|---|
| Examples | 20 | **200** (10× stable) |
| `max_tokens` | 256 | **768** |
| Extraction | last number | boxed → post-think → "Answer:" |
| Accuracy | 55% (noisy) | **79.5%** |

**What this means for your KPI targets:**

- PS06 asks for +5 points over baseline on ≥ 2 of 3 benchmarks (GSM8K ≥50, MMLU ≥45, StrategyQA ≥65).
- Since GSM8K is already at ~80%, getting +5 there is **possible but tight** — the easy wins are gone.
- Best plan: bank **StrategyQA** as the reliable second KPI win (Phi-3-Mini has more room there), keep MMLU as stretch, and treat any GSM8K gain as bonus. This is a strategy shift from the original plan and is flagged in the memory file.

---

## What's ready to go on the H100

Everything below was authored, dry-run tested, and git-committed. When the GPU lands, it's launch-and-wait, not debug.

| File | What it does |
|---|---|
| `requirements/gpu.txt` | Installable dependency list for CUDA 12.1 + bf16 LoRA. bitsandbytes commented out (only needed if you drop to A100-40GB). |
| `scripts/setup/2026-04-25-check_gpu_env.py` | First thing to run on GPU host. Verifies CUDA, bf16, all library versions. Prints PASS/FAIL. |
| `src/ps06/data.py` | Added MMLU and StrategyQA loaders with proper CoT prompts. Both dataset downloads were tested locally and work. |
| `scripts/train/2026-04-25-sft_phi3.py` | SFT training: Phi-3-Mini + LoRA + bf16 + flash-attn-2, TRL `SFTTrainer`. Has `--dry-run` mode that validates config on CPU. |
| `scripts/train/2026-04-25-grpo_phi3.py` | GRPO training with rule-based rewards. Takes optional `--sft-adapter` to warm-start. Dry-run validates reward functions too. |
| `scripts/data/2026-04-25-prepare_sft_data.py` | Streams OpenMathInstruct-2 from HuggingFace, filters+dedups, writes `{prompt, completion}` JSONL. |
| `scripts/eval/2026-04-25-lm_eval_baselines.sh` | Runs lm-evaluation-harness on base Phi-3-Mini for gsm8k_cot, mmlu, and strategyqa. This gives you the REAL baseline (not just my local MLX one). |
| `scripts/eval/2026-04-25-lm_eval_finetuned.sh` | Same but for a fine-tuned LoRA adapter. |
| `scripts/eval/2026-04-25-compare_lm_eval.py` | Diffs two result dirs and prints a table: baseline vs finetuned, deltas in %. |

---

## First 8 moves when the H100 arrives

```
1. pip install -r requirements/gpu.txt
2. python scripts/setup/2026-04-25-check_gpu_env.py               # must say PASS
3. python scripts/data/2026-04-25-prepare_sft_data.py \
     --output data/sft/phi3_sft.jsonl --num-keep 5000 --max-scan 150000
4. bash scripts/eval/2026-04-25-lm_eval_baselines.sh              # ~1-2 hrs
5. python scripts/train/2026-04-25-sft_phi3.py \
     --train-file data/sft/phi3_sft.jsonl \
     --output-dir outputs/sft/phi3-mini-ft                        # ~30-60 min
6. python scripts/train/2026-04-25-grpo_phi3.py \
     --sft-adapter outputs/sft/phi3-mini-ft \
     --output-dir outputs/grpo/phi3-mini-rl                       # many hours
7. bash scripts/eval/2026-04-25-lm_eval_finetuned.sh outputs/grpo/phi3-mini-rl grpo
8. python scripts/eval/2026-04-25-compare_lm_eval.py \
     outputs/eval/baseline outputs/eval/grpo
```

Total expected GPU wall-clock: **6–12 hours** including eval. That leaves room for 1–2 retrains if the first GRPO run has issues (entropy collapse, reward hacking, etc.).

---

## Things to check on the GPU host (known risks)

1. **flash-attention**: both train scripts request `flash_attention_2`. If the wheel isn't pre-installed on your rental, either `pip install flash-attn --no-build-isolation` or just delete that arg from the script.
2. **StrategyQA task name**: I used `bbh_cot_fewshot_strategyqa` in the eval shell. If your lm-eval version uses a different name, run `lm_eval --tasks list | grep -i strategyqa` and swap it in.
3. **TRL version**: pinned to `>=0.12,<0.14`. If `GRPOConfig` kwargs differ in your installed version, error messages will tell you which arg to rename.

All three are 2-minute fixes, just calling them out so you don't get surprised.

---

## Git log

```
0dba880 overnight: 200-example baseline (79.5%), memory hand-off update
1fb6139 overnight: README update, eval compare, SFT data prep, smoke verified
50ea89f overnight: GPU-side scripts, reward extraction fix, loaders
88e1fa9 pre-overnight snapshot
```

Nothing uncommitted. If any overnight change broke something, `git revert <sha>` gets you back.
