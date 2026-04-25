# PS06 — Enhancing Reasoning in a Small Language Model via Reinforcement Learning

This repository is our entry for **Ennovate-X Problem Statement 06**. The problem asks us to take a small language model (≤7B parameters) and improve its *reasoning* ability — solving math word problems, answering multiple-choice questions, and handling yes/no common-sense questions — using reinforcement learning (RL).

This README walks through the **what, the why, and the how** from first principles. If you know roughly what an LLM is and have seen the words "fine-tuning" before, you should be able to follow along.

---

## 1. The problem in plain English

Large models like GPT-4 and Claude can reason step-by-step very well, but they're expensive to run. Small models (1B–7B parameters) are cheap and fast but weaker at multi-step reasoning — they hallucinate, skip steps, or get arithmetic wrong. The PS06 challenge is:

> Take a small model, teach it to reason better, and show at least **+5 percentage points** of improvement over the unmodified baseline on **at least 2 of 3** benchmarks:
>
> | Benchmark | What it tests | Gate |
> |---|---|---|
> | **GSM8K** | Grade-school math word problems | ≥ 50% |
> | **MMLU** | 57-subject multiple-choice knowledge test | ≥ 45% |
> | **StrategyQA** | Yes/no questions needing implicit reasoning | ≥ 65% |

We picked **Phi-3-Mini (3.8B parameters)** as our base model because it's small enough to train cheaply on a single GPU and has strong pre-training that gives it a good starting point for reasoning.

---

## 2. How we're going to make the model reason better

We're using a three-stage pipeline. Each stage has a specific purpose; they build on each other.

```
          ┌─────────────────┐      ┌─────────────────┐      ┌────────────────┐
          │  1. Baseline    │      │  2. SFT warmup  │      │  3. GRPO       │
raw ────▶ │  (measure)      │ ───▶ │  (teach format) │ ───▶ │  (reward good  │
Phi-3     │  no training    │      │  supervised     │      │  reasoning)    │
          └─────────────────┘      └─────────────────┘      └────────────────┘
           "what's the           "teach it to use         "reward answers
            starting score?"      <think> tags             that are correct
                                  before answering"        AND well-formatted"
```

### Stage 1 — Baseline
We run Phi-3-Mini on all three benchmarks with **no changes** and record its scores. This is what we have to beat. If we skip this we can't prove improvement, and proving improvement is the whole PS06 deliverable.

### Stage 2 — SFT (Supervised Fine-Tuning) warmup
SFT is the "easy" kind of fine-tuning: show the model thousands of examples of **"question → good answer"** and update the weights so it imitates the good answers. We use examples from **OpenMathInstruct-2** (a 14M-problem math dataset) where each solution is written as `<think>step-by-step reasoning</think>` followed by the final answer.

Why SFT first, before RL?
- The model needs to *know* what a good reasoning trace looks like before RL can reward it for producing one.
- SFT is fast and stable; RL from a cold start is slow and often diverges.
- This is the same recipe that DeepSeek-R1 and other frontier reasoning models use.

We only need ~5,000 high-quality examples — not millions — because SFT for style/format converges quickly.

### Stage 3 — GRPO (Group Relative Policy Optimization)
This is where the real improvement comes from. GRPO is an RL algorithm. For each question:

1. The model generates **several candidate answers** (e.g. 4 attempts per question).
2. We score each attempt with a **reward function**:
   - **+1.0** if the final numeric answer matches the ground truth
   - **+0.5** if the output is wrapped in `<think>...</think>` (format reward)
   - **0** otherwise
3. Within the group of 4 attempts, the ones with above-average reward get their probabilities pushed up; below-average ones get pushed down. That's the "group relative" part.
4. A **KL penalty** stops the model from drifting too far from its SFT starting point — this prevents catastrophic forgetting and reward hacking.

Why GRPO and not something else?
- Classic PPO keeps 4 copies of the model in memory — too heavy for small-GPU training.
- GRPO drops the value network (critic) entirely and works just fine for math/logic tasks with verifiable answers. DeepSeek-Math's 7B model jumped from ~30% to 51% on MATH using this exact technique.
- DPO and similar offline methods need preference-labeled data we don't have.

We use **LoRA (Low-Rank Adaptation)** for both SFT and GRPO: instead of updating all 3.8 billion parameters, we train ~10M small "adapter" matrices. This drops VRAM use by ~10× with negligible quality cost.

---

## 3. Why we care about the reward function details

A lot of RL projects fail because the model learns to game the reward rather than actually get better. Common failure modes:

- **Reward hacking**: if the reward just counts whether *any* number in the output matches, the model learns to spam common answers like "42" and get lucky.
- **Format collapse**: if only format is rewarded (not correctness), the model outputs empty `<think></think>` tags and random numbers.
- **Length explosion**: the model learns that longer reasoning = higher chance of hitting the right number by accident, and produces 10,000-token rambles.

Our reward design avoids these because:
- Accuracy is the dominant signal (1.0 vs 0.5).
- Format is a small additive bonus — the model can't win by only being well-formatted.
- Our **answer-extraction code** (more on this in §5) is strict: we only count numbers that appear in the "final answer zone" (after `</think>` or in `\boxed{...}`), not numbers from scratch work.

---

## 4. Current status & what we've done so far (as of 2026-04-26)

Everything in this repo was set up during two local sessions on a MacBook Air M4 (16 GB RAM, no GPU). We can't *train* on a Mac — the M4 isn't fast enough and doesn't have the right libraries — but we can do all the scaffolding, data prep, and baseline work. The actual training will happen on a rented **H100 GPU** (expected access soon).

### Done
- [x] Project structure, git repo, dependency manifests.
- [x] Data loaders for GSM8K, MMLU, and StrategyQA (smoke-tested — all three download and parse correctly).
- [x] Reward function (accuracy + format) with robust answer extraction (handles `\boxed{}`, `"Answer:"`, post-`</think>` numbers).
- [x] Local MLX baseline on 200 GSM8K examples → **79.5% accuracy** (see §5 for why this matters).
- [x] SFT training script with LoRA + bf16 + flash-attention. Dry-run verified locally.
- [x] GRPO training script with our reward function wired in. Dry-run verified locally (reward function sanity-checked on synthetic inputs).
- [x] OpenMathInstruct-2 streaming data-prep script. Smoke run produced 50 clean training examples.
- [x] `lm-evaluation-harness` wrapper scripts for the official benchmark runs (baseline + fine-tuned).
- [x] Comparison script that diffs baseline vs fine-tuned result JSONs and prints a delta table.
- [x] Project memory doc (`docs/planning/project-memory_claude.md`) for long-context continuity.
- [x] Overnight work report (`docs/planning/2026-04-26-overnight-report.md`).

### Not done — blocked on GPU access
- [ ] Install GPU dependencies and verify CUDA + bf16 on the H100.
- [ ] Pull 5,000 high-quality SFT examples from OpenMathInstruct-2.
- [ ] Run the **real** 3-benchmark baseline via `lm-evaluation-harness` (our MLX number is a sanity check, but the official scores need a GPU).
- [ ] Run SFT warmup (~30–60 min on H100).
- [ ] Run GRPO training (several hours on H100).
- [ ] Run 3-benchmark eval on the fine-tuned model.
- [ ] Produce the delta table and the final write-up.

---

## 5. The baseline number — and why it changed our strategy

Our first baseline attempt gave **55% on GSM8K (20 examples)**. After investigating, we found two bugs:

1. **Truncation**: the model's max output was set to 256 tokens, but Phi-3-Mini writes multiple `<think>` blocks and was being cut off mid-reasoning.
2. **Answer extraction**: the code was grabbing the last number anywhere in the response — so when the model was truncated, a random intermediate number was being graded as "the final answer".

After fixing both and re-running on 200 examples:

| | Before fix | After fix |
|---|---|---|
| Sample size | 20 | 200 |
| Max output tokens | 256 | 768 |
| Answer extraction | last number anywhere | `\boxed{}` → post-`</think>` → `"Answer: N"` → last number |
| **GSM8K accuracy** | 55% | **79.5%** |

This matters for strategy:
- The PS06 target for GSM8K is **≥ 50%** — we already clear that easily.
- But getting **+5 points** over an 80% baseline is harder than +5 over 55%. The easy wins are gone.
- So we're planning to **bank StrategyQA as our reliable second-benchmark win** (Phi-3-Mini has more headroom there), treat any GSM8K gain as bonus, and treat MMLU as stretch.

---

## 6. Repository layout

```
docs/
  planning/             Long-form planning docs + session memory
    project-memory_claude.md         ← continuity doc; read for full context
    2026-04-26-overnight-report.md   ← short human-readable report of changes
  problem-statement/    Official PS06 PDF
  research/             Literature survey + cloud-compute research
  setup/                Local execution notes

requirements/
  local.txt             Mac development dependencies (datasets, transformers, mlx)
  gpu.txt               GPU host dependencies (torch+cu121, trl, peft, lm-eval, ...)

src/ps06/
  data.py               Dataset loaders for GSM8K / MMLU / StrategyQA + prompt templates
  rewards.py            Answer extraction + reward scoring (used by both eval and GRPO)

scripts/
  setup/
    check_env.py                         Verify local (Mac) environment
    2026-04-25-check_gpu_env.py          Verify GPU host (run on H100 first)
  smoke/
    2026-04-25-smoke_gsm8k_rewards.py          Verify GSM8K loader + reward logic
    2026-04-25-smoke_mmlu_strategyqa_loaders.py   Verify MMLU + StrategyQA loaders
  baseline/
    2026-04-25-baseline_gsm8k_mlx.py     Local MLX inference on GSM8K (Mac only)
  data/
    2026-04-25-prepare_sft_data.py       Stream + filter OpenMathInstruct-2 → JSONL
  train/
    2026-04-25-sft_phi3.py               SFT warmup (LoRA, bf16, GPU required)
    2026-04-25-grpo_phi3.py              GRPO training (LoRA, bf16, GPU required)
  eval/
    2026-04-25-summarize_gsm8k_predictions.py   Print accuracy + first N errors
    2026-04-25-lm_eval_baselines.sh             Run lm-eval on the 3 KPI benchmarks
    2026-04-25-lm_eval_finetuned.sh             Same, for a fine-tuned adapter
    2026-04-25-compare_lm_eval.py                Print baseline-vs-finetuned delta table

outputs/
  baselines/            Local baseline prediction JSONL files
  sft/                  SFT LoRA adapter checkpoints (populated on GPU)
  grpo/                 GRPO LoRA adapter checkpoints (populated on GPU)
  eval/                 lm-eval-harness result JSONs (populated on GPU)

data/
  sft/                  Prepared {prompt, completion} JSONL for SFT training
```

---

## 7. How to run things — annotated

### 7a. On the Mac (local development, no training)

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Verify the local environment is healthy
python scripts/setup/check_env.py
# Expected: all packages [ok], MPS (Apple GPU) detected.

# 3. Smoke-test the dataset loaders + reward function (tiny; <1 min)
python scripts/smoke/2026-04-25-smoke_gsm8k_rewards.py --limit 5
python scripts/smoke/2026-04-25-smoke_mmlu_strategyqa_loaders.py

# 4. (Optional) Rerun the 200-example MLX baseline — takes ~90 min on M4
python scripts/baseline/2026-04-25-baseline_gsm8k_mlx.py \
    --limit 200 --max-tokens 768 \
    --output outputs/baselines/your-run.jsonl

# 5. Summarize baseline results
python scripts/eval/2026-04-25-summarize_gsm8k_predictions.py \
    outputs/baselines/2026-04-25-gsm8k-baseline-mlx-200.jsonl

# 6. Pull a few thousand SFT examples (streamed — doesn't download the full 14M)
python scripts/data/2026-04-25-prepare_sft_data.py \
    --output data/sft/phi3_sft.jsonl \
    --num-keep 5000 --max-scan 150000
# Takes ~10-30 min depending on network. Output is ~50 MB JSONL.
```

### 7b. On the GPU host (H100 recommended, A100 80 GB works)

```bash
# --- Setup (once per rental) -------------------------------------------
pip install -r requirements/gpu.txt
python scripts/setup/2026-04-25-check_gpu_env.py
# Must print PASS. If anything FAILs, fix before continuing.

# --- Step 1: Real baseline across all 3 KPI benchmarks ------------------
bash scripts/eval/2026-04-25-lm_eval_baselines.sh
# Takes ~1-2 hrs. Results land in outputs/eval/baseline/{gsm8k,mmlu,strategyqa}/
# This is THE number we have to beat.

# --- Step 2: SFT warmup --------------------------------------------------
python scripts/train/2026-04-25-sft_phi3.py \
    --train-file data/sft/phi3_sft.jsonl \
    --output-dir outputs/sft/phi3-mini-ft \
    --num-epochs 1 --per-device-batch-size 4 --grad-accum 4 \
    --learning-rate 2e-5
# ~30-60 min for 5K examples on H100. Produces a LoRA adapter we can load later.

# --- Step 3: GRPO reinforcement learning ---------------------------------
python scripts/train/2026-04-25-grpo_phi3.py \
    --sft-adapter outputs/sft/phi3-mini-ft \
    --output-dir outputs/grpo/phi3-mini-rl \
    --num-train-epochs 2 \
    --per-device-batch-size 2 --num-generations 4 \
    --learning-rate 1e-5 --kl-beta 0.02
# Takes several hours. Watch logs for reward curves and KL — if KL explodes
# (>1.0 avg) or reward stalls, stop and lower learning rate or raise kl_beta.

# --- Step 4: Evaluate the fine-tuned model ------------------------------
bash scripts/eval/2026-04-25-lm_eval_finetuned.sh outputs/grpo/phi3-mini-rl grpo
# Runs the same 3 benchmarks with the LoRA adapter attached.

# --- Step 5: Print the delta table --------------------------------------
python scripts/eval/2026-04-25-compare_lm_eval.py \
    outputs/eval/baseline outputs/eval/grpo
# Final submission numbers come from this output.
```

### 7c. Diagnosing problems during training

- **Reward goes up but accuracy doesn't**: reward hacking. Check the samples the model is generating — probably exploiting a loophole in the extractor or the format check.
- **KL divergence climbs past ~1.0**: the policy is drifting too far from the reference. Raise `--kl-beta` to 0.05 or 0.1.
- **Reward stalls near zero**: SFT wasn't strong enough — the base model isn't producing `<think>` tags often enough to get any reward signal. Re-run SFT with more epochs.
- **Entropy drops to near zero**: the model has collapsed to one single answer style. Raise `--temperature` to 1.0 or add entropy regularization.

---

## 8. Key technical choices and why we made them

| Decision | What we picked | Why |
|---|---|---|
| Base model | Phi-3-Mini (3.8B) | Fits on one cheap GPU; strong pre-training; validated by rStar-Math paper |
| RL algorithm | GRPO | Memory-efficient (no critic), proven on math reasoning, mature TRL implementation |
| Fine-tuning | LoRA (r=8–16) | ~10× VRAM savings vs full fine-tuning, minimal quality loss |
| Precision | bf16 (not 4-bit) | H100 has plenty of memory; bf16 gives cleaner gradients for RL |
| Reward type | Rule-based (verifiable) | Matches DeepSeek-R1 recipe; no learned reward model needed for math |
| SFT data | OpenMathInstruct-2 (streaming subset) | Open license, 14M examples, includes `<think>`-style CoT solutions |
| RL data | GSM8K train split | Verifiable answers, right difficulty for a small model |
| Eval | `lm-evaluation-harness` | Standard tool — avoids "did you evaluate correctly?" disputes in the final writeup |

---

## 9. Pointers to deeper context

- **[`docs/planning/2026-04-26-overnight-report.md`](docs/planning/2026-04-26-overnight-report.md)** — Short, plain-English report of what was done during the overnight setup session, including the baseline-bug debrief.
- **[`docs/problem-statement/ps06-ennovatex-hackathon.pdf`](docs/problem-statement/ps06-ennovatex-hackathon.pdf)** — Official problem statement with all rules and targets.
- **[`docs/research/2026-04-22-ps06-slm-reasoning-research.docx`](docs/research/2026-04-22-ps06-slm-reasoning-research.docx)** — Literature survey: DeepSeek-R1, rStar-Math, Qwen2.5-Math, OpenR1, etc.

---

## 10. Glossary (if any term above was unfamiliar)

- **SLM** — *Small Language Model.* Typically 1B–7B parameters. Runs on a single GPU.
- **SFT** — *Supervised Fine-Tuning.* Show the model examples of correct answers and update weights to match.
- **RL** — *Reinforcement Learning.* The model generates answers, gets scored by a reward function, and its weights are updated to produce higher-scoring outputs.
- **GRPO** — *Group Relative Policy Optimization.* An RL algorithm: sample N answers per question, reward each, update the policy relative to the group's average.
- **LoRA** — *Low-Rank Adaptation.* A parameter-efficient fine-tuning method: freeze the base model, train only small low-rank "adapter" matrices injected into the attention layers.
- **bf16** — *Brain Float 16.* A 16-bit floating-point format with the same range as fp32 (unlike fp16). Standard for modern GPU training.
- **KL divergence / KL penalty** — A measure of how much a new policy differs from a reference policy. Adding it to the loss prevents the model from drifting too far during RL.
- **CoT** — *Chain-of-Thought.* The model writes its reasoning step-by-step before the final answer (instead of jumping straight to the answer).
- **lm-evaluation-harness** — The open-source benchmark runner (EleutherAI). Standard tool for reporting LLM benchmark scores.
- **`<think>...</think>`** — A conventional tag used by DeepSeek-R1 and others to delimit the reasoning portion of a response. Our format reward checks for this tag pair.
