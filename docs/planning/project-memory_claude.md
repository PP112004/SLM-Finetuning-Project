# Project Memory — PS 06 SLM Reasoning Hackathon

> **Purpose of this file.** Drop this into a new Claude conversation (attach it, or paste its contents) and Claude will have full context to continue the project without re-doing research or re-asking questions.

**Last updated:** 2026-04-25 (session 2 — local-prep phase)
**Owner:** Pranjal (pranjalprakhar2000@gmail.com)
**Working folder:** `SLM finetunig Project/`
**Local machine:** MacBook Air M4, 16 GB RAM (MLX-friendly, no CUDA).
**GPU access:** H100 expected but not yet available. Current phase = offline prep so the first GPU hour is pure training, not debugging.

---

## 1. What this project is

Pranjal is competing in the **Ennovate-X hackathon** and has chosen **Problem Statement 06: "Enhancing Reasoning in Small Language Models (SLMs) using Reinforcement Learning."**

The full PS is in `docs/problem-statement/ps06-ennovatex-hackathon.pdf`. Key parameters:

- **Goal:** take an SLM (≤ 7B parameters) and use RL to improve its reasoning while keeping it efficient.
- **Suggested base models:** Qwen 2.5 7B, Gemma 4 E4B, Phi-3-Mini (3.8B).
- **Target KPIs (need ≥+5% improvement over baseline on at least 2 of 3):**
  - GSM8K ≥ 50%
  - MMLU ≥ 45%
  - StrategyQA ≥ 65%
- **Suggested datasets:** GSM8K, AQuA-RAT, MMLU.
- **Suggested directions:** outcome + process-based rewards, lightweight reward mechanisms, stability (KL tuning, entropy, curriculum), hybrid pipelines (SFT+RL, distillation+RL).
- **Deliverables:** RL training code, final model + inference code, detailed approach/results/insights.
- **Infra recommended:** 2× GPU with ≥24 GB VRAM, 64 GB RAM.

---

## 2. User constraints (decisions already made)

From the clarifying questions asked on 2026-04-22:

| Question | Pranjal's answer |
|---|---|
| Goal for first session | **Plan the approach** |
| Base model | **Not sure — help me decide** |
| Compute access | Originally "Colab/Kaggle free tier," then updated to **"cloud — find all options besides Colab/Kaggle"** |
| Timeline | **More than a week** (so optimize for quality over absolute speed) |
| First deliverable | **Literature summary first** (before any code) |

---

## 3. Approved plan (already recommended and not yet overridden)

### Base model
**Start with Phi-3-Mini (3.8B).** Fits 16 GB VRAM with 4-bit QLoRA, lets early work happen on free-tier cloud GPU (Lightning AI), and rStar-Math already validated the +45-point gain on MATH. Switch to Qwen2.5-Math-7B later if compute is comfortable.

### Training recipe
1. **SFT warmup** (1–2 days) — 1 epoch on ~2K CoT examples from OpenMathInstruct-2 or GSM8K CoT.
2. **GRPO** via TRL's GRPOTrainer (Unsloth optional) with LoRA (r=8, α=16, target q_proj + v_proj).
3. **Reward:** rule-based. `reward = accuracy + 0.5 × format_flag`, normalized. Format check: response wrapped in `<think>…</think>`.
4. **Hyperparameters:** LR 1e-5, KL β 0.02, temperature 0.8, batch 4, K=4 completions/prompt, 2–3 epochs over GSM8K + MATH (~15K examples).
5. **Eval:** `lm-eval-harness` for MMLU + StrategyQA, in-repo regex evaluator for GSM8K.

**Precision update (2026-04-25 session 2):** Since the GPU plan is now H100 (80 GB HBM3), drop the 4-bit QLoRA requirement. Default to **bf16 + LoRA** — fewer library headaches (no bitsandbytes kernel issues), cleaner gradients for GRPO, and Phi-3-Mini + reference + K=4 completions still fits easily. Keep 4-bit path only as a fallback for A100-40GB rentals.

### Cloud plan (10+ days)
- **Days 1–2 (setup + baseline + SFT):** Lightning AI Studios free tier (80 GPU-hrs/month on H200/A100).
- **Days 3–7 (main GRPO training):** Vast.ai A100 PCIe on-demand, ~$30–50 for 24–48 compute hours. Fallback to RunPod Secure Cloud ($1.89/hr A100 80GB) if Vast supply is thin.
- **Days 8–9 (eval + submission):** Leftover Lightning AI hours + Google Cloud $300 trial credit.
- **Total:** $30–$100 out-of-pocket; can be $0 with credit stacking.

### Credits to apply for immediately
- GitHub Student Pack → $200 DigitalOcean (1 yr)
- Google Cloud free trial → $300 over 90 days
- AWS Educate → $200 (if student)
- Azure for Students → $100 (if student)
- Lightning AI free tier → 80 GPU-hrs/month (no card)

### Expected outcomes
- GSM8K: +5 to +10 points (exceeds the +5% PS target)
- StrategyQA: +3 to +7 points
- MMLU: +1 to +3 points (math-focused RL transfers poorly; stretch target)

Plan is to hit GSM8K + StrategyQA confidently, treat MMLU as bonus.

---

## 4. Tech stack decisions

| Layer | Choice | Why |
|---|---|---|
| RL algorithm | **GRPO** (Group Relative Policy Optimization) | Memory-efficient (no critic), mature library support, proven on exact target problem |
| Training library | **Unsloth** (with TRL fallback) | ~90% VRAM reduction vs naive GRPO; supports LoRA + 4-bit + GRPO combo |
| Quantization | **4-bit NF4** via bitsandbytes | Required to fit in free/cheap GPUs |
| Adapter | **LoRA** (r=8, α=16) | Standard for constrained training |
| Reward type | **Rule-based verifiable** (accuracy + format) | Matches DeepSeek-R1 recipe; no learned RM needed for math |
| SFT dataset | OpenMathInstruct-2 filtered subset | 14M Q-A pairs, open license |
| RL dataset | GSM8K (7.5K) + MATH (7.5K), optional OpenMathInstruct-2 top-up | Verifiable answers, right difficulty |
| Eval harness | `lm-eval-harness` | Standardized, avoids methodology disputes in submission |

---

## 5. Files in this project

| Path | Purpose |
|---|---|
| `README.md` | Current project overview, directory map, and common commands |
| `docs/problem-statement/ps06-ennovatex-hackathon.pdf` | Official hackathon problem-statement PDF |
| `docs/research/2026-04-22-ps06-slm-reasoning-research.docx` | Consolidated research brief: literature survey + cloud options + synthesized plan |
| `docs/planning/project-memory_claude.md` | This continuity/context file |
| `docs/setup/local-setup.md` | Local Mac setup notes |
| `docs/setup/2026-04-25-execution-setup.md` | Current execution checklist |
| `requirements/local.txt` | Local development dependencies |
| `src/ps06/data.py` | GSM8K loading and prompt formatting |
| `src/ps06/rewards.py` | Numeric answer extraction and rule-based reward scoring |
| `scripts/setup/check_env.py` | Local environment verification |
| `scripts/smoke/2026-04-25-smoke_gsm8k_rewards.py` | GSM8K/reward smoke test |
| `scripts/baseline/2026-04-25-baseline_gsm8k_mlx.py` | Local MLX baseline runner |
| `scripts/eval/2026-04-25-summarize_gsm8k_predictions.py` | JSONL baseline/error summarizer |

Current execution status: local setup, GSM8K loading, reward scoring, and tiny MLX baseline are working. Current local baseline is 11/20 correct on the first 20 GSM8K test examples with `mlx-community/Phi-3-mini-4k-instruct-4bit`.

---

## 6. Key literature (what Claude should know without re-searching)

### Flagship papers
- **DeepSeek-R1** (arXiv:2501.12948, Jan 2025) — proved pure RL on a base model can induce reasoning; R1-Zero at 71% AIME 2024.
- **DeepSeek-Math** (arXiv:2402.03300, Feb 2024) — origin of GRPO; 7B → 51.7% MATH.
- **Qwen2.5-Math** (arXiv:2409.12122, Sep 2024) — current open-source 7B SOTA; 88.5% GSM8K, 58.8% MATH.
- **rStar-Math** (arXiv:2501.04519, Jan 2025) — **most relevant paper for this PS**; Phi-3-Mini 41.4% → 86.4% on MATH using PRM + MCTS + 4 rounds of self-evolution.
- **OpenR1** (HuggingFace, 2025+) — open reproduction of DeepSeek-R1; OpenR1-Math-220K dataset + training code.
- **MiMo-7B-RL** (Xiaomi, Apr 2025) — 7B with curriculum learning + GRPO; beat o1-mini on AIME 2025.

### Technique references
- **GRPO** — no critic, group-relative advantage. Default choice.
- **PPO** — 4 models in memory, too heavy for SLM hackathon, skip.
- **RLOO** — vanilla REINFORCE with leave-one-out baseline; cleaner than PPO, less mature than GRPO. Fallback.
- **DPO/KTO/SimPO** — offline preference optimization; incompatible with online math exploration. Skip as primary.
- **Rejection sampling / STaR / ReST** — cheap "poor man's RL"; useful as SFT-warmup booster.

### Reward design canon
- **Rule-based verifiable rewards** are the 2025–2026 winner for math/code (DeepSeek-R1 proof).
- **Process reward models (PRMs)** — Math-Shepherd, PRM800K, Skywork-o1-PRM. Expensive to train; skip round 1.
- **Outcome reward models (ORMs)** — noisy for training; useful for test-time best-of-N.

### Failure modes
- **Reward hacking** (format-only degenerate outputs) — mitigate with rule-based rewards + KL β ≥ 0.02.
- **Entropy collapse** — mitigate with temperature 0.8–1.0 and moderate KL penalty.
- **Cross-benchmark transfer** — math RL helps math + logic, not MMLU. Mix in non-math data if MMLU is a must-win.

### Engineering: memory math for Phi-3-Mini on 24 GB GPU
Model 4-bit ~2 GB + LoRA ~200 MB + optimizer ~4–8 GB + activations ~2–4 GB + reference ~2 GB + reward ~2 GB ≈ 12–14 GB total, leaving headroom for batch 4, K=4 completions.

---

## 7. Cloud compute summary

| Rank | Provider | Deal | When to use |
|---|---|---|---|
| 1 | Vast.ai | A100 PCIe 40GB from ~$0.56/hr, no session limits | Main GRPO training (cheapest) |
| 2 | RunPod Secure | A100 80GB $1.89/hr, guaranteed $5–$500 signup bonus | Reliability-first fallback |
| 3 | Lightning AI Studios | Free 80 GPU-hrs/month on H200/A100 | SFT warmup + prototyping (zero cost) |
| 4 | Lambda Labs | A100 $1.29/hr, $5K research grants for academics | Grant path if eligible |
| 5 | Google Cloud | $300 free trial credits | Phase 3 (eval + polish) |

**Skip:** Paperspace (mandatory $39/mo subscription), Modal (3.75× serverless multiplier), HuggingFace ZeroGPU (API only, 25 min/day), SageMaker Studio Lab (4 hrs/day T4 only), Azure (expensive at $3.40/hr A100).

---

## 8. Current session status (2026-04-25 session 2)

### Findings flagged
- **Baseline truncation bug.** `max_tokens=256` in `scripts/baseline/2026-04-25-baseline_gsm8k_mlx.py` is too low for multi-`<think>` outputs. idx=2 in `outputs/baselines/2026-04-25-gsm8k-baseline-mlx-20.jsonl` is cut off mid-sentence and `extract_final_number` pulls a scratch number ($130000) instead of the real answer. Raise to **768**.
- **Answer extraction is too permissive.** `src/ps06/rewards.py::extract_final_number` just takes the last number anywhere. Needs to prefer (in order): `\boxed{}`, text after `"Answer:"`, then last number. Otherwise GRPO may reward-hack by emitting correct numbers inside `<think>`.
- **Under-sampled baseline.** 20 examples → ±10pt variance. Need ≥200 for a defensible pre-training number.
- **Only GSM8K is wired up.** KPI requires 2/3 of GSM8K / MMLU / StrategyQA. Need loaders + lm-eval-harness baselines for the other two before first GPU session.
- **GPU requirements file not written yet** (flagged in `docs/setup/2026-04-25-execution-setup.md`).

### Local-prep roadmap (what to do WITHOUT a GPU)
Goal: when H100 access lands, the first hour is training, not debugging.

1. **Fix baseline script**: bump `max_tokens` to 768, tighten answer extraction, re-run on 200 GSM8K examples. (Local-MLX runnable.)
2. **Write `requirements/gpu.txt`** — torch+cu121, transformers, trl (≥0.12 for GRPO), peft, accelerate, datasets, lm-eval, sentencepiece, wandb. bitsandbytes optional (fallback only). unsloth optional (test after basic TRL path works).
3. **Write `scripts/setup/2026-04-25-check_gpu_env.py`** — prints torch/CUDA/GPU/bf16-support/transformers/trl versions; fails loud on missing pieces.
4. **Add StrategyQA + MMLU loaders** in `src/ps06/data.py` and matching smoke tests.
5. **Write `scripts/eval/2026-04-25-lm_eval_baselines.sh`** — exact `lm-eval` invocations for gsm8k, mmlu, strategyqa on Phi-3-Mini bf16.
6. **Draft SFT script** `scripts/train/2026-04-25-sft_phi3.py` — TRL `SFTTrainer`, bf16 LoRA, OpenMathInstruct-2 subset. Dry-run on Mac with `--dry-run` flag (no actual train).
7. **Draft GRPO script** `scripts/train/2026-04-25-grpo_phi3.py` — TRL `GRPOTrainer`, reward from `src.ps06.rewards`, same bf16 LoRA config.
8. **OpenMathInstruct-2 filtering script** `scripts/data/2026-04-25-prepare_sft_data.py` — dedup, length filter, format to chat template, write `data/sft/phi3_sft.jsonl`. Can run locally (CPU-only).

### Still queued (not yet done)
- Starter training notebook (Vast.ai / RunPod compatible).
- One-page team pitch.
- MMLU auxiliary reward design.
- Curriculum schedule for Phi-3-Mini.
- Credits applications (GitHub Student Pack, AWS Educate, Azure for Students, Google Cloud trial) — student-status unconfirmed.

---

## 9. User's global working rules (from their CLAUDE.md)

Claude must always:

- **Prevent accidental damage.** Before deleting/overwriting/renaming any existing file, show what will change and wait for confirmation. Never modify files outside the current working folder unless explicitly asked.
- **Naming.** New files use `YYYY-MM-DD-descriptive-name` format.
- **Track files.** At end of each task, list all files created/modified with locations.
- **Control the pace.** For multi-step tasks, outline the plan first and wait for approval before executing. After each major step, briefly summarize what was done and what's next.

---

## 10. How to resume in a new conversation

Paste into the new Claude session:

> I'm continuing the PS 06 SLM Reasoning hackathon project. Read `docs/planning/project-memory_claude.md` in the working folder for full context, then wait for my next instruction.

Claude should then:
1. Read this file in full.
2. Confirm it has the context (base model, recipe, cloud plan, next-decision queue).
3. Ask what Pranjal wants to work on from the "open questions / next decisions" list.
4. Follow the user's global rules (plan first, wait for approval on multi-step work, use YYYY-MM-DD naming).

---

*End of memory file. Do not assume any progress beyond what is documented here.*
