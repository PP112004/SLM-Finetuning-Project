# Local Setup Plan — PS06 SLM Reasoning

This workspace is configured for:

- local dataset preparation
- local prompt and reward-function development
- local baseline inference on Apple Silicon
- local evaluation and report-writing

This workspace is not intended for full RL training on the MacBook Air. Use the college GPU machine for SFT and GRPO runs.

## Directory layout

- `data/` — local datasets and prepared files
- `docs/` — problem statement, research, planning, and setup notes
- `notebooks/` — exploratory notebooks
- `outputs/` — smoke, baseline, and evaluation artifacts
- `requirements/` — local/GPU dependency lists
- `scripts/` — runnable setup, smoke, baseline, and eval commands
- `src/ps06/` — project Python package

## Python environment

Use the project-local virtual environment:

- `.venv/`

Activate it with:

```bash
cd "/Users/ppsfolder/Codex Playground/SLM finetunig Project"
source .venv/bin/activate
```

## Main packages

- `torch`, `torchvision`, `torchaudio`
- `transformers`
- `datasets`
- `evaluate`
- `accelerate`
- `peft`
- `trl`
- `sentencepiece`
- `safetensors`
- `huggingface_hub`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `jupyter`
- `ipykernel`
- `mlx`
- `mlx-lm`

## First tasks after setup

1. Verify Apple Silicon inference works with a 1B to 3B model.
2. Build the GSM8K preprocessing pipeline.
3. Implement the answer-extraction and reward functions locally.
4. Run baseline evaluation locally on a small sample.
5. Move the same scripts to the college GPU machine for training.

## Useful commands

Verify the environment:

```bash
./.venv/bin/python scripts/setup/check_env.py
```

Register the Jupyter kernel:

```bash
./.venv/bin/python -m ipykernel install --user --name ps06-slm-local --display-name "Python (ps06-slm-local)"
```

Launch JupyterLab:

```bash
source .venv/bin/activate
jupyter lab
```

Quick MLX check:

```bash
source .venv/bin/activate
python -c "import mlx.core as mx; print(mx.default_device())"
```
