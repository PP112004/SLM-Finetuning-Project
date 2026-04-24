from __future__ import annotations

from datasets import Dataset, load_dataset


GSM8K_DATASET = "gsm8k"
GSM8K_CONFIG = "main"


def load_gsm8k_split(split: str = "test", limit: int | None = None) -> Dataset:
    dataset = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def make_reasoning_prompt(question: str) -> str:
    return (
        "Solve the math problem. Put your reasoning inside <think> and </think>, "
        "then give the final numeric answer.\n\n"
        f"Problem: {question}\n\n"
        "Answer:"
    )


def gsm8k_prompt_record(example: dict[str, str]) -> dict[str, str]:
    return {
        "question": example["question"],
        "answer": example["answer"],
        "prompt": make_reasoning_prompt(example["question"]),
    }


# --- MMLU ---------------------------------------------------------------
# cais/mmlu has configs per subject and an "all" config covering every subject.
MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_CHOICE_LETTERS = ["A", "B", "C", "D"]


def load_mmlu_split(split: str = "test", config: str = MMLU_CONFIG, limit: int | None = None) -> Dataset:
    dataset = load_dataset(MMLU_DATASET, config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def make_mmlu_prompt(question: str, choices: list[str]) -> str:
    lines = [f"{letter}. {text}" for letter, text in zip(MMLU_CHOICE_LETTERS, choices)]
    choice_block = "\n".join(lines)
    return (
        "Answer the multiple-choice question. Put brief reasoning inside <think> and </think>, "
        "then output the final answer as a single letter (A, B, C, or D).\n\n"
        f"Question: {question}\n\n"
        f"{choice_block}\n\n"
        "Answer:"
    )


def mmlu_prompt_record(example: dict) -> dict:
    choices = list(example["choices"])
    answer_idx = int(example["answer"])
    return {
        "question": example["question"],
        "choices": choices,
        "answer_index": answer_idx,
        "answer_letter": MMLU_CHOICE_LETTERS[answer_idx],
        "subject": example.get("subject", ""),
        "prompt": make_mmlu_prompt(example["question"], choices),
    }


# --- StrategyQA ---------------------------------------------------------
# ChilleD/StrategyQA is a commonly used HF mirror of the original dataset.
STRATEGYQA_DATASET = "ChilleD/StrategyQA"


def load_strategyqa_split(split: str = "test", limit: int | None = None) -> Dataset:
    dataset = load_dataset(STRATEGYQA_DATASET, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def make_strategyqa_prompt(question: str) -> str:
    return (
        "Answer the yes/no question. Put brief reasoning inside <think> and </think>, "
        "then output the final answer as either 'yes' or 'no'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def strategyqa_prompt_record(example: dict) -> dict:
    answer_bool = example.get("answer")
    if isinstance(answer_bool, str):
        answer_letter = "yes" if answer_bool.strip().lower() in {"yes", "true", "1"} else "no"
    else:
        answer_letter = "yes" if bool(answer_bool) else "no"
    return {
        "question": example["question"],
        "answer_bool": answer_letter,
        "facts": example.get("facts", []),
        "prompt": make_strategyqa_prompt(example["question"]),
    }
