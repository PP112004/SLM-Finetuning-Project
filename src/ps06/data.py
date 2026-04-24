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
