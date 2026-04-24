from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class RewardBreakdown:
    extracted_prediction: str | None
    extracted_reference: str | None
    accuracy_reward: float
    format_reward: float

    @property
    def total(self) -> float:
        return self.accuracy_reward + self.format_reward


def has_think_format(text: str) -> bool:
    return bool(THINK_PATTERN.search(text or ""))


def extract_gsm8k_reference(answer: str) -> str | None:
    """Extract the canonical GSM8K answer after the #### marker."""
    if not answer:
        return None

    if "####" in answer:
        answer = answer.rsplit("####", maxsplit=1)[-1]

    return extract_final_number(answer)


def extract_final_number(text: str) -> str | None:
    """Return the last numeric-looking value in a generated solution."""
    if not text:
        return None

    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        text = boxed[-1]

    matches = NUMBER_PATTERN.findall(text.replace("$", ""))
    if not matches:
        return None

    return normalize_number(matches[-1])


def normalize_number(value: str) -> str | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None

    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None

    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def score_math_response(
    prediction: str,
    reference_answer: str,
    *,
    format_weight: float = 0.5,
) -> RewardBreakdown:
    pred = extract_final_number(prediction)
    ref = extract_gsm8k_reference(reference_answer)
    accuracy = 1.0 if pred is not None and ref is not None and pred == ref else 0.0
    fmt = format_weight if has_think_format(prediction) else 0.0
    return RewardBreakdown(
        extracted_prediction=pred,
        extracted_reference=ref,
        accuracy_reward=accuracy,
        format_reward=fmt,
    )
