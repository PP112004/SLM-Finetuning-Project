from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
THINK_CLOSE_PATTERN = re.compile(r"</think>", re.IGNORECASE)


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
    """Extract the predicted numeric answer.

    Priority: \\boxed{...} > text after last </think> > "Answer: N" > last number.
    Searching after </think> prevents picking up numbers from scratch work when
    the response is truncated mid-reasoning.
    """
    if not text:
        return None

    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return normalize_number(_last_number(boxed[-1]))

    post_think = text
    close_matches = list(THINK_CLOSE_PATTERN.finditer(text))
    if close_matches:
        post_think = text[close_matches[-1].end():]

    answer_match = ANSWER_PATTERN.findall(post_think)
    if answer_match:
        return normalize_number(answer_match[-1])

    post_think_num = _last_number(post_think)
    if post_think_num is not None:
        return normalize_number(post_think_num)

    return normalize_number(_last_number(text))


def _last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall((text or "").replace("$", ""))
    return matches[-1] if matches else None


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
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
