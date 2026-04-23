"""LLM-as-judge scoring.

The judge takes (input, expected, actual) and returns a 0..1 score plus rationale.
We keep the prompt rubric explicit so it can be calibrated against human labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import providers


DEFAULT_RUBRIC = """You are grading whether a model's answer matches the expected reference.

Score on this rubric:
  1.0 — semantically equivalent to the reference, same facts, no material omissions
  0.75 — mostly correct, minor omissions or phrasing differences
  0.5 — partially correct, one major fact wrong or missing
  0.25 — tangentially related, mostly incorrect
  0.0 — unrelated, fabricated, or contradicts the reference

Return ONLY valid JSON in this exact shape:
{"score": <float>, "rationale": "<one sentence>"}
"""


@dataclass
class Judgment:
    score: float
    rationale: str
    raw: str = ""


class LLMJudge:
    def __init__(
        self,
        model: str,
        rubric: str = DEFAULT_RUBRIC,
        temperature: float = 0.0,
    ):
        self.model = model
        self.rubric = rubric
        self.temperature = temperature

    def judge(self, input_: str, expected: str, actual: str) -> Judgment:
        prompt = (
            f"QUESTION:\n{input_}\n\n"
            f"REFERENCE:\n{expected}\n\n"
            f"MODEL ANSWER:\n{actual}\n\n"
            f"Return the JSON now."
        )
        resp = providers.call(
            model=self.model,
            prompt=prompt,
            system=self.rubric,
            temperature=self.temperature,
            max_tokens=256,
        )
        return _parse_judgment(resp.text)


class RubricJudge(LLMJudge):
    """Convenience wrapper for a custom rubric."""

    def __init__(self, model: str, rubric: str, temperature: float = 0.0):
        super().__init__(model=model, rubric=rubric, temperature=temperature)


def _parse_judgment(text: str) -> Judgment:
    # Extract the first {...} block — LLMs often wrap it in prose.
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = float(data.get("score", 0.0))
            rationale = str(data.get("rationale", ""))
            return Judgment(score=_clamp01(score), rationale=rationale, raw=text)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback: if judge just said "1.0" or "correct", try to salvage.
    number = re.search(r"([01](?:\.\d+)?)", text)
    if number:
        return Judgment(
            score=_clamp01(float(number.group(1))),
            rationale="(unparsed judge output)",
            raw=text,
        )
    return Judgment(score=0.0, rationale="(judge returned no score)", raw=text)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
