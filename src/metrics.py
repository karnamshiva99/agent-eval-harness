"""Aggregate metrics over an eval run + multilingual consistency check."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import providers
from .judges import LLMJudge

if TYPE_CHECKING:
    from .harness import RunResult


@dataclass
class Metrics:
    n: int
    accuracy: float
    pass_at_threshold: float
    latency_ms_p50: float
    latency_ms_p95: float
    cost_usd_total: float
    error_rate: float
    tag_breakdown: dict = field(default_factory=dict)
    lang_breakdown: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"n={self.n}",
            f"accuracy={self.accuracy:.3f}",
            f"pass@0.75={self.pass_at_threshold:.3f}",
            f"latency p50={self.latency_ms_p50:.0f}ms  p95={self.latency_ms_p95:.0f}ms",
            f"total cost=${self.cost_usd_total:.4f}",
            f"error rate={self.error_rate:.3f}",
        ]
        return "\n".join(lines)


def compute(result: "RunResult", pass_threshold: float = 0.75) -> Metrics:
    rows = result.rows
    n = len(rows)
    errored = [r for r in rows if r.error]
    scored = [r for r in rows if not r.error]

    if n == 0:
        return Metrics(
            n=0,
            accuracy=0.0,
            pass_at_threshold=0.0,
            latency_ms_p50=0.0,
            latency_ms_p95=0.0,
            cost_usd_total=0.0,
            error_rate=0.0,
        )

    scores = [r.score for r in scored]
    latencies = [r.latency_ms for r in scored]
    accuracy = sum(scores) / len(scores) if scores else 0.0
    pass_rate = (
        sum(1 for s in scores if s >= pass_threshold) / len(scores) if scores else 0.0
    )
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = _percentile(latencies, 95) if latencies else 0.0
    cost = sum(r.cost_usd for r in scored)
    error_rate = len(errored) / n if n else 0.0

    tag_breakdown = _group_mean(scored, key=lambda r: r.tags)
    lang_breakdown = _group_mean(scored, key=lambda r: [r.lang] if r.lang else [])

    return Metrics(
        n=n,
        accuracy=accuracy,
        pass_at_threshold=pass_rate,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        cost_usd_total=cost,
        error_rate=error_rate,
        tag_breakdown=tag_breakdown,
        lang_breakdown=lang_breakdown,
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _group_mean(rows, key) -> dict:
    buckets: dict[str, list[float]] = {}
    for r in rows:
        for k in key(r) or []:
            buckets.setdefault(k, []).append(r.score)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


@dataclass
class ConsistencyResult:
    languages: list[str]
    scores: dict[str, float]
    min_score: float
    max_score: float
    spread: float
    consistent: bool

    def summary(self) -> str:
        rows = [f"  {lang}: {score:.3f}" for lang, score in self.scores.items()]
        verdict = "CONSISTENT" if self.consistent else "INCONSISTENT"
        return (
            f"Multilingual check: {verdict} (spread={self.spread:.3f})\n"
            + "\n".join(rows)
        )


class MultilingualCheck:
    """Runs the same test case across multiple languages and checks that the
    judge scores are all within `tolerance` of each other.

    This mirrors the multilingual-consistency guard we ran in the Alexa Answers
    pipeline — if English scores 0.92 and Spanish scores 0.41, that's a release
    blocker even if the average looks fine.
    """

    TRANSLATIONS_PROMPT = (
        "Translate the following text to {lang} accurately, preserving meaning. "
        "Return ONLY the translated text.\n\nTEXT:\n{text}"
    )

    def __init__(
        self,
        model: str,
        judge: LLMJudge,
        languages: list[str],
        tolerance: float = 0.15,
    ):
        self.model = model
        self.judge = judge
        self.languages = languages
        self.tolerance = tolerance

    def run(self, test_case: str, expected: str) -> ConsistencyResult:
        scores: dict[str, float] = {}
        for lang in self.languages:
            if lang == "en":
                prompt = test_case
                ref = expected
            else:
                prompt = providers.call(
                    model=self.model,
                    prompt=self.TRANSLATIONS_PROMPT.format(lang=lang, text=test_case),
                    temperature=0.0,
                    max_tokens=512,
                ).text
                ref = providers.call(
                    model=self.model,
                    prompt=self.TRANSLATIONS_PROMPT.format(lang=lang, text=expected),
                    temperature=0.0,
                    max_tokens=512,
                ).text
            actual = providers.call(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
            ).text
            judgment = self.judge.judge(input_=prompt, expected=ref, actual=actual)
            scores[lang] = judgment.score
        lo = min(scores.values())
        hi = max(scores.values())
        return ConsistencyResult(
            languages=self.languages,
            scores=scores,
            min_score=lo,
            max_score=hi,
            spread=hi - lo,
            consistent=(hi - lo) <= self.tolerance,
        )
