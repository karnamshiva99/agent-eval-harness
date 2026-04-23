"""Core harness: run an LLM against a dataset, score with a judge, serialize.

Design principles:
  - Every run is reproducible: model, prompt, judge, dataset hash, seed, outputs.
  - Plain dataclasses, no framework.
  - Works with any LiteLLM-supported provider (OpenAI, Anthropic, Groq, Ollama).
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import metrics as metrics_mod
from . import providers
from .datasets import Dataset, load_dataset
from .judges import Judgment, LLMJudge


@dataclass
class RowResult:
    id: str
    input: str
    expected: str
    actual: str
    score: float
    rationale: str
    latency_ms: float
    cost_usd: float
    tags: list[str] = field(default_factory=list)
    lang: str = "en"
    error: str = ""


@dataclass
class RunResult:
    model: str
    judge: str
    dataset_path: str
    dataset_hash: str
    timestamp: str
    seed: int
    rows: list[RowResult] = field(default_factory=list)

    def summary(self) -> str:
        m = metrics_mod.compute(self)
        return m.summary()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "judge": self.judge,
            "dataset_path": self.dataset_path,
            "dataset_hash": self.dataset_hash,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "rows": [asdict(r) for r in self.rows],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [RowResult(**r) for r in data.pop("rows", [])]
        return cls(**data, rows=rows)


class EvalHarness:
    def __init__(
        self,
        model: str,
        dataset: str | Dataset,
        judge: str | LLMJudge,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
    ):
        self.model = model
        self.dataset = (
            dataset if isinstance(dataset, Dataset) else load_dataset(dataset)
        )
        self.judge = (
            judge if isinstance(judge, LLMJudge) else LLMJudge(model=judge)
        )
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def run(self) -> RunResult:
        random.seed(self.seed)
        result = RunResult(
            model=self.model,
            judge=self.judge.model,
            dataset_path=self.dataset.path,
            dataset_hash=self.dataset.hash(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=self.seed,
        )
        for case in self.dataset:
            try:
                resp = providers.call(
                    model=self.model,
                    prompt=case.input,
                    system=self.system,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                judgment: Judgment = self.judge.judge(
                    input_=case.input,
                    expected=case.expected,
                    actual=resp.text,
                )
                result.rows.append(
                    RowResult(
                        id=case.id,
                        input=case.input,
                        expected=case.expected,
                        actual=resp.text,
                        score=judgment.score,
                        rationale=judgment.rationale,
                        latency_ms=resp.latency_ms,
                        cost_usd=resp.cost_usd,
                        tags=case.tags,
                        lang=case.lang,
                    )
                )
            except Exception as e:
                result.rows.append(
                    RowResult(
                        id=case.id,
                        input=case.input,
                        expected=case.expected,
                        actual="",
                        score=0.0,
                        rationale="",
                        latency_ms=0.0,
                        cost_usd=0.0,
                        tags=case.tags,
                        lang=case.lang,
                        error=str(e),
                    )
                )
        return result


@dataclass
class ABVerdict:
    a_wins: int
    b_wins: int
    ties: int
    a_score_mean: float
    b_score_mean: float
    delta: float
    significant: bool
    p_value: float

    def summary(self) -> str:
        return (
            f"A wins: {self.a_wins}   B wins: {self.b_wins}   ties: {self.ties}\n"
            f"A mean: {self.a_score_mean:.3f}   B mean: {self.b_score_mean:.3f}   "
            f"delta: {self.delta:+.3f}\n"
            f"significant (p<0.05): {self.significant}   p={self.p_value:.4f}"
        )


class ABCompare:
    """Run both models on the same dataset and judge each pairwise."""

    def __init__(
        self,
        model_a: str,
        model_b: str,
        dataset: str | Dataset,
        judge: str | LLMJudge,
        system: str | None = None,
        seed: int = 42,
    ):
        self.harness_a = EvalHarness(
            model=model_a, dataset=dataset, judge=judge, system=system, seed=seed
        )
        self.harness_b = EvalHarness(
            model=model_b, dataset=dataset, judge=judge, system=system, seed=seed
        )

    def run(self) -> ABVerdict:
        ra = self.harness_a.run()
        rb = self.harness_b.run()
        by_id_b = {r.id: r for r in rb.rows}
        a_wins = b_wins = ties = 0
        diffs: list[float] = []
        for row_a in ra.rows:
            row_b = by_id_b.get(row_a.id)
            if row_b is None:
                continue
            d = row_a.score - row_b.score
            diffs.append(d)
            if d > 1e-6:
                a_wins += 1
            elif d < -1e-6:
                b_wins += 1
            else:
                ties += 1
        a_mean = statistics.mean(r.score for r in ra.rows) if ra.rows else 0.0
        b_mean = statistics.mean(r.score for r in rb.rows) if rb.rows else 0.0
        p = _bootstrap_pvalue(diffs) if diffs else 1.0
        return ABVerdict(
            a_wins=a_wins,
            b_wins=b_wins,
            ties=ties,
            a_score_mean=a_mean,
            b_score_mean=b_mean,
            delta=a_mean - b_mean,
            significant=p < 0.05,
            p_value=p,
        )


def _bootstrap_pvalue(diffs: list[float], n_boot: int = 2000) -> float:
    """Two-sided bootstrap p-value for H0: mean(diffs) == 0."""
    if not diffs:
        return 1.0
    observed = statistics.mean(diffs)
    rng = random.Random(0)
    n = len(diffs)
    extremes = 0
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        m = statistics.mean(sample)
        if abs(m - observed) >= abs(observed):
            extremes += 1
    return extremes / n_boot


@dataclass
class GateResult:
    passed: bool
    violations: list[str]

    def passes(self) -> bool:
        return self.passed


class RegressionGate:
    """Compare candidate run against a saved baseline with tolerance thresholds.

    Tolerance spec:
      {"accuracy": -0.02}          allows up to -2pp regression in accuracy
      {"latency_p95": 1.20}        allows up to +20% regression in p95 latency
      {"cost_usd_total": 1.50}     allows up to 50% cost increase
    """

    DIRECTION = {
        "accuracy": "higher_is_better",
        "pass_at_threshold": "higher_is_better",
        "latency_ms_p50": "lower_is_better",
        "latency_ms_p95": "lower_is_better",
        "cost_usd_total": "lower_is_better",
        "error_rate": "lower_is_better",
    }
    # alias map so users can write `latency_p95` instead of `latency_ms_p95`
    ALIASES = {"latency_p50": "latency_ms_p50", "latency_p95": "latency_ms_p95"}

    def __init__(
        self,
        baseline: str | Path | RunResult,
        candidate: str | Path | RunResult,
        tolerance: dict[str, float],
    ):
        self.baseline = (
            baseline
            if isinstance(baseline, RunResult)
            else RunResult.load(baseline)
        )
        self.candidate = (
            candidate
            if isinstance(candidate, RunResult)
            else RunResult.load(candidate)
        )
        self.tolerance = {
            self.ALIASES.get(k, k): v for k, v in tolerance.items()
        }

    def check(self) -> GateResult:
        base_m = metrics_mod.compute(self.baseline)
        cand_m = metrics_mod.compute(self.candidate)
        violations: list[str] = []
        for metric, tol in self.tolerance.items():
            base_v = getattr(base_m, metric, None)
            cand_v = getattr(cand_m, metric, None)
            if base_v is None or cand_v is None:
                violations.append(f"{metric}: missing in baseline or candidate")
                continue
            direction = self.DIRECTION.get(metric)
            if direction == "higher_is_better":
                # tol is an additive delta (usually negative, e.g. -0.02)
                delta = cand_v - base_v
                if delta < tol:
                    violations.append(
                        f"{metric} dropped {delta:+.4f} (baseline {base_v:.4f}, "
                        f"candidate {cand_v:.4f}, tolerance {tol:+.4f})"
                    )
            elif direction == "lower_is_better":
                # tol is a multiplicative ratio (e.g. 1.20 = allow +20%)
                if base_v == 0:
                    if cand_v > 0:
                        violations.append(
                            f"{metric} increased from 0 to {cand_v}"
                        )
                    continue
                ratio = cand_v / base_v
                if ratio > tol:
                    violations.append(
                        f"{metric} ratio {ratio:.3f}x > tolerance {tol:.3f}x "
                        f"(baseline {base_v:.4f}, candidate {cand_v:.4f})"
                    )
        return GateResult(passed=not violations, violations=violations)

    def passes(self) -> bool:
        return self.check().passed

    def violations(self) -> list[str]:
        return self.check().violations
