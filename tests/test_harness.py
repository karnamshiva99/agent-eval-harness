"""Unit tests that do not require network or API keys."""

import json
from pathlib import Path

import pytest

from src.datasets import Dataset, TestCase, load_dataset
from src.harness import (
    ABVerdict,
    RegressionGate,
    RowResult,
    RunResult,
    _bootstrap_pvalue,
)
from src.judges import Judgment, _parse_judgment
from src.metrics import compute


def test_load_dataset_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text(
        json.dumps({"input": "q1", "expected": "a1"}) + "\n"
        + json.dumps({"input": "q2", "expected": "a2", "tags": ["t"]}) + "\n",
        encoding="utf-8",
    )
    ds = load_dataset(p)
    assert len(ds) == 2
    assert ds.cases[0].id  # auto-generated
    assert ds.cases[1].tags == ["t"]


def test_dataset_filter() -> None:
    ds = Dataset(
        cases=[
            TestCase(input="a", expected="A", tags=["x"], lang="en"),
            TestCase(input="b", expected="B", tags=["y"], lang="es"),
        ]
    )
    assert len(ds.filter(tag="x")) == 1
    assert len(ds.filter(lang="es")) == 1


def test_parse_judgment_valid_json() -> None:
    j = _parse_judgment('{"score": 0.8, "rationale": "close"}')
    assert j.score == 0.8
    assert j.rationale == "close"


def test_parse_judgment_wrapped_prose() -> None:
    j = _parse_judgment('Here is the grade:\n{"score": 1.0, "rationale": "exact"}\n')
    assert j.score == 1.0


def test_parse_judgment_clamps() -> None:
    j = _parse_judgment('{"score": 1.5, "rationale": ""}')
    assert j.score == 1.0
    j = _parse_judgment('{"score": -0.3, "rationale": ""}')
    assert j.score == 0.0


def test_parse_judgment_fallback_number() -> None:
    j = _parse_judgment("the score is 0.9 I think")
    assert j.score == 0.9


def test_metrics_basic() -> None:
    result = RunResult(
        model="m",
        judge="j",
        dataset_path="",
        dataset_hash="abc",
        timestamp="",
        seed=0,
        rows=[
            RowResult(id="1", input="", expected="", actual="", score=1.0,
                      rationale="", latency_ms=100, cost_usd=0.01, tags=["a"]),
            RowResult(id="2", input="", expected="", actual="", score=0.5,
                      rationale="", latency_ms=200, cost_usd=0.02, tags=["a"]),
            RowResult(id="3", input="", expected="", actual="", score=0.0,
                      rationale="", latency_ms=300, cost_usd=0.01, tags=["b"]),
        ],
    )
    m = compute(result)
    assert m.n == 3
    assert m.accuracy == pytest.approx(0.5)
    assert m.pass_at_threshold == pytest.approx(1 / 3)
    assert m.cost_usd_total == pytest.approx(0.04)
    assert m.tag_breakdown["a"] == pytest.approx(0.75)
    assert m.tag_breakdown["b"] == pytest.approx(0.0)


def test_regression_gate_pass_and_fail(tmp_path: Path) -> None:
    def make_run(path: Path, scores: list[float], latencies: list[float]) -> None:
        rows = [
            RowResult(
                id=str(i), input="", expected="", actual="",
                score=s, rationale="", latency_ms=l, cost_usd=0.0,
            )
            for i, (s, l) in enumerate(zip(scores, latencies))
        ]
        r = RunResult(
            model="m", judge="j", dataset_path="", dataset_hash="x",
            timestamp="", seed=0, rows=rows,
        )
        r.save(path)

    base_path = tmp_path / "base.json"
    cand_path = tmp_path / "cand.json"
    make_run(base_path, [1.0, 1.0, 1.0, 1.0], [100, 100, 100, 100])
    make_run(cand_path, [1.0, 1.0, 0.99, 1.0], [110, 110, 110, 110])
    gate = RegressionGate(
        base_path, cand_path, tolerance={"accuracy": -0.02, "latency_p95": 1.20}
    )
    assert gate.passes()

    make_run(cand_path, [0.5, 0.5, 0.5, 0.5], [300, 300, 300, 300])
    gate2 = RegressionGate(
        base_path, cand_path, tolerance={"accuracy": -0.02, "latency_p95": 1.20}
    )
    assert not gate2.passes()
    violations = gate2.violations()
    assert any("accuracy" in v for v in violations)
    assert any("latency_ms_p95" in v for v in violations)


def test_bootstrap_pvalue_symmetric() -> None:
    # diffs centered on 0 -> large p
    assert _bootstrap_pvalue([0.1, -0.1, 0.05, -0.05]) > 0.1
    # diffs clearly positive -> small p
    assert _bootstrap_pvalue([0.3, 0.4, 0.35, 0.28, 0.32]) < 0.05
