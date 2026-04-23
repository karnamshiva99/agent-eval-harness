"""Demonstrate using RegressionGate to block a bad deploy.

Typical usage in CI:

    python examples/regression_gate.py \
        --baseline runs/production-v1.2.json \
        --candidate runs/candidate-v1.3.json
"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import RegressionGate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    gate = RegressionGate(
        baseline=args.baseline,
        candidate=args.candidate,
        tolerance={
            "accuracy": -0.02,        # allow up to -2 percentage points
            "latency_p95": 1.20,      # allow up to +20% latency
            "cost_usd_total": 1.50,   # allow up to +50% cost
            "error_rate": 1.50,       # allow up to +50% errors
        },
    )
    result = gate.check()
    if result.passed:
        print("PASS — no regressions detected.")
        return 0
    print("FAIL — regression gate violations:")
    for v in result.violations:
        print(f"  - {v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
