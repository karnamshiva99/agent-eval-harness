"""Minimal end-to-end example.

Set GROQ_API_KEY (free tier at https://console.groq.com/) then:

    python examples/run_eval.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import EvalHarness, LLMJudge


def main() -> None:
    harness = EvalHarness(
        model="groq/llama-3.1-70b-versatile",
        dataset=str(ROOT / "datasets" / "example_qa.jsonl"),
        judge=LLMJudge(model="groq/llama-3.1-8b-instant"),
    )
    result = harness.run()
    print(result.summary())
    out = ROOT / "runs" / "example.json"
    result.save(out)
    print(f"\nSerialized run to {out}")


if __name__ == "__main__":
    main()
