"""A/B compare two models on the same dataset with bootstrap significance."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import ABCompare, LLMJudge


def main() -> None:
    ab = ABCompare(
        model_a="groq/llama-3.3-70b-versatile",
        model_b="groq/llama-3.1-8b-instant",
        dataset=str(ROOT / "datasets" / "example_qa.jsonl"),
        judge=LLMJudge(model="groq/llama-3.1-8b-instant"),
    )
    verdict = ab.run()
    print(verdict.summary())


if __name__ == "__main__":
    main()
