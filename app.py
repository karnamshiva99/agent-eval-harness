"""Gradio UI for Agent Eval Harness.

Run locally:  python app.py
Deploy free:  push this repo to Hugging Face Spaces (SDK: gradio).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from src import ABCompare, EvalHarness, LLMJudge, load_dataset
from src.providers import has_credentials


SAMPLE_MODELS = [
    "groq/llama-3.1-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-haiku-latest",
    "gemini/gemini-1.5-flash",
    "ollama/llama3.1",
]

DEFAULT_DATASET = str(Path(__file__).parent / "datasets" / "example_qa.jsonl")


def run_single_eval(model: str, judge_model: str, dataset_path: str, max_cases: int):
    if not has_credentials(model):
        return (
            f"⚠ No credentials for {model}. Set the relevant env var "
            f"(OPENAI_API_KEY / GROQ_API_KEY / etc) or use ollama/... for local.",
            None,
            None,
        )
    if not has_credentials(judge_model):
        return (
            f"⚠ No credentials for judge {judge_model}. See note above.",
            None,
            None,
        )
    try:
        dataset = load_dataset(dataset_path)
        if max_cases > 0:
            dataset.cases = dataset.cases[:max_cases]
        harness = EvalHarness(
            model=model,
            dataset=dataset,
            judge=LLMJudge(model=judge_model),
        )
        result = harness.run()
        summary = result.summary()
        rows = [
            {
                "id": r.id[:8],
                "lang": r.lang,
                "score": round(r.score, 3),
                "latency_ms": round(r.latency_ms),
                "input": r.input[:80],
                "actual": r.actual[:120],
                "rationale": r.rationale[:120],
                "error": r.error[:80],
            }
            for r in result.rows
        ]
        df = pd.DataFrame(rows)
        out_path = Path("runs") / f"{model.replace('/', '_')}.json"
        result.save(out_path)
        return summary, df, str(out_path)
    except Exception as e:  # pragma: no cover
        return f"Error: {e}", None, None


def run_ab_compare(
    model_a: str,
    model_b: str,
    judge_model: str,
    dataset_path: str,
    max_cases: int,
):
    for m in (model_a, model_b, judge_model):
        if not has_credentials(m):
            return f"⚠ No credentials for {m}.", None
    try:
        dataset = load_dataset(dataset_path)
        if max_cases > 0:
            dataset.cases = dataset.cases[:max_cases]
        ab = ABCompare(
            model_a=model_a,
            model_b=model_b,
            dataset=dataset,
            judge=LLMJudge(model=judge_model),
        )
        verdict = ab.run()
        df = pd.DataFrame(
            [
                {
                    "metric": "A wins",
                    "value": verdict.a_wins,
                },
                {
                    "metric": "B wins",
                    "value": verdict.b_wins,
                },
                {
                    "metric": "Ties",
                    "value": verdict.ties,
                },
                {
                    "metric": "A mean score",
                    "value": round(verdict.a_score_mean, 3),
                },
                {
                    "metric": "B mean score",
                    "value": round(verdict.b_score_mean, 3),
                },
                {
                    "metric": "Delta (A - B)",
                    "value": round(verdict.delta, 3),
                },
                {
                    "metric": "p-value",
                    "value": round(verdict.p_value, 4),
                },
                {
                    "metric": "Significant (p<0.05)",
                    "value": verdict.significant,
                },
            ]
        )
        return verdict.summary(), df
    except Exception as e:  # pragma: no cover
        return f"Error: {e}", None


with gr.Blocks(title="Agent Eval Harness") as demo:
    gr.Markdown(
        "# Agent Eval Harness\n"
        "A production-grade evaluation layer for LLM agents. "
        "[GitHub](https://github.com/karnamshiva/agent-eval-harness) · "
        "Set `GROQ_API_KEY` (free) or any LiteLLM-supported provider."
    )

    with gr.Tab("Single eval"):
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    SAMPLE_MODELS,
                    value="groq/llama-3.1-70b-versatile",
                    label="Model under test",
                    allow_custom_value=True,
                )
                judge = gr.Dropdown(
                    SAMPLE_MODELS,
                    value="groq/llama-3.1-8b-instant",
                    label="Judge model",
                    allow_custom_value=True,
                )
                dataset = gr.Textbox(value=DEFAULT_DATASET, label="Dataset JSONL path")
                max_cases = gr.Slider(
                    minimum=1, maximum=100, value=5, step=1, label="Max cases"
                )
                run_btn = gr.Button("Run evaluation", variant="primary")
            with gr.Column():
                summary_out = gr.Textbox(label="Summary", lines=7)
                saved_path = gr.Textbox(label="Saved run", interactive=False)
        rows_out = gr.DataFrame(label="Per-case results")
        run_btn.click(
            run_single_eval,
            inputs=[model, judge, dataset, max_cases],
            outputs=[summary_out, rows_out, saved_path],
        )

    with gr.Tab("A/B compare"):
        with gr.Row():
            model_a = gr.Dropdown(
                SAMPLE_MODELS,
                value="groq/llama-3.1-70b-versatile",
                label="Model A",
                allow_custom_value=True,
            )
            model_b = gr.Dropdown(
                SAMPLE_MODELS,
                value="groq/llama-3.1-8b-instant",
                label="Model B",
                allow_custom_value=True,
            )
        judge_ab = gr.Dropdown(
            SAMPLE_MODELS,
            value="groq/llama-3.1-8b-instant",
            label="Judge",
            allow_custom_value=True,
        )
        dataset_ab = gr.Textbox(value=DEFAULT_DATASET, label="Dataset JSONL path")
        max_cases_ab = gr.Slider(
            minimum=1, maximum=100, value=5, step=1, label="Max cases"
        )
        ab_btn = gr.Button("Run A/B", variant="primary")
        ab_summary = gr.Textbox(label="Verdict", lines=5)
        ab_table = gr.DataFrame(label="Details")
        ab_btn.click(
            run_ab_compare,
            inputs=[model_a, model_b, judge_ab, dataset_ab, max_cases_ab],
            outputs=[ab_summary, ab_table],
        )

    gr.Markdown(
        "### Why this exists\n"
        "Read the design story in the project README. "
        "Built by [Shivakumar Karnam](https://www.linkedin.com/in/shivakumarkarnam)."
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
