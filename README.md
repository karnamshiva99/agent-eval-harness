---
title: Agent Eval Harness
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Eval harness for LLM agents — A/B + regression gates
---

# Agent Eval Harness

> **A production-grade evaluation harness for LLM agents — the layer you build before you ship a model to real users.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HF Spaces Demo](https://img.shields.io/badge/🤗-Try%20on%20HF%20Spaces-yellow)](https://huggingface.co/spaces/shiva585/agent-eval-harness)

## Why this exists

When I owned the Alexa Answers LLM post-training pipeline at Panasonic, the single most important thing we built was **not the model** — it was the A/B evaluation harness that gated every production rollout. We refused to ship without it because regressions would reach millions of users otherwise. That harness caught three bugs before they hit production and was the reason we confidently deployed a 4-bit quantized Mixtral-8x7B replacement for the BERT baseline, lifting automation +15.81%, auto-approving 500K answers, and saving $100K in labeling.

After leaving, I kept seeing small teams ship LLM agents with basically no eval layer. They vibe-check 10 prompts, deploy, and then find out in production. The open-source tools that exist (DeepEval, Ragas, LangSmith) are great but either heavyweight, SaaS-coupled, or focused on RAG. I wanted something simple enough to drop into any project, but rigorous enough to actually catch regressions.

**Agent Eval Harness** is that minimum viable eval layer:

- ✅ Provider-agnostic (OpenAI / Anthropic / Groq / Ollama via [LiteLLM](https://github.com/BerriAI/litellm))
- ✅ LLM-as-judge with configurable rubrics
- ✅ A/B comparison between models or prompts
- ✅ Regression detection against a saved baseline
- ✅ Multilingual consistency checks
- ✅ Gradio UI + programmatic API
- ✅ Zero paid infra — runs locally or on Hugging Face Spaces free tier

## Quickstart

```bash
git clone https://github.com/karnamshiva99/agent-eval-harness
cd agent-eval-harness
pip install -r requirements.txt

# Run the Gradio UI (opens at http://localhost:7860)
python app.py

# Or run programmatically
python examples/run_eval.py
```

## What it does

### 1. Score any LLM against a test set

```python
from harness import EvalHarness

harness = EvalHarness(
    model="groq/llama-3.1-70b-versatile",
    dataset="datasets/example_qa.jsonl",
    judge="groq/llama-3.1-8b-instant",
)
results = harness.run()
print(results.summary())  # accuracy, latency p50/p95, cost estimate
```

### 2. A/B compare two candidates

```python
from harness import ABCompare

ab = ABCompare(
    model_a="openai/gpt-4o-mini",
    model_b="anthropic/claude-3-5-haiku",
    dataset="datasets/example_qa.jsonl",
)
verdict = ab.run()
# verdict: { a_wins: 42, b_wins: 51, ties: 7, significant: True (p<0.05) }
```

### 3. Gate rollouts with regression detection

```python
from harness import RegressionGate

gate = RegressionGate(
    baseline="runs/production-v1.2.json",
    candidate="runs/candidate-v1.3.json",
    tolerance={"accuracy": -0.02, "latency_p95": 1.20},  # allow -2pp acc, +20% latency
)
if not gate.passes():
    raise Exception(f"Regression detected: {gate.violations()}")
```

### 4. Multilingual consistency

```python
from harness import MultilingualCheck

check = MultilingualCheck(
    model="groq/llama-3.1-70b-versatile",
    languages=["en", "es", "fr", "de", "ja"],
    test_case="What is the capital of France?",
    expected_semantic="Paris",
)
# verifies the model gives semantically equivalent answers across languages
```

## The design philosophy

### Three decisions that matter

**1. Build the harness before the model.** The single highest-leverage thing I learned at Panasonic is that you should not train, fine-tune, or deploy anything until the measurement layer is trustworthy. If you cannot detect a regression, you *will* ship one. Everything in this harness is designed so you can plug it in day one, even if your model is still a stub.

**2. LLM-as-judge with calibration, not as oracle.** LLM judges drift. The harness supports calibrating your judge against a small set of human-labeled examples (we do a simple Cohen's kappa agreement check) before you trust it on large runs. If agreement is low, the harness warns you and refuses to produce a confidence score.

**3. Reproducibility over convenience.** Every run serializes: model, prompt, judge, dataset hash, seed, and exact outputs. You can re-run any historical evaluation bit-for-bit, which is what makes rollback gates actually work.

### What it is not

- **Not a SaaS.** No account, no API key to our servers, no telemetry. You run it, you own the data.
- **Not a framework.** No class hierarchies to subclass. Plain Python functions and dataclasses.
- **Not a benchmark curator.** Bring your own test cases. The `datasets/example_qa.jsonl` is a seed, not a gold standard.

## Architecture

```
┌────────────────────────────────────────────────────┐
│                     Gradio UI                      │
│          (app.py - optional, for humans)           │
└────────────────────────────────────────────────────┘
                          ▲
                          │
┌────────────────────────────────────────────────────┐
│                      Harness                       │
│    (src/harness.py - run loop, result I/O)         │
└────────────────────────────────────────────────────┘
       ▲              ▲              ▲         ▲
       │              │              │         │
┌───────────┐  ┌──────────────┐  ┌────────┐  ┌──────────┐
│ Providers │  │    Judges    │  │Metrics │  │ Datasets │
│ (LiteLLM) │  │ (LLM + code) │  │        │  │ (JSONL)  │
└───────────┘  └──────────────┘  └────────┘  └──────────┘
```

## Roadmap

- [x] Core harness + LLM-as-judge
- [x] A/B comparison with bootstrap significance
- [x] Regression gate with YAML config
- [x] Gradio UI
- [x] Multilingual consistency
- [ ] Agent trace evaluation (multi-step workflows)
- [ ] Cost/latency SLO dashboards
- [ ] Integration with W&B + MLflow
- [ ] Judge calibration helpers (Cohen's kappa, labeler UI)

## About me

I'm Shivakumar Karnam — ML engineer with 5+ years shipping production LLM systems. I built this harness because I kept explaining the same evaluation patterns to every team I talked to. If it's useful to you, let me know — and if you want help wiring up a real eval layer for your agents, I'm available: [LinkedIn](https://www.linkedin.com/in/shivakumarkarnam) · [Email](mailto:karnamshiva85@gmail.com).

## License

MIT — use it, fork it, ship it.
