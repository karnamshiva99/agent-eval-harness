# Quickstart

## 1. Install

```bash
git clone https://github.com/karnamshiva/agent-eval-harness
cd agent-eval-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get a free LLM API key

The examples default to **Groq** (free tier, 30 req/min on Llama 3.1 70B).

1. Sign up at [console.groq.com](https://console.groq.com/)
2. Create an API key
3. Export it: `export GROQ_API_KEY=gsk_...`

Other supported providers (set the relevant env var):

| Provider | Env var | Free tier |
|----------|---------|-----------|
| Groq | `GROQ_API_KEY` | Yes |
| Google Gemini | `GEMINI_API_KEY` | Yes (1500 req/day) |
| OpenAI | `OPENAI_API_KEY` | No (paid only) |
| Anthropic | `ANTHROPIC_API_KEY` | $5 signup credit |
| Ollama (local) | — | 100% free, local |

## 3. Run your first eval

```bash
python examples/run_eval.py
```

Expected output (times vary):

```
n=15
accuracy=0.933
pass@0.75=0.933
latency p50=420ms  p95=850ms
total cost=$0.0012
error rate=0.000

Serialized run to runs/example.json
```

## 4. Try the Gradio UI

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## 5. Ship your own dataset

Create `datasets/my_cases.jsonl` (one JSON per line):

```json
{"input": "What does our product do?", "expected": "It is an eval harness for LLM agents.", "tags": ["product"], "lang": "en"}
{"input": "¿Qué hace nuestro producto?", "expected": "Es un arnés de evaluación para agentes LLM.", "tags": ["product"], "lang": "es"}
```

Then:

```python
from src import EvalHarness, LLMJudge

harness = EvalHarness(
    model="groq/llama-3.1-70b-versatile",
    dataset="datasets/my_cases.jsonl",
    judge=LLMJudge(model="groq/llama-3.1-8b-instant"),
)
result = harness.run()
print(result.summary())
result.save("runs/my-first-real-run.json")
```

## 6. Set up a regression gate in CI

Commit a baseline run once (`runs/baseline.json`). Then in CI:

```bash
python examples/regression_gate.py \
  --baseline runs/baseline.json \
  --candidate runs/latest.json
```

The script exits non-zero if any tolerance is violated — perfect for a GitHub Actions check.
