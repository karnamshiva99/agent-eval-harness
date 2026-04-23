# Design notes: why Agent Eval Harness looks the way it does

*A companion piece to the [README](../README.md). Written as a blog post.*

## The question that drove this

About a year ago I was on a call with a founder of an AI agent startup. They were three months from their first enterprise pilot and they showed me their test suite: a Notion page with 18 example prompts and hand-written "good" / "bad" labels. They were about to fine-tune a model and ship it.

I asked: "How will you know if the new model is better than the old one?"

The answer was: "We'll run the 18 prompts and see."

Two weeks later they shipped, the agent regressed on a subset of customer queries, and nobody noticed for a week because nobody was measuring anything beyond "does it still run." This is not a rare story. The hardest part of shipping LLM systems is not the model — it is the measurement layer you build around it. I built this harness to be the minimum viable version of that layer, because every team I talked to was reinventing it badly.

## Principle 1 — build the harness before the model

At Panasonic, the Alexa Answers pipeline I owned had one non-negotiable rule: *nothing ships to production without passing the A/B harness.* Not fine-tuning experiments, not prompt tweaks, not a new quantization. The harness had to exist and be trustworthy before we touched the model. This sounds obvious but it's violated constantly because "set up evals" feels like meta-work when you have a model to ship.

The payoff is invisible right up until the moment it saves you. When I moved our production stack from a BERT baseline to a 4-bit quantized Mixtral-8x7B (LoRA-tuned), the harness caught three regressions on multilingual consistency and one on false-approval rate. We fixed them before they reached users. That one deploy alone saved the team weeks of incident response.

So the first design goal of Agent Eval Harness is: **you can drop it into a project that has no model yet.** The `EvalHarness` class runs against any provider, including `ollama/llama3.1` running on your laptop. You can build the dataset and the rubric before you write any production code.

## Principle 2 — the judge needs calibration, not trust

LLM-as-judge is fashionable and genuinely useful, but I've watched multiple teams treat it as an oracle. It is not. Judge models have biases: they favor longer answers, they are inconsistent on edge cases, and they drift when you change the rubric. If you're running 10,000 evals against a judge, you need to know whether the judge agrees with humans on a held-out set.

The harness has a hook for judge calibration (Cohen's kappa against a small labeled set). If the kappa is below a threshold, the harness *refuses* to produce a confidence number. I would rather have an honest "we cannot measure this yet" than a fake 0.84 accuracy.

## Principle 3 — reproducibility is what makes rollbacks work

Every `RunResult` serializes:

- the full model string (`groq/llama-3.1-70b-versatile`)
- the judge model
- the dataset path *and* the dataset hash
- the timestamp and random seed
- every input, every output, every score, every rationale

This matters because *the reason you have a gate is so you can roll back.* A rollback that says "last week's version was better" only works if you can re-run last week's version against today's test set and get the same numbers. Without reproducibility, your gate is a suggestion.

## Principle 4 — multilingual consistency is a release blocker

One pattern that burned us at Panasonic: the English eval looked great, the Spanish eval looked great, but on the joint run we saw a 30-point accuracy gap in French because of one prompt token that tokenized differently. `MultilingualCheck` exists because an averaged accuracy hides this. If the spread between your best and worst language is > 0.15, the harness flags it and the gate fails. This is the kind of thing you only learn by shipping into a multilingual product and getting burned once.

## What's explicitly *not* in the scope

- **A benchmark.** The bundled `example_qa.jsonl` is 15 trivia questions. It's a smoke test so the tool runs end-to-end, not an evaluation standard. Bring your own dataset.
- **A model registry.** Integrate with W&B or MLflow for that — they are good at it.
- **A SaaS.** Your evaluation data is frequently your most sensitive IP. The harness runs where you run it, end of story.

## What comes next

On the roadmap, in order:

1. **Agent trace evaluation** — scoring multi-step agent trajectories, not just single-turn Q&A
2. **SLO dashboards** — tying latency and cost to SLOs so the gate can block on p95 regressions
3. **Judge calibration helpers** — a small Gradio tab for labeling 50 examples by hand and spitting out a kappa
4. **W&B / MLflow adapters** — one-line `result.log_to_wandb()`

If you ship LLMs in production and any of this resonates, I'd love to hear what you'd add, remove, or change. Open an issue or email me: [karnamshiva85@gmail.com](mailto:karnamshiva85@gmail.com).

— Shivakumar Karnam
