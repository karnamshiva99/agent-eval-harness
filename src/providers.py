"""Provider-agnostic LLM inference via LiteLLM.

Model strings follow LiteLLM conventions:
  - openai/gpt-4o-mini
  - anthropic/claude-3-5-haiku
  - groq/llama-3.1-70b-versatile
  - ollama/llama3.1
  - gemini/gemini-1.5-flash

Set API keys via environment variables:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, ...
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

try:
    from litellm import completion as _completion  # type: ignore
except Exception:  # pragma: no cover
    _completion = None


@dataclass
class LLMResponse:
    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw: Any = None


class LLMError(RuntimeError):
    pass


def call(
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> LLMResponse:
    """Single LLM call. Returns normalized LLMResponse or raises LLMError."""
    if _completion is None:
        raise LLMError(
            "litellm is not installed. Run `pip install -r requirements.txt`."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    try:
        resp = _completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as e:
        raise LLMError(f"{model}: {e}") from e
    latency_ms = (time.perf_counter() - start) * 1000

    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    # LiteLLM attaches `_hidden_params.response_cost` when pricing is known.
    cost_usd = 0.0
    hidden = getattr(resp, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        cost_usd = float(hidden.get("response_cost") or 0.0)

    return LLMResponse(
        text=text.strip(),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        raw=resp,
    )


def has_credentials(model: str) -> bool:
    """Return True if the provider's key env var appears to be set."""
    prefix = model.split("/", 1)[0] if "/" in model else model
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "ollama": None,  # local, no key required
    }
    key = env_map.get(prefix, None)
    if key is None:
        return prefix == "ollama"
    return bool(os.environ.get(key))
