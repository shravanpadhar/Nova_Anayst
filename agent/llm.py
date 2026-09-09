"""Provider-agnostic LLM client. This is the ONE module to touch to swap
providers: Anthropic, OpenAI, or a local OSS model via Ollama. Every node in
agent/graph.py calls `complete()` -- nothing elsewhere imports a provider SDK
directly, so "any LLM" is a real property of this codebase, not a slide claim.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


def complete(system: str, user: str, max_tokens: int = 1500, temperature: float = 0.0, retries: int = 2) -> str:
    """Single text-in/text-out completion call. Swap the provider by setting
    LLM_PROVIDER in .env; add a new branch here for any other SDK.

    Retries transient failures (network hiccups, rate limits) with a short
    backoff -- a single flaky API call should not take down an entire agent
    run. Re-raises after the final attempt so a genuinely broken call (bad
    API key, model not found) still surfaces as an error rather than being
    silently swallowed.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if LLM_PROVIDER == "anthropic":
                return _complete_anthropic(system, user, max_tokens, temperature)
            elif LLM_PROVIDER == "openai":
                return _complete_openai(system, user, max_tokens, temperature)
            elif LLM_PROVIDER == "ollama":
                return _complete_ollama(system, user, max_tokens, temperature)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use anthropic|openai|ollama.")
        except ValueError:
            raise  # config error, not transient -- don't retry
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
    raise last_err  # type: ignore[misc]


def complete_json(system: str, user: str, max_tokens: int = 1500, temperature: float = 0.0, retries: int = 2) -> dict[str, Any]:
    """Like complete(), but asks for and parses a JSON object response.
    Falls back to extracting the first {...} block if the model wraps it in
    prose. LLM output is occasionally empty or malformed (rate limiting,
    truncation, a model that ignores the format instruction) -- retry a
    couple of times before giving up, and NEVER raise out to the caller: a
    transient bad response from one node must not crash the whole agent run.
    Every caller in agent/graph.py already reads this via .get()/.setdefault()
    with sensible defaults, so an empty dict on total failure degrades
    gracefully rather than crashing.
    """
    json_system = system + "\n\nYou MUST respond with ONLY a valid JSON object, no prose, no markdown fences."
    for attempt in range(retries + 1):
        try:
            text = complete(json_system, user, max_tokens=max_tokens, temperature=temperature)
            return _extract_json(text)
        except Exception:
            if attempt == retries:
                return {}
            continue
    return {}


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


_anthropic_client = None


def _complete_anthropic(system: str, user: str, max_tokens: int, temperature: float) -> str:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=LLM_API_KEY)

    # Note: newer Anthropic SDK/model versions (1.x SDK, Claude 5 family)
    # dropped the `temperature` sampling param from messages.create, so it's
    # intentionally not passed here even though complete() accepts it --
    # kept in the shared signature for parity with the other providers below.
    resp = _anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


_openai_client = None


def _complete_openai(system: str, user: str, max_tokens: int, temperature: float) -> str:
    global _openai_client
    from openai import OpenAI

    if _openai_client is None:
        _openai_client = OpenAI(api_key=LLM_API_KEY)

    resp = _openai_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def _complete_ollama(system: str, user: str, max_tokens: int, temperature: float) -> str:
    import requests

    resp = requests.post(
        os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat"),
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
