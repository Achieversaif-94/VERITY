"""LLM providers with fallback and JSON-schema validation.

Groq primary (llama-3.3-70b-versatile), Gemini fallback. Each provider gets
one retry on parse/validation failure and on 429/5xx. If all providers fail,
raise LLMError -- the caller decides whether that's a task failure or a replan trigger.

Every call is logged to the trace with the provider's own token counts.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .trace import Tracer

M = TypeVar("M", bound=BaseModel)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_PROVIDER_NAMES = ("groq", "gemini")


class LLMError(RuntimeError):
    pass


def _load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env()


def _parse_json(text: str) -> Any:
    """Tolerate code fences around JSON. Reject anything else non-JSON."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return json.loads(t)


def _groq_call(system: str, user: str, timeout: float = 30.0) -> tuple[str, dict]:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    model = os.environ.get("VERITY_GROQ_MODEL", "llama-3.3-70b-versatile")
    r = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    return text, {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def _gemini_call(system: str, user: str, timeout: float = 30.0) -> tuple[str, dict]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    model = os.environ.get("VERITY_GEMINI_MODEL", "gemini-1.5-flash-latest")
    r = httpx.post(
        GEMINI_URL.format(model=model),
        params={"key": key},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {}) or {}
    return text, {
        "prompt_tokens": int(usage.get("promptTokenCount", 0)),
        "completion_tokens": int(usage.get("candidatesTokenCount", 0)),
    }


def _provider(name: str) -> Any:
    """Dynamic lookup so monkeypatch on verity.llm._<name>_call takes effect."""
    fn = globals().get(f"_{name}_call")
    if fn is None:
        raise LLMError(f"provider {name} not registered")
    return fn


def complete_json(
    system: str,
    user: str,
    schema: type[M],
    *,
    tracer: Tracer | None = None,
    node_id: str | None = None,
) -> M:
    errors: list[str] = []
    for name in _PROVIDER_NAMES:
        fn = _provider(name)
        try:
            return _call_with_retry(name, fn, system, user, schema, tracer, node_id)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise LLMError("all providers failed :: " + " | ".join(errors))


def _call_with_retry(
    name: str,
    fn: Any,
    system: str,
    user: str,
    schema: type[M],
    tracer: Tracer | None,
    node_id: str | None,
) -> M:
    last_err: Exception | None = None
    for attempt in range(2):
        t0 = time.monotonic()
        try:
            text, usage = fn(system, user)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            latency = (time.monotonic() - t0) * 1000
            _log(tracer, node_id, name, attempt, latency, f"http_{status}", 0, 0)
            if status == 429 or status >= 500:
                time.sleep(min(2 ** attempt, 8))
                last_err = e
                continue
            raise
        except LLMError:
            raise
        except Exception as e:
            _log(tracer, node_id, name, attempt, (time.monotonic() - t0) * 1000,
                 f"transport_{type(e).__name__}", 0, 0)
            last_err = e
            time.sleep(min(2 ** attempt, 4))
            continue

        latency = (time.monotonic() - t0) * 1000
        try:
            obj = _parse_json(text)
            validated = schema.model_validate(obj)
        except (json.JSONDecodeError, ValidationError) as e:
            _log(tracer, node_id, name, attempt, latency,
                 f"invalid_{type(e).__name__}",
                 usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            last_err = e
            user = user + f"\n\nPrevious attempt failed with: {e}. Return only valid JSON matching the schema."
            continue

        _log(tracer, node_id, name, attempt, latency, "ok",
             usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        return validated
    raise LLMError(f"{type(last_err).__name__}: {last_err}")


def _log(
    tracer: Tracer | None,
    node_id: str | None,
    provider: str,
    attempt: int,
    latency_ms: float,
    status: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    if tracer is None:
        return
    tracer.emit(
        "llm_call",
        node_id=node_id,
        provider=provider,
        attempt=attempt,
        latency_ms=round(latency_ms, 1),
        status=status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )