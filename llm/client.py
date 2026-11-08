"""Minimal OpenAI-compatible chat client using the standard library."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import cast

from preprocess.jsonl import JsonValue


@dataclass(frozen=True, slots=True)
class LlmSettings:
    base_url: str
    model: str
    api_key: str
    timeout: int
    max_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class LoraScale:
    id: int
    scale: float


def endpoints(base_url: str, base_urls: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in base_urls.split(",") if item.strip())
    return values if values else (base_url,)


def settings_list(
    urls: tuple[str, ...],
    model: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> tuple[LlmSettings, ...]:
    return tuple(
        LlmSettings(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for base_url in urls
    )


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _content(data: dict[str, JsonValue]) -> str:
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LLM choice is not an object")
    message = first.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("LLM message is not an object")
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response is empty")
    return content


def chat(
    settings: LlmSettings,
    system: str,
    user: str,
    lora: tuple[LoraScale, ...] = (),
) -> str:
    effective_lora = lora if lora else (LoraScale(0, 0.0),)
    payload: dict[str, JsonValue] = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "lora": [
            {"id": item.id, "scale": item.scale}
            for item in effective_lora
        ],
    }
    request = urllib.request.Request(
        _endpoint(settings.base_url, "chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            data = cast(
                dict[str, JsonValue],
                json.loads(response.read().decode("utf-8")),
            )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM HTTP {error.code} from {settings.base_url}: {detail}"
        ) from error
    return _content(data)
