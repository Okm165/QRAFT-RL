"""Parse a JSON object from free-form LLM output."""

from __future__ import annotations

import json
import re
from typing import cast

from preprocess.jsonl import JsonValue

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_DOUBLE_COMMA = re.compile(r",\s*,")
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")


def _object_text(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response has no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _candidates(text: str) -> tuple[str, ...]:
    body = _object_text(text)
    repaired = _TRAILING_COMMA.sub(r"\1", body)
    repaired = _DOUBLE_COMMA.sub(",", repaired)
    repaired = _BARE_KEY.sub(r'\1"\2":', repaired)
    return (body, repaired)


def _matching_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position
    return -1


def _scan_objects(text: str) -> tuple[dict[str, JsonValue], ...]:
    objects: list[dict[str, JsonValue]] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        end = _matching_end(text, start)
        if end < 0:
            break
        try:
            value = json.loads(text[start : end + 1])
        except ValueError:
            value = None
        if isinstance(value, dict):
            objects.append(cast(dict[str, JsonValue], value))
        # Continue one character after the start so a malformed outer object
        # does not hide the valid objects nested inside it.
        index = start + 1
    return tuple(objects)


def _merge_objects(
    objects: tuple[dict[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    chunks: list[JsonValue] = []
    facts: list[JsonValue] = []
    concepts: list[JsonValue] = []
    for value in objects:
        for key, target in (
            ("chunks", chunks),
            ("facts", facts),
            ("concepts", concepts),
        ):
            items = value.get(key)
            if isinstance(items, list):
                target.extend(item for item in items if isinstance(item, dict))
        if "chunks" in value or "facts" in value:
            continue
        if "name" in value and "definition" in value:
            concepts.append(value)
        elif "section" in value and "text" in value:
            chunks.append(value)
        elif "text" in value:
            facts.append(value)
    return {"chunks": chunks, "facts": facts, "concepts": concepts}


def parse_object(text: str) -> dict[str, JsonValue]:
    last = ""
    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError as exc:
            last = str(exc)
            continue
        if not isinstance(value, dict):
            raise ValueError("LLM response JSON is not an object")
        return cast(dict[str, JsonValue], value)
    merged = _merge_objects(_scan_objects(text))
    if merged["chunks"] or merged["facts"] or merged["concepts"]:
        return merged
    raise ValueError(f"LLM response is not valid JSON: {last}")
