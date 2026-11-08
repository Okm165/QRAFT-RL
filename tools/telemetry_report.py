"""Aggregate per-stage JSONL telemetry into bounded quality summaries."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

from preprocess.jsonl import JsonValue, read_jsonl

STAGES = (
    "preprocess",
    "questions",
    "raft",
    "train",
    "eval",
    "behavior-base",
    "behavior-adapter",
)


def _number(value: JsonValue) -> tuple[bool, float]:
    if isinstance(value, bool):
        return (True, float(value))
    if isinstance(value, (int, float)):
        return (True, float(value))
    return (False, 0.0)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _stats(values: list[float]) -> dict[str, JsonValue]:
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "p50": round(_percentile(values, 0.50), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "max": round(max(values), 6),
    }


def summarize(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    events = [dict(row) for row in read_jsonl(path)]
    if not events:
        return {"events": 0}
    counts = Counter(str(event.get("event", "")) for event in events)
    numeric: dict[str, list[float]] = {}
    errors: Counter[str] = Counter()
    summary: dict[str, JsonValue] = {}
    for event in events:
        if event.get("event") == "summary":
            summary = {
                key: value
                for key, value in event.items()
                if key not in {"ts", "stage", "run", "event"}
            }
            continue
        for key, value in event.items():
            if key in {"ts", "stage", "run", "event"}:
                continue
            present, number = _number(value)
            if present:
                numeric.setdefault(key, []).append(number)
        error = event.get("error")
        if isinstance(error, str) and error:
            errors[error.split(":", 1)[0]] += 1
    return {
        "events": len(events),
        "kinds": dict(sorted(counts.items())),
        "errors": dict(sorted(errors.items())),
        "summary": summary,
        "numeric": {
            key: _stats(values)
            for key, values in sorted(numeric.items())
            if len(values) >= 2
        },
    }


def collect(root: Path) -> dict[str, JsonValue]:
    return {
        stage: summarize(root / "metrics" / f"{stage}.jsonl")
        for stage in STAGES
    }


def stage_timings(root: Path) -> dict[str, JsonValue]:
    timings: dict[str, JsonValue] = {}
    for path in sorted((root / "metrics").glob("*.json")):
        if path.name in {"quality.json", "hardware_report.json"}:
            continue
        try:
            value = cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            timings[path.stem] = value
    return timings
