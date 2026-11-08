"""Quality collection and readable run summary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from preprocess.jsonl import JsonValue
from tools.run_layout import layout
from tools import telemetry_report
from tools.run_metrics import (
    _behavior_report,
    _chunk_report,
    _concept_report,
    _fact_report,
    _files,
    _golden_report,
    _gpu_report,
    _json_file,
    _raft_report,
    _retrieval_report,
)


def _behavior_log(logs: Path, name: str) -> dict[str, JsonValue]:
    for candidate in (f"behavior-{name}.log", f"behavior_{name}.log"):
        path = logs / candidate
        if path.is_file():
            return _behavior_report(path)
    return {}


def _int(value: JsonValue) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _token_health(train: dict[str, JsonValue]) -> dict[str, JsonValue]:
    max_length = _int(train.get("max_length", 0))
    max_tokens = _int(train.get("max_tokens", 0))
    truncated = _int(train.get("truncated", 0))
    examples = _int(train.get("examples", 0))
    return {
        "max_length": max_length,
        "max_tokens": max_tokens,
        "margin_tokens": max(0, max_length - max_tokens),
        "truncated": truncated,
        "truncation_rate": round(truncated / examples, 6) if examples else 0.0,
        "prompt_tokens_max": _int(train.get("prompt_tokens_max", 0)),
        "answer_tokens_max": _int(train.get("answer_tokens_max", 0)),
        "mean_weight": train.get("mean_weight", 0.0),
        "weighted_examples": _int(train.get("weighted_examples", 0)),
    }


def collect_quality(root: Path) -> dict[str, JsonValue]:
    run = layout(root)
    train = _json_file(run.train / "adapter" / "summary.json")
    return cast(dict[str, JsonValue], {
        "run": root.name,
        "manifest": _json_file(root / "manifest.json"),
        "preprocess": {
            "audit": _json_file(run.preprocess / "audit.json"),
            "normalize": _json_file(run.preprocess / "normalize_summary.json"),
            "structure": _json_file(run.preprocess / "structure_summary.json"),
        },
        "golden": _golden_report(run.golden / "golden.generated.jsonl"),
        "golden_summary": _json_file(run.golden / "summary.json"),
        "holdout": _golden_report(run.golden / "holdout.jsonl"),
        "chunks": _chunk_report(run.preprocess / "chunks.jsonl"),
        "facts": _fact_report(run.preprocess / "facts.jsonl"),
        "concepts": _concept_report(run.preprocess / "concepts.jsonl"),
        "raft": _raft_report(run.raft / "raft.jsonl"),
        "raft_summary": _json_file(run.raft / "summary.json"),
        "retrieval": _retrieval_report(run.logs / "eval.log")
        or _retrieval_report(run.logs / "03_retrieval.log"),
        "train": train,
        "token_health": _token_health(train),
        "behavior": {
            "base": _behavior_log(run.logs, "base"),
            "adapter": _behavior_log(run.logs, "adapter"),
        },
        "hardware": _json_file(run.metrics / "hardware_report.json"),
        "gpu": _gpu_report(run.metrics / "gpu.csv"),
        "stages": telemetry_report.stage_timings(root),
        "telemetry": telemetry_report.collect(root),
        "files": _files(root),
    })


def _section(title: str, value: JsonValue) -> tuple[str, ...]:
    return (
        f"## {title}",
        "",
        "```json",
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    )


def write_summary(root: Path, quality: Mapping[str, JsonValue]) -> Path:
    lines = [
        f"# Run {root.name}",
        "",
        f"- root: `{root}`",
        f"- manifest: `{root / 'manifest.json'}`",
        f"- quality: `{root / 'metrics' / 'quality.json'}`",
        "",
    ]
    for title, key in (
        ("Preprocess", "preprocess"),
        ("Golden", "golden"),
        ("Question acceptance", "golden_summary"),
        ("Held-out set", "holdout"),
        ("Chunks", "chunks"),
        ("Facts", "facts"),
        ("Concepts", "concepts"),
        ("RAFT", "raft"),
        ("RAFT gates", "raft_summary"),
        ("Retrieval", "retrieval"),
        ("Training", "train"),
        ("Token health", "token_health"),
        ("Behavior", "behavior"),
        ("Stage telemetry", "telemetry"),
        ("Stage timings", "stages"),
        ("Hardware", "hardware"),
        ("GPU", "gpu"),
        ("Files", "files"),
    ):
        lines.extend(_section(title, quality[key]))
    path = root / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
