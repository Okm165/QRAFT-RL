"""Live run progress and data-quality dashboard.

Reads the telemetry a running pipeline writes under ``metrics/`` and prints a
compact health view: stage state, throughput, acceptance rates, rejection
reasons, loss/gradient health, and token margins. It never mutates the run.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from preprocess.jsonl import JsonValue, read_jsonl


def _rows(path: Path) -> tuple[Mapping[str, JsonValue], ...]:
    if not path.exists():
        return ()
    return tuple(read_jsonl(path))


def _object(path: Path) -> Mapping[str, JsonValue]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return cast(Mapping[str, JsonValue], value) if isinstance(value, dict) else {}


def _int(row: Mapping[str, JsonValue], key: str) -> int:
    value = row.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _float(row: Mapping[str, JsonValue], key: str) -> float:
    value = row.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _text(row: Mapping[str, JsonValue], key: str) -> str:
    value = row.get(key, "")
    return value if isinstance(value, str) else ""


def _epoch(row: Mapping[str, JsonValue]) -> float:
    text = _text(row, "ts")
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _command_arg(command: str, flag: str) -> int:
    parts = command.split()
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            try:
                return int(parts[index + 1])
            except ValueError:
                return 0
    return 0


def _stage_lines(run: Path) -> list[str]:
    rows = _rows(run / "metrics" / "pipeline.jsonl")
    starts: dict[str, Mapping[str, JsonValue]] = {}
    finished: dict[str, Mapping[str, JsonValue]] = {}
    for row in rows:
        name = _text(row, "name")
        event = _text(row, "event")
        if event == "start":
            starts[name] = row
        elif event == "finish":
            finished[name] = row
    lines = ["stages:"]
    for name, start in starts.items():
        if name in finished:
            done = finished[name]
            state = "failed" if _int(done, "exit") else "done"
            lines.append(f"  {name:18s} {state:7s} {_float(done, 'seconds'):8.1f}s")
        else:
            elapsed = max(0.0, time.time() - _epoch(start))
            lines.append(f"  {name:18s} running {elapsed:8.1f}s")
    return lines


def _questions_lines(run: Path) -> list[str]:
    rows = _rows(run / "metrics" / "questions.jsonl")
    chunks = [row for row in rows if _text(row, "event") == "chunk"]
    generated = sum(_int(row, "generated") for row in chunks)
    accepted = sum(_int(row, "accepted") for row in chunks)
    multi_fact = sum(_int(row, "multi_fact") for row in chunks)
    rejected = sum(_int(row, "rejected") for row in chunks)
    errors = sum(1 for row in chunks if _text(row, "error"))
    total_chunks = 0
    for row in rows:
        if _text(row, "event") == "start":
            total_chunks = _int(row, "chunks")
            break
    if not total_chunks:
        for row in _rows(run / "metrics" / "pipeline.jsonl"):
            if _text(row, "event") == "start" and _text(row, "name") == "questions":
                total_chunks = _command_arg(_text(row, "command"), "--chunks-limit")
    rate = accepted / generated if generated else 0.0
    lines = [
        "questions:",
        f"  chunks {len(chunks)}/{total_chunks or '?'}  generated {generated}"
        f"  accepted {accepted}  rejected {rejected}  rate {rate:.3f}"
        f"  errors {errors}",
    ]
    if accepted and any("multi_fact" in row for row in chunks):
        lines.append(
            f"  multi-fact {multi_fact}/{accepted} = {multi_fact / accepted:.3f}"
        )
    finished = any(
        _text(row, "event") == "finish" and _text(row, "name") == "questions"
        for row in _rows(run / "metrics" / "pipeline.jsonl")
    )
    if finished:
        lines.append("  complete")
    elif len(chunks) >= 2:
        first = _epoch(chunks[0])
        last = _epoch(chunks[-1])
        seconds = max(last - first, 1.0)
        per_chunk = seconds / (len(chunks) - 1)
        if total_chunks > len(chunks):
            lines.append(
                f"  {per_chunk:.0f}s/chunk  eta {per_chunk * (total_chunks - len(chunks)) / 60:.0f}m"
            )
    rejected_rows = _rows(run / "golden" / "rejected.jsonl")
    reasons = Counter(_text(row, "reason") for row in rejected_rows)
    if reasons:
        detail = ", ".join(f"{reason} {count}" for reason, count in reasons.most_common(4))
        lines.append(f"  rejection reasons: {detail}")
    return lines


def _preprocess_lines(run: Path) -> list[str]:
    rows = _rows(run / "metrics" / "preprocess.jsonl")
    windows = [row for row in rows if _text(row, "event") == "window"]
    if not windows:
        return []
    chunks = sum(_int(row, "chunks") for row in windows)
    facts = sum(_int(row, "facts") for row in windows)
    concepts = sum(_int(row, "concepts") for row in windows)
    errors = sum(1 for row in windows if _text(row, "error"))
    mean_seconds = sum(_float(row, "seconds") for row in windows) / len(windows)
    summary = _object(run / "preprocess" / "structure_summary.json")
    label = "concept_mentions"
    if summary:
        concepts = _int(summary, "concepts")
        label = "concepts"
    return [
        "preprocess:",
        f"  windows {len(windows)}  chunks {chunks}  facts {facts}"
        f"  {label} {concepts}  errors {errors}  mean {mean_seconds:.1f}s",
    ]


def _train_lines(run: Path) -> list[str]:
    rows = _rows(run / "metrics" / "train.jsonl")
    step_rows = [row for row in rows if _text(row, "event") == "step"]
    if not step_rows:
        return []
    by_step: dict[int, Mapping[str, JsonValue]] = {}
    for row in step_rows:
        by_step[_int(row, "step")] = row
    steps = [by_step[key] for key in sorted(by_step)]
    losses = [_float(row, "loss") for row in steps]
    grads = [_float(row, "grad_norm") for row in steps]
    evals = [row for row in rows if _text(row, "event") == "eval"]
    total_steps = 0
    for row in _rows(run / "metrics" / "pipeline.jsonl"):
        if _text(row, "event") == "start" and _text(row, "name") == "train":
            total_steps = _command_arg(_text(row, "command"), "--max-steps")
    eta = ""
    if len(steps) >= 2 and total_steps > len(steps):
        first = _epoch(steps[0])
        last = _epoch(steps[-1])
        per_step = max(last - first, 1.0) / (len(steps) - 1)
        eta = f"  eta {per_step * (total_steps - len(steps)) / 60:.0f}m"
    return [
        "train:",
        f"  steps {len(steps)}/{total_steps or '?'}  epoch {_float(steps[-1], 'epoch'):.2f}"
        f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}"
        f"  grad max {max(grads):.2f}  lr {_float(steps[-1], 'learning_rate'):.2e}{eta}",
        *(
            [
                f"  eval loss {_float(evals[-1], 'eval_loss'):.4f}"
                f" @ step {_int(evals[-1], 'step')}"
            ]
            if evals
            else []
        ),
        *(
            [
                f"  vram {_float(steps[-1], 'vram_allocated_mb') / 1024:.1f}"
                f" alloc / {_float(steps[-1], 'vram_reserved_mb') / 1024:.1f}"
                " reserved GiB"
            ]
            if "vram_allocated_mb" in steps[-1]
            else []
        ),
    ]


def _summary_lines(run: Path) -> list[str]:
    lines: list[str] = []
    for name in ("eval", "behavior-base", "behavior-adapter", "raft"):
        rows = _rows(run / "metrics" / f"{name}.jsonl")
        summaries = [row for row in rows if _text(row, "event") == "summary"]
        if summaries:
            summary = summaries[-1]
            fields = ", ".join(
                f"{key} {value}"
                for key, value in sorted(summary.items())
                if key not in {"event", "run", "stage", "ts"}
            )
            lines.append(f"{name}: {fields}")
    return lines


def _health_lines(run: Path) -> list[str]:
    flags: list[str] = []
    chunks = [
        row
        for row in _rows(run / "metrics" / "questions.jsonl")
        if _text(row, "event") == "chunk"
    ]
    generated = sum(_int(row, "generated") for row in chunks)
    accepted = sum(_int(row, "accepted") for row in chunks)
    multi_fact = sum(_int(row, "multi_fact") for row in chunks)
    if generated and accepted / generated < 0.25:
        flags.append(f"low acceptance {accepted}/{generated}")
    if (
        any("multi_fact" in row for row in chunks)
        and accepted >= 20
        and multi_fact / accepted < 0.4
    ):
        flags.append(f"low multi-fact {multi_fact}/{accepted}")
    question_errors = sum(1 for row in chunks if _text(row, "error"))
    if question_errors:
        flags.append(f"{question_errors} question chunks failed")
    train = _rows(run / "metrics" / "train.jsonl")
    steps = [row for row in train if _text(row, "event") == "step"]
    if len(steps) >= 2:
        if _float(steps[-1], "loss") > _float(steps[0], "loss") * 1.5:
            flags.append("loss increased >50% from first step")
        if max(_float(row, "grad_norm") for row in steps) > 15.0:
            flags.append("grad norm >15")
    facts = [
        row
        for row in _rows(run / "metrics" / "preprocess.jsonl")
        if _text(row, "event") == "window" and _text(row, "error")
    ]
    if facts:
        flags.append(f"{len(facts)} preprocess windows failed")
    return ["health: " + ("; ".join(flags) if flags else "green")]


def _export_lines(run: Path) -> list[str]:
    log = run / "logs" / "export.log"
    if not log.is_file():
        return []
    text = log.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith("lora:"):
            return ["export:", f"  {line}"]
    return []


def dashboard(run: Path) -> str:
    run = run.resolve()
    lines = [f"run: {run.name}"]
    lines.extend(_stage_lines(run))
    lines.extend(_preprocess_lines(run))
    lines.extend(_questions_lines(run))
    lines.extend(_train_lines(run))
    lines.extend(_export_lines(run))
    lines.extend(_summary_lines(run))
    lines.extend(_health_lines(run))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live run progress and quality.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0.0)
    args = parser.parse_args()
    if args.watch <= 0:
        print(dashboard(args.run), flush=True)
        return 0
    try:
        while True:
            print(dashboard(args.run), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
