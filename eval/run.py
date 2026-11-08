"""Evaluate retrieval on a held-out golden set.

Evidence recall is a semantic judgement made by the configured local LLM.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from preprocess.jsonl import JsonValue, json_texts, path_set, read_jsonl
from rag.fts import SearchHit, search
from telemetry import Telemetry, disabled


class EvidenceJudge(Protocol):
    def evidence_index(self, question: str, expected: str, context: str) -> int: ...


@dataclass(frozen=True, slots=True)
class EvalResult:
    total: int
    evidence_hits: int


def _evaluate_row(
    db_path: Path,
    row: Mapping[str, JsonValue],
    top_k: int,
    judge: EvidenceJudge,
) -> tuple[bool, float, tuple[SearchHit, ...]]:
    started = time.perf_counter()
    question = str(row["question"])
    expected = ", ".join(json_texts(row.get("expected_terms", [])))
    found = search(db_path, question, top_k)
    context = "\n\n".join(
        f"[{index}] {item.text}" for index, item in enumerate(found, start=1)
    )
    support = judge.evidence_index(question, expected, context) if expected else 0
    evidence_hit = 1 <= support <= len(found)
    return evidence_hit, time.perf_counter() - started, found


def evaluate(
    db_path: Path,
    golden_path: Path,
    top_k: int,
    judge: EvidenceJudge,
    telemetry: Telemetry = disabled("eval"),
    workers: int = 1,
) -> EvalResult:
    rows = tuple(read_jsonl(golden_path))
    total = 0
    evidence_hits = 0
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        results = pool.map(
            lambda row: _evaluate_row(db_path, row, top_k, judge),
            rows,
        )
        for row, (evidence_hit, seconds, found) in zip(rows, results):
            question = str(row["question"])
            expected = ", ".join(json_texts(row.get("expected_terms", [])))
            print(f"{'HIT ' if evidence_hit else 'MISS'}  {question}")
            if expected:
                print(f"      expected: {expected}")
            for item in found[:3]:
                print(f"      {item.source} | {item.section} | p{item.page}")
            total += 1
            evidence_hits += int(evidence_hit)
            telemetry.timed(
                "question",
                seconds,
                question=question,
                evidence_hit=evidence_hit,
            )
    return EvalResult(total=total, evidence_hits=evidence_hits)


def _recall(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a golden set.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/processed/rag.sqlite"))
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fail-below", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--telemetry", type=Path, default=Path(""))
    args = parser.parse_args()
    from cli.config import load_config

    telemetry = (
        Telemetry(args.telemetry, "eval", args.telemetry.parent.parent.name)
        if path_set(args.telemetry)
        else disabled("eval")
    )
    result = evaluate(
        args.db,
        args.golden,
        args.top_k,
        load_config(args.config).judge(),
        telemetry,
        args.workers,
    )
    evidence = _recall(result.evidence_hits, result.total)
    print(
        f"evidence recall@{args.top_k}: {result.evidence_hits}/{result.total} "
        f"= {evidence:.3f}"
    )
    telemetry.emit(
        "summary",
        total=result.total,
        evidence_hits=result.evidence_hits,
        evidence_recall=round(evidence, 6),
    )
    return 1 if evidence < args.fail_below else 0


if __name__ == "__main__":
    raise SystemExit(main())
