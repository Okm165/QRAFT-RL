"""Build RAFT training examples from the local index and a golden set.

Oracle context selection is a semantic judgement: the configured local LLM
decides which retrieved context supports the expected answer. The code never
compares expected terms with context text.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from preprocess.jsonl import (
    JsonValue,
    json_texts,
    path_set,
    read_jsonl,
    record_to_json,
    write_json,
    write_jsonl,
)
from rag.fts import SearchHit, search
from telemetry import Telemetry, disabled


class EvidenceJudge(Protocol):
    def evidence_index(self, question: str, expected: str, context: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ContextRecord:
    doc_id: str
    source: str
    section: str
    page: int
    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class RaftRecord:
    id: str
    question: str
    contexts: tuple[ContextRecord, ...]
    answer: str
    reasoning: str
    oracle_index: int
    status: str
    weight: float
    oracle_supported: bool
    absent_verified: bool


@dataclass(frozen=True, slots=True)
class RaftSummary:
    records: int
    skipped: int
    absent: int
    weight_mean: float
    reasoning_chars_mean: float


class SkipExample(Exception):
    """Raised when a golden row has no judge-supported oracle context."""


def _context(hit: SearchHit) -> ContextRecord:
    return ContextRecord(
        doc_id=hit.doc_id,
        source=hit.source,
        section=hit.section,
        page=hit.page,
        kind=hit.kind,
        text=hit.text,
    )


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _support(text: str, expected: str) -> bool:
    """Deterministic provenance gate; the judge remains the semantic decision."""
    normalized_expected = _normal(expected)
    normalized_text = _normal(text)
    if not normalized_expected:
        return False
    if normalized_expected in normalized_text:
        return True
    expected_terms = set(normalized_expected.split())
    text_terms = set(normalized_text.split())
    return len(expected_terms & text_terms) / len(expected_terms) >= 0.6


def _quote(text: str, expected: str, evidence: str) -> str:
    """Return a verbatim quote from the oracle context.

    The question stage validates evidence against the source chunk, but the
    judge may select a different oracle chunk. Fall back to the sentence in the
    oracle context with the strongest expected-term overlap so the target never
    teaches a quote that is absent from the context.
    """
    if evidence and _normal(evidence) in _normal(text):
        return evidence
    expected_terms = set(_normal(expected).split())
    best = ""
    best_score = 0.0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        terms = set(_normal(sentence).split())
        if not terms:
            continue
        score = len(expected_terms & terms) / max(len(expected_terms), 1)
        if score > best_score:
            best = sentence
            best_score = score
    return best if best_score > 0 else text.strip()[:300]


def _oracle_index(
    hits: tuple[SearchHit, ...],
    question: str,
    expected: str,
    source_chunk_id: str,
    judge: EvidenceJudge,
) -> int:
    for position, hit in enumerate(hits):
        if (
            hit.record_type == "chunk"
            and hit.record_id == source_chunk_id
            and _support(hit.text, expected)
        ):
            return position
    context = "\n\n".join(
        f"[{index}] {hit.text}" for index, hit in enumerate(hits, start=1)
    )
    index = judge.evidence_index(question, expected, context)
    if 1 <= index <= len(hits) and _support(hits[index - 1].text, expected):
        return index - 1
    for position, hit in enumerate(hits):
        if _support(hit.text, expected):
            return position
    return -1


def _record(
    index: int,
    row: dict[str, JsonValue],
    hits: tuple[SearchHit, ...],
    distractors: int,
    seed: int,
    absent_every: int,
    answer_template: str,
    refusal: str,
    judge: EvidenceJudge,
) -> RaftRecord:
    question = str(row["question"])
    answer_terms = json_texts(row.get("expected_terms", []))
    if not answer_terms:
        raise SkipExample("no expected terms")
    expected = answer_terms[0]
    oracle_index = _oracle_index(
        hits,
        question,
        expected,
        str(row.get("chunk_id", "")),
        judge,
    )
    if oracle_index < 0:
        raise SkipExample("no judge-supported oracle context")
    absent = absent_every > 0 and index % absent_every == absent_every - 1
    if absent:
        chosen = [
            hit
            for position, hit in enumerate(hits)
            if position != oracle_index and not _support(hit.text, expected)
        ][: distractors + 1]
        if len(chosen) != distractors + 1:
            raise SkipExample("not enough verified-absent contexts")
    else:
        chosen = [
            hit for position, hit in enumerate(hits) if position != oracle_index
        ][:distractors]
        chosen.insert(0, hits[oracle_index])
    if not chosen:
        raise SkipExample("no contexts")
    order = list(range(len(chosen)))
    random.Random(seed + index).shuffle(order)
    contexts = tuple(_context(chosen[position]) for position in order)
    reasoning = str(row.get("reasoning", "")).strip()
    evidence = str(row.get("evidence", "")).strip()
    quality_value = row.get("quality", 3)
    quality = (
        int(quality_value)
        if isinstance(quality_value, int) and not isinstance(quality_value, bool)
        else 3
    )
    quality = max(1, min(5, quality))
    if absent:
        answer = refusal
        reasoning = "The provided context does not contain the required evidence."
        oracle_position = -1
        weight = 0.8
    else:
        oracle_position = order.index(0)
        evidence = _quote(hits[oracle_index].text, expected, evidence)
        if not reasoning:
            reasoning = (
                f"The supporting evidence is in context [{oracle_position + 1}]."
            )
        answer = (
            answer_template.replace("%REASONING%", reasoning)
            .replace("%ANSWER%", expected)
            .replace("%INDEX%", str(oracle_position + 1))
            .replace("%EVIDENCE%", evidence)
        )
        weight = max(0.5, min(1.0, quality / 5.0))
    return RaftRecord(
        id=f"r{index + 1:03d}",
        question=question,
        contexts=contexts,
        answer=answer,
        reasoning=reasoning,
        oracle_index=oracle_position,
        status="absent" if absent else "oracle",
        weight=weight,
        oracle_supported=not absent,
        absent_verified=absent,
    )


def build(
    db_path: Path,
    golden_path: Path,
    out_path: Path,
    distractors: int,
    seed: int,
    judge: EvidenceJudge,
    answer_template: str,
    refusal: str,
    absent_every: int = 5,
    rejected_path: Path = Path(""),
    telemetry: Telemetry = disabled("raft"),
) -> RaftSummary:
    records: list[RaftRecord] = []
    rejected: list[dict[str, JsonValue]] = []
    skipped = 0
    for index, row in enumerate(read_jsonl(golden_path)):
        question = str(row["question"])
        if not json_texts(row.get("expected_terms", [])):
            skipped += 1
            rejected.append({"question": question, "reason": "no expected terms"})
            continue
        # Retrieve extra candidates so a verified-absent example can still
        # carry the same number of contexts as an oracle example. A context
        # count mismatch would let the model learn "few contexts => refuse".
        hits = search(db_path, question, distractors + 5)
        if not hits:
            skipped += 1
            rejected.append({"question": question, "reason": "no retrieval hits"})
            continue
        try:
            record = _record(
                index,
                dict(row),
                hits,
                distractors,
                seed,
                absent_every,
                answer_template,
                refusal,
                judge,
            )
        except SkipExample as exc:
            skipped += 1
            rejected.append({"question": question, "reason": str(exc)})
            continue
        except (OSError, RuntimeError, ValueError) as exc:
            skipped += 1
            rejected.append(
                {"question": question, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        records.append(record)
        telemetry.emit(
            "record",
            raft_id=record.id,
            status=record.status,
            contexts=len(record.contexts),
            oracle_index=record.oracle_index,
            answer_chars=len(record.answer),
            reasoning_chars=len(record.reasoning),
            weight=round(record.weight, 4),
            oracle_supported=record.oracle_supported,
            absent_verified=record.absent_verified,
        )
    if not records:
        raise ValueError("no RAFT examples with source-grounded contexts")
    write_jsonl(out_path, (record_to_json(record) for record in records))
    if path_set(rejected_path):
        write_jsonl(rejected_path, rejected)
    absent = sum(1 for record in records if record.status == "absent")
    weights = [record.weight for record in records]
    summary = RaftSummary(
        records=len(records),
        skipped=skipped,
        absent=absent,
        weight_mean=round(sum(weights) / len(weights), 6) if weights else 0.0,
        reasoning_chars_mean=round(
            sum(len(record.reasoning) for record in records) / len(records), 2
        ),
    )
    summary_json = record_to_json(summary)
    write_json(out_path.parent / "summary.json", summary_json)
    telemetry.emit("summary", **summary_json)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local RAFT training examples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/processed/rag.sqlite"))
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/raft.jsonl"))
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--absent-every", type=int, default=5)
    parser.add_argument("--rejected", type=Path, default=Path(""))
    parser.add_argument("--telemetry", type=Path, default=Path(""))
    args = parser.parse_args()
    from cli.config import load_config

    config = load_config(args.config)
    telemetry = (
        Telemetry(args.telemetry, "raft", args.telemetry.parent.parent.name)
        if path_set(args.telemetry)
        else disabled("raft")
    )
    summary = build(
        args.db,
        args.golden,
        args.out,
        args.distractors,
        args.seed,
        config.judge(),
        config.raft_answer_template,
        config.answer_refusal,
        args.absent_every,
        args.rejected,
        telemetry,
    )
    print(
        f"raft: {summary.records} examples, {summary.absent} oracle-absent, "
        f"{summary.skipped} skipped (see rejected.jsonl), "
        f"weight mean {summary.weight_mean:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
