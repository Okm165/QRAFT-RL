"""Generate source-grounded, chain-of-thought question/answer pairs.

Generation and validation both use local LLM endpoints. Evidence must be a
verbatim quote from the source chunk; the configured judge decides support and
retrievability. Every accepted row keeps its source chunk, evidence quote,
short rationale, and judge quality.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from preprocess.jsonl import (
    JsonValue,
    path_set,
    read_jsonl,
    write_json,
    write_jsonl,
)
from rag.fts import search
from telemetry import Telemetry, disabled

from .client import LlmSettings, chat, endpoints, settings_list
from .judge import QuestionJudgement
from .structured import parse_object


class QuestionJudge(Protocol):
    def question(
        self,
        context: str,
        question: str,
        answer: str,
        reasoning: str = "",
    ) -> QuestionJudgement: ...


def _normal_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _normal_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.casefold()))


def _near_duplicate(tokens: frozenset[str], seen: list[frozenset[str]]) -> bool:
    for other in seen:
        union = tokens | other
        if union and len(tokens & other) / len(union) >= 0.8:
            return True
    return False


def _record_docs(record: Mapping[str, JsonValue]) -> set[str]:
    value = record.get("expected_doc_ids", [])
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _ask(
    settings: LlmSettings,
    context: str,
    system: str,
    prompt_template: str,
    count: int,
) -> dict[str, JsonValue]:
    prompt = prompt_template.replace("%CONTEXT%", context).replace(
        "%COUNT%", str(max(count, 1))
    )
    try:
        return parse_object(chat(settings, system, prompt))
    except ValueError:
        # Retry only a parse failure. A timeout/connection error should
        # reject the chunk instead of multiplying the wait.
        larger = replace(settings, max_tokens=max(settings.max_tokens * 2, 2048))
        return parse_object(
            chat(larger, system, prompt + "\nReturn one valid JSON object only.")
        )


def _items(value: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    raw = value.get("items")
    if isinstance(raw, list):
        return tuple(
            {str(key): item for key, item in row.items()}
            for row in raw
            if isinstance(row, dict)
        )
    if "question" in value:
        return (dict(value),)
    return ()


def _interleave(
    chunks: tuple[Mapping[str, JsonValue], ...],
) -> tuple[dict[str, JsonValue], ...]:
    groups: dict[str, list[dict[str, JsonValue]]] = {}
    for chunk in chunks:
        groups.setdefault(str(chunk.get("doc_id", "")), []).append(dict(chunk))
    if not groups:
        return ()
    ordered: list[dict[str, JsonValue]] = []
    for index in range(max(len(group) for group in groups.values())):
        for doc_id in sorted(groups):
            if index < len(groups[doc_id]):
                ordered.append(groups[doc_id][index])
    return tuple(ordered)


def _candidate(
    settings: LlmSettings,
    chunk: Mapping[str, JsonValue],
    db_path: Path,
    top_k: int,
    judge: QuestionJudge,
    system: str,
    prompt_template: str,
    per_chunk: int,
) -> tuple[str, list[dict[str, JsonValue]], list[dict[str, JsonValue]], str]:
    chunk_id = str(chunk.get("chunk_id", ""))
    context = str(chunk.get("text", "")).strip()
    if not context:
        return chunk_id, [], [], "empty chunk"
    started = time.perf_counter()
    try:
        value = _ask(settings, context, system, prompt_template, per_chunk)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return chunk_id, [], [], f"{type(exc).__name__}: {exc}"
    generation_ms = int((time.perf_counter() - started) * 1000)
    accepted: list[dict[str, JsonValue]] = []
    rejected: list[dict[str, JsonValue]] = []
    for item in _items(value):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        reasoning = str(item.get("reasoning", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        kind = str(item.get("kind", "lookup")).strip().casefold()
        if kind not in {"synthesis", "lookup"}:
            kind = "lookup"
        if not question or not answer:
            rejected.append({"chunk_id": chunk_id, "reason": "invalid question/answer"})
            continue
        if not reasoning:
            rejected.append({"chunk_id": chunk_id, "reason": "missing reasoning"})
            continue
        if not evidence:
            rejected.append({"chunk_id": chunk_id, "reason": "missing evidence"})
            continue
        if _normal_text(evidence) not in _normal_text(context):
            rejected.append(
                {
                    "chunk_id": chunk_id,
                    "question": question,
                    "reason": "evidence not in source",
                }
            )
            continue
        judge_started = time.perf_counter()
        try:
            judgement = judge.question(context, question, answer, reasoning)
        except (OSError, RuntimeError, ValueError) as exc:
            rejected.append(
                {"chunk_id": chunk_id, "reason": f"judge failed: {type(exc).__name__}: {exc}"}
            )
            continue
        judge_ms = int((time.perf_counter() - judge_started) * 1000)
        if not judgement.valid:
            rejected.append(
                {
                    "chunk_id": chunk_id,
                    "question": question,
                    "reason": "question not supported by source",
                    "quality": judgement.quality,
                }
            )
            continue
        evidence_index = 0
        if db_path.is_file():
            hits = search(db_path, question, top_k)
            for position, hit in enumerate(hits, start=1):
                if hit.record_type == "chunk" and hit.record_id == chunk_id:
                    evidence_index = position
                    break
            if evidence_index <= 0:
                rejected.append(
                    {
                        "chunk_id": chunk_id,
                        "question": question,
                        "reason": "source chunk not retrievable",
                        "quality": judgement.quality,
                    }
                )
                continue
        accepted.append(
            {
                "id": "",
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
                "evidence": evidence,
                "kind": kind,
                "multi_fact": judgement.multi_fact,
                "expected_doc_ids": [str(chunk.get("doc_id", ""))],
                "expected_terms": [answer],
                "chunk_id": chunk_id,
                "quality": judgement.quality,
                "judge_valid": judgement.valid,
                "judge_evidence_index": evidence_index,
                "generation_ms": generation_ms,
                "judge_ms": judge_ms,
            }
        )
    return chunk_id, accepted, rejected, ""


def generate(
    chunks_path: Path,
    out_path: Path,
    settings: tuple[LlmSettings, ...],
    limit: int,
    judge: QuestionJudge,
    system: str,
    prompt_template: str,
    db_path: Path = Path(""),
    top_k: int = 5,
    workers: int = 1,
    per_chunk: int = 1,
    chunks_limit: int = 0,
    holdout_every: int = 5,
    rejected_path: Path = Path(""),
    telemetry: Telemetry = disabled("questions"),
) -> int:
    if not settings:
        raise ValueError("at least one LLM endpoint is required")
    chunks = _interleave(
        tuple(
            chunk
            for chunk in read_jsonl(chunks_path)
            if str(chunk.get("kind", "")) != "code"
            and str(chunk.get("text", "")).strip()
        )
    )
    if chunks_limit > 0:
        chunks = chunks[:chunks_limit]
    if not chunks:
        raise ValueError("no source chunks available for question generation")
    telemetry.emit(
        "start",
        chunks=len(chunks),
        per_chunk=per_chunk,
        limit=limit,
    )
    records: list[dict[str, JsonValue]] = []
    rejected: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    seen_tokens: list[frozenset[str]] = []
    multi_fact_total = 0
    batch_size = max(workers * 2, 1)
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        for start in range(0, len(chunks), batch_size):
            if limit > 0 and len(records) >= limit:
                break
            batch = chunks[start : start + batch_size]
            jobs = tuple(
                (settings[index % len(settings)], chunk)
                for index, chunk in enumerate(batch, start=start)
            )
            results = pool.map(
                lambda job: _candidate(
                    job[0],
                    job[1],
                    db_path,
                    top_k,
                    judge,
                    system,
                    prompt_template,
                    per_chunk,
                ),
                jobs,
            )
            for chunk_id, accepted, rejected_rows, error in results:
                if error:
                    telemetry.emit(
                        "chunk",
                        chunk_id=chunk_id,
                        generated=0,
                        accepted=0,
                        rejected=0,
                        error=error,
                    )
                    print(f"skip {chunk_id}: {error}")
                    continue
                accepted_count = 0
                multi_fact_count = 0
                rejected_count = len(rejected_rows)
                for record in accepted:
                    if limit > 0 and len(records) >= limit:
                        break
                    key = _normal_question(str(record["question"]))
                    if key in seen:
                        rejected.append(
                            {
                                "chunk_id": chunk_id,
                                "question": record["question"],
                                "reason": "duplicate question",
                            }
                        )
                        rejected_count += 1
                        continue
                    tokens = _tokens(str(record["question"]))
                    if _near_duplicate(tokens, seen_tokens):
                        rejected.append(
                            {
                                "chunk_id": chunk_id,
                                "question": record["question"],
                                "reason": "near-duplicate question",
                            }
                        )
                        rejected_count += 1
                        continue
                    if (
                        not bool(record.get("multi_fact"))
                        and len(records) >= 100
                        and multi_fact_total / len(records) < 0.5
                    ):
                        rejected.append(
                            {
                                "chunk_id": chunk_id,
                                "question": record["question"],
                                "reason": "single-fact quota",
                            }
                        )
                        rejected_count += 1
                        continue
                    seen.add(key)
                    seen_tokens.append(tokens)
                    record["id"] = f"g{len(records) + 1:04d}"
                    records.append(record)
                    accepted_count += 1
                    multi_fact_count += int(bool(record.get("multi_fact")))
                    multi_fact_total += int(bool(record.get("multi_fact")))
                    print(f"question: {record['question']}")
                    print(f"answer:   {record['answer']}")
                rejected.extend(rejected_rows)
                telemetry.emit(
                    "chunk",
                    chunk_id=chunk_id,
                    generated=len(accepted) + len(rejected_rows),
                    accepted=accepted_count,
                    multi_fact=multi_fact_count,
                    rejected=rejected_count,
                    error="",
                )
                print(
                    f"chunk {min(start + len(batch), len(chunks))}/{len(chunks)}"
                    f" generated={len(accepted) + len(rejected_rows)}"
                    f" accepted={accepted_count} multi_fact={multi_fact_count}"
                    f" total={len(records)}",
                    flush=True,
                )
                if limit > 0 and len(records) >= limit:
                    break
    if not records:
        raise ValueError("LLM generated no valid source-grounded questions")
    write_jsonl(out_path, records)
    docs = sorted({doc for record in records for doc in _record_docs(record)})
    holdout_docs = {
        doc
        for index, doc in enumerate(docs)
        if holdout_every > 0 and (index + 1) % holdout_every == 0
    }
    if not holdout_docs and holdout_every > 0 and len(docs) >= 2:
        # Small corpora still get a real document holdout: keep the last
        # document out of training rather than leaking near-duplicate chunks.
        holdout_docs = {docs[-1]}
    if holdout_docs:
        train = [
            record for record in records if not (_record_docs(record) & holdout_docs)
        ]
        holdout = [
            record for record in records if _record_docs(record) & holdout_docs
        ]
    else:
        train = records
        holdout = records
    if not train or not holdout:
        train = records
        holdout = records
        holdout_docs = set()
    write_jsonl(out_path.parent / "train.jsonl", train)
    write_jsonl(out_path.parent / "holdout.jsonl", holdout)
    if path_set(rejected_path):
        write_jsonl(rejected_path, rejected)
    accepted_rate = len(records) / max(len(records) + len(rejected), 1)
    multi_fact = sum(1 for record in records if bool(record.get("multi_fact")))
    summary: dict[str, JsonValue] = {
        "accepted": len(records),
        "rejected": len(rejected),
        "acceptance_rate": round(accepted_rate, 6),
        "multi_fact": multi_fact,
        "multi_fact_rate": round(multi_fact / max(len(records), 1), 6),
        "chunks": len(chunks),
        "per_chunk": per_chunk,
        "limit": limit,
        "documents": len(docs),
        "holdout_documents": len(holdout_docs),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
    }
    write_json(out_path.parent / "summary.json", summary)
    telemetry.emit("summary", **summary)
    return len(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate LLM question/answer pairs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, default=Path(""))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--base-urls", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--chunks-limit", type=int, default=0)
    parser.add_argument("--per-chunk", type=int, default=1)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--db", type=Path, default=Path(""))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--telemetry", type=Path, default=Path(""))
    return parser


def main() -> int:
    args = _parser().parse_args()
    from cli.config import load_config

    config = load_config(args.config)
    urls = endpoints(str(args.base_url), str(args.base_urls))
    settings = settings_list(
        urls,
        str(args.model),
        str(args.api_key),
        int(args.timeout),
        int(args.max_tokens),
        float(args.temperature),
    )
    workers = int(args.workers) or len(settings)
    telemetry = (
        Telemetry(args.telemetry, "questions", args.telemetry.parent.parent.name)
        if path_set(args.telemetry)
        else disabled("questions")
    )
    count = generate(
        args.chunks,
        args.out,
        settings,
        int(args.limit),
        config.judge(),
        config.questions_system,
        config.questions_prompt,
        args.db,
        int(args.top_k),
        workers,
        int(args.per_chunk),
        int(args.chunks_limit),
        int(args.holdout_every),
        args.rejected,
        telemetry,
    )
    print(f"wrote {count} questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
