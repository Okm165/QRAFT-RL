"""Stage 2: LLM-driven document structuring.

The code knows only a generic contract: the configured prompt receives a text
window through ``%TEXT%`` and returns a JSON object with ``chunks``, ``facts``,
and optional ``concepts``. All domain-specific instructions live in TOML.

Every record keeps source provenance: document, window, character span, and a
stable content hash. Concepts are merged into a document-level glossary.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias, TypeVar

from llm.client import LlmSettings, chat
from llm.structured import parse_object
from telemetry import Telemetry, disabled

from .domain import (
    ChunkRecord,
    ConceptRecord,
    ContentKind,
    DocumentRecord,
    FactRecord,
    StructureSummary,
)
from .jsonl import JsonValue, record_to_json, write_json, write_jsonl
from .util import read_normalized, stable_id

ChatFn: TypeAlias = Callable[[LlmSettings, str, str], str]
RecordT = TypeVar("RecordT")
TokenCounter: TypeAlias = Callable[[str], int]

KIND_VALUES = frozenset(kind.value for kind in ContentKind)


@dataclass(frozen=True, slots=True)
class StructureSettings:
    system: str
    prompt: str
    window_chars: int
    overlap_chars: int
    limit: int
    workers: int


def windows(text: str, size: int, overlap: int) -> tuple[str, ...]:
    if size < 1:
        raise ValueError("window_chars must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap_chars must be in [0, window_chars)")
    if not text:
        return ()
    step = size - overlap
    result: list[str] = []
    start = 0
    while start < len(text):
        result.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return tuple(result)


def windows_with_offsets(
    text: str, size: int, overlap: int
) -> tuple[tuple[int, str], ...]:
    if size < 1:
        raise ValueError("window_chars must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap_chars must be in [0, window_chars)")
    if not text:
        return ()
    step = size - overlap
    result: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        result.append((start, text[start : start + size]))
        if start + size >= len(text):
            break
        start += step
    return tuple(result)


def _text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _texts(value: JsonValue) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _section(value: Mapping[str, JsonValue], index: int) -> str:
    section = str(value.get("section", "")).strip()
    return section if section else f"Window {index + 1}"


def _kind(value: Mapping[str, JsonValue]) -> ContentKind:
    kind = str(value.get("kind", ContentKind.PROSE.value)).strip().lower()
    return ContentKind(kind) if kind in KIND_VALUES else ContentKind.PROSE


def _concept_names(value: Mapping[str, JsonValue]) -> tuple[str, ...]:
    names: list[str] = []
    raw = value.get("concepts", [])
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if isinstance(item, str) and item.strip():
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = ""
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _span(window: str, window_start: int, text: str) -> tuple[int, int]:
    position = window.find(text)
    if position < 0:
        return (window_start, window_start + len(window))
    return (window_start + position, window_start + position + len(text))


def _chunk(
    document: DocumentRecord,
    window_index: int,
    chunk_index: int,
    window_start: int,
    window: str,
    value: JsonValue,
    counter: TokenCounter,
) -> ChunkRecord:
    if isinstance(value, str):
        item: dict[str, JsonValue] = {"text": value}
    elif isinstance(value, dict):
        item = {str(key): entry for key, entry in value.items()}
    else:
        item = {}
    text = _text(item.get("text", item.get("chunk", "")))
    section = _section(item, window_index)
    kind = _kind(item)
    start, end = _span(window, window_start, text) if text else (window_start, window_start)
    return ChunkRecord(
        chunk_id=(
            stable_id(document.doc_id, str(window_index), str(chunk_index), text)
            if text
            else ""
        ),
        doc_id=document.doc_id,
        source=document.source,
        section=section,
        page=window_index + 1,
        kind=kind,
        text=text,
        token_count=counter(text),
        summary=_text(item.get("summary", "")),
        concepts=_concept_names(item),
        window_index=window_index + 1,
        chunk_index=chunk_index,
        char_start=start,
        char_end=end,
        content_sha256=(
            hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
        ),
    )


def _tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _overlap(text: str, source: str) -> float:
    expected = _tokens(text)
    if not expected:
        return 0.0
    return len(expected & _tokens(source)) / len(expected)


def _grounded_chunk(text: str, chunks: tuple[ChunkRecord, ...]) -> str:
    """Attribute a fact to the chunk that best supports it.

    Exact substrings are ideal, but an LLM fact is often a close paraphrase.
    A normalized token-overlap threshold keeps grounded paraphrases while
    dropping facts that were attached to an unrelated chunk.
    """
    best_id = ""
    best = 0.0
    for chunk in chunks:
        score = _overlap(text, chunk.text)
        if score > best:
            best = score
            best_id = chunk.chunk_id
    return best_id if best >= 0.6 else ""


def _fact(
    document: DocumentRecord,
    window_index: int,
    fact_index: int,
    window_start: int,
    window: str,
    value: JsonValue,
    chunks: tuple[ChunkRecord, ...],
) -> FactRecord:
    if isinstance(value, str):
        item: dict[str, JsonValue] = {"text": value}
    elif isinstance(value, dict):
        item = {str(key): entry for key, entry in value.items()}
    else:
        item = {}
    text = _text(item.get("text", item.get("fact", "")))
    section = _section(item, window_index)
    start, end = _span(window, window_start, text) if text else (window_start, window_start)
    chunk_id = _grounded_chunk(text, chunks)
    return FactRecord(
        fact_id=(
            stable_id(document.doc_id, str(window_index), str(fact_index), text)
            if text
            else ""
        ),
        doc_id=document.doc_id,
        source=document.source,
        section=section,
        page=window_index + 1,
        kind=_text(item.get("kind", "")) or "claim",
        text=text,
        chunk_id=chunk_id,
        concepts=_concept_names(item),
        window_index=window_index + 1,
        char_start=start,
        char_end=end,
        grounded=bool(chunk_id),
    )


def _concepts(
    document: DocumentRecord,
    window_index: int,
    value: Mapping[str, JsonValue],
    chunks: tuple[ChunkRecord, ...],
    facts: tuple[FactRecord, ...],
) -> tuple[ConceptRecord, ...]:
    records: list[ConceptRecord] = []
    raw = value.get("concepts", [])
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            definition = ""
            aliases: tuple[str, ...] = ()
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            definition = str(item.get("definition", "")).strip()
            aliases = _texts(item.get("aliases", []))
        else:
            name = ""
            definition = ""
            aliases = ()
        if not name or not definition:
            continue
        records.append(
            ConceptRecord(
                concept_id=stable_id(document.doc_id, name.casefold()),
                doc_id=document.doc_id,
                source=document.source,
                name=name,
                definition=definition,
                aliases=aliases,
                chunk_ids=tuple(
                    chunk.chunk_id
                    for chunk in chunks
                    if name.casefold() in {item.casefold() for item in chunk.concepts}
                ),
                fact_ids=tuple(
                    fact.fact_id
                    for fact in facts
                    if name.casefold() in {item.casefold() for item in fact.concepts}
                ),
            )
        )
    return tuple(records)


def _dedupe_records(records: list[RecordT], text: Callable[[RecordT], str]) -> list[RecordT]:
    seen: set[str] = set()
    result: list[RecordT] = []
    for record in records:
        key = " ".join(text(record).casefold().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _merge_concepts(
    existing: dict[str, ConceptRecord],
    records: tuple[ConceptRecord, ...],
) -> int:
    for record in records:
        current = existing.get(record.concept_id)
        if current is None:
            existing[record.concept_id] = record
            continue
        aliases = tuple(dict.fromkeys((*current.aliases, *record.aliases)))
        chunk_ids = tuple(dict.fromkeys((*current.chunk_ids, *record.chunk_ids)))
        fact_ids = tuple(dict.fromkeys((*current.fact_ids, *record.fact_ids)))
        existing[record.concept_id] = ConceptRecord(
            concept_id=current.concept_id,
            doc_id=current.doc_id,
            source=current.source,
            name=current.name,
            definition=current.definition or record.definition,
            aliases=aliases,
            chunk_ids=chunk_ids,
            fact_ids=fact_ids,
        )
    return len(records)


def _records(
    document: DocumentRecord,
    window_index: int,
    window_start: int,
    window: str,
    value: Mapping[str, JsonValue],
    counter: TokenCounter,
) -> tuple[tuple[ChunkRecord, ...], tuple[FactRecord, ...], tuple[ConceptRecord, ...]]:
    chunks_value = value.get("chunks", [])
    facts_value = value.get("facts", [])
    chunks = (
        tuple(
            chunk
            for index, item in enumerate(chunks_value)
            if (
                chunk := _chunk(
                    document,
                    window_index,
                    index,
                    window_start,
                    window,
                    item,
                    counter,
                )
            ).text
        )
        if isinstance(chunks_value, list)
        else ()
    )
    facts = (
        tuple(
            fact
            for index, item in enumerate(facts_value)
            if (
                fact := _fact(
                    document,
                    window_index,
                    index,
                    window_start,
                    window,
                    item,
                    chunks,
                )
            ).text
        )
        if isinstance(facts_value, list)
        else ()
    )
    return chunks, facts, _concepts(document, window_index, value, chunks, facts)


@dataclass(frozen=True, slots=True)
class WindowResult:
    chunks: tuple[ChunkRecord, ...]
    facts: tuple[FactRecord, ...]
    concepts: tuple[ConceptRecord, ...]
    error: str
    split: int


def _split_window(window: str) -> tuple[str, str]:
    midpoint = len(window) // 2
    split = window.rfind("\n", 0, midpoint)
    if split < len(window) // 4:
        split = midpoint
    return window[:split], window[split:]


def _window(
    document: DocumentRecord,
    window_index: int,
    window_start: int,
    window: str,
    settings: StructureSettings,
    llm: LlmSettings,
    chat_fn: ChatFn,
    counter: TokenCounter,
    depth: int = 0,
) -> WindowResult:
    prompt = settings.prompt.replace("%TEXT%", window)
    try:
        value = parse_object(chat_fn(llm, settings.system, prompt))
    except ValueError as first:
        try:
            larger = replace(llm, max_tokens=max(llm.max_tokens * 2, 4096))
            value = parse_object(
                chat_fn(
                    larger,
                    settings.system,
                    prompt + "\nReturn one valid JSON object only.",
                )
            )
        except (OSError, RuntimeError) as second:
            # A timeout or connection failure is not a syntax problem; do
            # not split and multiply the wait.
            return WindowResult(
                (),
                (),
                (),
                f"retry {type(second).__name__}: {second}",
                0,
            )
        except ValueError as second:
            if depth >= 1 or len(window) < 1000:
                return WindowResult(
                    (),
                    (),
                    (),
                    f"{type(first).__name__}: {first}; retry {type(second).__name__}: {second}",
                    0,
                )
            first_text, second_text = _split_window(window)
            left = _window(
                document,
                window_index,
                window_start,
                first_text,
                settings,
                llm,
                chat_fn,
                counter,
                depth + 1,
            )
            right = _window(
                document,
                window_index,
                window_start + len(first_text),
                second_text,
                settings,
                llm,
                chat_fn,
                counter,
                depth + 1,
            )
            error = ""
            if left.error and right.error:
                error = f"split left: {left.error}; split right: {right.error}"
            return WindowResult(
                left.chunks + right.chunks,
                left.facts + right.facts,
                left.concepts + right.concepts,
                error,
                1 + left.split + right.split,
            )
    except (OSError, RuntimeError) as exc:
        return WindowResult((), (), (), f"{type(exc).__name__}: {exc}", 0)
    chunks, facts, concepts = _records(
        document,
        window_index,
        window_start,
        window,
        value,
        counter,
    )
    return WindowResult(chunks, facts, concepts, "", 0)


def _jobs(
    documents: tuple[DocumentRecord, ...],
    data_root: Path,
    out_dir: Path,
    settings: StructureSettings,
) -> tuple[tuple[DocumentRecord, int, int, str], ...]:
    groups: list[list[tuple[DocumentRecord, int, int, str]]] = []
    for document in documents:
        text = read_normalized(document, data_root, out_dir)
        groups.append(
            [
                (document, index, start, window)
                for index, (start, window) in enumerate(
                    windows_with_offsets(
                        text,
                        settings.window_chars,
                        settings.overlap_chars,
                    )
                )
            ]
        )
    if settings.limit <= 0:
        return tuple(job for group in groups for job in group)
    jobs: list[tuple[DocumentRecord, int, int, str]] = []
    depth = 0
    while len(jobs) < settings.limit:
        added = False
        for group in groups:
            if depth < len(group) and len(jobs) < settings.limit:
                jobs.append(group[depth])
                added = True
        if not added:
            break
        depth += 1
    return tuple(jobs)


def run(
    documents: tuple[DocumentRecord, ...],
    data_root: Path,
    out_dir: Path,
    settings: StructureSettings,
    llm_settings: tuple[LlmSettings, ...],
    counter: TokenCounter,
    chat_fn: ChatFn = chat,
    telemetry: Telemetry = disabled("preprocess"),
) -> StructureSummary:
    if not llm_settings:
        raise ValueError("at least one LLM endpoint is required")
    if "%TEXT%" not in settings.prompt:
        raise ValueError("preprocess.prompt must contain %TEXT%")
    unique_documents = tuple(
        document for document in documents if not document.duplicate_of
    )
    jobs = _jobs(unique_documents, data_root, out_dir, settings)

    chunks: list[ChunkRecord] = []
    facts: list[FactRecord] = []
    concepts: dict[str, ConceptRecord] = {}
    skipped = 0
    empty_windows = 0
    structured_documents: set[str] = set()

    def process(
        job: tuple[DocumentRecord, int, int, str],
        position: int,
    ) -> tuple[tuple[DocumentRecord, int, int, str], WindowResult, float]:
        document, window_index, window_start, window = job
        started = time.perf_counter()
        result = _window(
            document,
            window_index,
            window_start,
            window,
            settings,
            llm_settings[position % len(llm_settings)],
            chat_fn,
            counter,
        )
        return job, result, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=max(settings.workers, 1)) as pool:
        results = pool.map(
            lambda pair: process(pair[1], pair[0]),
            enumerate(jobs),
        )
        for job, result, seconds in results:
            document, window_index, window_start, window = job
            if result.error:
                skipped += 1
                telemetry.timed(
                    "window",
                    seconds,
                    doc_id=document.doc_id,
                    source=document.source,
                    window_index=window_index + 1,
                    chars=len(window),
                    chunks=0,
                    facts=0,
                    concepts=0,
                    split=result.split,
                    error=result.error,
                )
                print(f"skip window {document.source} #{window_index + 1}: {result.error}")
                continue
            if not result.chunks and not result.facts:
                empty_windows += 1
            if result.chunks or result.facts:
                structured_documents.add(document.doc_id)
            chunks.extend(result.chunks)
            facts.extend(result.facts)
            _merge_concepts(concepts, result.concepts)
            telemetry.timed(
                "window",
                seconds,
                doc_id=document.doc_id,
                source=document.source,
                window_index=window_index + 1,
                chars=len(window),
                char_start=window_start,
                chunks=len(result.chunks),
                facts=len(result.facts),
                concepts=len(result.concepts),
                split=result.split,
                error="",
            )
            print(
                f"window {document.source} #{window_index + 1}"
                f" chunks={len(result.chunks)} facts={len(result.facts)}"
                f" concepts={len(result.concepts)} seconds={seconds:.1f}",
                flush=True,
            )

    chunks = _dedupe_records(chunks, lambda row: row.text)
    facts = _dedupe_records(facts, lambda row: row.text)
    write_jsonl(out_dir / "chunks.jsonl", (record_to_json(row) for row in chunks))
    write_jsonl(out_dir / "facts.jsonl", (record_to_json(row) for row in facts))
    write_jsonl(
        out_dir / "concepts.jsonl",
        (record_to_json(row) for row in concepts.values()),
    )
    summary = StructureSummary(
        documents=len(unique_documents),
        windows=len(jobs),
        chunks=len(chunks),
        facts=len(facts),
        skipped=skipped,
        concepts=len(concepts),
        documents_structured=len(structured_documents),
        documents_empty=len(unique_documents) - len(structured_documents),
        empty_windows=empty_windows,
    )
    write_json(out_dir / "structure_summary.json", record_to_json(summary))
    telemetry.emit("summary", **record_to_json(summary))
    return summary
