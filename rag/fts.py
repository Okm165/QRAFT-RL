"""Local lexical retrieval with SQLite FTS5. No server, no model, no network."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from preprocess.jsonl import JsonValue, read_jsonl


@dataclass(frozen=True, slots=True)
class SearchHit:
    doc_id: str
    source: str
    section: str
    page: int
    kind: str
    text: str
    score: float
    record_id: str = ""
    record_type: str = ""


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _insert_record(
    connection: sqlite3.Connection,
    row: Mapping[str, JsonValue],
    record_type: str,
) -> int:
    doc_id = str(row["doc_id"])
    source = str(row["source"])
    section = str(row.get("section", ""))
    page = _as_int(row.get("page", 0))
    kind = str(row.get("kind", ""))
    text = str(row["text"])
    if record_type == "fact":
        record_id = str(row.get("fact_id", ""))
    elif record_type == "concept":
        record_id = str(row.get("concept_id", ""))
    else:
        record_id = str(row.get("chunk_id", ""))
    indexed = f"{source} {section} {text}".strip()
    connection.execute(
        "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, source, section, page, kind, text, record_id, record_type),
    )
    connection.execute(
        "INSERT INTO records_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, source, section, page, kind, indexed, record_id, record_type),
    )
    return 1


def build(
    chunks_path: Path,
    db_path: Path,
    limit: int,
    facts_path: Path = Path(""),
    concepts_path: Path = Path(""),
) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE records (doc_id TEXT, source TEXT, section TEXT, page INTEGER, kind TEXT, text TEXT, record_id TEXT, record_type TEXT)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE records_fts USING fts5("
            "doc_id UNINDEXED, source UNINDEXED, section UNINDEXED, page UNINDEXED, kind UNINDEXED, text, record_id UNINDEXED, record_type UNINDEXED)"
        )
        count = 0
        for row in read_jsonl(chunks_path):
            count += _insert_record(connection, row, "chunk")
            if limit > 0 and count >= limit:
                break
        if facts_path.is_file():
            for row in read_jsonl(facts_path):
                if limit > 0 and count >= limit:
                    break
                if row.get("grounded") is False:
                    continue
                count += _insert_record(connection, row, "fact")
        if concepts_path.is_file():
            for row in read_jsonl(concepts_path):
                if limit > 0 and count >= limit:
                    break
                definition = str(row.get("definition", "")).strip()
                if not definition:
                    continue
                concept = dict(row)
                concept["text"] = f"{concept.get('name', '')}: {definition}".strip()
                concept["section"] = "Glossary"
                concept["page"] = 0
                concept["kind"] = "concept"
                count += _insert_record(connection, concept, "concept")
        connection.commit()
    finally:
        connection.close()
    return count


def _tokens(query: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for char in query:
        if char.isalnum() or char == "_":
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _terms(query: str) -> tuple[str, ...]:
    return _tokens(query)


def _rows(
    connection: sqlite3.Connection,
    terms: tuple[str, ...],
    operator: str,
    top_k: int,
) -> list[tuple[object, ...]]:
    query = f" {operator} ".join(f'"{term}"' for term in terms)
    return connection.execute(
        "SELECT r.rowid, r.doc_id, r.source, r.section, r.page, r.kind, r.text, "
        "bm25(records_fts) AS score, r.record_id, r.record_type "
        "FROM records_fts JOIN records r ON r.rowid = records_fts.rowid "
        "WHERE records_fts MATCH ? ORDER BY score LIMIT ?",
        (query, max(top_k, 1)),
    ).fetchall()


def search(db_path: Path, query: str, top_k: int) -> tuple[SearchHit, ...]:
    terms = _terms(query)
    if not terms:
        return ()
    connection = sqlite3.connect(db_path)
    try:
        # Require every content term first; fill the remaining slots with any
        # term. Strict matches stay first, but a single table-of-contents hit
        # cannot hide the table that actually contains the answer.
        rows = _rows(connection, terms, "AND", top_k)
        if len(rows) < top_k:
            seen = {_as_int(row[0]) for row in rows}
            for row in _rows(connection, terms, "OR", top_k):
                if _as_int(row[0]) in seen:
                    continue
                rows.append(row)
                seen.add(_as_int(row[0]))
                if len(rows) >= top_k:
                    break
    finally:
        connection.close()
    return tuple(
        SearchHit(
            doc_id=str(row[1]),
            source=str(row[2]),
            section=str(row[3]),
            page=_as_int(row[4]),
            kind=str(row[5]),
            text=str(row[6]),
            score=_as_float(row[7]),
            record_id=str(row[8]),
            record_type=str(row[9]),
        )
        for row in rows
    )
