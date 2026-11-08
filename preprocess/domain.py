"""Typed domain records for the preprocessing pipeline.

Records stay total and JSON-safe: absent values use empty strings, zero, or
empty tuples instead of ``None``/``Any``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContentKind(StrEnum):
    PROSE = "prose"
    TABLE = "table"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    doc_id: str
    source_path: str
    source: str
    sha256: str
    bytes: int
    lines: int
    warnings: tuple[str, ...]
    duplicate_of: str


@dataclass(frozen=True, slots=True)
class InventoryAudit:
    root: str
    files: int
    unique_hashes: int
    duplicate_files: int
    bytes: int
    warnings: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class NormalizedText:
    text: str
    changed_lines: int
    html_entities: int
    control_chars: int


@dataclass(frozen=True, slots=True)
class NormalizeSummary:
    documents: int
    changed_lines: int
    html_entities: int
    control_chars: int


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    section: str
    page: int
    kind: ContentKind
    text: str
    token_count: int
    summary: str = ""
    concepts: tuple[str, ...] = ()
    window_index: int = 0
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    content_sha256: str = ""


@dataclass(frozen=True, slots=True)
class FactRecord:
    fact_id: str
    doc_id: str
    source: str
    section: str
    page: int
    kind: str
    text: str
    chunk_id: str = ""
    concepts: tuple[str, ...] = ()
    window_index: int = 0
    char_start: int = 0
    char_end: int = 0
    grounded: bool = False


@dataclass(frozen=True, slots=True)
class ConceptRecord:
    concept_id: str
    doc_id: str
    source: str
    name: str
    definition: str
    aliases: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructureSummary:
    documents: int
    windows: int
    chunks: int
    facts: int
    skipped: int
    concepts: int = 0
    documents_structured: int = 0
    documents_empty: int = 0
    empty_windows: int = 0
