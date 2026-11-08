"""Stage 0: enumerate input files and record provenance."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from .domain import DocumentRecord, InventoryAudit
from .jsonl import record_to_json, write_json, write_jsonl
from .util import iter_files, sha256_bytes, source_name


def _decode(raw: bytes) -> tuple[str, list[str]]:
    try:
        return raw.decode("utf-8"), []
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), ["decode_error"]


def _warnings(text: str) -> list[str]:
    warnings: list[str] = []
    if "\ufffd" in text:
        warnings.append("replacement_chars")
    return warnings


def _read_document(path: Path, data_root: Path) -> DocumentRecord:
    raw = path.read_bytes()
    sha = sha256_bytes(raw)
    text, warnings = _decode(raw)
    warnings.extend(_warnings(text))
    return DocumentRecord(
        doc_id=f"sha256:{sha}",
        source_path=str(path.relative_to(data_root)),
        source=source_name(path),
        sha256=sha,
        bytes=len(raw),
        lines=len(text.splitlines()),
        warnings=tuple(warnings),
        duplicate_of="",
    )


def _with_duplicates(records: list[DocumentRecord]) -> tuple[DocumentRecord, ...]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.sha256, []).append(index)
    result = list(records)
    for indexes in groups.values():
        if len(indexes) > 1:
            canonical = records[indexes[0]].source_path
            for index in indexes[1:]:
                result[index] = replace(result[index], duplicate_of=canonical)
    return tuple(result)


def _audit(records: tuple[DocumentRecord, ...], data_root: Path) -> InventoryAudit:
    warnings = Counter(warning for record in records for warning in record.warnings)
    return InventoryAudit(
        root=str(data_root),
        files=len(records),
        unique_hashes=len({record.sha256 for record in records}),
        duplicate_files=sum(1 for record in records if record.duplicate_of),
        bytes=sum(record.bytes for record in records),
        warnings=tuple(warnings.most_common()),
    )


def scan(
    data_root: Path,
    globs: tuple[str, ...],
    limit: int,
) -> tuple[tuple[DocumentRecord, ...], InventoryAudit]:
    files = iter_files(data_root, globs)
    if limit > 0:
        files = files[:limit]
    records = _with_duplicates([_read_document(path, data_root) for path in files])
    return records, _audit(records, data_root)


def run(data_root: Path, out_dir: Path, globs: tuple[str, ...], limit: int) -> InventoryAudit:
    records, audit = scan(data_root, globs, limit)
    write_jsonl(out_dir / "docs.jsonl", (record_to_json(record) for record in records))
    write_json(out_dir / "audit.json", record_to_json(audit))
    return audit
