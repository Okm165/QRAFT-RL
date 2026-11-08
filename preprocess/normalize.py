"""Stage 1: minimal text normalization without semantic heuristics."""

from __future__ import annotations

import html
import unicodedata
from pathlib import Path

from .domain import NormalizeSummary, NormalizedText
from .inventory import scan
from .jsonl import JsonValue, record_to_json, write_json, write_jsonl

ZERO_WIDTH = frozenset({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"})
CONTROL = frozenset(chr(code) for code in range(32)) - frozenset({"\n", "\t"})


def _strip_controls(text: str) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for char in text:
        if char in ZERO_WIDTH:
            continue
        if char in CONTROL:
            removed += 1
            continue
        kept.append(char)
    return "".join(kept).replace("\u00a0", " "), removed


def normalize_text(text: str) -> NormalizedText:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, control_chars = _strip_controls(text)

    lines: list[str] = []
    changed_lines = 0
    html_entities = 0
    for line in text.split("\n"):
        unescaped = html.unescape(line)
        if unescaped != line:
            html_entities += 1
            changed_lines += 1
        lines.append(unescaped)

    return NormalizedText(
        text="\n".join(lines),
        changed_lines=changed_lines,
        html_entities=html_entities,
        control_chars=control_chars,
    )


def run(data_root: Path, out_dir: Path, globs: tuple[str, ...], limit: int) -> NormalizeSummary:
    documents, _ = scan(data_root, globs, limit)
    normalized_dir = out_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, JsonValue]] = []
    totals = NormalizeSummary(0, 0, 0, 0)
    for document in documents:
        source = data_root / document.source_path
        raw = source.read_text(encoding="utf-8", errors="replace")
        normalized = normalize_text(raw)
        (normalized_dir / f"{document.sha256}.md").write_text(normalized.text, encoding="utf-8")
        reports.append(
            {
                "doc_id": document.doc_id,
                "source_path": document.source_path,
                "changed_lines": normalized.changed_lines,
                "html_entities": normalized.html_entities,
                "control_chars": normalized.control_chars,
            }
        )
        totals = NormalizeSummary(
            documents=totals.documents + 1,
            changed_lines=totals.changed_lines + normalized.changed_lines,
            html_entities=totals.html_entities + normalized.html_entities,
            control_chars=totals.control_chars + normalized.control_chars,
        )
    write_jsonl(out_dir / "normalize_report.jsonl", reports)
    write_json(out_dir / "normalize_summary.json", record_to_json(totals))
    return totals
