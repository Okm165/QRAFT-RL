from __future__ import annotations

import json
from pathlib import Path

from rag.fts import build, search


def test_fts_build_and_search(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    rows = (
        {"doc_id": "d1", "source": "a.md", "section": "Pins", "page": 1, "kind": "table", "text": "Pin Signal GND VDD"},
        {"doc_id": "d2", "source": "a.md", "section": "Data", "page": 2, "kind": "table", "text": "Register reset value 42"},
        {"doc_id": "d3", "source": "a.md", "section": "Electrical", "page": 3, "kind": "table", "text": "Supply voltage min typ max"},
    )
    chunks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    db = tmp_path / "rag.sqlite"
    assert build(chunks, db, 0) == 3
    assert search(db, "pin signal", 5)[0].doc_id == "d1"
    assert search(db, "register reset", 5)[0].doc_id == "d2"
    assert search(db, "supply voltage", 5)[0].doc_id == "d3"


def test_short_prose_is_indexed(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "source": "clock",
                "section": "Clock",
                "page": 1,
                "kind": "prose",
                "text": "The maximum voltage is 5.5 V.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db = tmp_path / "rag.sqlite"
    assert build(chunks, db, 0) == 1
    assert search(db, "maximum voltage", 5)[0].doc_id == "d1"


def test_search_fills_with_or_hits(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    rows = (
        {"doc_id": "d1", "source": "a.md", "section": "Pins", "page": 1, "kind": "table", "text": "Pin Signal GND"},
        {"doc_id": "d2", "source": "a.md", "section": "Pins", "page": 2, "kind": "table", "text": "Pin GND"},
    )
    chunks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    db = tmp_path / "rag.sqlite"
    build(chunks, db, 0)
    found = search(db, "pin signal gnd", 5)
    assert {hit.doc_id for hit in found} == {"d1", "d2"}


def test_facts_are_indexed(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "source": "a.md",
                "section": "Pins",
                "page": 1,
                "kind": "table",
                "text": "Pin Signal GND VDD",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    facts = tmp_path / "facts.jsonl"
    facts.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "source": "a.md",
                "section": "Pins",
                "page": 1,
                "kind": "table",
                "text": "Pin: 1 | Name: GND",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db = tmp_path / "rag.sqlite"
    assert build(chunks, db, 0, facts) == 2
    assert search(db, "pin 1 name gnd", 5)[0].text == "Pin: 1 | Name: GND"
