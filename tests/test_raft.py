from __future__ import annotations

import json
from pathlib import Path

from raft.build import _quote
from raft.build import build as build_raft
from rag.fts import build as build_index
from tests.fakes import RuleJudge

REFUSAL = "I do not have enough evidence in the corpus to answer this."
TEMPLATE = "The answer is %ANSWER% [%INDEX%]."


def _fixture(tmp_path: Path, count: int) -> tuple[Path, Path, Path]:
    chunks = tmp_path / "chunks.jsonl"
    rows = (
        {"doc_id": "d1", "source": "a.md", "section": "Pins", "page": 1, "kind": "table", "text": "Pin 1 GND"},
        {"doc_id": "d2", "source": "a.md", "section": "Pins", "page": 2, "kind": "table", "text": "Pin 2 VDD"},
        {"doc_id": "d3", "source": "a.md", "section": "Pins", "page": 3, "kind": "table", "text": "Pin 3 GND"},
        {"doc_id": "d4", "source": "a.md", "section": "Pins", "page": 4, "kind": "table", "text": "Pin 4 VDD"},
        {"doc_id": "d5", "source": "a.md", "section": "Pins", "page": 5, "kind": "table", "text": "Pin 5 VDD"},
    )
    chunks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    db = tmp_path / "rag.sqlite"
    build_index(chunks, db, 0)
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps(
                {
                    "question": "pin 1",
                    "expected_doc_ids": ["d1"],
                    "expected_terms": ["GND"],
                }
            )
            for _ in range(count)
        )
        + "\n",
        encoding="utf-8",
    )
    return db, golden, tmp_path / "raft.jsonl"


def test_quote_falls_back_to_the_oracle_sentence() -> None:
    text = "The device has two ports. Pin 1 is GND."
    assert _quote(text, "GND", "Pin 1 is GND.") == "Pin 1 is GND."
    assert _quote(text, "GND", "a paraphrase") == "Pin 1 is GND."


def test_raft_record_uses_oracle_and_distractor(tmp_path: Path) -> None:
    db, golden, out = _fixture(tmp_path, 1)
    summary = build_raft(db, golden, out, 1, 42, RuleJudge(), TEMPLATE, REFUSAL)
    assert (summary.records, summary.absent, summary.skipped) == (1, 0, 0)
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert len(record["contexts"]) == 2
    assert {context["page"] for context in record["contexts"]} == {1, 2}
    assert record["status"] == "oracle"
    assert 0 <= record["oracle_index"] < len(record["contexts"])
    assert record["answer"] == f"The answer is GND [{record['oracle_index'] + 1}]."


def test_raft_record_refuses_when_oracle_is_absent(tmp_path: Path) -> None:
    db, golden, out = _fixture(tmp_path, 5)
    summary = build_raft(db, golden, out, 1, 42, RuleJudge(), TEMPLATE, REFUSAL)
    assert (summary.records, summary.absent, summary.skipped) == (5, 1, 0)
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    assert record["status"] == "absent"
    assert record["oracle_index"] == -1
    assert record["answer"] == REFUSAL
    pages = {context["page"] for context in record["contexts"]}
    assert len(pages) == 2
    assert pages <= {2, 4, 5}


def test_raft_skips_absent_without_distractors(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "source": "a.md",
                "section": "Pins",
                "page": 1,
                "kind": "table",
                "text": "Pin 1 GND",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db = tmp_path / "rag.sqlite"
    build_index(chunks, db, 0)
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps(
                {
                    "question": "pin 1",
                    "expected_doc_ids": ["d1"],
                    "expected_terms": ["GND"],
                }
            )
            for _ in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "raft.jsonl"
    summary = build_raft(db, golden, out, 1, 42, RuleJudge(), TEMPLATE, REFUSAL)
    assert (summary.records, summary.absent, summary.skipped) == (4, 0, 1)
