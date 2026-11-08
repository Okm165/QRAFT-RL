from __future__ import annotations

import json
from pathlib import Path

from eval.run import EvalResult, evaluate
from rag.fts import build
from tests.fakes import RuleJudge


def _evaluate(tmp_path: Path, expected_terms: list[str]) -> EvalResult:
    chunks = tmp_path / "chunks.jsonl"
    rows = (
        {"doc_id": "d1", "source": "a.md", "section": "Pins", "page": 1, "kind": "table", "text": "Pin Signal GND VDD"},
        {"doc_id": "d2", "source": "a.md", "section": "Data", "page": 2, "kind": "table", "text": "Register reset value 42"},
    )
    chunks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    db = tmp_path / "rag.sqlite"
    build(chunks, db, 0)
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "question": "pin signal",
                "expected_doc_ids": ["d1"],
                "expected_terms": expected_terms,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return evaluate(db, golden, 5, RuleJudge())


def test_evidence_hit(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, ["GND"])
    assert (result.total, result.evidence_hits) == (1, 1)


def test_evidence_miss_when_document_matches_but_term_does_not(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, ["VCC"])
    assert (result.total, result.evidence_hits) == (1, 0)
