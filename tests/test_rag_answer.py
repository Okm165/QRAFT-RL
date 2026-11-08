from __future__ import annotations

import json
from pathlib import Path

from rag.answer import answer_question
from rag.fts import build

PROMPT = "Cite sources as [1], [2].\nQuestion: %QUESTION%\nContext:\n%CONTEXT%\nAnswer:"
REFUSAL = "I do not have enough evidence in the corpus to answer this."


def _db(tmp_path: Path) -> Path:
    chunks = tmp_path / "chunks.jsonl"
    rows = (
        {"doc_id": "d1", "source": "a.md", "section": "Pins", "page": 1, "kind": "table", "text": "Pin Signal GND VDD"},
        {"doc_id": "d2", "source": "a.md", "section": "Data", "page": 2, "kind": "table", "text": "Register reset value 42"},
    )
    chunks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    db = tmp_path / "rag.sqlite"
    build(chunks, db, 0)
    return db


def test_answer_uses_retrieved_context_and_citations(tmp_path: Path) -> None:
    prompts: list[str] = []

    def generator(prompt: str) -> str:
        prompts.append(prompt)
        return "Answer [1]"

    answer = answer_question(_db(tmp_path), "pin signal", 5, generator, PROMPT, REFUSAL)
    assert answer.text == "Answer [1]"
    assert answer.citations[0].doc_id == "d1"
    assert "Cite sources as [1], [2]" in prompts[0]
    assert "Context:" in prompts[0]


def test_answer_generates_when_any_context_matches(tmp_path: Path) -> None:
    prompts: list[str] = []

    def generator(prompt: str) -> str:
        prompts.append(prompt)
        return "Answer [1]"

    answer = answer_question(_db(tmp_path), "pin register", 5, generator, PROMPT, REFUSAL)
    assert answer.text == "Answer [1]"
    assert answer.citations
    assert prompts


def test_answer_refuses_without_any_context(tmp_path: Path) -> None:
    prompts: list[str] = []

    def generator(prompt: str) -> str:
        prompts.append(prompt)
        return "should not run"

    answer = answer_question(_db(tmp_path), "nonexistentterm", 5, generator, PROMPT, REFUSAL)
    assert answer.text == REFUSAL
    assert answer.citations == ()
    assert prompts == []
