from __future__ import annotations

import json
from pathlib import Path

from eval.behavior import BehaviorResult, evaluate
from rag.fts import build
from tests.fakes import RuleJudge

PROMPT = "Question: %QUESTION%\nContext:\n%CONTEXT%\nAnswer:"
REFUSAL = "I do not have enough evidence in the corpus to answer this."


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    db = tmp_path / "rag.sqlite"
    build(chunks, db, 0)
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "question": "pin signal",
                "expected_doc_ids": ["d1"],
                "expected_terms": ["GND"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    refusal = tmp_path / "refusal_questions.jsonl"
    refusal.write_text(
        json.dumps({"question": "price USD"}) + "\n",
        encoding="utf-8",
    )
    return db, golden, refusal


def _run(tmp_path: Path, answer: str) -> tuple[BehaviorResult, list[str]]:
    prompts: list[str] = []

    def generator(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    result = evaluate(*_fixture(tmp_path), generator, RuleJudge(), PROMPT, REFUSAL, 5)
    return result, prompts


def test_behavior_counts_answer_citation_and_refusal(tmp_path: Path) -> None:
    result, prompts = _run(tmp_path, "GND [1]")
    assert (result.answerable, result.answer_hits, result.citation_hits) == (1, 1, 1)
    assert (result.refusal_probes, result.refusals) == (1, 1)
    assert len(prompts) == 1


def test_behavior_rejects_unsupported_citation(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "GND [2]")
    assert (result.answer_hits, result.citation_hits) == (1, 0)


def test_behavior_limit_samples_golden(tmp_path: Path) -> None:
    db, golden, refusal = _fixture(tmp_path)
    with golden.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "question": "second pin",
                    "expected_doc_ids": ["d1"],
                    "expected_terms": ["VDD"],
                }
            )
            + "\n"
        )
    prompts: list[str] = []

    def generator(prompt: str) -> str:
        prompts.append(prompt)
        return "GND [1]"

    result = evaluate(db, golden, refusal, generator, RuleJudge(), PROMPT, REFUSAL, 5, 1)
    assert result.answerable == 1
    assert len(prompts) == 1


def test_behavior_stratifies_multi_fact_and_lookup(tmp_path: Path) -> None:
    db, golden, refusal = _fixture(tmp_path)
    with golden.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "question": "pin signal synthesis",
                    "expected_doc_ids": ["d1"],
                    "expected_terms": ["GND"],
                    "multi_fact": True,
                }
            )
            + "\n"
        )

    def generator(prompt: str) -> str:
        return "GND [1]"

    result = evaluate(db, golden, refusal, generator, RuleJudge(), PROMPT, REFUSAL, 5)
    assert (result.multi_fact, result.multi_fact_answer_hits) == (1, 1)
    assert (result.lookup, result.lookup_answer_hits) == (1, 1)
    assert result.answer_hits == 2
