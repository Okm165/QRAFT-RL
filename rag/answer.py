"""Grounded answer generation over the local FTS5 index."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from .fts import SearchHit, search
from .prompt import ContextView, render_answer_prompt

Generator: TypeAlias = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    text: str
    citations: tuple[SearchHit, ...]


def _view(hit: SearchHit) -> ContextView:
    return ContextView(hit.source, hit.section, hit.page, hit.kind, hit.text)


def build_prompt(
    question: str,
    hits: tuple[SearchHit, ...],
    template: str,
    refusal: str = "",
) -> str:
    return render_answer_prompt(
        template,
        question,
        tuple(_view(hit) for hit in hits),
        refusal,
    )


def answer_question(
    db_path: Path,
    question: str,
    top_k: int,
    generator: Generator,
    prompt_template: str,
    refusal: str,
) -> Answer:
    hits = search(db_path, question, top_k)
    if not hits:
        return Answer(
            question=question,
            text=refusal,
            citations=(),
        )
    prompt = build_prompt(question, hits, prompt_template, refusal)
    return Answer(
        question=question,
        text=generator(prompt).strip(),
        citations=hits,
    )
