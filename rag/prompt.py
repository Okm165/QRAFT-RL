"""Generic prompt rendering shared by training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextView:
    source: str
    section: str
    page: int
    kind: str
    text: str


def render(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"%{key.upper()}%", value)
    return rendered


def render_answer_prompt(
    template: str,
    question: str,
    contexts: tuple[ContextView, ...],
    refusal: str = "",
) -> str:
    lines: list[str] = []
    for index, context in enumerate(contexts, start=1):
        lines.append(
            f"[{index}] {context.source} | {context.section} | "
            f"page {context.page} | {context.kind}"
        )
        lines.append(context.text)
        lines.append("")
    return render(
        template,
        {
            "question": question,
            "context": "\n".join(lines),
            "refusal": refusal,
        },
    )
