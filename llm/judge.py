"""Semantic judging through the local OpenAI-compatible endpoint.

The judge is deliberately generic: every prompt lives in TOML and receives
named placeholders. The code only renders the prompt and parses a JSON object
with boolean fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TypeAlias

from preprocess.jsonl import JsonValue

from .client import LlmSettings, chat
from .structured import parse_object

ChatFn: TypeAlias = Callable[[LlmSettings, str, str], str]


@dataclass(frozen=True, slots=True)
class JudgeSettings:
    system: str
    answer_prompt: str
    evidence_prompt: str
    question_prompt: str
    refusal_prompt: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class AnswerJudgement:
    correct: bool
    citation: bool
    refused: bool
    reasoning: bool = False


@dataclass(frozen=True, slots=True)
class QuestionJudgement:
    valid: bool
    quality: int
    multi_fact: bool = False


def _render(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"%{key.upper()}%", value)
    return rendered


def _flag(value: JsonValue) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


@dataclass(frozen=True, slots=True)
class LlmJudge:
    llm: LlmSettings
    settings: JudgeSettings
    chat_fn: ChatFn = chat

    def _ask(self, prompt: str) -> dict[str, JsonValue]:
        llm = replace(
            self.llm,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )
        try:
            return parse_object(self.chat_fn(llm, self.settings.system, prompt))
        except ValueError:
            # A parse failure is usually a truncated JSON object. Retry once
            # with a larger completion budget instead of repeating the same
            # cut-off response.
            larger = replace(llm, max_tokens=max(self.settings.max_tokens * 2, 1024))
            return parse_object(
                self.chat_fn(
                    larger,
                    self.settings.system,
                    prompt + "\nReturn one valid JSON object only.",
                )
            )

    def evidence_index(self, question: str, expected: str, context: str) -> int:
        prompt = _render(
            self.settings.evidence_prompt,
            {"question": question, "expected": expected, "context": context},
        )
        value = self._ask(prompt).get("index", 0)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return 0

    def evidence(self, question: str, expected: str, context: str) -> bool:
        return self.evidence_index(question, expected, context) > 0

    def answer(
        self,
        question: str,
        expected: str,
        answer: str,
        context: str,
    ) -> AnswerJudgement:
        prompt = _render(
            self.settings.answer_prompt,
            {
                "question": question,
                "expected": expected,
                "answer": answer,
                "context": context,
            },
        )
        value = self._ask(prompt)
        return AnswerJudgement(
            correct=_flag(value.get("correct", False)),
            citation=_flag(value.get("citation", False)),
            refused=_flag(value.get("refused", False)),
            reasoning=_flag(value.get("reasoning", False)),
        )

    def question(
        self,
        context: str,
        question: str,
        answer: str,
        reasoning: str = "",
    ) -> QuestionJudgement:
        prompt = _render(
            self.settings.question_prompt,
            {
                "context": context,
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
            },
        )
        value = self._ask(prompt)
        quality = value.get("quality", 1)
        if isinstance(quality, bool):
            score = 5 if quality else 1
        elif isinstance(quality, int):
            score = quality
        elif isinstance(quality, str) and quality.strip().isdigit():
            score = int(quality.strip())
        else:
            score = 1
        return QuestionJudgement(
            valid=_flag(value.get("valid", False)),
            quality=max(1, min(5, score)),
            multi_fact=_flag(value.get("multi_fact", False)),
        )

    def refusal(self, question: str, answer: str) -> bool:
        prompt = _render(
            self.settings.refusal_prompt,
            {"question": question, "answer": answer},
        )
        return _flag(self._ask(prompt).get("refused", False))
