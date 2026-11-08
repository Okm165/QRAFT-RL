"""Deterministic test judge. Production judging always uses the local LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm.judge import AnswerJudgement


@dataclass(frozen=True, slots=True)
class RuleJudge:
    def evidence_index(self, question: str, expected: str, context: str) -> int:
        for line in context.splitlines():
            match = re.match(r"\[(\d+)\]", line)
            if match and expected.lower() in line.lower():
                return int(match.group(1))
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
        refused = answer.strip().lower().startswith("i do not")
        correct = bool(expected) and expected.lower() in answer.lower() and not refused
        citation = False
        if not refused:
            lines = [line for line in context.splitlines() if line.startswith("[")]
            for value in re.findall(r"\[(\d+)\]", answer):
                position = int(value) - 1
                if 0 <= position < len(lines) and expected.lower() in lines[position].lower():
                    citation = True
                    break
        return AnswerJudgement(correct=correct, citation=citation, refused=refused)

    def question(self, context: str, question: str, answer: str) -> bool:
        return answer.lower() in context.lower()

    def refusal(self, question: str, answer: str) -> bool:
        return answer.strip().lower().startswith("i do not")
