"""Measure answer quality, citation support, and refusal behavior.

Every semantic decision is made by the configured local LLM judge. The code
only builds prompts and counts the judge's boolean verdicts.
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, TypeVar

from llm.client import LlmSettings, LoraScale, chat
from llm.judge import AnswerJudgement
from preprocess.jsonl import JsonValue, json_texts, path_set, read_jsonl
from rag.answer import Answer, answer_question
from telemetry import Telemetry, disabled

Generator: TypeAlias = Callable[[str], str]
RecordT = TypeVar("RecordT")
ResultT = TypeVar("ResultT")


class AnswerJudge(Protocol):
    def answer(
        self,
        question: str,
        expected: str,
        answer: str,
        context: str,
    ) -> AnswerJudgement: ...

    def refusal(self, question: str, answer: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class BehaviorResult:
    answerable: int
    answer_hits: int
    citation_hits: int
    answerable_refusals: int
    refusal_probes: int
    refusals: int
    multi_fact: int = 0
    multi_fact_answer_hits: int = 0
    multi_fact_citation_hits: int = 0
    lookup: int = 0
    lookup_answer_hits: int = 0
    lookup_citation_hits: int = 0


@dataclass(frozen=True, slots=True)
class _Answerable:
    row: Mapping[str, JsonValue]
    answer: Answer
    seconds: float


@dataclass(frozen=True, slots=True)
class _Judged:
    answered: _Answerable
    judgement: AnswerJudgement
    seconds: float


@dataclass(frozen=True, slots=True)
class _RefusalJudged:
    answered: _Answerable
    refused: bool
    seconds: float


def _context_text(answer: Answer) -> str:
    return "\n\n".join(
        f"[{index}] {hit.text}"
        for index, hit in enumerate(answer.citations, start=1)
    )


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _refused(answer: str, refusal: str) -> bool:
    normalized_answer = _normal(answer)
    normalized_refusal = _normal(refusal)
    return bool(normalized_refusal) and normalized_refusal in normalized_answer


def _complete(answer: str) -> bool:
    return answer.rstrip().endswith((".", "]", "!", "`"))


def _map(
    items: Sequence[RecordT],
    function: Callable[[RecordT], ResultT],
    workers: int,
) -> list[ResultT]:
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        return list(pool.map(function, items))


def _answer_row(
    db_path: Path,
    row: Mapping[str, JsonValue],
    generator: Generator,
    prompt_template: str,
    refusal: str,
    top_k: int,
) -> _Answerable:
    started = time.perf_counter()
    question = str(row["question"])
    answer = answer_question(
        db_path,
        question,
        top_k,
        generator,
        prompt_template,
        refusal,
    )
    return _Answerable(
        row=row,
        answer=answer,
        seconds=time.perf_counter() - started,
    )


def evaluate(
    db_path: Path,
    golden_path: Path,
    refusal_path: Path,
    generator: Generator,
    judge: AnswerJudge,
    prompt_template: str,
    refusal: str,
    top_k: int,
    limit: int = 0,
    telemetry: Telemetry = disabled("behavior"),
    workers: int = 1,
) -> BehaviorResult:
    rows = tuple(read_jsonl(golden_path))
    if limit > 0:
        rows = rows[:limit]

    def answer_row(row: Mapping[str, JsonValue]) -> _Answerable:
        return _answer_row(
            db_path,
            row,
            generator,
            prompt_template,
            refusal,
            top_k,
        )

    def judge_row(answered: _Answerable) -> _Judged:
        started = time.perf_counter()
        question = str(answered.row["question"])
        expected = ", ".join(json_texts(answered.row.get("expected_terms", [])))
        judgement = judge.answer(
            question,
            expected,
            answered.answer.text,
            _context_text(answered.answer),
        )
        return _Judged(
            answered=answered,
            judgement=judgement,
            seconds=time.perf_counter() - started,
        )

    answered_rows = _map(rows, answer_row, workers)
    judged_rows = _map(answered_rows, judge_row, workers)

    answerable = 0
    answer_hits = 0
    citation_hits = 0
    answerable_refusals = 0
    multi_fact_total = 0
    multi_fact_answer_hits = 0
    multi_fact_citation_hits = 0
    lookup_total = 0
    lookup_answer_hits = 0
    lookup_citation_hits = 0
    for judged in judged_rows:
        answered = judged.answered
        row = answered.row
        answer = answered.answer
        judgement = judged.judgement
        question = str(row["question"])
        expected = ", ".join(json_texts(row.get("expected_terms", [])))
        refused = judgement.refused or _refused(answer.text, refusal)
        hit = judgement.correct and not refused
        cited = judgement.citation and not refused
        multi_fact = bool(row.get("multi_fact", False))
        answerable += 1
        answer_hits += int(hit)
        citation_hits += int(cited)
        answerable_refusals += int(refused)
        if multi_fact:
            multi_fact_total += 1
            multi_fact_answer_hits += int(hit)
            multi_fact_citation_hits += int(cited)
        else:
            lookup_total += 1
            lookup_answer_hits += int(hit)
            lookup_citation_hits += int(cited)
        telemetry.timed(
            "question",
            answered.seconds + judged.seconds,
            question=question,
            correct=judgement.correct,
            hit=hit,
            citation=judgement.citation,
            reasoning=judgement.reasoning,
            refused=refused,
            multi_fact=multi_fact,
            answer_chars=len(answer.text),
            answer_long=len(answer.text) > 400,
            answer_complete=_complete(answer.text) or refused,
            answer=answer.text[:400],
        )
        print(f"{'HIT ' if hit else 'MISS'}  {question}")
        if expected:
            print(f"      expected: {expected}")
        print(f"      answer: {answer.text[:180].replace(chr(10), ' ')}")

    refusal_rows = tuple(read_jsonl(refusal_path))

    def answer_refusal(row: Mapping[str, JsonValue]) -> _Answerable:
        return _answer_row(
            db_path,
            row,
            generator,
            prompt_template,
            refusal,
            top_k,
        )

    def judge_refusal(answered: _Answerable) -> _RefusalJudged:
        started = time.perf_counter()
        question = str(answered.row["question"])
        refused = judge.refusal(question, answered.answer.text) or _refused(
            answered.answer.text,
            refusal,
        )
        return _RefusalJudged(
            answered=answered,
            refused=refused,
            seconds=time.perf_counter() - started,
        )

    refusal_answered = _map(refusal_rows, answer_refusal, workers)
    refusal_judged = _map(refusal_answered, judge_refusal, workers)
    probes = 0
    refusals = 0
    for judged in refusal_judged:
        answered = judged.answered
        question = str(answered.row["question"])
        probes += 1
        refusals += int(judged.refused)
        telemetry.timed(
            "refusal",
            answered.seconds + judged.seconds,
            question=question,
            refused=judged.refused,
            answer_chars=len(answered.answer.text),
            answer_long=len(answered.answer.text) > 400,
            answer_complete=_complete(answered.answer.text),
            answer=answered.answer.text[:400],
        )
        print(f"{'REFUSE' if judged.refused else 'ANSWER'}  {question}")
        if not judged.refused:
            print(f"      answer: {answered.answer.text[:180].replace(chr(10), ' ')}")
    return BehaviorResult(
        answerable=answerable,
        answer_hits=answer_hits,
        citation_hits=citation_hits,
        answerable_refusals=answerable_refusals,
        refusal_probes=probes,
        refusals=refusals,
        multi_fact=multi_fact_total,
        multi_fact_answer_hits=multi_fact_answer_hits,
        multi_fact_citation_hits=multi_fact_citation_hits,
        lookup=lookup_total,
        lookup_answer_hits=lookup_answer_hits,
        lookup_citation_hits=lookup_citation_hits,
    )


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure local answer and refusal behavior.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/processed/rag.sqlite"))
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--refusal-questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--lora-id", type=int, default=0)
    parser.add_argument("--lora-scale", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-below-answer", type=float, default=0.0)
    parser.add_argument("--fail-below-refusal", type=float, default=0.0)
    parser.add_argument("--telemetry", type=Path, default=Path(""))
    return parser


def main() -> int:
    args = _parser().parse_args()
    from cli.config import load_config

    config = load_config(args.config)
    llm = LlmSettings(
        base_url=(
            config.llm_base_urls[0] if config.llm_base_urls else config.llm_base_url
        ),
        model=config.llm_model,
        api_key=config.llm_api_key,
        timeout=config.llm_timeout,
        max_tokens=int(args.max_new_tokens),
        temperature=0.0,
    )
    lora = (
        (LoraScale(int(args.lora_id), float(args.lora_scale)),)
        if float(args.lora_scale) > 0.0
        else ()
    )

    def generator(prompt: str) -> str:
        return chat(llm, "", prompt, lora)

    telemetry = (
        Telemetry(args.telemetry, "behavior", args.telemetry.parent.parent.name)
        if path_set(args.telemetry)
        else disabled("behavior")
    )
    result = evaluate(
        args.db,
        args.golden,
        args.refusal_questions,
        generator,
        config.judge(),
        config.answer_prompt,
        config.answer_refusal,
        int(args.top_k),
        int(args.limit),
        telemetry,
        int(args.workers),
    )
    answer_rate = _rate(result.answer_hits, result.answerable)
    citation_rate = _rate(result.citation_hits, result.answerable)
    multi_fact_rate = _rate(result.multi_fact_answer_hits, result.multi_fact)
    multi_fact_citation_rate = _rate(
        result.multi_fact_citation_hits,
        result.multi_fact,
    )
    lookup_rate = _rate(result.lookup_answer_hits, result.lookup)
    lookup_citation_rate = _rate(result.lookup_citation_hits, result.lookup)
    refusal_rate = _rate(result.refusals, result.refusal_probes)
    false_refusal_rate = _rate(result.answerable_refusals, result.answerable)
    leakage_rate = _rate(
        result.refusal_probes - result.refusals,
        result.refusal_probes,
    )
    print(f"answer hits:            {result.answer_hits}/{result.answerable} = {answer_rate:.3f}")
    print(f"citation support hits:  {result.citation_hits}/{result.answerable} = {citation_rate:.3f}")
    print(
        f"multi-fact answers:     {result.multi_fact_answer_hits}/{result.multi_fact} "
        f"= {multi_fact_rate:.3f}  citations {result.multi_fact_citation_hits}/"
        f"{result.multi_fact} = {multi_fact_citation_rate:.3f}"
    )
    print(
        f"lookup answers:         {result.lookup_answer_hits}/{result.lookup} "
        f"= {lookup_rate:.3f}  citations {result.lookup_citation_hits}/"
        f"{result.lookup} = {lookup_citation_rate:.3f}"
    )
    print(f"refusals:               {result.refusals}/{result.refusal_probes} = {refusal_rate:.3f}")
    print(
        f"answerable refusals:    {result.answerable_refusals}/{result.answerable} "
        f"= {false_refusal_rate:.3f}"
    )
    print(
        f"refusal leakage:        {result.refusal_probes - result.refusals}/"
        f"{result.refusal_probes} = {leakage_rate:.3f}"
    )
    telemetry.emit(
        "summary",
        answerable=result.answerable,
        answer_hits=result.answer_hits,
        citation_hits=result.citation_hits,
        answerable_refusals=result.answerable_refusals,
        refusal_probes=result.refusal_probes,
        refusals=result.refusals,
        multi_fact=result.multi_fact,
        multi_fact_answer_hits=result.multi_fact_answer_hits,
        multi_fact_citation_hits=result.multi_fact_citation_hits,
        lookup=result.lookup,
        lookup_answer_hits=result.lookup_answer_hits,
        lookup_citation_hits=result.lookup_citation_hits,
        answer_rate=round(answer_rate, 6),
        citation_rate=round(citation_rate, 6),
        multi_fact_answer_rate=round(multi_fact_rate, 6),
        multi_fact_citation_rate=round(multi_fact_citation_rate, 6),
        lookup_answer_rate=round(lookup_rate, 6),
        lookup_citation_rate=round(lookup_citation_rate, 6),
        refusal_rate=round(refusal_rate, 6),
        false_refusal_rate=round(false_refusal_rate, 6),
        leakage_rate=round(leakage_rate, 6),
    )
    if answer_rate < args.fail_below_answer or refusal_rate < args.fail_below_refusal:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
