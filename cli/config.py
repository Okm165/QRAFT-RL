"""Strict TOML configuration for the generic pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llm.client import LlmSettings
from llm.judge import JudgeSettings, LlmJudge
from preprocess.jsonl import JsonValue

DEFAULT_PREPROCESS_PROMPT = """Text:
%TEXT%

Extract a compact, source-grounded knowledge library from this OCR text.
Return exactly this JSON object:
{"chunks": [{"text": "...", "section": "...", "kind": "prose", "concepts": ["..."]}], "facts": [{"text": "...", "concepts": ["..."]}], "concepts": [{"name": "...", "definition": "..."}]}
Rules:
- Return up to 4 chunks, 8 facts, and 6 concepts.
- Each chunk must be a contiguous, verbatim span copied from the text, roughly 40-300 words. Do not paraphrase, rewrite, or merge distant passages; keep numbers, signal names, and register names exact.
- Use kind "prose", "table", or "code".
- Facts are atomic, source-grounded statements copied or closely paraphrased from the text.
- Only include a concept when you can give a short definition grounded in the text; otherwise omit it.
- Do not invent information. Return empty lists when nothing is useful.
- Do not add explanations or markdown fences."""

DEFAULT_QUESTIONS_SYSTEM = (
    "You build a source-grounded question-answer dataset. Return JSON only."
)
DEFAULT_QUESTIONS_PROMPT = """Context:
%CONTEXT%

Generate up to %COUNT% diverse, retrieval-friendly, source-grounded question-answer pairs.
Return exactly this JSON object:
{"items": [{"question": "...", "answer": "...", "reasoning": "...", "evidence": "...", "kind": "synthesis"}]}
Rules:
- Use the exact entity, signal, register, and section terms from the context so the question is retrievable by keyword search.
- kind must be "synthesis" when answering requires combining two or more distinct facts, comparing alternatives, or explaining a procedure/constraint; use "lookup" only for a single fact.
- At least half of the items must be "synthesis". If the context cannot support a synthesis question, return fewer items rather than padding with lookups.
- For synthesis questions, the answer must explicitly state both facts or both sides of the comparison; do not answer with only one side.
- Do not use "and" to join two unrelated facts; the facts must be needed together to answer.
- Ask one focused question per item. Prefer "why", "how", "which", "compare", and "what happens if" forms over "What is X?".
- If the context describes multiple device variants, revisions, or packages with different values for the same property, name the variant in the question. If the variant cannot be identified, skip that fact.
- The answer must be a short phrase copied or closely paraphrased from the context.
- The reasoning must be a concise 2-4 step chain that cites the exact evidence and uses no outside knowledge.
- The evidence must be the shortest exact quote that supports the answer.
- Ask about definitions, values, procedures, constraints, comparisons, and multi-fact synthesis.
- Do not ask yes/no questions, table titles, or headings.
- Do not invent information. Return fewer items when the context is thin.
- Do not add explanations, markdown fences, or any other text."""

DEFAULT_ANSWER_PROMPT = """You are a source-grounded assistant. Quote the decisive evidence, think through it step by step, then answer.
Use only the supplied context. Do not use prior or world knowledge.
Cite sources as [1], [2], ... for every factual claim.
If the context gives different values for different device variants, state the variant that matches the question; do not blend them.
If the context does not explicitly contain the answer, output exactly:
%REFUSAL%

Format:
Quote: <shortest exact supporting quote> [n]
Reasoning:
<2-4 short evidence-linked steps>
Answer: <short final answer> [n]

Question: %QUESTION%

Context:
%CONTEXT%
Answer:"""

DEFAULT_ANSWER_REFUSAL = "I do not have enough evidence in the corpus to answer this."
DEFAULT_RAFT_ANSWER_TEMPLATE = """Quote: %EVIDENCE% [%INDEX%]

Reasoning:
%REASONING%

Answer: %ANSWER% [%INDEX%]"""

DEFAULT_JUDGE_SYSTEM = "You are a strict evaluator. Return JSON only."

DEFAULT_JUDGE_ANSWER_PROMPT = """Question: %QUESTION%
Expected answer: %EXPECTED%
Model answer: %ANSWER%
Retrieved context:
%CONTEXT%

Return exactly this JSON object:
{"correct": true, "citation": true, "refused": false, "reasoning": true}

Rules:
- correct is true only when the model answer states the expected answer, allowing paraphrases and extra detail.
- If the model answer refuses or says the context is insufficient, correct must be false and citation must be false.
- citation is true when the model answer cites at least one numbered context that supports the expected answer.
- refused is true when the model answer declines to answer or says the context is insufficient.
- reasoning is true when the model answer contains a short evidence-linked rationale before the final answer.
- Judge only from the provided context. Do not add explanations."""

DEFAULT_JUDGE_EVIDENCE_PROMPT = """Question: %QUESTION%
Expected answer: %EXPECTED%
Contexts:
%CONTEXT%

Return exactly this JSON object:
{"index": 1}

Rules:
- index is the number of the first context that contains enough information to answer the question with the expected answer.
- Use 0 when no context contains the answer.
- Paraphrases and synonyms count. Do not require the exact expected wording.
- Do not add explanations."""

DEFAULT_JUDGE_QUESTION_PROMPT = """Context:
%CONTEXT%
Question: %QUESTION%
Answer: %ANSWER%
Reasoning:
%REASONING%

Return exactly this JSON object:
{"valid": true, "quality": 5, "multi_fact": true}

Rules:
- valid is true only when the answer is supported by the context and the question is answerable from this context alone.
- multi_fact is true only when the answer explicitly combines at least two distinct facts from the context, states both sides of a comparison, or explains a procedure/constraint with at least two steps. A single-fact lookup is false.
- quality is 1-5: 5 = precise multi-fact synthesis, comparison, or procedure; 4 = precise single-fact lookup; 3 = acceptable but shallow; 2 = vague or ambiguous; 1 = trivial, tautological, or unsupported.
- quality 5 requires multi_fact true and the answer explicitly combines the facts.
- valid is false when the question is ambiguous across device variants or the context contains conflicting values for the requested fact; set quality to 2 or lower.
- The reasoning must be grounded in the context; ignore unsupported steps.
- Do not add explanations."""

DEFAULT_JUDGE_REFUSAL_PROMPT = """Question: %QUESTION%
Model answer: %ANSWER%

Return exactly this JSON object:
{"refused": true}

Rules:
- refused is true when the model answer declines to answer because the corpus lacks evidence.
- refused is false when the model answer provides any factual answer.
- Do not add explanations."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    name: str
    output: Path
    input_path: Path
    glob: str
    limit: int
    preprocess_llm_limit: int
    preprocess_llm_max_tokens: int
    preprocess_window_chars: int
    preprocess_overlap_chars: int
    preprocess_system: str
    preprocess_prompt: str
    questions_system: str
    questions_prompt: str
    questions_base_url: str
    questions_base_urls: tuple[str, ...]
    questions_model: str
    questions_api_key: str
    questions_timeout: int
    questions_max_tokens: int
    questions_temperature: float
    questions_workers: int
    questions_per_chunk: int
    questions_limit: int
    questions_chunks: int
    questions_holdout_every: int
    answer_prompt: str
    answer_refusal: str
    raft_answer_template: str
    judge_system: str
    judge_answer_prompt: str
    judge_evidence_prompt: str
    judge_question_prompt: str
    judge_refusal_prompt: str
    judge_max_tokens: int
    judge_temperature: float
    llm_base_url: str
    llm_base_urls: tuple[str, ...]
    llm_workers: int
    llm_context: int
    llm_model: str
    llm_api_key: str
    llm_timeout: int
    llm_max_tokens: int
    llm_temperature: float
    question_chunks: int
    rag_top_k: int
    raft_distractors: int
    raft_absent_every: int
    raft_seed: int
    train_model: str
    train_dtype: str
    train_quantize: str
    train_parallel: str
    world_size: int
    max_steps: int
    batch_size: int
    grad_accum: int
    warmup_steps: int
    max_length: int
    learning_rate: float
    save_steps: int
    eval_steps: int
    val_fraction: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: str
    train_min_weight: float
    train_allow_truncation: bool
    train_attn: str
    train_optimizer: str
    train_gradient_checkpointing: bool
    eval_top_k: int
    eval_max_new_tokens: int
    eval_fail_below: float
    eval_limit: int
    eval_fail_below_answer: float
    eval_fail_below_refusal: float
    refusal_questions: tuple[str, ...]
    env: dict[str, str]

    def judge(self) -> LlmJudge:
        return LlmJudge(
            llm=LlmSettings(
                base_url=(
                    self.llm_base_urls[0] if self.llm_base_urls else self.llm_base_url
                ),
                model=self.llm_model,
                api_key=self.llm_api_key,
                timeout=self.llm_timeout,
                max_tokens=self.judge_max_tokens,
                temperature=self.judge_temperature,
            ),
            settings=JudgeSettings(
                system=self.judge_system,
                answer_prompt=self.judge_answer_prompt,
                evidence_prompt=self.judge_evidence_prompt,
                question_prompt=self.judge_question_prompt,
                refusal_prompt=self.judge_refusal_prompt,
                max_tokens=self.judge_max_tokens,
                temperature=self.judge_temperature,
            ),
        )


def _section(raw: dict[str, JsonValue], name: str) -> dict[str, JsonValue]:
    value = raw.get(name, {})
    if isinstance(value, dict):
        return cast(dict[str, JsonValue], value)
    return {}


def _text(section: dict[str, JsonValue], key: str, default: str) -> str:
    value = section.get(key, default)
    return str(value) if isinstance(value, (str, int, float, bool)) else default


def _int(section: dict[str, JsonValue], key: str, default: int) -> int:
    value = section.get(key, default)
    return int(value) if isinstance(value, (int, float, str)) else default


def _float(section: dict[str, JsonValue], key: str, default: float) -> float:
    value = section.get(key, default)
    return float(value) if isinstance(value, (int, float, str)) else default


def _bool(section: dict[str, JsonValue], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _required_text(section: dict[str, JsonValue], name: str, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"[{name}].{key} is required")
    return value.strip()


def _texts(section: dict[str, JsonValue], key: str) -> tuple[str, ...]:
    value = section.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(
        str(item).strip()
        for item in value
        if isinstance(item, (str, int, float)) and str(item).strip()
    )


def load_config(path: Path) -> PipelineConfig:
    raw = cast(dict[str, JsonValue], tomllib.loads(path.read_text(encoding="utf-8")))
    run = _section(raw, "run")
    input_config = _section(raw, "input")
    preprocess = _section(raw, "preprocess")
    questions = _section(raw, "questions")
    answer = _section(raw, "answer")
    judge = _section(raw, "judge")
    llm = _section(raw, "llm")
    rag = _section(raw, "rag")
    raft = _section(raw, "raft")
    train = _section(raw, "train")
    evaluation = _section(raw, "eval")
    runtime = _section(raw, "runtime")
    env_value = runtime.get("env", {})
    env: dict[str, str] = {}
    if isinstance(env_value, dict):
        env = {str(key): str(value) for key, value in env_value.items()}
    llm_model = _required_text(llm, "llm", "model")
    train_model = _required_text(train, "train", "model")
    train_quantize = _text(train, "quantize", "none")
    train_parallel = _text(train, "parallel", "data")
    if train_quantize not in {"none", "4bit", "pre4bit"}:
        raise ValueError("[train].quantize must be none, 4bit, or pre4bit")
    if train_parallel not in {"data", "model"}:
        raise ValueError("[train].parallel must be data or model")
    train_attn = _text(train, "attn", "sdpa")
    if train_attn not in {"eager", "sdpa"}:
        raise ValueError("[train].attn must be eager or sdpa")
    train_optimizer = _text(train, "optimizer", "auto")
    if train_optimizer not in {"auto", "paged_adamw_8bit", "adamw_torch"}:
        raise ValueError(
            "[train].optimizer must be auto, paged_adamw_8bit, or adamw_torch"
        )
    window_chars = _int(preprocess, "window_chars", 6000)
    overlap_chars = _int(preprocess, "overlap_chars", 400)
    if window_chars < 1:
        raise ValueError("[preprocess].window_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= window_chars:
        raise ValueError("[preprocess].overlap_chars must be in [0, window_chars)")
    questions_system = _text(questions, "system", DEFAULT_QUESTIONS_SYSTEM)
    questions_prompt = _text(questions, "prompt", DEFAULT_QUESTIONS_PROMPT)
    answer_prompt = _text(answer, "prompt", DEFAULT_ANSWER_PROMPT)
    answer_refusal = _text(answer, "refusal", DEFAULT_ANSWER_REFUSAL)
    raft_answer_template = _text(
        raft,
        "answer_template",
        DEFAULT_RAFT_ANSWER_TEMPLATE,
    )
    for name, template, keys in (
        ("questions.prompt", questions_prompt, ("context",)),
        ("answer.prompt", answer_prompt, ("question", "context", "refusal")),
        (
            "raft.answer_template",
            raft_answer_template,
            ("reasoning", "answer", "index"),
        ),
    ):
        missing = [key for key in keys if f"%{key.upper()}%" not in template]
        if missing:
            raise ValueError(f"[{name}] is missing placeholders: {', '.join(missing)}")
    judge_system = _text(judge, "system", DEFAULT_JUDGE_SYSTEM)
    judge_answer_prompt = _text(judge, "answer_prompt", DEFAULT_JUDGE_ANSWER_PROMPT)
    judge_evidence_prompt = _text(judge, "evidence_prompt", DEFAULT_JUDGE_EVIDENCE_PROMPT)
    judge_question_prompt = _text(judge, "question_prompt", DEFAULT_JUDGE_QUESTION_PROMPT)
    judge_refusal_prompt = _text(judge, "refusal_prompt", DEFAULT_JUDGE_REFUSAL_PROMPT)
    for name, template, keys in (
        ("answer_prompt", judge_answer_prompt, ("question", "expected", "answer", "context")),
        ("evidence_prompt", judge_evidence_prompt, ("question", "expected", "context")),
        (
            "question_prompt",
            judge_question_prompt,
            ("context", "question", "answer", "reasoning"),
        ),
        ("refusal_prompt", judge_refusal_prompt, ("question", "answer")),
    ):
        missing = [key for key in keys if f"%{key.upper()}%" not in template]
        if missing:
            raise ValueError(f"[judge].{name} is missing placeholders: {', '.join(missing)}")
    llm_base_url = _text(llm, "base_url", "http://127.0.0.1:8000/v1")
    llm_base_urls = _texts(llm, "base_urls") or (llm_base_url,)
    llm_workers = _int(llm, "workers", len(llm_base_urls))
    llm_context = _int(llm, "context", 196608)
    if llm_workers < 1:
        raise ValueError("[llm].workers must be >= 1")
    if llm_context < 32768 or llm_context > 262144:
        raise ValueError("[llm].context must be in [32768, 262144]")
    if llm_context // llm_workers < 4096:
        raise ValueError(
            "[llm].context / workers must give at least 4096 tokens per slot"
        )
    questions_base_url = _text(questions, "base_url", llm_base_url)
    questions_base_urls = _texts(questions, "base_urls") or (questions_base_url,)
    questions_model = _text(questions, "model", llm_model)
    questions_api_key = _text(questions, "api_key", _text(llm, "api_key", "local"))
    questions_timeout = _int(questions, "timeout", _int(llm, "timeout", 300))
    questions_max_tokens = _int(questions, "max_tokens", _int(llm, "max_tokens", 2048))
    questions_temperature = _float(questions, "temperature", 0.0)
    questions_workers = _int(questions, "workers", llm_workers)
    questions_per_chunk = _int(questions, "per_chunk", 1)
    questions_limit = _int(questions, "limit", _int(llm, "question_chunks", 20))
    questions_chunks = _int(questions, "chunks", 0)
    questions_holdout_every = _int(questions, "holdout_every", 5)
    preprocess_max_tokens = _int(preprocess, "llm_max_tokens", 2048)
    judge_max_tokens = _int(judge, "max_tokens", 512)
    eval_max_new_tokens = _int(evaluation, "max_new_tokens", 512)
    max_length = _int(train, "max_length", 2048)
    floors = (
        ("[preprocess].llm_max_tokens", preprocess_max_tokens, 2048),
        ("[questions].max_tokens", questions_max_tokens, 2048),
        ("[judge].max_tokens", judge_max_tokens, 512),
        ("[eval].max_new_tokens", eval_max_new_tokens, 512),
        ("[train].max_length", max_length, 2048),
    )
    for name, value, floor in floors:
        if value < floor:
            raise ValueError(
                f"{name}={value} is below the token floor {floor}; "
                "raise it so generation and training never truncate silently"
            )
    if questions_per_chunk < 1:
        raise ValueError("[questions].per_chunk must be >= 1")
    if questions_limit < 1:
        raise ValueError("[questions].limit must be >= 1")
    refusal_questions = _texts(evaluation, "refusal_questions")
    if not refusal_questions:
        raise ValueError("[eval].refusal_questions must contain at least one probe")
    return PipelineConfig(
        name=_text(run, "name", path.stem),
        output=Path(_text(run, "output", "data/runs")),
        input_path=Path(_required_text(input_config, "input", "path")),
        glob=_text(input_config, "glob", "*.md"),
        limit=_int(input_config, "limit", 0),
        preprocess_llm_limit=_int(preprocess, "llm_limit", 40),
        preprocess_llm_max_tokens=preprocess_max_tokens,
        preprocess_window_chars=window_chars,
        preprocess_overlap_chars=overlap_chars,
        preprocess_system=_text(
            preprocess,
            "system",
            "Extract retrieval chunks and facts from the text. Return JSON only.",
        ),
        preprocess_prompt=_text(
            preprocess,
            "prompt",
            DEFAULT_PREPROCESS_PROMPT,
        ),
        questions_system=questions_system,
        questions_prompt=questions_prompt,
        questions_base_url=questions_base_url,
        questions_base_urls=questions_base_urls,
        questions_model=questions_model,
        questions_api_key=questions_api_key,
        questions_timeout=questions_timeout,
        questions_max_tokens=questions_max_tokens,
        questions_temperature=questions_temperature,
        questions_workers=questions_workers,
        questions_per_chunk=questions_per_chunk,
        questions_limit=questions_limit,
        questions_chunks=questions_chunks,
        questions_holdout_every=questions_holdout_every,
        answer_prompt=answer_prompt,
        answer_refusal=answer_refusal,
        raft_answer_template=raft_answer_template,
        judge_system=judge_system,
        judge_answer_prompt=judge_answer_prompt,
        judge_evidence_prompt=judge_evidence_prompt,
        judge_question_prompt=judge_question_prompt,
        judge_refusal_prompt=judge_refusal_prompt,
        judge_max_tokens=judge_max_tokens,
        judge_temperature=_float(judge, "temperature", 0.0),
        llm_base_url=llm_base_url,
        llm_base_urls=llm_base_urls,
        llm_workers=llm_workers,
        llm_context=llm_context,
        llm_model=llm_model,
        llm_api_key=_text(llm, "api_key", "local"),
        llm_timeout=_int(llm, "timeout", 300),
        llm_max_tokens=_int(llm, "max_tokens", 2048),
        llm_temperature=_float(llm, "temperature", 0.0),
        question_chunks=_int(llm, "question_chunks", 20),
        rag_top_k=_int(rag, "top_k", 5),
        raft_distractors=_int(raft, "distractors", 4),
        raft_absent_every=_int(raft, "absent_every", 5),
        raft_seed=_int(raft, "seed", 42),
        train_model=train_model,
        train_dtype=_text(train, "dtype", "auto"),
        train_quantize=train_quantize,
        train_parallel=train_parallel,
        world_size=_int(train, "world_size", 1),
        max_steps=_int(train, "max_steps", 10),
        batch_size=_int(train, "batch_size", 2),
        grad_accum=_int(train, "grad_accum", 1),
        warmup_steps=_int(train, "warmup_steps", 0),
        max_length=max_length,
        learning_rate=_float(train, "learning_rate", 0.0001),
        save_steps=_int(train, "save_steps", 0),
        eval_steps=_int(train, "eval_steps", 0),
        val_fraction=_float(train, "val_fraction", 0.0),
        lora_r=_int(train, "lora_r", 8),
        lora_alpha=_int(train, "lora_alpha", 16),
        lora_dropout=_float(train, "lora_dropout", 0.0),
        lora_targets=_text(train, "lora_targets", "q_proj,v_proj"),
        train_min_weight=_float(train, "min_weight", 0.0),
        train_allow_truncation=_bool(train, "allow_truncation", False),
        train_attn=train_attn,
        train_optimizer=train_optimizer,
        train_gradient_checkpointing=_bool(train, "gradient_checkpointing", True),
        eval_top_k=_int(evaluation, "top_k", 5),
        eval_max_new_tokens=eval_max_new_tokens,
        eval_fail_below=_float(evaluation, "fail_below", 0.9),
        eval_limit=_int(evaluation, "limit", 0),
        eval_fail_below_answer=_float(evaluation, "fail_below_answer", 0.0),
        eval_fail_below_refusal=_float(evaluation, "fail_below_refusal", 0.0),
        refusal_questions=refusal_questions,
        env=env,
    )
