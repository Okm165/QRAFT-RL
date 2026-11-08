"""Tokenized SFT data, collation, and deterministic splits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llm.chat import render_chat
from rag.prompt import ContextView, render_answer_prompt

from preprocess.jsonl import JsonValue, json_texts, read_jsonl

DEFAULT_MAX_LENGTH = 1024


class TrainingRow(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    weight: float


@dataclass(frozen=True, slots=True)
class TrainingExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    truncated: bool
    doc_id: str = ""
    weight: float = 1.0
    prompt_tokens: int = 0
    answer_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ListDataset(Dataset[TrainingRow]):
    rows: list[TrainingRow]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainingRow:
        return self.rows[index]


@dataclass(frozen=True, slots=True)
class PadCollator:
    pad_token_id: int

    def __call__(self, features: list[TrainingRow]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor(
                [feature["weight"] for feature in features],
                dtype=torch.float32,
            ),
        }


def _int_value(value: JsonValue) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _views(contexts: list[dict[str, JsonValue]]) -> tuple[ContextView, ...]:
    return tuple(
        ContextView(
            source=str(context.get("source", "")),
            section=str(context.get("section", "")),
            page=_int_value(context.get("page", 0)),
            kind=str(context.get("kind", "")),
            text=str(context.get("text", "")),
        )
        for context in contexts
    )


def _prompt(
    question: str,
    contexts: list[dict[str, JsonValue]],
    template: str,
    refusal: str,
) -> str:
    return render_answer_prompt(template, question, _views(contexts), refusal)


def _contexts(value: JsonValue) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        return []
    return [cast(dict[str, JsonValue], item) for item in value if isinstance(item, dict)]


def _render(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    if not tokenizer.chat_template:
        return prompt
    return render_chat(tokenizer, ({"role": "user", "content": prompt},), True)


def _tokenize(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    answer: str,
    max_length: int,
    doc_id: str,
    weight: float = 1.0,
    allow_truncation: bool = False,
) -> TrainingExample:
    prompt_ids = cast(
        list[int],
        tokenizer(_render(tokenizer, prompt), add_special_tokens=False)["input_ids"],
    )
    eos = tokenizer.eos_token
    answer_text = answer + (eos if isinstance(eos, str) else "")
    answer_ids = cast(
        list[int],
        tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"],
    )
    total = len(prompt_ids) + len(answer_ids)
    truncated = total > max_length
    labels = [-100] * len(prompt_ids) + answer_ids
    if truncated and allow_truncation:
        # Explicit opt-in only. The default is to fail in train() so a silent
        # cut can never corrupt supervision or remove the EOS token.
        input_ids = (prompt_ids + answer_ids)[:max_length]
        labels = labels[:max_length]
    else:
        input_ids = prompt_ids + answer_ids
    return TrainingExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        truncated=truncated,
        doc_id=doc_id,
        weight=weight,
        prompt_tokens=len(prompt_ids),
        answer_tokens=len(answer_ids),
    )


def _text_pair(
    row: Mapping[str, JsonValue],
    template: str,
    refusal: str = "",
) -> tuple[str, str]:
    if "prompt" in row:
        return str(row["prompt"]), str(row["answer"])
    return (
        _prompt(str(row["question"]), _contexts(row["contexts"]), template, refusal),
        str(row["answer"]),
    )


def _doc_id(row: Mapping[str, JsonValue]) -> str:
    contexts = _contexts(row.get("contexts", []))
    oracle = row.get("oracle_index")
    if isinstance(oracle, int) and 0 <= oracle < len(contexts):
        for key in ("doc_id", "source"):
            value = contexts[oracle].get(key)
            if isinstance(value, str) and value:
                return value
    for context in contexts:
        for key in ("doc_id", "source"):
            value = context.get(key)
            if isinstance(value, str) and value:
                return value
    expected = json_texts(row.get("expected_doc_ids", []))
    if expected:
        return expected[0]
    value = row.get("source")
    return value if isinstance(value, str) else ""


def _weight(row: Mapping[str, JsonValue]) -> float:
    value = row.get("weight", 1.0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(2.0, float(value)))
    return 1.0


def _examples(
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    template: str,
    refusal: str = "",
    allow_truncation: bool = False,
) -> tuple[TrainingExample, ...]:
    examples: list[TrainingExample] = []
    for row in read_jsonl(path):
        prompt, answer = _text_pair(row, template, refusal)
        examples.append(
            _tokenize(
                tokenizer,
                prompt,
                answer,
                max_length,
                _doc_id(row),
                _weight(row),
                allow_truncation,
            )
        )
    return tuple(examples)


def _tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _pad_id(tokenizer: PreTrainedTokenizerBase) -> int:
    value = tokenizer.pad_token_id
    return value if isinstance(value, int) else 0


def _split(
    examples: tuple[TrainingExample, ...],
    val_fraction: float,
) -> tuple[tuple[TrainingExample, ...], tuple[TrainingExample, ...]]:
    if val_fraction <= 0.0 or len(examples) < 2:
        return examples, ()
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.doc_id or f"row-{index}", []).append(index)
    target = max(1, min(len(examples) - 1, int(round(len(examples) * val_fraction))))
    if len(groups) < 2:
        indexes = {int(index * len(examples) / target) for index in range(target)}
    else:
        # Prefer whole documents, but start with the smallest groups so one
        # large document cannot consume the entire training set. Always leave
        # at least one training example.
        selected: set[str] = set()
        count = 0
        for key, members in sorted(groups.items(), key=lambda item: len(item[1])):
            if count >= target:
                break
            if len(examples) - count - len(members) < 1:
                continue
            selected.add(key)
            count += len(members)
        if not selected:
            indexes = {
                int(index * len(examples) / target) for index in range(target)
            }
        else:
            indexes = {index for key in selected for index in groups[key]}
    train = tuple(example for index, example in enumerate(examples) if index not in indexes)
    validation = tuple(
        example for index, example in enumerate(examples) if index in indexes
    )
    return train, validation
