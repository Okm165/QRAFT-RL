"""Training summary and history artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from preprocess.jsonl import JsonValue, record_to_json, write_json, write_jsonl
from sft.data import TrainingExample


@dataclass(frozen=True, slots=True)
class TrainSummary:
    examples: int
    steps: int
    loss: float
    loss_first: float
    loss_delta: float
    mean_weight: float
    weighted_examples: int
    seconds: float
    max_length: int
    max_tokens: int
    min_tokens: int
    prompt_tokens_max: int
    answer_tokens_max: int
    truncated: int
    val_examples: int
    best_eval_loss: float
    quantize: str
    output: str
    history: str
    process_rank: int
    world_size: int
    parallel: str
    devices: int


def _history_rows(history: list[dict[str, object]]) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for entry in history:
        row: dict[str, JsonValue] = {}
        for key, value in entry.items():
            if isinstance(value, (str, int, float, bool)):
                row[key] = value
            else:
                row[key] = str(value)
        rows.append(row)
    return rows


def _write_history(output_path: Path, history: list[dict[str, object]]) -> Path:
    path = output_path / "history.jsonl"
    write_jsonl(path, _history_rows(history))
    return path


def _float_value(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _summary(
    examples: tuple[TrainingExample, ...],
    validation: tuple[TrainingExample, ...],
    steps: int,
    max_steps: int,
    max_length: int,
    seconds: float,
    history: list[dict[str, object]],
    output_path: Path,
    rank: int,
    world_size: int,
    is_main: bool,
    quantize: str,
    parallel: str,
    devices: int,
) -> TrainSummary:
    losses = [
        _float_value(entry["loss"]) for entry in history if "loss" in entry
    ]
    eval_losses = [
        _float_value(entry["eval_loss"]) for entry in history if "eval_loss" in entry
    ]
    all_examples = examples + validation
    tokens = [len(example.input_ids) for example in all_examples]
    weights = [example.weight for example in all_examples]
    summary = TrainSummary(
        examples=len(examples),
        steps=steps,
        loss=losses[-1] if losses else 0.0,
        loss_first=losses[0] if losses else 0.0,
        loss_delta=round(losses[-1] - losses[0], 8) if len(losses) > 1 else 0.0,
        mean_weight=round(sum(weights) / len(weights), 6) if weights else 0.0,
        weighted_examples=sum(1 for weight in weights if weight != 1.0),
        seconds=seconds,
        max_length=max_length,
        max_tokens=max(tokens),
        min_tokens=min(tokens),
        prompt_tokens_max=max((example.prompt_tokens for example in all_examples), default=0),
        answer_tokens_max=max((example.answer_tokens for example in all_examples), default=0),
        truncated=sum(1 for example in all_examples if example.truncated),
        val_examples=len(validation),
        best_eval_loss=min(eval_losses) if eval_losses else 0.0,
        quantize=quantize,
        output=str(output_path),
        history=str(output_path / "history.jsonl"),
        process_rank=rank,
        world_size=world_size,
        parallel=parallel,
        devices=devices,
    )
    if is_main:
        write_json(output_path / "summary.json", record_to_json(summary))
    return summary
