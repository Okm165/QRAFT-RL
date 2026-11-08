"""Minimal local SFT trainer: one process, model-parallel shards."""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from preprocess.jsonl import path_set
from sft.data import (
    DEFAULT_MAX_LENGTH,
    PadCollator,
    TrainingExample,
    TrainingRow,
    _examples,
    _pad_id,
    _split,
    _tokenizer,
)
from sft.model import (
    PARALLEL_MODES,
    LoraSettings,
    _load_model,
    _lora_settings,
    _save_adapter,
)
from sft.summary import TrainSummary, _summary, _write_history
from telemetry import Telemetry, disabled, gpu_snapshot

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "auto" else DTYPES[name]


def _rows(examples: tuple[TrainingExample, ...]) -> list[TrainingRow]:
    return [
        {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
            "weight": example.weight,
        }
        for example in examples
    ]


def _weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    valid = (shift_labels != -100).sum(dim=1).clamp(min=1)
    per_example = token_loss.sum(dim=1) / valid
    return (per_example * weights).sum() / weights.sum().clamp(min=1e-6)


def _optimizer(
    parameters: list[torch.nn.Parameter],
    name: str,
    learning_rate: float,
    quantize: str,
) -> torch.optim.Optimizer:
    selected = (
        "paged_adamw_8bit"
        if name == "auto" and quantize in {"4bit", "pre4bit"}
        else "adamw_torch"
        if name == "auto"
        else name
    )
    if selected == "paged_adamw_8bit":
        from bitsandbytes.optim import PagedAdamW8bit

        return PagedAdamW8bit(parameters, lr=learning_rate)
    return torch.optim.AdamW(parameters, lr=learning_rate)


def _auto_steps(examples: int, batch_size: int, grad_accum: int) -> int:
    """Two epochs at the effective batch size, with a one-step floor."""
    return max(1, math.ceil(examples * 2 / max(1, batch_size * grad_accum)))


def _batches(
    rows: list[TrainingRow],
    batch_size: int,
) -> Iterator[list[TrainingRow]]:
    index = 0
    while True:
        yield [
            rows[(index + offset) % len(rows)]
            for offset in range(batch_size)
        ]
        index += batch_size


def _eval_loss(
    model: PeftModel,
    collator: PadCollator,
    rows: list[TrainingRow],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for row in rows:
            inputs = collator([row])
            weights = inputs.pop("weights").to(device)
            labels = inputs.pop("labels").to(device)
            moved = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=dtype):
                logits = model(**moved).logits
                loss = _weighted_loss(logits, labels, weights)
            total += float(loss)
    model.train()
    return total / len(rows) if rows else 0.0


def _train_loop(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    training: tuple[TrainingExample, ...],
    validation: tuple[TrainingExample, ...],
    output_path: Path,
    max_steps: int,
    learning_rate: float,
    seed: int,
    dtype: torch.dtype,
    batch_size: int,
    grad_accum: int,
    save_steps: int,
    eval_steps: int,
    quantize: str,
    optimizer_name: str,
    telemetry: Telemetry,
    warmup_steps: int = 0,
) -> tuple[list[dict[str, object]], int, float]:
    device = torch.device("cuda:0")
    train_rows = _rows(training)
    eval_rows = _rows(validation)
    random.Random(seed).shuffle(train_rows)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("the model has no trainable parameters")
    optimizer = _optimizer(parameters, optimizer_name, learning_rate, quantize)

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scheduler.step(0)
    collator = PadCollator(_pad_id(tokenizer))
    batches = _batches(train_rows, batch_size)
    history: list[dict[str, object]] = []
    model.train()
    step = 0
    started = time.perf_counter()
    while step < max_steps:
        step += 1
        losses: list[float] = []
        for _ in range(grad_accum):
            batch = next(batches)
            inputs = collator(batch)
            weights = inputs.pop("weights").to(device)
            labels = inputs.pop("labels").to(device)
            moved = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=dtype):
                logits = model(**moved).logits
                loss = _weighted_loss(logits, labels, weights)
            losses.append(float(loss.detach()))
            (loss / grad_accum).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        entry: dict[str, object] = {
            "step": step,
            "epoch": round(step * batch_size * grad_accum / len(train_rows), 6),
            "loss": round(sum(losses) / len(losses), 8),
            "grad_norm": round(grad_norm, 8),
            "learning_rate": round(scheduler.get_last_lr()[0], 8),
        }
        if torch.cuda.is_available():
            entry["vram_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 2
            )
            entry["vram_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / (1024 * 1024), 2
            )
            entry["gpus"] = gpu_snapshot()
        history.append(entry)
        telemetry.emit("step", **entry)
        print(
            f"step {step}/{max_steps} loss={entry['loss']}"
            f" grad={entry['grad_norm']} lr={entry['learning_rate']}",
            flush=True,
        )
        if eval_rows and eval_steps > 0 and step % eval_steps == 0:
            eval_loss = _eval_loss(model, collator, eval_rows, device, dtype)
            history.append(
                {
                    "step": step,
                    "epoch": entry["epoch"],
                    "eval_loss": round(eval_loss, 8),
                }
            )
            telemetry.emit("eval", step=step, eval_loss=eval_loss)
        if save_steps > 0 and step % save_steps == 0:
            _save_adapter(model, output_path / f"checkpoint-{step}")
    seconds = time.perf_counter() - started
    _save_adapter(model, output_path)
    _write_history(output_path, history)
    telemetry.emit("train_end", step=step, epoch=history[-1].get("epoch", 0.0))
    return history, step, seconds


def train(
    data_path: Path,
    model_path: str,
    output_path: Path,
    max_steps: int,
    max_length: int,
    prompt_template: str,
    learning_rate: float,
    seed: int,
    dtype_name: str,
    batch_size: int,
    grad_accum: int,
    save_steps: int,
    eval_steps: int,
    val_fraction: float,
    quantize: str,
    parallel: str,
    devices: int,
    lora: LoraSettings,
    refusal: str = "",
    min_weight: float = 0.0,
    allow_truncation: bool = False,
    attn: str = "sdpa",
    gradient_checkpointing: bool = True,
    optimizer: str = "auto",
    telemetry: Telemetry = disabled("train"),
    warmup_steps: int = 0,
) -> TrainSummary:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/HIP device is required for training")
    if parallel not in PARALLEL_MODES:
        raise ValueError(f"unsupported parallel mode: {parallel}")
    if devices < 1:
        raise ValueError("devices must be at least 1")
    if devices > torch.cuda.device_count():
        raise ValueError(
            f"requested {devices} devices but only {torch.cuda.device_count()} are visible"
        )
    dtype = _dtype(dtype_name)
    torch.manual_seed(seed)
    tokenizer = _tokenizer(model_path)
    examples = _examples(
        data_path,
        tokenizer,
        max_length,
        prompt_template,
        refusal,
        allow_truncation,
    )
    examples = tuple(example for example in examples if example.weight >= min_weight)
    if not examples:
        raise ValueError(f"no training examples in {data_path} above weight {min_weight}")
    truncated = [example for example in examples if example.truncated]
    if truncated and not allow_truncation:
        raise ValueError(
            f"{len(truncated)} examples exceed max_length {max_length}; "
            "raise max_length or allow truncation"
        )
    training, validation = _split(examples, val_fraction)
    if not training:
        raise ValueError("no training examples after the validation split")
    if max_steps <= 0:
        max_steps = _auto_steps(len(training), batch_size, grad_accum)
    model = _load_model(
        model_path,
        dtype,
        torch.device("cuda:0"),
        lora,
        quantize,
        parallel,
        devices,
        attn,
        gradient_checkpointing,
    )
    history, steps, seconds = _train_loop(
        model,
        tokenizer,
        training,
        validation,
        output_path,
        max_steps,
        learning_rate,
        seed,
        dtype,
        batch_size,
        grad_accum,
        save_steps,
        eval_steps,
        quantize,
        optimizer,
        telemetry,
        warmup_steps,
    )
    return _summary(
        training,
        validation,
        steps,
        max_steps,
        max_length,
        seconds,
        history,
        output_path,
        0,
        1,
        True,
        quantize,
        parallel,
        devices,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local SFT train from RAFT or prompt/answer JSONL."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/sft/adapter"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument(
        "--quantize",
        choices=("none", "4bit", "pre4bit"),
        default="none",
    )
    parser.add_argument("--parallel", choices=PARALLEL_MODES, default="model")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument(
        "--optimizer",
        choices=("auto", "paged_adamw_8bit", "adamw_torch"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--telemetry", type=Path, default=Path(""))
    return parser


def main() -> int:
    args = _parser().parse_args()
    from cli.config import load_config

    config = load_config(args.config)
    lora = _lora_settings(
        int(args.lora_r),
        int(args.lora_alpha),
        float(args.lora_dropout),
        str(args.lora_targets),
    )
    telemetry = (
        Telemetry(args.telemetry, "train", args.telemetry.parent.parent.name)
        if path_set(args.telemetry)
        else disabled("train")
    )
    summary = train(
        args.data,
        str(args.model),
        args.output,
        int(args.max_steps),
        int(args.max_length),
        config.answer_prompt,
        float(args.learning_rate),
        int(args.seed),
        str(args.dtype),
        int(args.batch_size),
        int(args.grad_accum),
        int(args.save_steps),
        int(args.eval_steps),
        float(args.val_fraction),
        str(args.quantize),
        str(args.parallel),
        int(args.devices),
        lora,
        config.answer_refusal,
        float(args.min_weight),
        bool(args.allow_truncation),
        str(args.attn),
        bool(args.gradient_checkpointing),
        str(args.optimizer),
        telemetry,
        int(args.warmup_steps),
    )
    if summary.process_rank == 0:
        print(
            f"sft-train: {summary.examples} examples, {summary.steps} steps, "
            f"parallel {summary.parallel} on {summary.devices} devices, "
            f"world {summary.world_size}, loss {summary.loss:.4f}, "
            f"{summary.seconds:.1f}s, tokens {summary.min_tokens}-{summary.max_tokens}, "
            f"truncated {summary.truncated}, val {summary.val_examples}, "
            f"best eval {summary.best_eval_loss:.4f}, "
            f"quantize {summary.quantize}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
