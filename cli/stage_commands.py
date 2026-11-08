"""Small host-side command builders, one per stage."""

from __future__ import annotations

import sys
from pathlib import Path

from cli.config import PipelineConfig


def _preprocess(config: PipelineConfig, config_path: Path, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cli.preprocess",
        "--config",
        str(config_path),
        "--stage",
        "all",
        "--data",
        str(config.input_path),
        "--out",
        str(root / "preprocess"),
        "--globs",
        config.glob,
        "--limit",
        str(config.limit),
    ]


def _index(root: Path) -> list[str]:
    preprocess = root / "preprocess"
    return [
        sys.executable,
        "-m",
        "rag.build",
        "--chunks",
        str(preprocess / "chunks.jsonl"),
        "--facts",
        str(preprocess / "facts.jsonl"),
        "--concepts",
        str(preprocess / "concepts.jsonl"),
        "--db",
        str(preprocess / "rag.sqlite"),
    ]


def _questions(config: PipelineConfig, config_path: Path, root: Path) -> list[str]:
    preprocess = root / "preprocess"
    return [
        sys.executable,
        "-m",
        "llm.questions",
        "--config",
        str(config_path),
        "--chunks",
        str(preprocess / "chunks.jsonl"),
        "--out",
        str(root / "golden" / "golden.generated.jsonl"),
        "--rejected",
        str(root / "golden" / "rejected.jsonl"),
        "--base-urls",
        ",".join(config.questions_base_urls),
        "--workers",
        str(config.questions_workers),
        "--model",
        config.questions_model,
        "--api-key",
        config.questions_api_key,
        "--timeout",
        str(config.questions_timeout),
        "--max-tokens",
        str(config.questions_max_tokens),
        "--temperature",
        str(config.questions_temperature),
        "--limit",
        str(config.questions_limit),
        "--chunks-limit",
        str(config.questions_chunks),
        "--per-chunk",
        str(config.questions_per_chunk),
        "--holdout-every",
        str(config.questions_holdout_every),
        "--db",
        str(preprocess / "rag.sqlite"),
        "--top-k",
        str(config.rag_top_k),
        "--telemetry",
        str(root / "metrics" / "questions.jsonl"),
    ]


def _raft(config: PipelineConfig, config_path: Path, root: Path) -> list[str]:
    preprocess = root / "preprocess"
    return [
        sys.executable,
        "-m",
        "raft.build",
        "--config",
        str(config_path),
        "--db",
        str(preprocess / "rag.sqlite"),
        "--golden",
        str(root / "golden" / "train.jsonl"),
        "--out",
        str(root / "raft" / "raft.jsonl"),
        "--rejected",
        str(root / "raft" / "rejected.jsonl"),
        "--distractors",
        str(config.raft_distractors),
        "--absent-every",
        str(config.raft_absent_every),
        "--seed",
        str(config.raft_seed),
        "--telemetry",
        str(root / "metrics" / "raft.jsonl"),
    ]


def _hardware(root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.hardware_check",
        "--run",
        str(root),
        "--json",
    ]


def _train(config: PipelineConfig, config_path: Path, root: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sft.train",
        "--config",
        str(config_path),
        "--data",
        str(root / "raft" / "raft.jsonl"),
        "--model",
        config.train_model,
        "--output",
        str(root / "train" / "adapter"),
        "--max-steps",
        str(config.max_steps),
        "--max-length",
        str(config.max_length),
        "--learning-rate",
        str(config.learning_rate),
        "--batch-size",
        str(config.batch_size),
        "--grad-accum",
        str(config.grad_accum),
        "--warmup-steps",
        str(config.warmup_steps),
        "--save-steps",
        str(config.save_steps),
        "--eval-steps",
        str(config.eval_steps),
        "--val-fraction",
        str(config.val_fraction),
        "--lora-r",
        str(config.lora_r),
        "--lora-alpha",
        str(config.lora_alpha),
        "--lora-dropout",
        str(config.lora_dropout),
        "--lora-targets",
        config.lora_targets,
        "--dtype",
        config.train_dtype,
        "--quantize",
        config.train_quantize,
        "--parallel",
        config.train_parallel,
        "--devices",
        str(config.world_size),
        "--min-weight",
        str(config.train_min_weight),
        "--attn",
        config.train_attn,
        "--optimizer",
        config.train_optimizer,
        "--telemetry",
        str(root / "metrics" / "train.jsonl"),
    ]
    command.append(
        "--gradient-checkpointing"
        if config.train_gradient_checkpointing
        else "--no-gradient-checkpointing"
    )
    if config.train_allow_truncation:
        command.append("--allow-truncation")
    if config.train_parallel == "data" and config.world_size > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(config.world_size),
            "-m",
            "sft.train",
            *command[command.index("-m") + 2 :],
        ]
    return command


def _export(config: PipelineConfig, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.export_lora",
        "--adapter",
        str(root / "train" / "adapter"),
        "--base",
        config.train_model,
        "--out",
        "/models/adapters/qwen3.8-27b-lora.gguf",
    ]


def _behavior(
    config: PipelineConfig,
    config_path: Path,
    root: Path,
    lora_scale: float,
) -> list[str]:
    preprocess = root / "preprocess"
    return [
        sys.executable,
        "-m",
        "eval.behavior",
        "--config",
        str(config_path),
        "--db",
        str(preprocess / "rag.sqlite"),
        "--golden",
        str(root / "golden" / "holdout.jsonl"),
        "--refusal-questions",
        str(root / "golden" / "refusal_questions.jsonl"),
        "--top-k",
        str(config.eval_top_k),
        "--max-new-tokens",
        str(config.eval_max_new_tokens),
        "--lora-id",
        "0",
        "--lora-scale",
        str(lora_scale),
        "--workers",
        str(config.llm_workers),
        "--limit",
        str(config.eval_limit),
        "--fail-below-answer",
        str(config.eval_fail_below_answer),
        "--fail-below-refusal",
        str(config.eval_fail_below_refusal),
        "--telemetry",
        str(root / "metrics" / ("behavior-adapter.jsonl" if lora_scale else "behavior-base.jsonl")),
    ]


def _eval(config: PipelineConfig, config_path: Path, root: Path) -> list[str]:
    preprocess = root / "preprocess"
    return [
        sys.executable,
        "-m",
        "eval.run",
        "--config",
        str(config_path),
        "--db",
        str(preprocess / "rag.sqlite"),
        "--golden",
        str(root / "golden" / "holdout.jsonl"),
        "--top-k",
        str(config.eval_top_k),
        "--fail-below",
        str(config.eval_fail_below),
        "--workers",
        str(config.llm_workers),
        "--telemetry",
        str(root / "metrics" / "eval.jsonl"),
    ]


def _report(root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.run_diagnostics",
        "report",
        "--run",
        str(root),
    ]


def _commands(
    config: PipelineConfig,
    root: Path,
    config_path: Path,
) -> dict[str, list[str]]:
    return {
        "preprocess": _preprocess(config, config_path, root),
        "index": _index(root),
        "questions": _questions(config, config_path, root),
        "raft": _raft(config, config_path, root),
        "hardware": _hardware(root),
        "train": _train(config, config_path, root),
        "export": _export(config, root),
        "behavior-base": _behavior(config, config_path, root, 0.0),
        "behavior-adapter": _behavior(config, config_path, root, 1.0),
        "eval": _eval(config, config_path, root),
        "report": _report(root),
    }
