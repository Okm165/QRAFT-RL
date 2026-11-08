"""Config-driven local pipeline runner.

All case-specific paths, model endpoints, and sizes live in TOML. The runner
only knows generic stages: preprocess, index, questions, raft, train, eval.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from cli.stage_commands import _commands
from cli.config import PipelineConfig, load_config
from preprocess.jsonl import write_jsonl
from telemetry import stage_telemetry
from tools.run_layout import create_run, write_stage

STAGES = (
    "preprocess",
    "index",
    "questions",
    "raft",
    "hardware",
    "train",
    "export",
    "behavior-base",
    "behavior-adapter",
    "eval",
    "report",
)


def _run_stage(
    root: Path,
    name: str,
    command: list[str],
    env: Mapping[str, str],
) -> int:
    log = root / "logs" / f"{name}.log"
    telemetry = stage_telemetry(root, "pipeline")
    telemetry.emit("start", name=name, command=" ".join(command))
    started = time.perf_counter()
    child_env = {**os.environ, **env, "PYTHONUNBUFFERED": "1"}
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )
        returncode = process.wait()
    seconds = time.perf_counter() - started
    output = log.read_text(encoding="utf-8")
    write_stage(root, name, seconds, returncode, f"logs/{name}.log")
    telemetry.timed(
        "finish",
        seconds,
        name=name,
        exit=returncode,
        log=f"logs/{name}.log",
        output_chars=len(output),
    )
    print(output[-4000:])
    if returncode:
        raise RuntimeError(f"{name} failed with exit {returncode}")
    return returncode



def run_pipeline(
    config: PipelineConfig,
    stage: str,
    run_dir: Path = Path("-"),
    config_path: Path = Path(""),
) -> Path:
    os.environ.update(config.env)
    if run_dir != Path("-"):
        root = run_dir.resolve()
        for subdir in ("logs", "metrics", "preprocess", "golden", "raft", "train"):
            (root / subdir).mkdir(parents=True, exist_ok=True)
        latest = root.parent / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(root.name)
    else:
        root = create_run(config.output, config.name)
    commands = _commands(config, root, config_path)
    write_jsonl(
        root / "golden" / "refusal_questions.jsonl",
        (
            {"id": f"u{index:03d}", "question": question}
            for index, question in enumerate(config.refusal_questions, start=1)
        ),
    )
    selected = STAGES if stage == "all" else (stage,)
    for name in selected:
        command = cast(list[str], commands[name])
        _run_stage(root, name, command, {})
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the config-driven local pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--run", type=Path, default=Path("-"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    root = run_pipeline(
        config,
        str(args.stage),
        args.run,
        args.config,
    )
    print(f"run: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
