"""Run directory layout, manifest, and stage timing."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from preprocess.jsonl import JsonValue, write_json

SUBDIRS = ("logs", "metrics", "preprocess", "golden", "raft", "train")


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path
    logs: Path
    metrics: Path
    preprocess: Path
    golden: Path
    raft: Path
    train: Path


def layout(root: Path) -> RunLayout:
    return RunLayout(
        root=root,
        logs=root / "logs",
        metrics=root / "metrics",
        preprocess=root / "preprocess",
        golden=root / "golden",
        raft=root / "raft",
        train=root / "train",
    )


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def create_run(base: Path, name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = (base / f"{stamp}-{name}").resolve()
    for subdir in SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    manifest: dict[str, JsonValue] = {
        "run": root.name,
        "root": str(root),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
    }
    write_json(root / "manifest.json", manifest)
    latest = base / "latest"
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    elif latest.is_dir():
        shutil.rmtree(latest)
    latest.symlink_to(root.name, target_is_directory=True)
    (root / "README.md").write_text(
        "\n".join(
            (
                f"# Run {root.name}",
                "",
                "Artifacts are grouped by stage:",
                "",
                "- `preprocess/` — docs, chunks, facts, concepts, normalized, audit",
                "- `golden/` — generated questions and config-driven refusal probes",
                "- `raft/` — RAFT training examples",
                "- `train/` — adapter and training summary",
                "- `logs/` — one log per stage",
                "- `metrics/` — stage timing, quality report, GPU samples",
                "- `summary.md` — readable roll-up",
                "",
            )
        ),
        encoding="utf-8",
    )
    return root


def write_stage(
    root: Path,
    name: str,
    seconds: float,
    exit_code: int,
    log: str,
) -> int:
    record: dict[str, JsonValue] = {
        "stage": name,
        "seconds": seconds,
        "exit": exit_code,
        "log": log,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(root / "metrics" / f"{name}.json", record)
    return exit_code
