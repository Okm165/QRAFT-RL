"""Create and inspect local run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.jsonl import write_json
from tools.run_layout import create_run, write_stage
from tools.run_quality import collect_quality, write_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and inspect local run artifacts.")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new")
    new.add_argument("--base", type=Path, default=Path("data/runs"))
    new.add_argument("--name", default="run")

    stage = commands.add_parser("stage")
    stage.add_argument("--run", type=Path, required=True)
    stage.add_argument("--name", required=True)
    stage.add_argument("--seconds", type=float, required=True)
    stage.add_argument("--exit", type=int, required=True)
    stage.add_argument("--log", required=True)

    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "new":
        print(create_run(args.base, str(args.name)))
        return 0
    if args.command == "stage":
        return write_stage(
            args.run,
            str(args.name),
            float(args.seconds),
            int(args.exit),
            str(args.log),
        )
    root = args.run
    quality = collect_quality(root)
    write_json(root / "metrics" / "quality.json", quality)
    print(write_summary(root, quality))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
