"""Build the local FTS5 index from chunks.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fts import build


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local FTS5 index.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--facts", type=Path, default=Path("data/processed/facts.jsonl"))
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path("data/processed/concepts.jsonl"),
    )
    parser.add_argument("--db", type=Path, default=Path("data/processed/rag.sqlite"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    count = build(args.chunks, args.db, args.limit, args.facts, args.concepts)
    print(f"indexed {count} records into {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
