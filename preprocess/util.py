"""Small, dependency-free helpers shared by the pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .domain import DocumentRecord


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_id(*parts: str) -> str:
    return sha256_text("\x1f".join(parts))


def iter_files(root: Path, globs: tuple[str, ...]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for pattern in globs:
        paths.update(path for path in root.rglob(pattern) if path.is_file())
    return tuple(sorted(paths, key=lambda path: str(path.relative_to(root))))


def source_name(path: Path) -> str:
    return path.name


def read_normalized(document: DocumentRecord, data_root: Path, out_dir: Path) -> str:
    normalized_path = out_dir / "normalized" / f"{document.sha256}.md"
    if normalized_path.is_file():
        return normalized_path.read_text(encoding="utf-8")
    return (data_root / document.source_path).read_text(encoding="utf-8", errors="replace")


def word_count(text: str) -> int:
    return len(text.split())
