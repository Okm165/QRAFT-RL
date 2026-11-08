"""JSON boundary for typed records.

Only JSON scalars, lists, and mappings cross this boundary. ``Any`` and
``None`` are not part of the public surface.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TypeAlias, TypeVar, cast

JsonScalar: TypeAlias = str | int | float | bool
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

RecordT = TypeVar("RecordT")


def path_set(path: Path) -> bool:
    """True when an optional path argument names a real destination."""
    return str(path) not in {"", "."}


def write_json(path: Path, value: JsonValue) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, JsonValue]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Mapping[str, JsonValue]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield cast(JsonValue, json.loads(line))


def json_texts(value: JsonValue) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in cast(list[JsonValue], value))


def json_rows(value: JsonValue) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        return ()
    rows: list[tuple[str, ...]] = []
    for row in value:
        if isinstance(row, list):
            rows.append(tuple(str(cell) for cell in cast(list[JsonValue], row)))
    return tuple(rows)


def record_to_json(record: RecordT) -> dict[str, JsonValue]:
    if not is_dataclass(record):
        raise TypeError(f"expected dataclass, got {type(record).__name__}")
    return cast(dict[str, JsonValue], asdict(record))
