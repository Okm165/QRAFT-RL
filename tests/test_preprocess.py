from __future__ import annotations

import json
from pathlib import Path

from llm.client import LlmSettings
from preprocess import inventory, normalize
from preprocess.pipeline import Stage, run_pipeline
from preprocess.structure import StructureSettings, windows
from preprocess.util import word_count


def _llm() -> tuple[LlmSettings, ...]:
    return (LlmSettings("http://local", "model", "key", 30, 128, 0.0),)


def _settings() -> StructureSettings:
    return StructureSettings(
        system="Return JSON.",
        prompt="Text:\n%TEXT%",
        window_chars=1000,
        overlap_chars=0,
        limit=0,
        workers=1,
    )


def _chat(settings: LlmSettings, system: str, user: str) -> str:
    return json.dumps(
        {
            "chunks": [{"text": "chunk one", "section": "Pins", "kind": "prose"}],
            "facts": ["fact one"],
        }
    )


def test_inventory_counts_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.md").write_text("alpha", encoding="utf-8")
    (root / "b.md").write_text("alpha", encoding="utf-8")
    (root / "c.md").write_text("gamma", encoding="utf-8")
    records, audit = inventory.scan(root, ("*.md",), 0)
    assert audit.files == 3
    assert audit.unique_hashes == 2
    assert audit.duplicate_files == 1
    duplicate = next(record for record in records if record.duplicate_of)
    assert duplicate.duplicate_of == "a.md"
    assert all(record.lines == 1 for record in records)


def test_normalize_is_minimal_and_entity_aware() -> None:
    result = normalize.normalize_text("A\r\nB &amp;\u200b\x01")
    assert result.text == "A\nB &"
    assert result.html_entities == 1
    assert result.control_chars == 1


def test_windows_cover_text_with_overlap() -> None:
    assert windows("abcdef", 4, 2) == ("abcd", "cdef")
    assert windows("", 4, 2) == ()
    try:
        windows("abc", 4, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap must be smaller than window")


def test_pipeline_all_uses_llm_structure(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "processed"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nPin 1 is GND.", encoding="utf-8")
    message = run_pipeline(
        Stage.ALL,
        root,
        out,
        ("*.md",),
        0,
        word_count,
        _llm(),
        _settings(),
        _chat,
    )
    assert "inventory:" in message
    assert "normalize:" in message
    assert "structure: 1 windows, 1 chunks, 1 facts" in message
    assert (out / "docs.jsonl").is_file()
    assert (out / "chunks.jsonl").is_file()
    assert (out / "facts.jsonl").is_file()
    assert (out / "structure_summary.json").is_file()
    chunk = json.loads((out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    fact = json.loads((out / "facts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert chunk["text"] == "chunk one"
    assert chunk["kind"] == "prose"
    assert chunk["token_count"] == 2
    assert fact["text"] == "fact one"
    assert fact["kind"] == "claim"
