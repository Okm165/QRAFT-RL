from __future__ import annotations

from pathlib import Path

from preprocess.jsonl import write_json, write_jsonl
from tools.run_layout import create_run, write_stage
from tools.run_metrics import _behavior_report
from tools.run_quality import collect_quality, write_summary


def test_run_diagnostics_collects_artifacts(tmp_path: Path) -> None:
    root = create_run(tmp_path, "test")
    preprocess = root / "preprocess"
    write_json(preprocess / "audit.json", {"files": 1})
    write_json(
        preprocess / "structure_summary.json",
        {"chunks": 2, "facts": 1, "windows": 1, "skipped": 0},
    )
    write_jsonl(
        preprocess / "chunks.jsonl",
        (
            {
                "kind": "table",
                "token_count": 10,
                "doc_id": "d1",
                "section": "s",
                "text": "a",
            },
            {
                "kind": "prose",
                "token_count": 20,
                "doc_id": "d1",
                "section": "s",
                "text": "a",
            },
        ),
    )
    write_jsonl(
        preprocess / "facts.jsonl",
        ({"doc_id": "d1", "text": "1 GND"},),
    )
    write_jsonl(
        root / "raft" / "raft.jsonl",
        (
            {
                "question": "q",
                "status": "oracle",
                "contexts": [{"text": "x"}],
                "answer": "The answer is GND [1].",
            },
        ),
    )
    write_json(
        root / "train" / "adapter" / "summary.json",
        {"examples": 1, "steps": 1, "loss": 0.1},
    )
    (root / "logs" / "03_retrieval.log").write_text(
        "doc recall@10: 1/1 = 1.000\n",
        encoding="utf-8",
    )
    (root / "metrics" / "gpu.csv").write_text(
        "timestamp,device,GPU use (%),GPU Memory Allocated (VRAM%),"
        "Average Graphics Package Power (W)\n"
        "1,card0,50,10,20\n"
        "2,card0,100,20,30\n",
        encoding="utf-8",
    )

    quality = collect_quality(root)
    assert quality["preprocess"]["structure"]["chunks"] == 2
    assert quality["chunks"]["chunks"] == 2
    assert quality["chunks"]["duplicate_texts"] == 1
    assert quality["raft"]["examples"] == 1
    assert quality["retrieval"]["doc"]["recall"] == 1.0
    assert quality["gpu"]["cards"]["card0"]["gpu_use_max"] == 100.0

    summary = write_summary(root, quality)
    assert summary.is_file()
    assert "## Chunks" in summary.read_text(encoding="utf-8")

    assert write_stage(root, "test", 1.5, 0, "logs/test.log") == 0
    assert (root / "metrics" / "test.json").is_file()

def test_collect_quality_on_empty_run(tmp_path: Path) -> None:
    root = create_run(tmp_path, "empty")
    quality = collect_quality(root)
    assert quality["chunks"] == {}
    assert quality["raft"] == {}

def test_create_run_updates_latest_symlink(tmp_path: Path) -> None:
    first = create_run(tmp_path, "first")
    assert (tmp_path / "latest").resolve() == first.resolve()
    second = create_run(tmp_path, "second")
    assert (tmp_path / "latest").resolve() == second.resolve()
    latest = tmp_path / "latest"
    latest.unlink()
    latest.mkdir()
    (latest / "stale").write_text("stale", encoding="utf-8")
    third = create_run(tmp_path, "third")
    assert latest.is_symlink()
    assert latest.resolve() == third.resolve()
    assert not (latest / "stale").exists()


def test_behavior_report_parses_rates(tmp_path: Path) -> None:
    path = tmp_path / "behavior.log"
    path.write_text(
        "answer hits:            10/50 = 0.200\n"
        "citation support hits:  1/50 = 0.020\n"
        "refusals:               8/8 = 1.000\n",
        encoding="utf-8",
    )
    report = _behavior_report(path)
    assert report["answer"] == {"hits": 10, "total": 50}
    assert report["citation"] == {"hits": 1, "total": 50}
    assert report["refusal"] == {"hits": 8, "total": 8}
