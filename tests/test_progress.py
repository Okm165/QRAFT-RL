from __future__ import annotations

import json
from pathlib import Path

from tools.progress import dashboard


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dashboard_reports_questions_health(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write(
        run / "metrics" / "pipeline.jsonl",
        [
            {
                "event": "start",
                "name": "questions",
                "command": "llm.questions --chunks-limit 10 --limit 100",
                "ts": "2026-11-07T20:00:00+0000",
            }
        ],
    )
    _write(
        run / "metrics" / "questions.jsonl",
        [
            {
                "event": "chunk",
                "generated": 4,
                "accepted": 3,
                "rejected": 1,
                "ts": "2026-11-07T20:01:00+0000",
            },
            {
                "event": "chunk",
                "generated": 4,
                "accepted": 3,
                "rejected": 1,
                "ts": "2026-11-07T20:02:00+0000",
            },
        ],
    )
    _write(
        run / "golden" / "rejected.jsonl",
        [{"reason": "question not retrievable"}],
    )
    text = dashboard(run)
    assert "chunks 2/10" in text
    assert "accepted 6" in text
    assert "rate 0.750" in text
    assert "question not retrievable" in text
    assert "health: green" in text
