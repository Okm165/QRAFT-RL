"""Append-only JSONL telemetry for stage health and data quality.

One record per meaningful unit of work (window, candidate, RAFT row, training
step, evaluated question). Records are bounded scalars only; raw text and model
outputs stay in the stage artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from preprocess.jsonl import JsonValue


@dataclass(frozen=True, slots=True)
class Telemetry:
    path: Path
    stage: str
    run: str
    active: bool = True

    def emit(self, event: str, **fields: JsonValue) -> None:
        if not self.active:
            return
        record: dict[str, JsonValue] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stage": self.stage,
            "run": self.run,
            "event": event,
        }
        record.update(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def timed(self, event: str, seconds: float, **fields: JsonValue) -> None:
        self.emit(event, seconds=round(seconds, 6), **fields)


def stage_telemetry(root: Path, stage: str) -> Telemetry:
    return Telemetry(root / "metrics" / f"{stage}.jsonl", stage, root.name)


def disabled(stage: str = "") -> Telemetry:
    return Telemetry(Path(""), stage, "", active=False)


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def gpu_snapshot() -> list[JsonValue]:
    """Read AMD GPU busy/temp/power/VRAM from sysfs.

    ``rocm-smi`` is not present in the runner image, but the kernel exposes
    the same counters under ``/sys/class/drm``. Values are per card and safe
    to emit as bounded JSON.
    """
    cards: list[JsonValue] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        busy = device / "gpu_busy_percent"
        if not busy.is_file():
            continue
        hwmons = sorted((device / "hwmon").glob("hwmon*"))
        hwmon = hwmons[0] if hwmons else None
        total = _read_int(device / "mem_info_vram_total")
        used = _read_int(device / "mem_info_vram_used")
        cards.append(
            {
                "card": card.name,
                "use": float(_read_int(busy)),
                "temp_edge": _read_int(hwmon / "temp1_input") / 1000.0
                if hwmon
                else 0.0,
                "temp_junction": _read_int(hwmon / "temp2_input") / 1000.0
                if hwmon
                else 0.0,
                "power": _read_int(hwmon / "power1_average") / 1_000_000.0
                if hwmon
                else 0.0,
                "vram_used_mib": used // (1024 * 1024),
                "vram_total_mib": total // (1024 * 1024),
            }
        )
    return cards
