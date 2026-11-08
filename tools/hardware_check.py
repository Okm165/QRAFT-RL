"""CLI for read-only GPU hardware inventory and health."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from preprocess.jsonl import JsonValue, write_json
from tools.hardware import GpuHealth, _host_memory, _number, collect


def _print(cards: tuple[GpuHealth, ...]):
    print("HIP  BDF          board                                      link   AER     temp(e/j/m)     power  use  VRAM(MiB)  status")
    for card in cards:
        board = card.board[:42]
        print(
            f"{card.hip:<4} {card.bdf:<12} {board:<42} "
            f"x{card.link_width}/{card.max_link_width:<2} {card.aer_correctable:<7} "
            f"{card.temp_edge:.0f}/{card.temp_junction:.0f}/{card.temp_memory:.0f}C"
            f"{card.power:>10.0f}W {card.use:>3.0f}%  "
            f"{card.vram_used_mib}/{card.vram_total_mib}  {card.status}"
        )
        for issue in card.issues:
            print(f"     - {issue}")


def _print_host(host: dict[str, JsonValue]):
    total = _number(host.get("total_kb", 0)) / 1024.0 / 1024.0
    available = _number(host.get("available_kb", 0)) / 1024.0 / 1024.0
    swap_total = _number(host.get("swap_total_kb", 0)) / 1024.0 / 1024.0
    swap_free = _number(host.get("swap_free_kb", 0)) / 1024.0 / 1024.0
    print(
        f"host: {total:.0f}GiB total, {available:.0f}GiB available, "
        f"swap {swap_total - swap_free:.1f}/{swap_total:.0f}GiB used"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report local GPU hardware health.")
    parser.add_argument("--run", type=Path, default=Path(""))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    cards = collect()
    if not cards:
        print("no HIP devices found")
        return 1
    host = _host_memory()
    payload: dict[str, JsonValue] = {
        "host": host,
        "cards": [asdict(card) for card in cards],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_host(host)
        _print(cards)
    if args.run != Path(""):
        metrics = args.run / "metrics"
        write_json(metrics / "hardware_report.json", payload)
        rows = [
            "GPU use (%),card,GPU Memory Allocated (VRAM%),"
            "Average Graphics Package Power (W)"
        ]
        for card in cards:
            vram_percent = (
                100.0 * card.vram_used_mib / card.vram_total_mib
                if card.vram_total_mib
                else 0.0
            )
            rows.append(
                f"{card.use:.1f},{card.rocm_card or f'card{card.hip}'},"
                f"{vram_percent:.1f},{card.power:.1f}"
            )
        (metrics / "gpu.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    low_memory = _number(host.get("available_kb", 0)) < 8 * 1024 * 1024
    if args.fail and (low_memory or any(card.status == "fail" for card in cards)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
