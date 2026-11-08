"""Read-only GPU hardware inventory and hop-level health collection.

Maps HIP device order to PCI BDFs and rocm-smi cards by unique ID, then
collects the narrowest link hop, the AER hop, thermals, power, and stuck-GPU
state. The hop distinction separates a GPU fault from an upstream riser fault.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from preprocess.jsonl import JsonValue

DRM_ROOT = Path("/sys/class/drm")


@dataclass(frozen=True, slots=True)
class GpuHealth:
    hip: int
    rocm_card: str
    unique_id: str
    bdf: str
    board: str
    link_width: int
    max_link_width: int
    link_speed: str
    link_hop: str
    endpoint_link_width: int
    aer_correctable: int
    aer_fatal: int
    aer_hop: str
    temp_edge: float
    temp_junction: float
    temp_memory: float
    power: float
    power_cap: float
    use: float
    vram_total_mib: int
    vram_free_mib: int
    vram_used_mib: int
    fan: str
    status: str
    issues: tuple[str, ...]


def _json(command: tuple[str, ...]) -> dict[str, JsonValue]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except (FileNotFoundError, PermissionError):
        return {}
    start = result.stdout.find("{")
    if start < 0:
        return {}
    value = json.loads(result.stdout[start:])
    if isinstance(value, dict):
        return cast(dict[str, JsonValue], value)
    return {}


def _cards(metric: str) -> dict[str, dict[str, JsonValue]]:
    raw = _json(("rocm-smi", f"--{metric}", "--json"))
    result: dict[str, dict[str, JsonValue]] = {}
    for key, value in raw.items():
        if key.startswith("card") and isinstance(value, dict):
            result[key] = cast(dict[str, JsonValue], value)
    return result


def _number(value: JsonValue) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _text(value: JsonValue) -> str:
    return str(value) if isinstance(value, (str, int, float)) else ""


def _int_file(path: Path) -> int:
    if not path.is_file():
        return 0
    return int(path.read_text(encoding="utf-8").strip())


def _text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _unique(value: str) -> str:
    return value.strip().lower().removeprefix("0x")


def _hip_unique_id(uuid: str) -> str:
    compact = uuid.replace("-", "")
    try:
        return bytes.fromhex(compact).decode("ascii").strip().lower()
    except (ValueError, UnicodeDecodeError):
        return uuid.strip().lower()


def _aer(path: Path) -> tuple[int, int, str]:
    correctable = 0
    fatal = 0
    hop = ""
    for parent in (path, *path.parents):
        cor_file = parent / "aer_dev_correctable"
        if cor_file.is_file():
            for line in cor_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("TOTAL_ERR_COR"):
                    value = int(line.split()[-1])
                    if value > correctable:
                        correctable = value
                        hop = parent.name
        fatal_file = parent / "aer_dev_fatal"
        if fatal_file.is_file():
            for line in fatal_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("TOTAL_ERR_FATAL"):
                    fatal = max(fatal, int(line.split()[-1]))
    return correctable, fatal, hop


def _milli(path: Path) -> float:
    return _int_file(path) / 1000.0


def _micro(path: Path) -> float:
    return _int_file(path) / 1_000_000.0


def _sysfs_cards() -> dict[str, dict[str, JsonValue]]:
    result: dict[str, dict[str, JsonValue]] = {}
    for card in sorted(DRM_ROOT.glob("card[0-9]*")):
        device = (card / "device").resolve()
        unique_file = device / "unique_id"
        if not unique_file.is_file():
            continue
        unique = _unique(_text_file(unique_file))
        widths: list[tuple[int, str]] = []
        max_widths: list[int] = []
        speeds: list[str] = []
        for parent in (device, *device.parents):
            if (parent / "current_link_width").is_file():
                widths.append((_int_file(parent / "current_link_width"), parent.name))
                max_widths.append(_int_file(parent / "max_link_width"))
                speeds.append(_text_file(parent / "current_link_speed"))
        link_width = 0
        link_hop = ""
        for width, hop in widths:
            if link_width == 0 or width <= link_width:
                link_width = width
                link_hop = hop
        correctable, fatal, aer_hop = _aer(device)
        hwmons = sorted((device / "hwmon").glob("hwmon*"))
        hwmon = hwmons[0] if hwmons else None
        result[unique] = {
            "bdf": device.name,
            "link_width": link_width,
            "max_link_width": min(max_widths) if max_widths else 0,
            "link_speed": speeds[0] if speeds else "",
            "link_hop": link_hop,
            "endpoint_link_width": _int_file(device / "current_link_width"),
            "aer_correctable": correctable,
            "aer_fatal": fatal,
            "aer_hop": aer_hop,
            "use": float(_int_file(device / "gpu_busy_percent")),
            "temp_edge": _milli(hwmon / "temp1_input") if hwmon else 0.0,
            "temp_junction": _milli(hwmon / "temp2_input") if hwmon else 0.0,
            "temp_memory": _milli(hwmon / "temp3_input") if hwmon else 0.0,
            "power": _micro(hwmon / "power1_average") if hwmon else 0.0,
            "fan": _int_file(hwmon / "fan1_input") if hwmon else 0,
        }
    return result


def _host_memory() -> dict[str, JsonValue]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.split()[0])
    return {
        "total_kb": values.get("MemTotal", 0),
        "available_kb": values.get("MemAvailable", 0),
        "swap_total_kb": values.get("SwapTotal", 0),
        "swap_free_kb": values.get("SwapFree", 0),
    }


def _subsystem(bdf: str) -> str:
    try:
        result = subprocess.run(
            ("lspci", "-v", "-s", bdf),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, PermissionError):
        return ""
    for line in result.stdout.splitlines():
        if "Subsystem:" in line:
            return line.split("Subsystem:", 1)[1].strip()
    return ""


def _has_kfd_processes() -> bool:
    try:
        result = subprocess.run(
            ("rocm-smi", "--showpids"),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, PermissionError):
        return False
    return "No KFD PIDs" not in result.stdout


def _status(
    link_width: int,
    max_link_width: int,
    link_hop: str,
    endpoint_link_width: int,
    aer_correctable: int,
    aer_hop: str,
    bdf: str,
    use: float,
    has_pids: bool,
    temp_junction: float,
) -> tuple[str, tuple[str, ...]]:
    issues: list[str] = []
    if link_width and max_link_width and link_width < max_link_width:
        where = f" at {link_hop}" if link_hop else ""
        endpoint = f"; endpoint x{endpoint_link_width}" if endpoint_link_width else ""
        issues.append(f"link x{link_width}{where} (max x{max_link_width}{endpoint})")
    if aer_correctable > 1000:
        where = f" at {aer_hop}" if aer_hop else ""
        endpoint = f"; endpoint {bdf} clean" if aer_hop and aer_hop != bdf else ""
        issues.append(f"{aer_correctable} correctable AER{where}{endpoint}")
    if use > 50.0 and not has_pids:
        issues.append("busy without KFD process")
    if temp_junction > 95.0:
        issues.append(f"junction {temp_junction:.0f}C")
    hard = (use > 50.0 and not has_pids) or temp_junction > 95.0
    return ("fail" if hard else "warn" if issues else "ok"), tuple(issues)


def collect() -> tuple[GpuHealth, ...]:
    unique = _cards("showuniqueid")
    temps = _cards("showtemp")
    power = _cards("showpower")
    caps = _cards("showmaxpower")
    use = _cards("showuse")
    fans = _cards("showfan")
    sysfs = _sysfs_cards()
    has_pids = _has_kfd_processes()

    rocm_by_unique: dict[str, str] = {}
    for card, value in unique.items():
        rocm_by_unique[_unique(_text(value.get("Unique ID", "")))] = card

    result: list[GpuHealth] = []
    for hip in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(hip)
        unique_id = _hip_unique_id(str(getattr(properties, "uuid", "")))
        rocm_card = rocm_by_unique.get(unique_id, "")
        system = sysfs.get(unique_id, {})
        thermal = temps.get(rocm_card, {})
        bdf = _text(system.get("bdf", ""))
        link_width = int(system.get("link_width", 0))
        max_link_width = int(system.get("max_link_width", 0))
        link_hop = _text(system.get("link_hop", ""))
        endpoint_link_width = int(system.get("endpoint_link_width", 0))
        aer_correctable = int(system.get("aer_correctable", 0))
        aer_hop = _text(system.get("aer_hop", ""))
        temp_edge = _number(thermal.get("Temperature (Sensor edge) (C)", 0))
        temp_junction = _number(thermal.get("Temperature (Sensor junction) (C)", 0))
        temp_memory = _number(thermal.get("Temperature (Sensor memory) (C)", 0))
        power_w = _number(
            power.get(rocm_card, {}).get("Average Graphics Package Power (W)", 0)
        )
        power_cap_w = _number(
            caps.get(rocm_card, {}).get("Max Graphics Package Power (W)", 0)
        )
        use_percent = _number(use.get(rocm_card, {}).get("GPU use (%)", 0))
        fan_rpm = _text(fans.get(rocm_card, {}).get("Fan RPM", ""))
        if not temp_edge:
            temp_edge = float(system.get("temp_edge", 0.0))
        if not temp_junction:
            temp_junction = float(system.get("temp_junction", 0.0))
        if not temp_memory:
            temp_memory = float(system.get("temp_memory", 0.0))
        if not power_w:
            power_w = float(system.get("power", 0.0))
        if not use_percent:
            use_percent = float(system.get("use", 0.0))
        if not fan_rpm:
            fan_value = int(system.get("fan", 0))
            fan_rpm = str(fan_value) if fan_value else "n/a"
        free_bytes, total_bytes = torch.cuda.mem_get_info(hip)
        vram_total_mib = int(total_bytes // (1024 * 1024))
        vram_free_mib = int(free_bytes // (1024 * 1024))
        status, issues = _status(
            link_width,
            max_link_width,
            link_hop,
            endpoint_link_width,
            aer_correctable,
            aer_hop,
            bdf,
            use_percent,
            has_pids,
            temp_junction,
        )
        result.append(
            GpuHealth(
                hip=hip,
                rocm_card=rocm_card,
                unique_id=unique_id,
                bdf=bdf,
                board=_subsystem(bdf) if bdf else "",
                link_width=link_width,
                max_link_width=max_link_width,
                link_speed=_text(system.get("link_speed", "")),
                link_hop=link_hop,
                endpoint_link_width=endpoint_link_width,
                aer_correctable=aer_correctable,
                aer_fatal=int(system.get("aer_fatal", 0)),
                aer_hop=aer_hop,
                temp_edge=temp_edge,
                temp_junction=temp_junction,
                temp_memory=temp_memory,
                power=power_w,
                power_cap=power_cap_w,
                use=use_percent,
                vram_total_mib=vram_total_mib,
                vram_free_mib=vram_free_mib,
                vram_used_mib=vram_total_mib - vram_free_mib,
                fan=fan_rpm,
                status=status,
                issues=issues,
            )
        )
    return tuple(result)
