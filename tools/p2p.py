"""Multi-rank all-reduce and P2P bandwidth checks."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from preprocess.jsonl import JsonValue

if TYPE_CHECKING:
    import torch

BANDWIDTH_PAYLOAD_MIB = 64
BANDWIDTH_ITERATIONS = 5


def _env(name: str) -> str:
    return os.environ.get(name, "")


def _bandwidth(report: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return cast(Mapping[str, JsonValue], report["bandwidth"])


def run_allreduce_check(world_size: int) -> dict[str, JsonValue]:
    import torch
    import torch.multiprocessing as spawn_mp

    if torch.cuda.device_count() < world_size:
        return {
            "ok": False,
            "skipped": True,
            "error": f"need {world_size} GPUs, found {torch.cuda.device_count()}",
            "world_size": world_size,
            "backend": "nccl",
        }
    port = _find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            spawn_mp.spawn(
                _allreduce_child,
                args=(world_size, tmpdir),
                nprocs=world_size,
                join=True,
            )
            reports = tuple(
                cast(
                    Mapping[str, JsonValue],
                    json.loads(Path(tmpdir, f"rank-{rank}.json").read_text(encoding="utf-8")),
                )
                for rank in range(world_size)
            )
            values = tuple(float(str(report["value"])) for report in reports)
            bandwidths = tuple(_bandwidth(report) for report in reports)
            bandwidth_ok = all(bool(item["ok"]) for item in bandwidths)
            return {
                "ok": all(value == float(world_size) for value in values) and bandwidth_ok,
                "values": list(values),
                "bandwidth_gb_s": sum(
                    float(str(item["bus_gb_s"])) for item in bandwidths
                )
                / len(bandwidths),
                "bandwidth": [dict(item) for item in bandwidths],
                "world_size": world_size,
                "backend": "nccl",
                "port": port,
                "hsa_enable_ipc_mode_legacy": _env("HSA_ENABLE_IPC_MODE_LEGACY"),
                "nccl_p2p_disable": _env("NCCL_P2P_DISABLE"),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "world_size": world_size,
                "backend": "nccl",
                "hsa_enable_ipc_mode_legacy": _env("HSA_ENABLE_IPC_MODE_LEGACY"),
                "nccl_p2p_disable": _env("NCCL_P2P_DISABLE"),
            }


def run_peer_check(world_size: int) -> dict[str, JsonValue]:
    import torch

    if torch.cuda.device_count() < world_size:
        return {
            "ok": False,
            "skipped": True,
            "error": f"need {world_size} GPUs, found {torch.cuda.device_count()}",
            "world_size": world_size,
        }
    payload = 8 * 1024 * 1024
    pairs: list[dict[str, JsonValue]] = []
    for source in range(world_size):
        for target in range(world_size):
            if source == target:
                continue
            accessible = bool(torch.cuda.can_device_access_peer(source, target))
            if not accessible:
                pairs.append(
                    {
                        "source": source,
                        "target": target,
                        "ok": False,
                        "peer_access": False,
                    }
                )
                continue
            tensor = torch.ones(payload // 4, dtype=torch.float32, device=f"cuda:{source}")
            destination = torch.empty_like(tensor, device=f"cuda:{target}")
            torch.cuda.synchronize(source)
            started = time.perf_counter()
            destination.copy_(tensor, non_blocking=False)
            torch.cuda.synchronize(target)
            seconds = time.perf_counter() - started
            pairs.append(
                {
                    "source": source,
                    "target": target,
                    "ok": True,
                    "peer_access": True,
                    "gb_s": payload / seconds / 1e9,
                }
            )
            del tensor, destination
            torch.cuda.empty_cache()
    return {
        "ok": all(bool(pair["ok"]) for pair in pairs),
        "world_size": world_size,
        "payload_mib": payload // (1024 * 1024),
        "pairs": pairs,
    }


def measure_allreduce_bandwidth(
    device: torch.device,
    world_size: int,
) -> dict[str, JsonValue]:
    import torch
    import torch.distributed as dist

    if world_size < 2:
        return {"ok": False, "skipped": True, "world_size": world_size}
    elements = BANDWIDTH_PAYLOAD_MIB * 1024 * 1024 // 4
    tensor = torch.ones(elements, dtype=torch.float32, device=device)
    for _ in range(2):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(BANDWIDTH_ITERATIONS):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    check = torch.ones(1, dtype=torch.float32, device=device)
    dist.all_reduce(check, op=dist.ReduceOp.SUM)
    ok = float(check.item()) == float(world_size)
    bus_bytes = (
        BANDWIDTH_PAYLOAD_MIB
        * 1024
        * 1024
        * 2
        * (world_size - 1)
        / world_size
        * BANDWIDTH_ITERATIONS
    )
    result: dict[str, JsonValue] = {
        "ok": ok,
        "skipped": False,
        "world_size": world_size,
        "payload_mib": BANDWIDTH_PAYLOAD_MIB,
        "iterations": BANDWIDTH_ITERATIONS,
        "seconds": seconds,
        "bus_gb_s": bus_bytes / seconds / 1e9,
        "hsa_enable_ipc_mode_legacy": _env("HSA_ENABLE_IPC_MODE_LEGACY"),
        "nccl_p2p_disable": _env("NCCL_P2P_DISABLE"),
    }
    del tensor, check
    torch.cuda.empty_cache()
    return result


def _allreduce_child(rank: int, world_size: int, tmpdir: str) -> int:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    tensor = torch.ones(1, device=f"cuda:{rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    bandwidth = measure_allreduce_bandwidth(torch.device(f"cuda:{rank}"), world_size)
    Path(tmpdir, f"rank-{rank}.json").write_text(
        json.dumps({"value": float(tensor.item()), "bandwidth": bandwidth}),
        encoding="utf-8",
    )
    dist.destroy_process_group()
    return 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
