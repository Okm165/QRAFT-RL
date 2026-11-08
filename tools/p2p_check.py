"""Run a local multi-GPU all-reduce and bandwidth check."""

from __future__ import annotations

import argparse
import json

import torch

from tools.p2p import run_allreduce_check, run_peer_check


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local multi-GPU P2P check.")
    parser.add_argument("--world-size", type=int, default=0)
    parser.add_argument("--mode", choices=("peer", "collective"), default="peer")
    args = parser.parse_args()
    world_size = args.world_size or torch.cuda.device_count()
    if world_size < 2:
        print(
            json.dumps(
                {"ok": False, "error": "need at least two GPUs", "world_size": world_size},
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    result = (
        run_peer_check(world_size)
        if args.mode == "peer"
        else run_allreduce_check(world_size)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
