"""Convert a trained HF LoRA adapter to a GGUF adapter for llama.cpp.

The converter needs a base-model config but not the base weights. The NF4
training checkpoint carries a bitsandbytes quantization block that the
converter cannot dequantize, so this wrapper writes a config-only base copy
with that block removed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONVERTER = Path("/opt/llama.cpp/convert_lora_to_gguf.py")
CONVERSION_ROOT = Path("/opt/llama.cpp")
GGUF_ROOT = Path("/opt/llama.cpp/gguf-py")
DEFAULT_OUTPUT = Path("/models/adapters/qwen3.8-27b-lora.gguf")
CONFIG_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def _config_only_base(base: Path, target: Path):
    config_path = base / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("quantization_config", None)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    for name in CONFIG_FILES:
        source = base / name
        if source.is_file():
            shutil.copy2(source, target / name)


def export(
    adapter: Path,
    base: Path,
    output: Path,
    outtype: str = "f16",
) -> int:
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter not found: {adapter}")
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"base config not found: {base}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qraft-lora-") as temporary:
        config_base = Path(temporary) / "base"
        _config_only_base(base, config_base)
        environment = {
            **os.environ,
            "PYTHONPATH": ":".join(
                item
                for item in (
                    str(CONVERSION_ROOT),
                    str(GGUF_ROOT),
                    os.environ.get("PYTHONPATH", ""),
                )
                if item
            ),
        }
        command = [
            sys.executable,
            str(CONVERTER),
            "--base",
            str(config_base),
            "--outfile",
            str(output),
            "--outtype",
            outtype,
            str(adapter),
        ]
        print("export:", " ".join(command), flush=True)
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode:
            return result.returncode
    print(f"lora: {output} ({output.stat().st_size} bytes)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an HF LoRA adapter to GGUF.")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--outtype",
        choices=("f32", "f16", "bf16", "q8_0", "auto"),
        default="f16",
    )
    args = parser.parse_args()
    return export(args.adapter, args.base, args.out, str(args.outtype))


if __name__ == "__main__":
    raise SystemExit(main())
