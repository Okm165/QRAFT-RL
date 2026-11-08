"""Composition root for the local preprocessing CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from cli.config import load_config
from llm.client import chat, settings_list
from preprocess.pipeline import Stage, run_pipeline
from preprocess.structure import StructureSettings
from preprocess.util import word_count
from telemetry import stage_telemetry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local preprocessing pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=tuple(stage.value for stage in Stage),
        default=Stage.ALL.value,
    )
    parser.add_argument("--data", type=Path, default=Path(""))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--globs", default="")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = args.data if str(args.data) else config.input_path
    globs = (
        tuple(item.strip() for item in args.globs.split(",") if item.strip())
        if args.globs
        else (config.glob,)
    )
    limit = config.limit if args.limit < 0 else args.limit
    counter = word_count
    llm_settings = settings_list(
        config.llm_base_urls,
        config.llm_model,
        config.llm_api_key,
        config.llm_timeout,
        config.preprocess_llm_max_tokens,
        config.llm_temperature,
    )
    structure_settings = StructureSettings(
        system=config.preprocess_system,
        prompt=config.preprocess_prompt,
        window_chars=config.preprocess_window_chars,
        overlap_chars=config.preprocess_overlap_chars,
        limit=config.preprocess_llm_limit,
        workers=config.llm_workers,
    )
    print(
        run_pipeline(
            Stage(args.stage),
            data_root,
            args.out,
            globs,
            limit,
            counter,
            llm_settings,
            structure_settings,
            chat,
            stage_telemetry(args.out.parent, "preprocess"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
