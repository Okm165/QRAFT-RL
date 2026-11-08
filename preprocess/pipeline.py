"""Generic, local-only preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from llm.client import LlmSettings
from telemetry import Telemetry, disabled

from . import inventory, normalize, structure
from .structure import ChatFn, StructureSettings

TokenCounter: TypeAlias = Callable[[str], int]


class Stage(StrEnum):
    INVENTORY = "inventory"
    NORMALIZE = "normalize"
    STRUCTURE = "structure"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class PipelineContext:
    data_root: Path
    out_dir: Path
    globs: tuple[str, ...]
    limit: int
    counter: TokenCounter
    llm_settings: tuple[LlmSettings, ...]
    structure_settings: StructureSettings
    chat_fn: ChatFn
    telemetry: Telemetry


StageHandler: TypeAlias = Callable[[PipelineContext], str]


def _inventory(context: PipelineContext) -> str:
    audit = inventory.run(context.data_root, context.out_dir, context.globs, context.limit)
    return (
        f"inventory: {audit.files} files, {audit.unique_hashes} unique, "
        f"{audit.duplicate_files} duplicates"
    )


def _normalize(context: PipelineContext) -> str:
    summary = normalize.run(context.data_root, context.out_dir, context.globs, context.limit)
    return (
        f"normalize: {summary.documents} docs, {summary.changed_lines} changed lines, "
        f"{summary.html_entities} html entities, {summary.control_chars} control chars"
    )


def _structure(context: PipelineContext) -> str:
    documents, _ = inventory.scan(context.data_root, context.globs, context.limit)
    summary = structure.run(
        documents,
        context.data_root,
        context.out_dir,
        context.structure_settings,
        context.llm_settings,
        context.counter,
        context.chat_fn,
        context.telemetry,
    )
    return (
        f"structure: {summary.windows} windows, {summary.chunks} chunks, "
        f"{summary.facts} facts, {summary.concepts} concepts, "
        f"{summary.documents_structured}/{summary.documents} docs, "
        f"{summary.skipped} skipped, {summary.empty_windows} empty windows"
    )


HANDLERS: dict[Stage, StageHandler] = {
    Stage.INVENTORY: _inventory,
    Stage.NORMALIZE: _normalize,
    Stage.STRUCTURE: _structure,
}

STAGES: tuple[Stage, ...] = (
    Stage.INVENTORY,
    Stage.NORMALIZE,
    Stage.STRUCTURE,
)


def run_stage(stage: Stage, context: PipelineContext) -> str:
    if stage is Stage.ALL:
        return "\n".join(HANDLERS[step](context) for step in STAGES)
    return HANDLERS[stage](context)


def run_pipeline(
    stage: Stage,
    data_root: Path,
    out_dir: Path,
    globs: tuple[str, ...],
    limit: int,
    counter: TokenCounter,
    llm_settings: tuple[LlmSettings, ...],
    structure_settings: StructureSettings,
    chat_fn: ChatFn,
    telemetry: Telemetry = disabled("preprocess"),
) -> str:
    return run_stage(
        stage,
        PipelineContext(
            data_root=data_root,
            out_dir=out_dir,
            globs=globs,
            limit=limit,
            counter=counter,
            llm_settings=llm_settings,
            structure_settings=structure_settings,
            chat_fn=chat_fn,
            telemetry=telemetry,
        ),
    )
