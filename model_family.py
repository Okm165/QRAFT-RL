"""Model-family selection for text-only loading of multimodal checkpoints."""

from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

MULTIMODAL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
LAYER_PREFIXES = ("model.layers.", "model.language_model.layers.")


def model_type(config: object) -> str:
    return str(getattr(config, "model_type", ""))


def is_multimodal(config: object) -> bool:
    return model_type(config) in MULTIMODAL_TYPES


def auto_class(config: object) -> type:
    return AutoModelForImageTextToText if is_multimodal(config) else AutoModelForCausalLM


def drop_vision_tower(model: object) -> None:
    """Release the unused vision tower of a text-only multimodal checkpoint."""
    inner = getattr(model, "model", None)
    if inner is None or getattr(inner, "visual", None) is None:
        return
    setattr(inner, "visual", None)
    gc.collect()
    torch.cuda.empty_cache()


def layer_group(name: str) -> str:
    parts = name.split(".")
    for prefix in LAYER_PREFIXES:
        depth = prefix.count(".")
        if name.startswith(prefix):
            return ".".join(parts[: depth + 1])
    # Do not create a parent key such as ``model.language_model``: it would
    # overlap the per-layer keys and confuse accelerate's device hooks.
    if parts[0] == "model" and len(parts) >= 3 and parts[1] == "language_model":
        return ".".join(parts[:3])
    if parts[0] == "model" and len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]
