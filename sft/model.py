"""LoRA model construction and adapter persistence."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, BitsAndBytesConfig

from model_family import auto_class, drop_vision_tower, is_multimodal, layer_group

PARALLEL_MODES = ("data", "model")


@dataclass(frozen=True, slots=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    targets: tuple[str, ...]


def _lora_settings(r: int, alpha: int, dropout: float, targets: str) -> LoraSettings:
    parsed = tuple(item.strip() for item in targets.split(",") if item.strip())
    if not parsed:
        raise ValueError("lora targets must not be empty")
    return LoraSettings(r=r, alpha=alpha, dropout=dropout, targets=parsed)


def _group(name: str) -> str:
    return layer_group(name)


def _fallback_map(config: object, devices: int) -> dict[str, int]:
    text_config = getattr(config, "text_config", config)
    layers = int(getattr(text_config, "num_hidden_layers", 0))
    if layers < 1:
        return {}
    prefix = "model.language_model" if is_multimodal(config) else "model"
    per_device = max(1, math.ceil(layers / devices))
    result = {
        f"{prefix}.layers.{index}": min(index // per_device, devices - 1)
        for index in range(layers)
    }
    result[f"{prefix}.embed_tokens"] = 0
    result[f"{prefix}.norm"] = devices - 1
    result["lm_head"] = 0
    return result


def _balanced_device_map(
    model_path: str,
    dtype: torch.dtype,
    devices: int,
) -> dict[str, int]:
    """Assign top-level parameter groups to devices by weight size."""
    if devices < 1:
        raise ValueError("devices must be at least 1")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if hasattr(config, "quantization_config"):
        config.quantization_config = None
    try:
        with torch.device("meta"):
            meta = auto_class(config).from_config(config, dtype=dtype)
    except (TypeError, ValueError, RuntimeError):
        return _fallback_map(config, devices)
    sizes: dict[str, int] = {}
    for name, parameter in meta.named_parameters():
        group = _group(name)
        sizes[group] = sizes.get(group, 0) + parameter.numel() * dtype.itemsize
    for name, buffer in meta.named_buffers():
        group = _group(name)
        sizes[group] = sizes.get(group, 0) + buffer.numel() * dtype.itemsize
    del meta
    gc.collect()
    tied = bool(getattr(config, "tie_word_embeddings", False))
    if tied:
        sizes.pop("lm_head", None)
    layers = sorted(
        (
            group
            for group in sizes
            if group.startswith("model.layers.")
            or group.startswith("model.language_model.layers.")
        ),
        key=lambda group: int(group.rsplit(".", 1)[1]),
    )
    result: dict[str, int] = {}
    if layers:
        total = sum(sizes[group] for group in layers)
        target = total / devices
        assigned = 0
        index = 0
        for group in layers:
            result[group] = index
            assigned += sizes[group]
            if index < devices - 1 and assigned >= target * (index + 1):
                index += 1
        for group in sizes:
            if group in result:
                continue
            if group in {"model.norm", "model.language_model.norm", "lm_head"}:
                result[group] = devices - 1
            else:
                result[group] = 0
    else:
        total = sum(sizes.values())
        target = total / devices
        assigned = 0
        index = 0
        for group, size in sizes.items():
            result[group] = index
            assigned += size
            if index < devices - 1 and assigned >= target * (index + 1):
                index += 1
    if tied:
        result["lm_head"] = result.get(
            "model.embed_tokens",
            result.get("model.language_model.embed_tokens", 0),
        )
    return result


def _pre4bit_config(config: object, dtype: torch.dtype) -> BitsAndBytesConfig:
    """Load the checkpoint's packed NF4 layout with a stable compute dtype."""
    raw = getattr(config, "quantization_config", None)
    if not isinstance(raw, dict):
        raise ValueError("pre4bit checkpoint has no quantization_config")
    values = cast(dict[str, object], raw)
    skip = values.get("llm_int8_skip_modules", [])
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(values.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(
            values.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_quant_storage=str(values.get("bnb_4bit_quant_storage", "uint8")),
        llm_int8_skip_modules=(
            [str(item) for item in skip] if isinstance(skip, list) else []
        ),
    )


def _load_model(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    lora: LoraSettings,
    quantize: str,
    parallel: str,
    devices: int,
    attn: str = "sdpa",
    gradient_checkpointing: bool = True,
) -> PeftModel:
    if parallel not in PARALLEL_MODES:
        raise ValueError(f"unsupported parallel mode: {parallel}")
    if devices > torch.cuda.device_count():
        raise ValueError(
            f"requested {devices} devices but only {torch.cuda.device_count()} are visible"
        )
    if attn not in {"eager", "sdpa"}:
        raise ValueError(f"unsupported attention implementation: {attn}")
    if quantize not in {"none", "4bit", "pre4bit"}:
        raise ValueError(f"unsupported quantize mode: {quantize}")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        if quantize == "4bit"
        else _pre4bit_config(config, dtype)
        if quantize == "pre4bit"
        else None
    )
    device_map: dict[str, int] | dict[str, str]
    if parallel == "model":
        device_map = _balanced_device_map(model_path, dtype, devices)
        if not device_map:
            raise ValueError("cannot build a GPU-only model-parallel device map")
    else:
        device_map = {"": str(device)}
    model_class = auto_class(config)
    if quantization_config is None:
        model = model_class.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            device_map=device_map,
            max_memory={index: "23GiB" for index in range(devices)},
            low_cpu_mem_usage=True,
            attn_implementation=attn,
        )
    else:
        model = model_class.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory={index: "23GiB" for index in range(devices)},
            low_cpu_mem_usage=True,
            attn_implementation=attn,
        )
    placement = getattr(model, "hf_device_map", {})
    if not isinstance(placement, dict) or any(
        str(value) in {"cpu", "disk"} for value in placement.values()
    ):
        raise RuntimeError(
            "model weights were offloaded to CPU/disk; GPU-only training requires a smaller model or more VRAM"
        )
    if is_multimodal(config):
        drop_vision_tower(model)
    if quantize == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif quantize == "pre4bit" and gradient_checkpointing:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    model.config.use_cache = False
    config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        target_modules=list(lora.targets),
        lora_dropout=lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model


def _save_adapter(model: PeftModel, output_path: Path) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    return output_path
