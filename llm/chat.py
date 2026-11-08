"""Chat-template rendering with thinking disabled when the template supports it."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase


def render_chat(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    template = tokenizer.chat_template or ""
    options: dict[str, object] = (
        {"enable_thinking": False} if "enable_thinking" in template else {}
    )
    return str(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **options,
        )
    )
