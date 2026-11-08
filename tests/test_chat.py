from __future__ import annotations

from typing import cast

from transformers import PreTrainedTokenizerBase

from llm.chat import render_chat


class _Tokenizer:
    chat_template = "{% if enable_thinking %}think{% endif %}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str:
        return f"{tokenize}:{add_generation_prompt}:{kwargs.get('enable_thinking')}"


def test_render_chat_disables_thinking() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, _Tokenizer())
    assert render_chat(tokenizer, ({"role": "user", "content": "q"},), True) == (
        "False:True:False"
    )
