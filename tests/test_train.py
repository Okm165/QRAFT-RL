from __future__ import annotations

from sft.data import TrainingExample, _doc_id, _split, _text_pair
from sft.model import LoraSettings, _lora_settings
from sft.summary import _history_rows
from sft.train import _auto_steps

TEMPLATE = "Question: %QUESTION%\nContext:\n%CONTEXT%\nAnswer:"


def test_auto_steps_targets_two_epochs() -> None:
    assert _auto_steps(100, 1, 2) == 100
    assert _auto_steps(101, 1, 2) == 101
    assert _auto_steps(0, 1, 2) == 1


def test_lora_settings_parse_targets() -> None:
    settings = _lora_settings(8, 16, 0.0, "q_proj, v_proj")
    assert settings == LoraSettings(r=8, alpha=16, dropout=0.0, targets=("q_proj", "v_proj"))


def test_text_pair_accepts_prompt_answer() -> None:
    assert _text_pair({"prompt": "p", "answer": "a"}, TEMPLATE) == ("p", "a")


def test_history_rows_keep_json_scalars() -> None:
    assert _history_rows([{"loss": 1.0, "step": 2}]) == [{"loss": 1.0, "step": 2}]


def test_text_pair_accepts_raft_row() -> None:
    prompt, answer = _text_pair(
        {
            "question": "Q?",
            "contexts": [
                {
                    "source": "S",
                    "section": "T",
                    "page": 1,
                    "kind": "prose",
                    "text": "X",
                }
            ],
            "answer": "A",
        },
        TEMPLATE,
    )
    assert answer == "A"
    assert "Question: Q?" in prompt
    assert "[1] S | T | page 1 | prose" in prompt


def test_validation_split_is_deterministic() -> None:
    examples = tuple(
        TrainingExample([index], [1], [index], False) for index in range(10)
    )
    train, validation = _split(examples, 0.2)
    assert len(train) == 8
    assert len(validation) == 2
    assert _split(examples, 0.2) == (train, validation)
    assert _split(examples, 0.0) == (examples, ())


def test_doc_id_uses_oracle_context() -> None:
    row = {
        "oracle_index": 1,
        "contexts": [{"doc_id": "a"}, {"doc_id": "b"}],
    }
    assert _doc_id(row) == "b"


def test_doc_id_falls_back_to_first_context() -> None:
    row = {
        "oracle_index": -1,
        "contexts": [{"doc_id": "a"}, {"doc_id": "b"}],
    }
    assert _doc_id(row) == "a"


def test_validation_split_keeps_documents_together() -> None:
    examples = tuple(
        TrainingExample([index], [1], [index], False, f"d{index // 3}")
        for index in range(9)
    )
    train, validation = _split(examples, 0.34)
    train_docs = {example.doc_id for example in train}
    validation_docs = {example.doc_id for example in validation}
    assert len(train_docs) == 2
    assert len(validation_docs) == 1
    assert train_docs.isdisjoint(validation_docs)
