from __future__ import annotations

from pathlib import Path

import pytest

from cli.config import load_config
from cli.stage_commands import _commands
from llm.client import _content, endpoints
from llm.questions import _interleave
from llm.structured import parse_object


def test_llm_content_parses_choice() -> None:
    data = {"choices": [{"message": {"content": "ok"}}]}
    assert _content(data) == "ok"


def test_llm_object_parses_fenced_json() -> None:
    assert parse_object('text {"question": "q", "answer": "a"} text') == {
        "question": "q",
        "answer": "a",
    }


def test_pipeline_config_reads_toml(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        "[run]\n"
        'name = "test"\n'
        'output = "runs"\n'
        "\n[input]\n"
        'path = "docs"\n'
        'glob = "*.md"\n'
        "limit = 3\n"
        "\n[llm]\n"
        'base_url = "http://llm/v1"\n'
        'base_urls = ["http://llm-a/v1", "http://llm-b/v1"]\n'
        "workers = 2\n"
        "context = 131072\n"
        'model = "test-llm"\n'
        "\n[train]\n"
        'model = "test-model"\n'
        'dtype = "float32"\n'
        'quantize = "4bit"\n'
        "world_size = 2\n"
        "max_steps = 7\n"
        "\n[eval]\n"
        "fail_below = 0.5\n"
        'refusal_questions = ["what is the weather?"]\n',
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded.name == "test"
    assert loaded.input_path == Path("docs")
    assert loaded.limit == 3
    assert loaded.llm_base_urls == ("http://llm-a/v1", "http://llm-b/v1")
    assert loaded.llm_workers == 2
    assert loaded.llm_context == 131072
    assert loaded.world_size == 2
    assert loaded.max_steps == 7
    assert loaded.train_dtype == "float32"
    assert loaded.train_quantize == "4bit"
    assert loaded.eval_fail_below == 0.5
    assert loaded.refusal_questions == ("what is the weather?",)


def test_pipeline_config_requires_input_path(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        "[llm]\nmodel = \"test-llm\"\n\n[train]\nmodel = \"m\"\n\n[eval]\n"
        "refusal_questions = [\"probe?\"]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"\[input\]\.path"):
        load_config(config)


def test_pipeline_config_requires_refusal_probes(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        "[input]\npath = \"docs\"\n\n[llm]\nmodel = \"test-llm\"\n\n[train]\nmodel = \"m\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refusal_questions"):
        load_config(config)

def test_pipeline_commands_are_gpu_only(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        "[input]\npath = \"docs\"\n\n[llm]\nmodel = \"test-llm\"\n\n"
        "[train]\nmodel = \"m\"\ndtype = \"float32\"\n\n"
        "[eval]\nfail_below = 0.5\n"
        "refusal_questions = [\"probe?\"]\n",
        encoding="utf-8",
    )
    commands = _commands(load_config(config), tmp_path / "run", config)
    train = commands["train"]
    evaluation = commands["eval"]
    assert "--device" not in train
    assert train[train.index("--dtype") + 1] == "float32"
    assert evaluation[evaluation.index("--fail-below") + 1] == "0.5"

def test_questions_interleave_documents() -> None:
    chunks = (
        {"doc_id": "a", "text": "a1"},
        {"doc_id": "a", "text": "a2"},
        {"doc_id": "b", "text": "b1"},
    )
    assert [chunk["text"] for chunk in _interleave(chunks)] == ["a1", "b1", "a2"]


def test_endpoints_parse_multiple_urls() -> None:
    assert endpoints("http://a/v1", "") == ("http://a/v1",)
    assert endpoints("http://a/v1", "http://b/v1, http://c/v1") == (
        "http://b/v1",
        "http://c/v1",
    )

