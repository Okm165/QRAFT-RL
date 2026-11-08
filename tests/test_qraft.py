from __future__ import annotations

from pathlib import Path

from cli.config import load_config
from tools.qraft import DEFAULT_OVERLAY, STAGES, _host_run, _parser, _stage_command


def test_qraft_stage_order() -> None:
    assert STAGES[0] == "preprocess"
    assert STAGES[-1] == "report"
    assert STAGES.index("train") < STAGES.index("export")
    assert STAGES.index("export") < STAGES.index("behavior-base")
    assert STAGES.index("behavior-adapter") < STAGES.index("eval")


def test_qraft_stage_command_uses_latest_run() -> None:
    preprocess = _stage_command("preprocess", "/run/latest")
    train = _stage_command("train", "/run/latest")
    assert preprocess[preprocess.index("--run") + 1] == "/run/latest"
    assert train[train.index("--run") + 1] == "/run/latest"


def test_qraft_parser_uses_the_local_overlay() -> None:
    parser = _parser()
    assert parser.parse_args(["pipeline"]).overlay == DEFAULT_OVERLAY
    assert parser.parse_args(["stage", "train"]).overlay == DEFAULT_OVERLAY
    assert parser.parse_args(["pipeline"]).ctx == 0
    assert parser.parse_args(["apply", "--ctx", "262144"]).ctx == 262144


def test_host_run_resolves_the_latest_symlink(tmp_path: Path) -> None:
    run = tmp_path / "20261108-000000-local"
    run.mkdir()
    (tmp_path / "latest").symlink_to(run.name)
    assert _host_run(tmp_path, "/run/latest") == run
    assert _host_run(tmp_path, f"/run/{run.name}") == run


def test_local_profile_uses_model_parallelism() -> None:
    config = load_config(Path("k8s/overlays/local/pipeline.toml"))
    assert config.train_parallel == "model"
    assert config.world_size == 3
    assert config.train_quantize == "pre4bit"
    assert config.train_model == "/models/qwen3.8-27b-nf4"
    assert config.llm_workers == 8
    assert config.llm_context == 196608
    assert config.llm_model == "qwen3.8-27b"
    assert config.questions_model == "qwen3.8-27b"
