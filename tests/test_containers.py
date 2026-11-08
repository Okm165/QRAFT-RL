from __future__ import annotations

from pathlib import Path


def test_local_model_base_is_a_local_bootstrap() -> None:
    text = Path("containers/model-base-local.Containerfile").read_text(encoding="utf-8")
    assert "FROM ${BASE_OS_IMAGE}" in text
    assert "/host-python" in text
    assert "/host-site" in text


def test_dockerignore_excludes_local_artifacts() -> None:
    text = Path(".dockerignore").read_text(encoding="utf-8")
    for item in (".venv", ".data", ".cache", "data", "runs"):
        assert item in text


def test_runner_image_is_model_work_only() -> None:
    text = Path("containers/runner.Containerfile").read_text(encoding="utf-8")
    for package in ("preprocess", "rag", "raft", "sft", "cli", "eval"):
        assert f"COPY {package} ./{package}" in text
    for item in ("llm/judge.py", "llm/questions.py", "tools/p2p.py", "tools/p2p_check.py"):
        assert f"COPY {item}" in text
    assert "COPY llm/server.py" not in text
    assert "COPY llm/proxy.py" not in text
    assert "COPY tools ./tools" not in text


def test_llama_image_is_a_rocm_build() -> None:
    text = Path("containers/llama-27b.Containerfile").read_text(encoding="utf-8")
    assert "rocm/dev-ubuntu" in text
    assert "GGML_HIP=ON" in text
    assert "gfx1100" in text
    assert "llama-server" in text


def test_base_llm_manifest_uses_the_27b_model() -> None:
    text = Path("k8s/base/llm.yaml").read_text(encoding="utf-8")
    assert "Qwen3.8-27B-UD-Q4_K_M.gguf" in text
    assert "docker.io/qraft-rl/llama-27b:local" in text
    assert "--parallel" in text
    assert "--tensor-split" in text
    assert "--threads" in text
    assert "LLM_CTX" in text
    assert "LLM_PARALLEL" in text
    assert "LLM_TENSOR_SPLIT" in text
    assert "LLM_SPEC_TYPE" in text
    assert "draft-mtp" in text
    assert "qwen3.8-27b-lora.gguf" in text


def test_local_overlay_sets_guaranteed_qos() -> None:
    text = Path("k8s/overlays/local/kustomization.yaml").read_text(encoding="utf-8")
    assert "value: 20Gi" in text
    assert "value: 24Gi" in text
    assert 'value: "8"' in text
