from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from cli.config import load_config

BASE = Path("k8s/base")


def _docs() -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    for path in sorted(BASE.glob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc is not None:
                docs.append(cast(dict[str, object], doc))
    return docs


def test_k8s_profile_is_valid() -> None:
    config = load_config(BASE / "pipeline.toml")
    assert config.input_path == Path("/data")
    assert config.output == Path("/run")
    assert len(config.llm_base_urls) == 1
    assert config.world_size == 3


def test_k8s_has_resident_workloads_and_no_jobs() -> None:
    kinds = [doc["kind"] for doc in _docs()]
    assert "Job" not in kinds
    assert kinds.count("StatefulSet") == 1
    assert kinds.count("Deployment") == 1


def test_k8s_llm_statefulset_uses_all_gpus() -> None:
    docs = _docs()
    statefulset = next(
        doc
        for doc in docs
        if doc["kind"] == "StatefulSet"
        and cast(dict[str, object], doc["metadata"])["name"] == "qraft-llm"
    )
    spec = cast(dict[str, object], statefulset["spec"])
    assert spec["replicas"] == 1
    template = cast(dict[str, object], spec["template"])
    pod = cast(dict[str, object], template["spec"])
    containers = cast(list[dict[str, object]], pod["containers"])
    assert [container["name"] for container in containers] == ["llm"]
    limits = cast(
        dict[str, object],
        cast(dict[str, object], containers[0]["resources"])["limits"],
    )
    assert limits["amd.com/gpu"] == "3"


def test_k8s_runner_is_resident_and_requests_all_gpus() -> None:
    docs = _docs()
    deployment = next(
        doc for doc in docs if doc["kind"] == "Deployment"
    )
    metadata = cast(dict[str, object], deployment["metadata"])
    assert metadata["name"] == "qraft-runner"
    spec = cast(dict[str, object], deployment["spec"])
    assert spec["replicas"] == 1
    template = cast(dict[str, object], spec["template"])
    pod = cast(dict[str, object], template["spec"])
    container = cast(list[dict[str, object]], pod["containers"])[0]
    assert container["name"] == "runner"
    assert container["command"] == ["sleep", "infinity"]
    limits = cast(dict[str, object], cast(dict[str, object], container["resources"])["limits"])
    assert limits["amd.com/gpu"] == "3"


def test_k8s_local_overlay_has_no_jobs() -> None:
    text = Path("k8s/overlays/local/kustomization.yaml").read_text(encoding="utf-8")
    assert "kind: Job" not in text


def test_k8s_configmap_is_generated_from_the_profile() -> None:
    text = (BASE / "kustomization.yaml").read_text(encoding="utf-8")
    assert "configMapGenerator" in text
    assert "- pipeline.toml" in text
