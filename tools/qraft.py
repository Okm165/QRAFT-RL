"""One action CLI for local builds, checks, and k8s runs."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from tools.progress import dashboard

STAGES = (
    "preprocess",
    "index",
    "questions",
    "raft",
    "hardware",
    "train",
    "export",
    "behavior-base",
    "behavior-adapter",
    "eval",
    "report",
)
DEFAULT_OVERLAY = "k8s/overlays/local"
DEFAULT_NAMESPACE = "qraft"
DEFAULT_RUNS = "/srv/qraft/runs"
LLM_STAGES = frozenset(
    {"preprocess", "questions", "raft", "eval", "behavior-base", "behavior-adapter"}
)
STOP_LLM_STAGES = frozenset({"train", "export"})
LLM_RESOURCE = "statefulset/qraft-llm"
LLM_LABEL = "app=qraft-llm"
RUNNER_RESOURCE = "deployment/qraft-runner"
RUNNER_LABEL = "app.kubernetes.io/name=qraft-runner"


def _run(
    command: Sequence[str],
    input_text: str = "",
    env: Mapping[str, str] = {},
) -> int:
    return subprocess.run(
        list(command),
        input=input_text,
        text=True,
        check=False,
        env={**os.environ, **env},
    ).returncode


def _kubectl(namespace: str, *args: str, input_text: str = "") -> int:
    return _run(("kubectl", "-n", namespace, *args), input_text)


def apply(overlay: str, namespace: str, context: int = 0) -> int:
    code = _kubectl(namespace, "apply", "-k", overlay)
    if code:
        return code
    return _sync_llm_env(namespace, overlay, context)


def _overlay_llm(overlay: str) -> tuple[int, int]:
    with (Path(overlay) / "pipeline.toml").open("rb") as handle:
        raw = cast(dict[str, object], tomllib.load(handle))
    section = raw.get("llm", {})
    llm = cast(dict[str, object], section) if isinstance(section, dict) else {}
    return int(llm.get("context", 196608)), int(llm.get("workers", 8))


def _sync_llm_env(namespace: str, overlay: str, context: int) -> int:
    default_context, workers = _overlay_llm(overlay)
    chosen = context if context > 0 else default_context
    if chosen < 32768 or chosen > 262144:
        print(f"context must be in [32768, 262144]: {chosen}")
        return 2
    if chosen // max(workers, 1) < 4096:
        print(f"context {chosen} leaves less than 4096 tokens per slot")
        return 2
    return _kubectl(
        namespace,
        "set",
        "env",
        "statefulset/qraft-llm",
        f"LLM_CTX={chosen}",
        f"LLM_PARALLEL={workers}",
        "LLM_TENSOR_SPLIT=1,1,1",
    )


def _state(
    namespace: str,
    resource: str,
    replicas: str,
    timeout: str,
    label: str,
) -> int:
    code = _kubectl(namespace, "scale", resource, "--replicas", replicas)
    if code:
        return code
    if replicas == "1":
        return _kubectl(
            namespace,
            "rollout",
            "status",
            resource,
            f"--timeout={timeout}",
        )
    return _kubectl(
        namespace,
        "wait",
        "--for=delete",
        "pod",
        "-l",
        label,
        f"--timeout={timeout}",
    )


def llm_state(namespace: str, state: str, timeout: str) -> int:
    replicas = "1" if state == "up" else "0"
    return _state(namespace, LLM_RESOURCE, replicas, timeout, LLM_LABEL)


def runner_state(namespace: str, state: str, timeout: str) -> int:
    replicas = "1" if state == "up" else "0"
    return _state(namespace, RUNNER_RESOURCE, replicas, timeout, RUNNER_LABEL)


def clean(namespace: str, llm: bool) -> int:
    code = _kubectl(
        namespace,
        "delete",
        "job",
        "--all",
        "--ignore-not-found",
        "--wait=true",
    )
    code = max(code, runner_state(namespace, "down", "600s"))
    if llm:
        code = max(code, llm_state(namespace, "down", "600s"))
    return code


def _seconds(value: str) -> int:
    text = value.strip().rstrip("s")
    return int(text) if text.isdigit() else 900


def _pod_config_hash(namespace: str) -> str:
    result = subprocess.run(
        (
            "kubectl",
            "-n",
            namespace,
            "exec",
            RUNNER_RESOURCE,
            "-c",
            "runner",
            "--",
            "python3",
            "-c",
            "import hashlib; print(hashlib.sha256(open('/config/pipeline.toml','rb').read()).hexdigest())",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _wait_config(namespace: str, path: Path, timeout: str) -> int:
    if not path.is_file():
        return 0
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    deadline = time.monotonic() + _seconds(timeout)
    while time.monotonic() < deadline:
        if _pod_config_hash(namespace) == expected:
            return 0
        time.sleep(5)
    print(f"config did not propagate: {path}")
    return 1


def _capture(namespace: str, *args: str) -> str:
    result = subprocess.run(
        ("kubectl", "-n", namespace, *args),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _host_run(runs: Path, pod_run: str) -> Path:
    root = runs / Path(pod_run).name
    if root.is_symlink():
        return root.resolve()
    return root


def _runner_restarts(namespace: str) -> str:
    return _capture(
        namespace,
        "get",
        "pods",
        "-l",
        RUNNER_LABEL,
        "-o",
        "jsonpath={.items[0].metadata.name}:{.items[0].status.containerStatuses[0].restartCount}",
    )


def _print_tail(path: Path, lines: int = 3):
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in text[-lines:]:
        if line.strip():
            print(f"  {line}", flush=True)


def _exec(
    namespace: str,
    command: Sequence[str],
    timeout: str,
    pod_run: str,
    runs: Path,
) -> int:
    stage = str(command[command.index("--stage") + 1])
    host_run = _host_run(runs, pod_run)
    state_path = f"{pod_run}/state"
    log_dir = f"{pod_run}/logs"
    exit_file = host_run / "state" / f"{stage}.exit"
    script_path = f"{state_path}/{stage}.sh"
    log_path = f"{log_dir}/{stage}.exec.log"
    exit_path = f"{state_path}/{stage}.exit"
    command_text = " ".join(shlex.quote(part) for part in command)
    script = (
        "#!/bin/sh\n"
        f"timeout {shlex.quote(timeout)} {command_text}\n"
        f"echo $? > {shlex.quote(exit_path)}\n"
    )
    launcher = (
        'set -e; rm -f "$4"; mkdir -p "$5" "$6"; '
        'printf "%s\\n" "$1" > "$2"; chmod +x "$2"; '
        'nohup "$2" > "$3" 2>&1 </dev/null &'
    )
    code = _kubectl(
        namespace,
        "exec",
        RUNNER_RESOURCE,
        "-c",
        "runner",
        "--",
        "sh",
        "-c",
        launcher,
        "sh",
        script,
        script_path,
        log_path,
        exit_path,
        state_path,
        log_dir,
    )
    if code:
        return code
    restarts = _runner_restarts(namespace)
    deadline = time.monotonic() + _seconds(timeout) + 300
    started = time.monotonic()
    next_note = started + 60
    while time.monotonic() < deadline:
        if exit_file.is_file():
            text = exit_file.read_text(encoding="utf-8").strip()
            return int(text) if text.isdigit() else 1
        if time.monotonic() >= next_note:
            print(
                f"stage {stage} running {int(time.monotonic() - started)}s",
                flush=True,
            )
            _print_tail(host_run / "logs" / f"{stage}.log")
            next_note += 60
            if _runner_restarts(namespace) != restarts:
                print(f"runner restarted during {stage}", flush=True)
                return 1
        time.sleep(5)
    print(f"stage {stage} timed out", flush=True)
    return 1


def _stage_command(name: str, run: str) -> tuple[str, ...]:
    command = [
        "python3",
        "-m",
        "cli.pipeline",
        "--config",
        "/config/pipeline.toml",
        "--stage",
        name,
    ]
    command.extend(("--run", run))
    return tuple(command)


def _run_path(overlay: str) -> str:
    with (Path(overlay) / "pipeline.toml").open("rb") as handle:
        raw = cast(dict[str, object], tomllib.load(handle))
    section = raw.get("run", {})
    run = cast(dict[str, object], section) if isinstance(section, dict) else {}
    name = str(run.get("name", "pipeline"))
    output = str(run.get("output", "/run")).rstrip("/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{output}/{stamp}-{name}"


def pipeline(args: argparse.Namespace) -> int:
    if apply(args.overlay, args.namespace, args.ctx):
        return 1
    if llm_state(args.namespace, "up", args.startup_timeout):
        return 1
    if runner_state(args.namespace, "up", args.startup_timeout):
        return 1
    if _wait_config(
        args.namespace,
        Path(args.overlay) / "pipeline.toml",
        args.startup_timeout,
    ):
        return 1
    run = _run_path(args.overlay)
    print(f"run: {run}", flush=True)
    llm_up = True
    for stage_name in STAGES:
        if stage_name in STOP_LLM_STAGES:
            if llm_up and llm_state(args.namespace, "down", args.startup_timeout):
                return 1
            llm_up = False
        if stage_name in LLM_STAGES and not llm_up:
            if llm_state(args.namespace, "up", args.startup_timeout):
                return 1
            llm_up = True
        code = _exec(
            args.namespace,
            _stage_command(stage_name, run),
            args.timeout,
            run,
            args.runs,
        )
        if code:
            if not llm_up:
                llm_state(args.namespace, "up", args.startup_timeout)
            return code
    if not llm_up:
        return llm_state(args.namespace, "up", args.startup_timeout)
    return 0


def stage(args: argparse.Namespace) -> int:
    if args.stop_llm and args.name not in STOP_LLM_STAGES:
        print("--stop-llm is valid for train and export")
        return 2
    if not args.stop_llm and args.name in LLM_STAGES and llm_state(
        args.namespace,
        "up",
        args.startup_timeout,
    ):
        return 1
    if runner_state(args.namespace, "up", args.startup_timeout):
        return 1
    if _wait_config(
        args.namespace,
        Path(args.overlay) / "pipeline.toml",
        args.startup_timeout,
    ):
        return 1
    if args.stop_llm and llm_state(args.namespace, "down", args.startup_timeout):
        return 1
    code = _exec(
        args.namespace,
        _stage_command(args.name, str(args.run)),
        args.timeout,
        str(args.run),
        args.runs,
    )
    if args.stop_llm:
        code = max(code, llm_state(args.namespace, "up", args.startup_timeout))
    return code


def status(namespace: str, runs: Path) -> int:
    code = _kubectl(namespace, "get", "deployment,statefulset,pods")
    latest = runs / "latest"
    if latest.is_symlink():
        print(f"latest run: {latest.resolve()}")
    return code


def progress(run: Path, watch: float) -> int:
    if watch <= 0:
        print(dashboard(run), flush=True)
        return 0
    try:
        while True:
            print(dashboard(run), flush=True)
            time.sleep(watch)
    except KeyboardInterrupt:
        return 0


def _check_env() -> dict[str, str]:
    root = Path(".cache/check-tmp").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return {
        "TMPDIR": str(root),
        "TORCHINDUCTOR_CACHE_DIR": str(root),
        "TRITON_CACHE_DIR": str(root),
    }


def hardware(run: Path) -> int:
    command = [sys.executable, "-m", "tools.hardware_check", "--json"]
    if run != Path(""):
        command.extend(("--run", str(run)))
    return _run(command)


def check() -> int:
    env = _check_env()
    code = _run((sys.executable, "-m", "pytest", "-q"), env=env)
    if code:
        return code
    code = _run(
        (
            sys.executable,
            "-m",
            "tools.type_audit",
            "--root",
            "preprocess",
            "--root",
            "rag",
            "--root",
            "raft",
            "--root",
            "sft",
            "--root",
            "llm",
            "--root",
            "tools",
            "--root",
            "cli",
            "--root",
            "eval",
            "--fail",
        ),
        env=env,
    )
    return code


def _build_runners() -> int:
    if _run(("podman", "image", "exists", "qraft-rl/model-base-local:local")):
        code = _run(("bash", "scripts/build_base.sh", "model"))
        if code:
            return code
    return _run(
        (
            "podman",
            "build",
            "--pull-never",
            "--network=none",
            "--security-opt",
            "label=disable",
            "-t",
            "qraft-rl/runner:local",
            "-f",
            "containers/runner.Containerfile",
            ".",
        )
    )


def build(target: str) -> int:
    if target == "model":
        return _run(("bash", "scripts/build_base.sh", "model"))
    if target == "llama":
        return _run(
            (
                "podman",
                "build",
                "--network=host",
                "--security-opt",
                "label=disable",
                "-t",
                "qraft-rl/llama-27b:local",
                "-f",
                "containers/llama-27b.Containerfile",
                ".",
            )
        )
    if target == "runners":
        return _build_runners()
    code = _run(("bash", "scripts/build_base.sh", "all"))
    return code or _build_runners()


def _common(parser: argparse.ArgumentParser) -> bool:
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--overlay", default=DEFAULT_OVERLAY)
    parser.add_argument("--runs", type=Path, default=Path(DEFAULT_RUNS))
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qraft", description="Local pipeline actions.")
    commands = parser.add_subparsers(dest="action", required=True)

    command = commands.add_parser(
        "pipeline",
        help="run all stages in the resident runner",
    )
    _common(command)
    command.add_argument("--startup-timeout", default="900s")
    command.add_argument("--timeout", default="3600s")
    command.add_argument("--ctx", type=int, default=0)
    command.set_defaults(handler=pipeline)

    command = commands.add_parser(
        "stage",
        help="run one stage in the resident runner",
    )
    _common(command)
    command.add_argument("name", choices=STAGES)
    command.add_argument("--run", default="/run/latest")
    command.add_argument("--startup-timeout", default="900s")
    command.add_argument("--timeout", default="3600s")
    command.add_argument("--stop-llm", action="store_true")
    command.set_defaults(handler=stage)

    command = commands.add_parser(
        "apply",
        help="apply the resident LLM and runner",
    )
    _common(command)
    command.add_argument("--ctx", type=int, default=0)
    command.set_defaults(handler=lambda args: apply(args.overlay, args.namespace, args.ctx))

    command = commands.add_parser(
        "status",
        help="show deployments, pods, and the latest run",
    )
    _common(command)
    command.set_defaults(handler=lambda args: status(args.namespace, args.runs))

    command = commands.add_parser(
        "progress",
        help="live run progress and data-quality dashboard",
    )
    command.add_argument("--run", type=Path, default=Path(DEFAULT_RUNS) / "latest")
    command.add_argument("--watch", type=float, default=0.0)
    command.set_defaults(handler=lambda args: progress(args.run, args.watch))

    command = commands.add_parser("llm", help="scale the resident LLM")
    command.add_argument("state", choices=("up", "down"))
    command.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    command.add_argument("--timeout", default="900s")
    command.set_defaults(handler=lambda args: llm_state(
        args.namespace, args.state, args.timeout
    ))

    command = commands.add_parser(
        "clean",
        help="stop the runner; optionally stop the LLM",
    )
    command.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    command.add_argument("--llm", action="store_true")
    command.set_defaults(handler=lambda args: clean(args.namespace, bool(args.llm)))

    command = commands.add_parser("hardware", help="report GPU/VRAM health")
    command.add_argument("--run", type=Path, default=Path(""))
    command.set_defaults(handler=lambda args: hardware(args.run))

    command = commands.add_parser("check", help="pytest and type audit")
    command.set_defaults(handler=lambda args: check())

    command = commands.add_parser("build", help="build local base/runner images")
    command.add_argument("target", choices=("model", "runners", "all"))
    command.set_defaults(handler=lambda args: build(args.target))
    return parser


def main() -> int:
    args = _parser().parse_args()
    handler = args.handler
    if not callable(handler):
        raise SystemExit("qraft: missing handler")
    result = handler(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
