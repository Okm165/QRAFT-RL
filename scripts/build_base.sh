#!/usr/bin/env bash
# Build the local model base image from the host Python and repo venv.
# Usage: ./scripts/build_base.sh [model|all]
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-all}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTHON_SOURCE="$(uv python find --managed-python --no-python-downloads --resolve-links "$PYTHON_VERSION")"
PYTHON_SOURCE_ROOT="$(dirname "$(dirname "$PYTHON_SOURCE")")"
PYTHON_ROOT="${PYTHON_ROOT:-$PWD/.cache/python${PYTHON_VERSION//./}}"
SITE_PACKAGES="${SITE_PACKAGES:-$PWD/.venv/lib/python$PYTHON_VERSION/site-packages}"
BASE_OS_IMAGE="${BASE_OS_IMAGE:-docker.io/library/rust:1.95-bookworm}"
MODEL_IMAGE="${MODEL_IMAGE:-qraft-rl/model-base-local:local}"

test -x "$PYTHON_SOURCE"

if [ ! -x "$PYTHON_ROOT/bin/python$PYTHON_VERSION" ]; then
  mkdir -p "$PYTHON_ROOT"
  cp -a "$PYTHON_SOURCE_ROOT/." "$PYTHON_ROOT/"
fi

context="$(mktemp -d)"
trap 'rm -rf "$context"' EXIT

build_model() {
  test -d "$SITE_PACKAGES"
  podman build \
    --pull-never \
    --network=none \
    --security-opt label=disable \
    --build-arg BASE_OS_IMAGE="$BASE_OS_IMAGE" \
    --volume "$PYTHON_ROOT:/host-python:ro" \
    --volume "$SITE_PACKAGES:/host-site:ro" \
    -t "$MODEL_IMAGE" \
    -f containers/model-base-local.Containerfile \
    "$context"
}

case "$TARGET" in
  model | all) build_model ;;
  *)
    echo "usage: $0 [model|all]" >&2
    exit 2
    ;;
esac
