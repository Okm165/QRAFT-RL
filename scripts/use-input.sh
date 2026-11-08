#!/usr/bin/env bash
# Point the local corpus PVC host path at this repository's data/input folder.
# The Kubernetes manifest keeps a host-neutral /srv/qraft/corpus path; this
# script makes that path a symlink to the repo folder.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-/srv/qraft/corpus}"

if [ -e "$target" ] && [ ! -L "$target" ]; then
  echo "Refusing to replace existing $target; move it aside first." >&2
  exit 1
fi

sudo mkdir -p "$(dirname "$target")"
sudo ln -sfn "$root/data/input" "$target"
echo "$root/data/input -> $target"
