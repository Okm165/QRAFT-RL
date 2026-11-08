# Kubernetes

One namespace, one resident 27B LLM StatefulSet, one resident runner
Deployment. No per-run Jobs.

## Layout

```
base/
  namespace.yaml   qraft namespace
  pvc.yaml         corpus, runs, models, HF cache claims
  llm.yaml         Service + Qwen3.8-27B StatefulSet (8 slots)
  runner.yaml      resident qraft-runner Deployment
  pipeline.toml    generic 27B configuration
overlays/local/    host-path PVs, GPU patches, local profile
overlays/smoke/    small prod-like profile (4 docs, 10-step QLoRA)
```

## Apply and run

```bash
./scripts/qraft apply --overlay k8s/overlays/smoke
./scripts/qraft pipeline --overlay k8s/overlays/smoke
./scripts/qraft status
./scripts/qraft clean            # stop the runner
./scripts/qraft clean --llm      # stop the runner and the 27B server
```

`qraft pipeline` applies the overlay, starts the 27B server and runner, waits
for the config hash, then runs every stage. It stops the 27B server for
`train` and `export`, and restarts it for `behavior-base`,
`behavior-adapter`, and `eval`. The run path is printed as
`/run/<timestamp>-<name>`; the host path is
`/srv/qraft/runs/<timestamp>-<name>` and `/srv/qraft/runs/latest` points to it.
`qraft` reads `/srv/qraft/runs` by default; pass `--runs <dir>` if you change
the host path.

`qraft stage <name>` runs one stage in the resident runner. Use
`--stop-llm` for `train` and `export`; the server is restored afterwards.
`behavior-base`, `behavior-adapter`, and `eval` need the server.

## LLM lifecycle

```
clients -> Service qraft-llm:8000 -> StatefulSet qraft-llm (replicas 1)
```

`llama-server` loads the 27B Q4_K_M GGUF with `--split-mode layer`,
`--tensor-split 1,1,1`, and `--parallel 8` continuous batching (196608 total
context, ~24.5k tokens per slot, ~176 tok/s aggregate with native MTP; ~101
tok/s without). The model is split across all three GPUs. Readiness is false
until the GGUF is loaded. If
`/models/adapters/qwen3.8-27b-lora.gguf` exists, the server also loads it with
`--lora`; requests select the base or adapter scale through the OpenAI-compatible
`lora` field, so one server serves both behavior variants. `LLM_CTX=262144`
raises the budget to 32.8k tokens/slot when a scenario needs the full native
window.

The context budget lives in `[llm] context` in the profile (default 196608).
`qraft apply --ctx <n>` or `qraft pipeline --ctx <n>` overrides it for a
scenario; the base manifest reads `LLM_CTX`, `LLM_PARALLEL`, and
`LLM_TENSOR_SPLIT`, so the server slot count always matches the stage workers.

The main GGUF carries the native MTP head; the server enables
`--spec-type draft-mtp --spec-draft-n-max 3` (`LLM_SPEC_TYPE`,
`LLM_SPEC_N_MAX`). Set `LLM_SPEC_TYPE=off` to compare against plain decoding.
The MTP is lossless and adds no model file.

`LLM_FLASH_ATTN` (default `auto`), `LLM_CACHE_K`/`LLM_CACHE_V` (default `f16`),
and `LLM_CACHE_REUSE` (default `0`) are exposed for big-context or throughput
scenarios; the defaults preserve full f16 KV quality.

Qwen3.8 has no 8B or smaller checkpoint. There is no small generator pod; the
27B serves Q&A and judging through the same endpoint.

## VRAM rules

| workload | placement | budget |
|---|---|---|
| 27B Q4_K_M serving, tensor split | GPUs 0/1/2 | ~5.5 GiB weights + ~4.3 GiB KV each |
| 27B QLoRA training | GPUs 0/1/2 | ~17-19 GiB each |
| behavior (base or adapter) | 8-slot server | no extra GPU allocation |
| LLM host RAM | - | 20 GiB Guaranteed |
| runner host RAM | - | 24 GiB Guaranteed |

The LLM and runner pods use requests = limits (Guaranteed QoS) so the host OOM
killer cannot take them under memory pressure from other namespaces. The
runner Deployment uses `strategy: Recreate`: a same-tag image update otherwise
tries to schedule a second 24 GiB Guaranteed pod while the old one is still
running, and the node has no memory headroom for the surge. The
server threads are capped at 8 to match its CPU budget.

The 27B server and 27B training cannot coexist on 24 GiB cards. The phased
pipeline stops the server for training and export, then restarts it for
behavior and eval. Do not run 27B training with the server resident.

## Local overlay specifics

- `qraft-corpus` -> `/srv/qraft/corpus` (the local setup symlinks it to `data/input`)
- `qraft-runs` -> `/srv/qraft/runs`
- `qraft-models` -> `/srv/qraft/models`
- `qraft-hf-cache` -> `/srv/qraft/hf-cache`

Edit these host paths in `k8s/overlays/local/pv.yaml` for your machine. They
are outside the repository by design.
- privileged pods with `/dev/kfd` and `/dev/dri`
- `HIP_VISIBLE_DEVICES` static pinning; no AMD device plugin
- one resident 27B llama.cpp pod with tensor split on GPUs 0/1/2
- one resident runner pod on GPUs 0/1/2
- `qraft-config` is a stable-name ConfigMap; editing `pipeline.toml` updates it
  in place, so no image rebuild is needed for prompt or sizing changes.

## Hardware notes

- Three 24 GiB cards. Serving uses llama.cpp layer split and training uses
  contiguous layer shards, so neither path depends on RCCL. Run the hardware
  stage to verify peer links on your host.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is intentionally not set:
  it corrupts bitsandbytes 4-bit gradients on this ROCm stack.
- `parallel = "data"` / DDP is not supported by the minimal trainer. Keep
  `parallel = "model"`.
- The host runs other namespaces; check `free -h` and `rocm-smi` before a run.
- Stages launch under `nohup` in the resident runner pod and write an exit
  marker under `state/`; the host polls the marker and the stage log. A dropped
  `kubectl exec` stream cannot orphan a stage, and a runner restart fails the
  stage instead of hanging.
- No per-run pod or Job is created.
