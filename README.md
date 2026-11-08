# QRAFT-RL

A fully local, GPU-only fine-tuning pipeline for Qwen3.8-27B on OCR/domain
documents. Data processing and orchestration run on CPU; every model load,
generation, judge call, and training step runs on the three GPUs.

## Model choice

The Qwen3.8 family has three public sizes: **27B**, **180B Flash-Next**, and
**2.4T MoE**. There is no official Qwen3.8 8B or smaller checkpoint. The 27B is
the only Qwen3.8 model that fits this host, so it serves every model role:

- document structuring,
- grounded Q&A generation,
- semantic judging,
- RAFT oracle/absence checks,
- evidence recall,
- behavior probes,
- the QLoRA training target.

One model, one serving image, one endpoint. The model store keeps only the
Qwen3.8-27B GGUF and the NF4 training checkpoint; the trained GGUF LoRA lives
in `adapters/` as a derived artifact. All other model families, GGUFs, and
images were removed.

## Pipeline

```
preprocess -> index -> questions -> raft -> hardware -> train
           -> export -> behavior-base -> behavior-adapter -> eval -> report
```

| stage | what it does |
|---|---|
| `preprocess` | inventory, normalize, LLM structure into chunks/facts/concepts |
| `index` | SQLite FTS5 retrieval index |
| `questions` | 27B generates grounded Q&A; 27B judge validates |
| `raft` | oracle support gate, verified absence, distractors, judge weights |
| `hardware` | GPU link, temperature, VRAM snapshot |
| `train` | weighted QLoRA SFT on 3-GPU model-parallel shards |
| `export` | convert the trained HF LoRA adapter to a GGUF adapter |
| `behavior` | base vs adapter answers, citations, refusals, leakage |
| `eval` | evidence recall on held-out documents |
| `report` | summary, quality, token health, GPU telemetry |

## Data

Put top-level Markdown files in `data/input` (gitignored) or point
`[input] path` at any folder. The local Kubernetes overlay mounts
`/srv/qraft/corpus` at `/data`; the local setup symlinks that path to
`data/input`:

```bash
./scripts/use-input.sh
```

Edit `k8s/overlays/local/pv.yaml` to use a different host folder. `[input] glob`
defaults to `*.md`, and `[input] limit = 0` reads every matching file.

The default profile generates **1,000 grounded Q&A pairs**, enforces a 50%
multi-fact quota, and runs **810 optimizer steps** (moderately long; about two
epochs on 1k Q&A).

A synthetic example is included at `data/input/example.md` for a quick smoke
test:

```bash
./scripts/qraft pipeline --overlay k8s/overlays/smoke
```

Replace the example with your own Markdown files for the full run.

## Orchestration

One `llama-server` pod serves every stage:

- **qraft-llm**: 27B Q4_K_M, `--split-mode layer --tensor-split 1,1,1`,
  `--parallel 8`, 196608 context (~24.5k tokens/slot, ~176 tok/s aggregate
  with native MTP; ~101 without. `LLM_CTX=262144` gives 32.8k/slot).

The budget is a platform setting: `[llm] context` in the profile is the
default, and `qraft apply --ctx <n>` or `qraft pipeline --ctx <n>` switches a
scenario without editing YAML. The base manifest reads `LLM_CTX`,
`LLM_PARALLEL`, and `LLM_TENSOR_SPLIT`; the server always uses the same
tensor-split and slot count as the stage workers.

The main Qwen3.8-27B GGUF already contains the native MTP head. The server
enables it with `--spec-type draft-mtp --spec-draft-n-max 3` (env
`LLM_SPEC_TYPE`/`LLM_SPEC_N_MAX`). On the reference host it is lossless and
gives ~1.75x decode (~176 tok/s aggregate versus ~101 without). Set
`LLM_SPEC_TYPE=off` to compare against plain decoding.

The server also exposes `LLM_FLASH_ATTN` (default `auto`), `LLM_CACHE_K` /
`LLM_CACHE_V` (default `f16`), and `LLM_CACHE_REUSE` (default `0`). Use
`q8_0`/`q4_0` KV or cache reuse for big-context scenarios; the defaults keep
full f16 KV quality.

The LLM and runner pods are **Guaranteed QoS** (20 GiB / 24 GiB, 8 CPU each),
and the server threads are capped at 8. This keeps the resident pods protected
when other namespaces apply host memory pressure.

Layer split is the supported multi-GPU path on the ROCm stack. Qwen3.8-27B has 64 language layers
(48 Gated DeltaNet + 16 full attention) plus the MTP head; only the 16
full-attention layers carry KV cache (4 KV heads, 256-dim), so a 192k cache
costs ~12.75 GiB total (~4.25 GiB per GPU). Serving stays around 10–11 GiB per
card and leaves large headroom.

Training and adapter export stop the server because 27B QLoRA training needs
~17 GiB on the GPUs. After training, `export` converts the HF LoRA adapter to a
GGUF adapter (~200 MB) on CPU; `behavior` then runs on the same 8-slot server
with per-request LoRA scale selection. Base requests send an empty `lora` list,
adapter requests send `{"id": 0, "scale": 1.0}`, and the judge always uses the
base scale. `qraft pipeline` performs the stop/start phases automatically;
`qraft stage train --stop-llm` and `qraft stage export --stop-llm` expose the
same lifecycle for individual stages.

Serving uses one tensor-split llama.cpp model; training uses contiguous
model-parallel shards. Neither path depends on RCCL/DDP collectives. Run
`qraft hardware` to verify peer links on your host.

`llama.cpp` is built once into `qraft-rl/llama-27b:local` from the ROCm
development image (`qraft build llama`). The GGUF and NF4 weights are local; the
runtime is offline. The 27B NF4 HF checkpoint remains the QLoRA target; the GGUF
is the serving format.

## RAFT and training

Each RAFT example contains the question, retrieved contexts (one oracle plus
distractors), a CoT target, and a citation:

```
Reasoning:
<2-4 evidence-linked steps>

Answer: <short answer> [n]
```

The oracle gate verifies that the selected context actually contains the
expected answer. A fraction of examples removes every supporting context and
teaches the exact refusal. The judge quality becomes a per-example SFT weight
(`0.6-1.0`); this is reward-weighted SFT, not PPO/GRPO. The local profile uses
20% verified-absence examples and four distractors, matching the RAFT
answerable/unanswerable mix.

Each RAFT target starts with a verbatim `Quote:` line before the reasoning and
answer, and the behavior prompt asks for the same quote. Research on grounded
QA shows that supervising the extraction/quote step improves multi-hop accuracy
and attributability. After the first 100 accepted questions, the pipeline
rejects single-fact lookups while the global multi-fact share is below 50%, so
the dataset honors the synthesis quota instead of only asking for it.

LoRA targets the full language stack: attention and MLP projections
(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`) plus the
Gated-DeltaNet input/output projections (`in_proj_qkv,out_proj`). Qwen3.8-27B
is a hybrid stack: 48 of its 64 layers are linear-attention blocks, and a
LLaMA-style target list adapts their MLPs but not their sequence-mixing
projections. The MLP carries most task knowledge, and the DeltaNet projections
carry long-context mixing; together they cover all 64 layers. The question
prompt requires a `synthesis`/`lookup` label, and the judge records
`multi_fact` so `qraft progress` can flag a run that drifts back to shallow
single-fact lookups.

The trainer is a small manual loop, not a `Trainer` subclass:

- one process, model-parallel shards,
- bf16 compute, NF4 weights, LoRA adapters,
- token-level causal loss with per-example weights,
- gradient clipping, cosine schedule,
- no CPU/disk offload,
- `PYTORCH_CUDA_ALLOC_CONF` is left unset: `expandable_segments:True` produces
  non-finite gradients with bitsandbytes 4-bit on this ROCm stack.

## Commands

```bash
# full 1k pass: 8 workers/slots, 810-step moderate training, MLP + DeltaNet LoRA
./scripts/qraft pipeline --overlay k8s/overlays/local --timeout 21600s

# scenario switch: full native 256k, or throughput-first 131k
./scripts/qraft pipeline --overlay k8s/overlays/local --ctx 262144
./scripts/qraft apply --overlay k8s/overlays/local --ctx 131072

# production-like small run: short QLoRA, MLP + DeltaNet LoRA, export, behavior
./scripts/qraft pipeline --overlay k8s/overlays/smoke

# one stage
./scripts/qraft stage questions --overlay k8s/overlays/local
./scripts/qraft stage train --overlay k8s/overlays/local --stop-llm
./scripts/qraft stage export --overlay k8s/overlays/local --stop-llm
./scripts/qraft stage behavior-adapter --overlay k8s/overlays/local

# live progress, multi-fact ratio, rejection reasons, loss health, ETA
./scripts/qraft progress --watch 30

# checks
uv run pytest -q
uv run python -m tools.type_audit --root preprocess --root rag \
  --root raft --root sft --root llm --root tools --root cli --root eval --fail
```

Kubernetes details: [`k8s/README.md`](k8s/README.md).
Container details: [`containers/README.md`](containers/README.md).
Benchmarks: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
