# Local containers

Podman-based local images. No host packages, no network at runtime.

## Images

| image | role |
|---|---|
| `qraft-rl/llama-27b:local` | ROCm llama.cpp server |
| `qraft-rl/runner:local` | all pipeline stages, training, export, behavior |

The 27B GGUF stays on the mounted `/models` volume; the serving image only
carries the `llama-server` binary and the ROCm runtime. The `runner` image
selects the HF NF4 model from the pipeline TOML for training and carries the
llama.cpp LoRA converter for the `export` stage.

## Build

```bash
./scripts/qraft build model      # ROCm/PyTorch base
./scripts/qraft build runners    # runner image (all stages)
./scripts/qraft build llama      # one-time ROCm llama.cpp image (network)
./scripts/qraft build all        # base + runner
```

Kubernetes is the only orchestration path: one resident 27B pod and one
resident runner pod, with no per-run Jobs. See [`../k8s/README.md`](../k8s/README.md).

## Training

`train` uses one process and model parallelism by default:

```
sft.train --parallel model --devices ${GPU_WORLD_SIZE:-3}
```

The 27B NF4 checkpoint is sharded across all visible GPUs; no RCCL collectives.
The resident 27B server must be stopped for training. On k8s use
`qraft stage train --stop-llm`; the phased `qraft pipeline` does this
automatically. LoRA targets attention and MLP projections
(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`) plus the
Gated-DeltaNet `in_proj_qkv,out_proj` projections; Qwen3.8-27B has 48 linear
attention layers and 16 full attention layers, so this covers the language
stack. The RAFT data uses 20% verified-absence examples. The `export` stage
then converts the adapter on CPU and writes
`/models/adapters/qwen3.8-27b-lora.gguf`; the models volume is mounted
read-write for that stage.

## Behavior

`behavior` uses the OpenAI-compatible server, not a local HF model. It sends an
empty `lora` list for the base run and `{"id": 0, "scale": 1.0}` for the adapter
run; the judge always uses the base scale. The server loads the adapter when
the file exists, so start it after `export` when comparing adapter behavior.
`BEHAVIOR_WORKERS` defaults to 8 to match the server slots.

## LLM endpoint

The server exposes `/v1/chat/completions` and `/health`. `llama-server` loads
the 27B Q4_K_M GGUF with `--split-mode layer --tensor-split 1,1,1` across the
three GPUs and `--parallel 8` continuous batching (196608 total context,
~24.5k tokens/slot). The OpenAI-compatible endpoint serves generation and
semantic judging. `LLM_TENSOR_SPLIT`, `LLM_PARALLEL`, and `LLM_CTX` tune the
split, slots, and context budget. If `/models/adapters/qwen3.8-27b-lora.gguf`
exists, the service adds `--lora` and requests select the adapter scale per
call. The main GGUF carries the native MTP head; `LLM_SPEC_TYPE=draft-mtp` and
`LLM_SPEC_N_MAX=3` enable lossless speculative decoding, and
`LLM_SPEC_TYPE=off` disables it. `LLM_FLASH_ATTN`, `LLM_CACHE_K`,
`LLM_CACHE_V`, and `LLM_CACHE_REUSE` tune flash attention and KV cache
behavior; the defaults are `auto`, `f16`, `f16`, and `0`.
`LLM_THREADS`/`LLM_THREADS_BATCH` default to 8, matching the Guaranteed CPU
budget.

Qwen3.8 has no 8B or smaller checkpoint: the family is 27B, 180B Flash-Next,
and 2.4T MoE. The 27B is the only local Qwen3.8 model, so one service handles
both roles.

## GPU and P2P

GPU services request `/dev/kfd` and `/dev/dri` and use
`HIP_VISIBLE_DEVICES`. Training uses contiguous model-parallel shards and
serving uses llama.cpp layer split; neither path depends on RCCL. Run the
hardware stage to verify peer links on your host.
