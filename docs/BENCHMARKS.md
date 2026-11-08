# Benchmarks

Reference host: three 24 GiB AMD GPUs, Qwen3.8-27B Q4_K_M, 8 slots, 192k
total context, native MTP depth 3.

## Serving

| metric | result |
|---|---|
| slots | 8 × 24,576 tokens |
| aggregate decode | ~176 tok/s with MTP; ~101 tok/s without |
| MTP acceptance | 0.93–0.98 on extraction prompts, 0.83–0.89 on JSON Q&A |
| prompt eval | ~1,300 tok/s |

## 1k Q&A reference run

| metric | result |
|---|---|
| accepted questions | 1,200 (50.0% multi-fact) |
| split | 1,051 train / 149 holdout |
| RAFT | 950 records, 142 verified-absent |
| training | 814 steps, eval loss **0.2082**, 0 truncation |
| adapter | 200.3 MB GGUF, 704 tensors |
| retrieval | 149/149 evidence recall |
| behavior | base 24/24, adapter 23/24 on a 24-question sample |
| preprocess | 833s |
| questions | 5,181s |
| RAFT | 132s |
| training | ~3.3h |

The default profile uses 1,000 questions and 810 steps (moderately long,
about two epochs on 1k Q&A); the numbers above are from the reference run used
to tune it. The default behavior
setting now evaluates the full holdout instead of a 24-question sample.
