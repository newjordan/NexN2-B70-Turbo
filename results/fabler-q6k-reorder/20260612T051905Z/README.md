# Fabler campaign: Q6_K MoE reorder + SYCL graph audit

Date: 2026-06-12. Hardware: Intel Arc Pro B70, oneAPI icpx 2026.0, Level Zero 1.15.38308.
Branch: llama.cpp `fabler` (= `b11935239` reorder-on-MoE + two new commits, exported as
[`patches/0002`](../../../patches/) and [`patches/0003`](../../../patches/)).
Models: `NX2-Q5_K_M.gguf` / `NX2-Q4_K_M.gguf` (checksums in `results/model-checksums.sha256`).

## What changed

1. **Q6_K MoE reorder** (`0003`): `ffn_down_exps` is Q6_K in both the Q4_K_M and Q5_K_M
   NX2 mixes — about a third of the expert bytes — and previously ran on the
   non-reordered fused MMVQ path. The per-expert SoA reorder + fused reordered GEMV now
   covers Q6_K, reusing the existing dense Q6_K reorder traits/readers (dense mmvq,
   reordered DMMV, and dequant-GEMM coverage already existed upstream).
2. **Concat made submit-only** (`0002`): dropped two host waits (in-order queue makes
   them redundant); also removes 17 host syncs/token from the NexN2 decode path and
   makes CONCAT graph-recordable.
3. **SYCL graph audit** (`0003`, default-off): `check_graph_compatibility` now admits
   `MUL_MAT_ID` nodes that take the fused no-host-sync TG path (ids stay on device,
   expert weights already reordered by the first eager pass).

## Decode, ctx0 (`llama-bench -ngl 99 -fa on -p 0 -n 128 -r 5`)

| model | before (0001 only) | after (+0002/0003) | gain |
|---|---:|---:|---:|
| NX2-Q5_K_M | 81.09 | **85.85** | **+5.9%** |
| NX2-Q4_K_M | 85.7 (retained frontier) | **91.02** | **+6.2%** |

## Depth sweep, Q5_K_M, same build, reorder OFF→ON (`GGML_SYCL_DISABLE_OPT`, `-r 3`, FA on both sides)

| depth | off | on | gain |
|---|---:|---:|---:|
| 0 | 68.91 | 85.09 | **+23.5%** |
| 4096 | 65.82 | 80.78 | +22.7% |
| 16384 | 62.25 | 75.64 | +21.5% |
| 32768 | 56.95 | 67.71 | +18.9% |

Previous off→on gains (Q4_K/Q5_K reorder only) were +18.2 / +16.1 / +12.5 / +10.4%.

## Correctness

- `test-backend-ops test -o MUL_MAT_ID`: **690/690** on the working branch,
  **714/714** on the pinned-base patch chain (`ac4cddeb0` + 0001..0003).
- PPL Q5_K_M wikitext-2 30 chunks: **5.5723 ± 0.15276** vs retained base
  5.5643 ± 0.15232 — statistically unchanged.
- Greedy sanity (temp 0): "The capital of France is" → "Paris."

## Negative result: SYCL graph replay

With graphs force-enabled (`GGML_SYCL_DISABLE_GRAPH=0`) the full NexN2 decode graph now
records (no incompatible nodes), but per-token executable-graph **update throws
"Cannot update using a graph with a different topology. Mismatch found in the number of
nodes"**, forcing a JIT re-finalize of ~3.7k nodes every token: decode collapses to
**15.5 t/s** vs 81.3 with graphs off (raw: `sycl-graph-ab.txt`). Graphs stay default-off;
the graph-safety changes are kept because they are correct, cost nothing when graphs are
off, and make the backend ready if the driver/runtime gains stable topology matching.

## Raw artifacts

- `q5km-ctx0-before-q6k.json` — fresh fabler-branch baseline before Q6_K work
- `ctx0-after-q6k.txt` — ctx0 results after (both models)
- `depth-sweep-and-greedy.txt` — depth sweep legs + greedy sanity output
- `ppl-q5km-30chunks.txt` — PPL final estimate
- `sycl-graph-ab.txt` — graphs off vs on A/B
