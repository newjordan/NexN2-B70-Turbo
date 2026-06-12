# llama.cpp SYCL backend patches

Reproducible patches generated against ggml-org/llama.cpp base commit `ac4cddeb0`
(`vendor : update LibreSSL to 4.3.2`, ggml-org/llama.cpp #24397).
The backend changes were developed and validated locally for the Intel Arc Pro B70.
Upstream PR (0001): https://github.com/ggml-org/llama.cpp/pull/24452

Apply on a clean checkout at the pinned base, in order:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout ac4cddeb0
git am /path/to/NexN2-B70-Turbo/patches/000*.patch
```

- **0001 — sycl: support reordered Q4_K and Q5_K MoE `MUL_MAT_ID`** — extends the SYCL
  weight-reorder optimization to the fused MoE expert GEMV (`mul_mat_id`), with
  reorder-aware reads across every decode and prefill sub-path (decode GEMV, dense
  mmvq, DMMV, and the dequant-to-fp16/fp32 GEMM path). Adds a Q5_K reorder DMMV
  kernel + per-expert reorder converters. Correctness-clean in retained targeted
  checks (714/714 MUL_MAT_ID op tests, PPL unchanged).
  **2026-06-10 re-validation** ([`../results/upstream-pr/SUMMARY.md`](../results/upstream-pr/SUMMARY.md)):
  treat this as a llama.cpp contribution-readiness check for one code patch, separate from
  the project-level NX2 GGUF benchmark results.
- **0002 — sycl: make concat submit-only** — drops two redundant host waits in
  `ggml_sycl_op_concat` (the queue is in-order); removes 17 host syncs per decoded
  token on NexN2 and makes CONCAT recordable into SYCL graphs.
- **0003 — sycl: extend MoE reorder to Q6_K `mul_mat_id`** — per-expert SoA reorder +
  fused reordered GEMV for Q6_K expert weights (`ffn_down_exps` in the NX2
  Q4_K_M/Q5_K_M mixes — about a third of the expert bytes), reusing the existing dense
  Q6_K reorder traits/readers. Also admits graph-safe `MUL_MAT_ID` nodes in
  `check_graph_compatibility` (SYCL graphs remain default-off; see the negative replay
  result in [`../results/fabler-q6k-reorder/`](../results/fabler-q6k-reorder/)).
  **2026-06-12 validation:** 714/714 MUL_MAT_ID op tests on the pinned-base chain,
  PPL statistically unchanged (5.5723 ± 0.153 vs 5.5643 ± 0.152), ctx0 decode
  Q5_K_M 81.1 → 85.8 t/s, Q4_K_M 85.7 → 91.0 t/s.

Build: oneAPI icpx 2026.0, SYCL backend, Intel Arc Pro B70 (Battlemage). WARP_SIZE 16.
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```
