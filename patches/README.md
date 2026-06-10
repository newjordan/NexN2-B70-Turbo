# llama.cpp SYCL backend patch

Reproducible patch generated against ggml-org/llama.cpp base commit `d2462f8`
(`chat: fix LFM2/LFM2.5 ignoring json_schema`, ggml-org/llama.cpp #24377).
The backend change was developed and validated locally for the Intel Arc Pro B70.

Apply on a clean checkout at the pinned base:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout d2462f8
git am /path/to/NexN2-B70-Turbo/patches/0001-*.patch
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

Build: oneAPI icpx 2026.0, SYCL backend, Intel Arc Pro B70 (Battlemage). WARP_SIZE 16.
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```
