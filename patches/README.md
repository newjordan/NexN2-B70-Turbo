# llama.cpp kernel patch — reorder-on-MoE (the "Turbo")

Reproducible patch generated against upstream base commit `f0156d1`
(`kv-cache: follow the source cache size when sharing cells`, ggml-org/llama.cpp #24267).
The kernel was developed and validated locally for the Intel Arc Pro B70.

Apply on a clean checkout at the pinned base:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout f0156d1
git am /path/to/NexN2-B70-Turbo/patches/0001-*.patch
```

- **0001 — sycl: reorder-on-MoE for Q4_K and Q5_K `mul_mat_id`** — extends the SYCL
  weight-reorder optimization to the fused MoE expert GEMV (`mul_mat_id`), with
  reorder-aware reads across every decode and prefill sub-path (decode GEMV, dense
  mmvq, DMMV, and the dequant-to-fp16/fp32 GEMM path). Adds a Q5_K reorder DMMV
  kernel + per-expert reorder converters. Correctness-clean vs upstream (716/716
  MUL_MAT_ID op tests, PPL unchanged; token-identical greedy with FA off).
  **2026-06-10 re-validation** ([`../results/upstream-pr/SUMMARY.md`](../results/upstream-pr/SUMMARY.md)):
  the +17–18% on/off delta in [`../results/longctx-reorder.csv`](../results/longctx-reorder.csv)
  belongs almost entirely to upstream's pre-existing dense reorder; this patch's own
  end-to-end contribution on NX2 is ~0–1% (cherry-picks cleanly onto master d2462f8).

Build: oneAPI icpx 2026.0, SYCL backend, Intel Arc Pro B70 (Battlemage). WARP_SIZE 16.
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```
