# llama.cpp kernel patch — reorder-on-MoE (the "Turbo")

Reproducible patch generated against upstream base commit `f0156d1`
(`kv-cache: follow the source cache size when sharing cells`, ggml-org/llama.cpp #24267).
The work is local to this project and is **not** submitted upstream.

Apply on a clean checkout at the pinned base:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout f0156d1
git am /path/to/nx2-b70-turbo/patches/0001-*.patch
```

- **0001 — sycl: reorder-on-MoE for Q4_K and Q5_K `mul_mat_id`** — the verified decode
  speedup. Extends the SYCL weight-reorder optimization to the fused MoE expert GEMV
  (`mul_mat_id`), with reorder-aware reads across every decode and prefill sub-path
  (decode GEMV, dense mmvq, DMMV, and the dequant-to-fp16/fp32 GEMM path). Adds a Q5_K
  reorder DMMV kernel + per-expert reorder converters. Token-identical greedy output vs
  reorder-off; +17–18% decode at short context. See [`../results/longctx-reorder.csv`](../results/longctx-reorder.csv).

Build: oneAPI icpx 2026.0, SYCL backend, Intel Arc Pro B70 (Battlemage). WARP_SIZE 16.
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```
