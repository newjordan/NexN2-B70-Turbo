# Methodology

All measurements are on the real Nex-N2-mini model and the real Intel Arc Pro B70.

## Hardware
Intel Arc Pro B70 (Battlemage, PCI `8086:e223`), 32 GB GDDR6 ECC, 256-bit, ~600 GB/s (measured ~598 read / ~525 copy). 32 Xe2 cores, 256 XMX engines @ 2.8 GHz. XMX peak ~367 TOPS int8 / ~183 TFLOPS fp16. Ubuntu 24.04, oneAPI (icpx 2026.0), llama.cpp SYCL backend. Subgroup / WARP_SIZE = **16** on Intel.

## Loading NexN2 in llama.cpp (MTP / NextN metadata)
NexN2 carries an MTP (multi-token-prediction) block in its metadata (`qwen35moe.block_count=41`, `qwen35moe.nextn_predict_layers=1`); that head is speculative-only and is absent from the published checkpoint. Point llama.cpp at the 40 standard layers by setting:

```bash
python gguf-py/gguf/scripts/gguf_set_metadata.py NexN2.gguf qwen35moe.block_count 40 --force
python gguf-py/gguf/scripts/gguf_set_metadata.py NexN2.gguf qwen35moe.nextn_predict_layers 0 --force
```
(or convert with `--no-mtp`). This is lossless for standard inference and derived quants inherit it — **the published GGUFs already have it applied**, so they load out of the box.

## Speed comes from the kernel
Decode is kernel-limited on this stack: Q4_K_M (4.88 bpw) decodes at 85.7 t/s while the lower-bit IQ4_XS (4.32 bpw) runs at 52.8 — Q4_K / Q5_K carry the optimized, reorder-capable kernels. The lever for more speed is a better kernel (reorder-on-MoE), which is what Turbo delivers.

## Project control data
The fresh reproducible control comparison is the Q5_K_M rerun in
`results/nx2-controls/20260612T052635Z/`, measured on the NX2 GGUF artifact and
Arc Pro B70 with the full patch chain (`0001`–`0003`, including the Q6_K MoE
reorder):

| config | ctx0 decode |
|---|---:|
| stock control: reorder off / FA off | 68.84 t/s |
| deployed: Turbo + FA | 85.48 t/s |

This is the reproducible project-level before/after: +24% at ctx0. (The prior
0001-only control, +17%, is retained in `results/nx2-controls/20260610T220943Z/`.)

NX2 path-debug evidence was retained in
`results/nx2-path-debug/20260610T223353Z/`. Q4_K_M and Q5_K_M both reach SYCL
`ggml_sycl_mul_mat_id` with 3D expert tensors:

```text
Q4_K_M: blk.0.ffn_gate_exps.weight type=q4_K ne=[2048, 512, 256, 1]
Q5_K_M: blk.0.ffn_gate_exps.weight type=q5_K ne=[2048, 512, 256, 1]
```

Model checksums are retained in `results/model-checksums.sha256`.

First-token timing is retained in `results/nx2-first-token/20260610T230504Z/`.
The cleaned-branch Q5_K_M control did not show a first-token lazy-reorder
penalty under `-fa on`, `-fit off`, `--no-warmup`; median prompt/first-token
eval was 1.12 s with reorder off and 1.05 s with reorder on. This is a whole
optimization-path control, not isolated single-line attribution.

## Reorder correctness checks
Greedy (`--temp 0`) decode is not primary correctness evidence for the SYCL reorder path. The retained FA-on greedy artifacts in `results/upstream-pr/` diverge across reorder variants, which is expected for small numerical differences amplified by argmax. Correctness evidence here relies on `test-backend-ops` and perplexity.

## Reorder-on-MoE: correctness and scope
Turbo extends the dense SYCL reorder to the fused MoE expert GEMV (`mul_mat_id`) for Q4_K and Q5_K. Correctness holds under the retained 2026-06-10 validation artifacts (`results/upstream-pr/SUMMARY.md`): `test-backend-ops` MUL_MAT_ID 714/714 vs CPU reference, the unpatched upstream and validated branch full-suite backend-op logs have the same 12 `GET_ROWS` failures, and PPL is statistically identical (5.5643 vs 5.5662 +/- 0.152). The earlier "byte-identical" / token-identical wording exceeded the retained evidence.

The project-level throughput numbers in this repo are measured on the NX2 GGUF variants, not on an abstract upstream model. The backend contribution audit in `results/upstream-pr/` is validation detail for one reusable SYCL component, separate from the NX2 model artifact benchmark.

## Accuracy reference
The KLD / PPL reference is **Q6_K** (27 GB, fits VRAM in full at `-ngl 99`), which is near-lossless. All tested quants (Q5_K and below) are lower precision, so the comparison is valid.

## Notes
- Qwen3.5 **Gated Delta Net** (chunked linear attention) runs on the **CPU** in this backend; the MoE experts stay on the GPU — so decode benefits from CPU headroom.
- Convert with **transformers ≥ 5.x** (NexN2's tokenizer is `TokenizersBackend`).
- `GGML_SYCL_F16=ON` ≈ 2.4× prefill.
- imatrix: full GPU on Q6_K, 129 chunks, Bartowski `calibration_datav3` (covers all 256 experts).
