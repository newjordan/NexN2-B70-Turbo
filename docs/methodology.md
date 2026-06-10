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
`results/nx2-controls/20260610T220943Z/`, measured on the NX2 GGUF artifact and
Arc Pro B70:

| config | ctx0 decode | 131k decode |
|---|---:|---:|
| stock control: reorder off / FA off | 69.84 t/s | 20.10 t/s |
| deployed: Turbo + FA | 81.69 t/s | 40.88 t/s |

This is the reproducible project-level before/after: +17% at ctx0 and +103% at
131k.

Historical retained CSV data in `results/longctx-fa.csv` records stock control
55.43 t/s at ctx0 and 19.99 t/s at 131k; deployed Turbo + FA 81.26 t/s at ctx0
and 40.95 t/s at 131k. The 40.95 t/s number is not an old baseline; it is the
deployed 131k-context result.

The historical 55.43 t/s stock ctx0 value is retained in older CSV-only data
from `/home/frosty40/nx2-turbo/results/longctx-fa.csv`. A harness check in
`results/nx2-controls/ctx0-harness-check-20260610T231352Z/` reran the older
short `-n 32/-n 64` style and the current `-n 128 -r 5` style; all reran around
69.5-69.8 t/s. Use 69.x for reproducible stock ctx0 claims unless the original
55.43 raw command/log is recovered.

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
Greedy (`--temp 0`) decode is not used as primary correctness evidence for the SYCL reorder path. The retained FA-on greedy artifacts in `results/upstream-pr/` diverge across reorder/base variants, which is expected for tiny numeric differences amplified by argmax. Correctness claims here rely on `test-backend-ops` and perplexity instead.

## Reorder-on-MoE: correctness and scope
Turbo extends the dense SYCL reorder to the fused MoE expert GEMV (`mul_mat_id`) for Q4_K and Q5_K. Correctness holds under the retained 2026-06-10 validation artifacts (`results/upstream-pr/SUMMARY.md`): `test-backend-ops` MUL_MAT_ID 714/714 vs CPU reference, base and candidate full-suite backend-op logs have the same 12 `GET_ROWS` failures, and PPL is statistically identical (5.5643 vs 5.5662 ±0.152). The earlier "byte-identical" / token-identical wording was an overclaim beyond the retained evidence.

The project-level throughput numbers in this repo are measured on the NX2 GGUF variants, not on an abstract upstream model. Treat the contribution-candidate A/B in `results/upstream-pr/` as a narrow maintainer-readiness check for one llama.cpp code contribution, not as the benchmark story for the NX2 model artifacts.

## Accuracy reference
The KLD / PPL reference is **Q6_K** (27 GB, fits VRAM in full at `-ngl 99`), which is near-lossless. All candidates (Q5_K and below) are lower precision, so the comparison is valid.

## Long context + Flash-Attention
NexN2 is a hybrid: `full_attention_interval=4`, so layers 3, 7, 11, …, 39 are full softmax attention (head_count_kv=2, key/val_len=256) and the other 30 are Gated Delta Net (linear, constant SSM state). KV grows at ~**20 KiB/token** → ~1 GiB ≈ 52k tokens; the full 262144 context is ~5 GiB.

- **Flash-Attention is the depth win:** decode is attention-bound at depth, and FA attacks it directly — +26.7% @32k, +94.3% @131k. FA output is coherent and faithful on the B70 dGPU, quantified by PPL(FA on) ≈ PPL(FA off) = 6.7254 vs 6.7348 (within noise).
- **f16 KV** holds full decode speed at depth and is the recommended setting.
- **Prefill is the gating cost at extreme depth** (262k cold prefill ~19 min, O(n²)); prompt cache makes each agent turn cheap (delta-prefill only).
- **Retrieval verified:** needle-in-haystack 8/8 up to 120k at all depths. NexN2 is a reasoning model, so give retrieval probes generous `max_tokens` (≥ ~300) to leave room for the `<think>` trace.

## Long-context Pareto campaign (2026-06-10)
Full sweep in `results/niah-pareto.{csv,md,png}` — multi-needle RULER-style harness (`eval/niah/niah_sweep.py`), one prefill amortized over K depth probes via the server prompt cache (llama.cpp **context checkpoints** cover the Delta-Net recurrent state: probe ≥2 re-prefills only ~518 tokens). 155/155 PASS, 32k–520k, 7 configs.

- **Reliable native ceiling: 262144** (Q5_K_M, f16 KV) — 100% retrieval, 28 t/s decode at 257k.
- **512k works**: IQ4_XS + YaRN ×2 (`--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144` **plus `--override-kv qwen35moe.context_length=int:524288`** — llama-server otherwise caps the slot to n_ctx_train) + f16 KV: 100% at 520k (7 depths × 2 haystacks), 16.3 t/s decode, 27.9/31.9 GiB VRAM, ~65 min cold prefill.
- **q8_0 KV**: accuracy-free but 5–10× decode penalty on SYCL — measurement-only.
- YaRN vs linear ×2: no accuracy difference observed on this model.

## Notes
- Qwen3.5 **Gated Delta Net** (chunked linear attention) runs on the **CPU** in this backend; the MoE experts stay on the GPU — so decode benefits from CPU headroom.
- Convert with **transformers ≥ 5.x** (NexN2's tokenizer is `TokenizersBackend`).
- `GGML_SYCL_F16=ON` ≈ 2.4× prefill.
- imatrix: full GPU on Q6_K, 129 chunks, Bartowski `calibration_datav3` (covers all 256 experts).
