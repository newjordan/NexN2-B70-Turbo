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

## Reorder is bit-safe on Battlemage
Greedy (`--temp 0`) decode of a fixed prompt is **token-identical** with the SYCL reorder optimization on (`GGML_SYCL_DISABLE_OPT=0`) vs off (`=1`) — only the t/s footer differs. The reorder layout is bit-safe for Q4_K / Q5_K decode on this card.

## Reorder-on-MoE: correctness
Turbo extends the dense SYCL reorder to the fused MoE expert GEMV (`mul_mat_id`) for Q4_K and Q5_K; depth-resolved A/B in `results/longctx-reorder.csv` (+17–18% @ctx0). Correctness is verified two ways: `test-backend-ops` MUL_MAT_ID against the CPU reference, and a server prefill-after-reorder differential — request 1 decode triggers the lazy reorder, request 2 prefills on the reordered weights; reorder-on output is byte-identical to reorder-off, including with `GGML_SYCL_PRIORITIZE_DMMV`. The patch covers every read path (decode GEMV, dense mmvq, DMMV, and the dequant-to-GEMM fallback) and adds a Q5_K reorder DMMV kernel for full-path parity.

## Accuracy reference
The KLD / PPL reference is **Q6_K** (27 GB, fits VRAM in full at `-ngl 99`), which is near-lossless. All candidates (Q5_K and below) are lower precision, so the comparison is valid.

## Long context + Flash-Attention
NexN2 is a hybrid: `full_attention_interval=4`, so layers 3, 7, 11, …, 39 are full softmax attention (head_count_kv=2, key/val_len=256) and the other 30 are Gated Delta Net (linear, constant SSM state). KV grows at ~**20 KiB/token** → ~1 GiB ≈ 52k tokens; the full 262144 context is ~5 GiB.

- **Flash-Attention is the depth win:** decode is attention-bound at depth, and FA attacks it directly — +26.7% @32k, +94.3% @131k. FA output is coherent and faithful on the B70 dGPU, quantified by PPL(FA on) ≈ PPL(FA off) = 6.7254 vs 6.7348 (within noise).
- **f16 KV** holds full decode speed at depth and is the recommended setting.
- **Prefill is the gating cost at extreme depth** (262k cold prefill ~19 min, O(n²)); prompt cache makes each agent turn cheap (delta-prefill only).
- **Retrieval verified:** needle-in-haystack 8/8 up to 120k at all depths. NexN2 is a reasoning model, so give retrieval probes generous `max_tokens` (≥ ~300) to leave room for the `<think>` trace.

## Notes
- Qwen3.5 **Gated Delta Net** (chunked linear attention) runs on the **CPU** in this backend; the MoE experts stay on the GPU — so decode benefits from CPU headroom.
- Convert with **transformers ≥ 5.x** (NexN2's tokenizer is `TokenizersBackend`).
- `GGML_SYCL_F16=ON` ≈ 2.4× prefill.
- imatrix: full GPU on Q6_K, 129 chunks, Bartowski `calibration_datav3` (covers all 256 experts).
