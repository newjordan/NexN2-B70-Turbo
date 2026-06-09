# Methodology & findings (hard-won)

All measurements are on the real Nex-N2-mini model and the real Intel Arc Pro B70. No proxy metrics, no requantize-from-Q4 shortcuts.

## Hardware
Intel Arc Pro B70 (Battlemage, PCI `8086:e223`), 32 GB GDDR6 ECC, 256-bit, ~600 GB/s (measured ~598 read / ~525 copy). 32 Xe2 cores, 256 XMX engines @ 2.8 GHz. XMX peak ~367 TOPS int8 / ~183 TFLOPS fp16. Ubuntu 24.04, oneAPI (icpx 2026.0), llama.cpp SYCL backend. Subgroup / WARP_SIZE = **16** on Intel.

## The MTP / NextN load bug (load NexN2 at all)
NexN2's convert writes `qwen35moe.block_count=41` and `qwen35moe.nextn_predict_layers=1`. Block 40 is an MTP (multi-token-prediction, speculative) layer with **zero** standard tensors. The runtime loader uses `n_layer = block_count - nextn_predict_layers` and then expects MTP tensors (`blk.i.nextn.eh_proj` etc.) on the trailing layer — which the checkpoint does not ship.

| block_count / nextn | n_layer | result |
|---|---|---|
| 41 / 1 (original) | 40 | FAIL: missing `blk.40.attn_norm.weight` |
| 40 / 1 (block_count alone) | 39 | FAIL: missing `blk.39.nextn.eh_proj.weight` |
| **40 / 0 (the fix)** | 40 | **loads, runs correctly** |

Fix (lossless for standard inference — the MTP head is speculative-only and absent here):
```bash
python gguf-py/gguf/scripts/gguf_set_metadata.py NexN2.gguf qwen35moe.block_count 40 --force
python gguf-py/gguf/scripts/gguf_set_metadata.py NexN2.gguf qwen35moe.nextn_predict_layers 0 --force
```
Or convert with `--no-mtp`. Derived quants inherit the metadata, so fix the bf16 source once. (`llama-bench` rejects `--override-kv`, so prefer fixing metadata.)

## Smaller ≠ faster on SYCL
Decode is **kernel-limited**, not bandwidth-limited, on this stack: IQ4_XS (4.32 bpw) = 52.8 t/s vs Q4_K_M (4.88 bpw) = 85.7 t/s. The mainline SYCL i-quant and Q3_K kernels are unoptimized; Q4_K / Q5_K have the good (reorder-capable) kernels. Choose Q4_K / Q5_K for speed; chase better kernels (reorder-on-MoE) for the next jump, not a smaller quant.

## Reorder safety on Battlemage
llama.cpp issue #21893 reports reorder corrupting weights on Battlemage with the optimization enabled. We tested it directly: greedy (`--temp 0`) decode of a fixed prompt with `GGML_SYCL_DISABLE_OPT=1` (reorder off) vs `=0` (reorder on) produced **token-identical** output (only the t/s footer differed). So reorder is bit-safe here for Q4_K / Q5_K decode. Re-validate after any kernel change, including a decode-then-prefill workload — `test-backend-ops` only covers the n=1 fused path.

## Reorder-on-MoE: verified
Extended the dense SYCL reorder to the fused MoE expert GEMV (`mul_mat_id`) for Q4_K and Q5_K. See `results/longctx-reorder.csv` for the depth-resolved A/B (+17–18% @ctx0, decaying to +10% @32k).

Correctness (the decode-then-prefill validation #21893 flagged): `test-backend-ops` MUL_MAT_ID vs CPU ref, plus a server **prefill-after-reorder differential** — req1 decode triggers the lazy reorder, req2 prefills on the reordered weights; reorder-ON output is byte-identical to reorder-OFF, including with forced `GGML_SYCL_PRIORITIZE_DMMV`. The "Q4_K / Q5_K only" guard is load-bearing: a reordered expert can reach the DMMV fallback (a single-routed-row expert during a 2nd-request prefill); a missing Q5_K reorder DMMV kernel there would silently corrupt, so one was written for full-path parity.

## Reference choice
The KLD reference is **Q6_K** (27 GB, fits VRAM full at `-ngl 99`), not Q8_0 (37 GB) — Q8_0 at `-ngl 32` partial-offload OOM-killed mid-imatrix on this 32 GB card. Q6_K is near-lossless; all candidates (Q5_K and below) are lower precision, so the comparison is valid. A pure bf16/Q8_0 reference would need a slow CPU run for the last ~0.002 KLD.

## Long context + Flash-Attention
NexN2 is a hybrid — `full_attention_interval=4`, so only layers 3, 7, 11, …, 39 are full softmax attention (head_count_kv=2, key/val_len=256); the other 30 are Gated Delta Net (linear, constant SSM state). KV grows at just **20 KiB/token** → ~1 GiB ≈ 52k tokens; the full 262144 context is only ~5 GiB.

- **Flash-Attention is the depth win:** decode is attention-bound at depth (reorder speeds only the MoE GEMV). FA attacks attention directly — +26.7% @32k, +94.3% @131k, −1.8% @ctx0. Correct on the B70 dGPU: FA on vs off is not bit-identical (FA reorders float accumulation, so greedy CoT diverges benignly) but both outputs are coherent and faithful — quantified by PPL(FA on) ≈ PPL(FA off) (6.7254 vs 6.7348, within noise; not the #19276 corruption).
- **q8_0 KV rejected:** 4.7× slower than f16 @131k (8.70 vs 40.95 t/s). The SYCL FA + quantized-KV path is unoptimized; never use it here.
- **Prefill is the real ceiling:** 262k cold prefill ~19 min (O(n²)). For loop agents, prompt cache makes each turn cheap (delta-prefill only).
- **Usable, not just allocatable:** needle-in-haystack 8/8 PASS up to 120k at all depths. **Measurement lesson:** NexN2 is a reasoning model (`<think>` consumes the token budget) — NIAH grading needs `max_tokens >= ~300` or the answer truncates and reads as a false FAIL.

## Other
- Qwen3.5 **Gated Delta Net** (chunked linear attention) is unsupported on SYCL → those 30 layers run on the **CPU**. The MoE experts stay on GPU. (Consequence: decode is CPU-sensitive — keep CPU headroom for it.)
- Convert needs **transformers ≥ 5.x** (NexN2 tokenizer is `TokenizersBackend`).
- `GGML_SYCL_F16=ON` ≈ 2.4× prefill, no decode change.
- imatrix: full GPU on Q6_K, 129 chunks, Bartowski `calibration_datav3` (covers the 256 experts).
