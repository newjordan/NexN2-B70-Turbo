<img width="1254" height="1254" alt="nx2_b70_turbo_x" src="https://github.com/user-attachments/assets/8c976d71-82e7-4a3c-bcc7-e26a1f774cdf" />



# NX2 B70 Turbo

**The fastest + most accurate _local_ deployment of [Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini) on the Intel Arc Pro B70 (Battlemage)** — measured on the real model, on the real card, with no shims.

NX2 is a **Qwen3.5-35B-A3B Mixture-of-Experts** reasoning model (35B total / ~3B active per token, 256 experts / 8 active, multimodal). **"Turbo"** is a custom SYCL kernel — *reorder-on-MoE* — that lifts the entire decode frontier at **zero accuracy cost**, paired with a tuned long-context serving profile.

> **Rule of the house: touch the real thing only.** Every number here is a real measurement from a real command on the real model and the real GPU — real KL-divergence vs a Q6_K reference, real wall-clock t/s on the card. No proxy metrics, no requantize-from-Q4 shortcuts, no estimates presented as results.

## Headline

| config | decode @ ctx0 | decode @ 131k |
|---|---:|---:|
| stock llama.cpp SYCL | 55.4 t/s | 20.0 t/s |
| **NX2 B70 Turbo** (reorder + Flash-Attention) | **81.3 t/s** | **41.0 t/s** |
| | **+47%** | **+105% (2.05×)** |

- **Turbo kernel (reorder-on-MoE):** +17–18% decode at short context (Q5_K_M 70.1 → 82.8 t/s; Q4_K_M 74.5 → 87.5 t/s) — token-identical greedy output, **zero accuracy cost**.
- **Flash-Attention:** +94% decode at 131k. Reorder and FA are complementary — reorder owns short context, FA owns deep context.
- **Long context is _usable_, not just allocatable:** 8/8 needle-in-haystack retrieval up to 120k tokens at every depth.
- Serves an **OpenAI-compatible** endpoint — drop-in for [opencode](serving/opencode.json) and [Hermes Agent](serving/hermes.md).

## The Pareto frontier (accuracy × speed)

Accuracy = KL-divergence + top-1 agreement of each quant's logits vs a **Q6_K reference** (PPL 6.572, wikitext-2 test, 100 chunks). Speed = `llama-bench` decode/prefill on the B70 (`-ngl 99`). imatrix from Bartowski `calibration_datav3` (full 129 chunks). Data: [`results/frontier.csv`](results/frontier.csv).

| quant | size | bpw | mean KLD ↓ | top-1 | **decode t/s** | prefill t/s |
|---|---:|---:|---:|---:|---:|---:|
| **Q5_K_M** ⭐ winner | 23.0 GB | 5.71 | **0.0201** | 94.0% | **81.7** | 1139 |
| **Q4_K_M** ⚡ fastest | 19.7 GB | 4.88 | 0.0389 | 91.6% | **85.7** | 1131 |
| Q4_K_dyn (Unsloth-style) | 21.3 GB | 5.27 | 0.0277 | 93.1% | 78.3 | 1136 |
| IQ4_XS | 17.4 GB | 4.32 | 0.0466 | 90.8% | 52.8 | 1105 |
| Q3_K_dyn | 17.1 GB | 4.24 | 0.0848 | 88.0% | 64.6 | 892 |
| Q3_K_M | 15.6 GB | 3.87 | 0.1048 | 86.3% | 62.1 | 874 |
| Q3_K_S | 14.1 GB | 3.50 | 0.1479 | 83.9% | 52.2 | 819 |

The frontier is two points; everything else is dominated. **Q5_K_M** is the all-rounder (best accuracy under Q6, near-fastest, ~9 GB free for context). **Q4_K_M** for max speed and more context headroom.

**Two findings drive the design:**
1. **Smaller is _not_ faster here.** IQ4_XS (4.32 bpw) decodes at 52.8 t/s; Q4_K_M (4.88 bpw — *bigger*) at 85.7. The SYCL i-quant / Q3_K kernels are unoptimized, so fewer bytes lose to a better kernel. **The path to speed is better kernels, not a smaller quant.**
2. **The whole frontier sits far below the ~600 GB/s roofline → it is kernel-limited.** That is exactly what Turbo attacks.

## Turbo: reorder-on-MoE

NX2's decode bottleneck is the fused MoE expert GEMV (`mul_mat_id`). Upstream llama.cpp has a SYCL "reorder" (Structure-of-Arrays weight layout) that the *dense* path uses, but the per-expert MoE path bails out of it. **Turbo extends reorder to the MoE expert GEMV** for Q4_K and Q5_K across every decode/prefill sub-path (decode GEMV, dense mmvq, DMMV, and the dequant-to-fp16/fp32 GEMM path), adding a Q5_K reorder DMMV kernel and per-expert reorder converters. Code: [`patches/0001`](patches/).

Clean A/B (`llama-bench`, reorder OFF = `GGML_SYCL_DISABLE_OPT=1` vs default ON) — [`results/longctx-reorder.csv`](results/longctx-reorder.csv):

| ctx depth | Q5_K_M off → on | gain | Q4_K_M off → on | gain |
|---|---|---:|---|---:|
| 0 | 70.07 → 82.84 | **+18.2%** | 74.53 → 87.45 | **+17.3%** |
| 4096 | 65.47 → 76.00 | +16.1% | 69.33 → 79.61 | +14.8% |
| 16384 | 55.65 → 62.59 | +12.5% | 58.28 → 65.26 | +12.0% |
| 32768 | 46.21 → 51.01 | +10.4% | 47.94 → 52.63 | +9.8% |

Gain decays with depth (the kernel speeds only the MoE GEMV; attention cost grows with depth and dilutes the relative win). **Zero accuracy cost:** greedy decode is token-identical reorder-on vs off, verified against the CPU reference via `test-backend-ops` (MUL_MAT_ID) and a server prefill-after-reorder differential.

## Long context: Flash-Attention + a tiny KV

NX2 is a **hybrid** model — only 10 of 40 layers are full attention (`full_attention_interval=4`); the rest are Gated Delta Net (linear). So KV grows at just **20 KiB/token**, and even the full 262144 context is ~5 GiB. Weights dominate VRAM, not KV.

FA decode by depth (Q5_K_M, f16 KV, reorder on) — [`results/longctx-fa.csv`](results/longctx-fa.csv):

| depth | FA off | FA on | gain |
|---|---:|---:|---:|
| 0 | 82.71 | 81.26 | −1.8% |
| 32768 | 50.97 | 64.57 | +26.7% |
| 131072 | 21.07 | 40.95 | **+94.3%** |
| 262144 | — | 26.80 | fits (f16 KV, no OOM) |

- **q8_0 KV rejected:** 4.7× *slower* than f16 at 131k (8.70 vs 40.95 t/s) — the SYCL FA + quantized-KV path is unoptimized.
- **The real ceiling is prefill, not VRAM.** Cold-prefilling 262k tokens takes ~19 min (O(n²) attention). For agents, prompt cache amortizes it: a 2-turn test reused 1303 cached prefix tokens and re-prefilled only 523 (delta).
- **Usable, not just allocatable:** needle-in-haystack **8/8 PASS** up to 120k at every depth, including 90% (near the 131072 serving limit) — [`results/niah.csv`](results/niah.csv).

## Serve it

```bash
./serving/llama-server.sh   # Q5_K_M + FA + 131072 ctx + f16 KV on 127.0.0.1:8090 (OpenAI-compatible)
```

Point any OpenAI client at `http://127.0.0.1:8090/v1`:
- **opencode:** drop [`serving/opencode.json`](serving/opencode.json) into `~/.config/opencode/`.
- **Hermes Agent:** see [`serving/hermes.md`](serving/hermes.md).

NX2 is a **reasoning model** — it emits a `<think>` trace first, so give it generous `max_tokens`.

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout f0156d1
git am /path/to/nx2-b70-turbo/patches/0001-*.patch
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```

## Reproduce the frontier

```bash
./scripts/nx2-phase1-sweep.sh   # quantize candidates, measure KLD/PPL + t/s -> results/frontier.csv
```

See [`docs/methodology.md`](docs/methodology.md) for the ground-truth setup: bf16 → GGUF convert, the **MTP/NextN metadata fix** (the one bug that blocks loading NX2 at all), the Q6_K reference, and the imatrix.

## Layout
- `patches/` — the reorder-on-MoE kernel (the "Turbo")
- `results/` — measured benchmarks (frontier, reorder × depth, FA × depth, needle-in-haystack)
- `serving/` — `llama-server` launcher + opencode / Hermes provider configs
- `scripts/` — the quant-sweep + measurement harness
- `docs/` — methodology, the MTP load fix, hardware notes

---
Hardware: Intel Arc Pro B70 (Battlemage, `8086:e223`), 32 GB GDDR6, ~600 GB/s, Ubuntu 24.04, oneAPI / SYCL (icpx 2026.0). Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). The kernel work is local to this project and is not submitted upstream.

```text
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠚⠓⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⡾⣅⠀⠀⠀⠀⣨⢷⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣀⡤⡦⣄⡀⠀⡇⠀⠙⢢⡔⠋⠀⢸⠀⢀⣠⢴⢤⣀⠀⠀
⡴⠊⠁⠀⢷⠀⠉⠲⣇⠀⠀⢸⡇⠀⠀⣸⠖⠉⠀⡼⠀⠈⠑⢦
⣧⠀⠀⢀⣸⡀⠀⠀⢿⠙⠲⢼⣧⠖⠋⡿⠀⠀⢀⣇⡀⠀⠀⣸
⢹⡴⠚⠉⠀⠈⠑⠦⣼⠀⠀⢸⡇⠀⠀⣧⡴⠊⠁⠀⠉⠓⢦⡏
⠀⠈⠓⢤⣀⡤⠖⠋⠁⠙⠲⣼⣧⠖⠋⠈⠙⠲⢤⣀⡤⠚⠁⠀
⢀⡠⠖⠉⠀⠉⠓⠦⣄⠴⠚⢹⡏⠓⠦⣠⡴⠚⠉⠀⠉⠲⢄⡀
⣼⠙⠲⢤⣀⡠⠔⠋⢹⠀⠀⣸⣇⠀⠀⣏⠙⠲⢄⣀⡤⠖⠋⣧
⡏⠀⠀⠀⢸⠀⠀⠀⣿⠴⠚⢹⡏⠓⠦⣿⠀⠀⠀⡇⠀⠀⠀⢸
⠙⠢⣄⡀⡟⣀⡤⠚⡇⠀⠀⢸⡇⠀⠀⢸⠓⠤⣀⢹⢀⣠⠔⠋
⠀⠀⠀⠉⠋⠁⠀⠀⡇⣀⠴⠊⠑⠦⣀⢸⠀⠀⠈⠙⠉⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠻⣅⠀⠀⠀⠀⣨⠟⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠲⠖⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

