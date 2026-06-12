<img width="1254" height="1254" alt="nx2_b70_turbo_x" src="https://github.com/user-attachments/assets/0670218d-63e6-4fa1-94e0-ffc4f36c53e4" />



# NexN2 B70 Turbo

**The fastest + most accurate _local_ deployment of [Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini) on the Intel Arc Pro B70 (Battlemage)** — every number measured on the real model and the real card.

NexN2 is a **Qwen3.5-35B-A3B Mixture-of-Experts** reasoning model (35B total / ~3B active per token, 256 experts / 8 active, multimodal). **"Turbo"** is the full B70 deployment package: corrected GGUF artifacts, imatrix quant selection, the SYCL MoE reorder path, Flash-Attention serving, and retained validation.

Canonical report: [`docs/lab-report.md`](docs/lab-report.md).

## Headline

| config | decode @ ctx0 |
|---|---:|
| fresh stock control: reorder off / FA off | 69.8 t/s |
| **NexN2 B70 Turbo** (reorder + Flash-Attention) | **81.7 t/s** |
| fresh reproducible gain | **+17%** |

- **Turbo kernel path:** targeted `MUL_MAT_ID` op tests pass (714/714) and PPL is unchanged on the retained NX2 checks.
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

**Q5_K_M** is the all-rounder: best accuracy under Q6 and near-fastest decode. **Q4_K_M** is the max-speed option.

**The design follows two measured facts:**
1. **Speed comes from better kernels.** Q4_K/Q5_K use the optimized reorder-capable kernels: Q4_K_M (4.88 bpw) decodes at 85.7 t/s, ahead of the lower-bit IQ4_XS (4.32 bpw) at 52.8. The lever is the kernel — so that's where Turbo invests.
2. **The whole frontier sits far below the ~600 GB/s roofline → it's kernel-limited.** Exactly what Turbo targets.

## Turbo: reorder-on-MoE

NexN2's decode bottleneck is the fused MoE expert GEMV (`mul_mat_id`). Upstream llama.cpp has a SYCL "reorder" (Structure-of-Arrays weight layout) that the *dense* path uses; **Turbo extends it to the per-expert MoE GEMV** for Q4_K and Q5_K across every decode/prefill sub-path (decode GEMV, dense mmvq, DMMV, and the dequant-to-fp16/fp32 GEMM path), adding a Q5_K reorder DMMV kernel and per-expert reorder converters. Code: [`patches/0001`](patches/).

Clean A/B (`llama-bench`, reorder ON vs OFF) — [`results/longctx-reorder.csv`](results/longctx-reorder.csv):

| ctx depth | Q5_K_M off → on | gain | Q4_K_M off → on | gain |
|---|---|---:|---|---:|
| 0 | 70.07 → 82.84 | **+18.2%** | 74.53 → 87.45 | **+17.3%** |
| 4096 | 65.47 → 76.00 | +16.1% | 69.33 → 79.61 | +14.8% |
| 16384 | 55.65 → 62.59 | +12.5% | 58.28 → 65.26 | +12.0% |
| 32768 | 46.21 → 51.01 | +10.4% | 47.94 → 52.63 | +9.8% |

The retained checks show zero accuracy cost: `test-backend-ops` MUL_MAT_ID passes 714/714 vs the CPU reference and perplexity is unchanged (5.5643 vs 5.5662 ±0.15).

The llama.cpp backend contribution audit in [`results/upstream-pr/SUMMARY.md`](results/upstream-pr/SUMMARY.md) is validation detail for a reusable SYCL MoE reorder component. Turbo remains the full deployment, model artifact, runtime, and validation package measured on the NX2 GGUF variants.

## Serve it

```bash
./serving/llama-server.sh   # Q5_K_M + FA + f16 KV on 127.0.0.1:8090 (OpenAI-compatible)
```

Point any OpenAI client at `http://127.0.0.1:8090/v1`:
- **opencode:** drop [`serving/opencode.json`](serving/opencode.json) into `~/.config/opencode/`.
- **Hermes Agent:** see [`serving/hermes.md`](serving/hermes.md).

NexN2 is a **reasoning model** — it emits a `<think>` trace first, so give it generous `max_tokens`.

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout ac4cddeb0
git am /path/to/NexN2-B70-Turbo/patches/0001-*.patch
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```

## Reproduce the frontier

```bash
./scripts/nx2-phase1-sweep.sh   # quantize variants, measure KLD/PPL + t/s -> results/frontier.csv
```

See [`docs/methodology.md`](docs/methodology.md) for the ground-truth setup: bf16 → GGUF convert, the MTP/NextN metadata setting, the Q6_K reference, and the imatrix.

## Layout
- `patches/` — the reorder-on-MoE kernel (the "Turbo")
- `results/` — measured benchmarks (frontier, reorder-depth controls, serving controls, memory bakeoff)
- `serving/` — `llama-server` launcher + opencode / Hermes provider configs
- `scripts/` — the quant-sweep + measurement scripts
- `docs/` — lab report, methodology, the MTP load setting, hardware notes
- `eval/` — measurement harnesses: `memory/` (memory-substrate bakeoff), `upstream/` (A/B validation matrix for the Turbo patch vs upstream master), and retained experimental runners
- `mempalace/` — rolling research memory substrates: `lean/` (files + FTS5 + local embeddings + MemGPT-style controller) and `letta/` (Letta against local endpoints)

---
Hardware: Intel Arc Pro B70 (Battlemage, `8086:e223`), 32 GB GDDR6, ~600 GB/s, Ubuntu 24.04, oneAPI / SYCL (icpx 2026.0). Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT); the SYCL MoE reorder path was developed and validated locally on the B70.

## Credits & licenses

NexN2 B70 Turbo is an independent community project. Nex-N2-mini and Qwen are the works of their respective creators; the names below identify the upstream model and hardware and do not imply endorsement.

Model lineage (all **Apache-2.0**):
- **[Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)** — foundation model, © Qwen Team, Alibaba Cloud.
- **[Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini)** — © [Nex-AGI](https://nex-agi.com), post-trained on Qwen3.5-35B-A3B-Base. **The model these GGUFs are built from.**
- The **GGUF quants** ([HuggingFace](https://huggingface.co/Frosty40/Nex-N2-mini-B70-Turbo-GGUF)) are derivative works of Nex-N2-mini, redistributed under Apache-2.0. Changes: GGUF conversion, imatrix quantization, and the `qwen35moe` MTP/NextN metadata set to `block_count=40` / `nextn_predict_layers=0` for llama.cpp loading (lossless for standard inference — see [`docs/methodology.md`](docs/methodology.md)).

**Tooling:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) — the engine, and the SYCL backend that `patches/0001` derives from (original copyright retained). **Imatrix calibration:** Bartowski `calibration_datav3`. **Eval:** WikiText-2 (perplexity / KLD).

This repository's own code and docs are **MIT** ([LICENSE](LICENSE)); the full attribution chain and change notice are in [NOTICE](NOTICE). If you use this, please cite the upstream work:

```bibtex
@misc{qwen3.5,
  title  = {{Qwen3.5}: Towards Native Multimodal Agents},
  author = {{Qwen Team}},
  month  = {February},
  year   = {2026},
  url    = {https://qwen.ai/blog?id=qwen3.5}
}
```

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
