---
license: apache-2.0
base_model: nex-agi/Nex-N2-mini
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
language:
  - en
tags:
  - gguf
  - llama.cpp
  - quantized
  - imatrix
  - moe
  - intel-arc
  - sycl
  - nex-n2
  - qwen3.5
  - iq3
---

<p align="center">
  <img src="https://huggingface.co/Frosty40/Nex-N2-mini-Turbo-Phase-Twin/resolve/main/cover.jpg" alt="Nex-N2-mini — Turbo Phase Twin" width="440">
</p>

# Nex-N2-mini — Turbo Phase Twin

A GGUF of **[Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini)** (Qwen3.5-35B-A3B MoE — ~3B active, multimodal reasoning) tuned for **Intel Arc**. One file carries **two expert precisions**; pick one at load with `--sm` (nothing is re-downloaded):

| `--sm` | precision | GPUs | resident | decode (B70) | quality (KLD vs Q6_K) |
|---|---|---|---|---:|---|
| **off** | IQ3_A770 3.19 bpw *(default)* | 1× 16 GB | ~15 GiB | **~85 tok/s** | 0.0547 / top-1 89.9% |
| **on** | Q4_K 4.5 bpw (`-sm layer`) | 2× 16 GB | ~18.8 GiB | bandwidth-positive | **0.0245 / 93.2% (near-lossless)** |

> ⚠️ **Requires a custom llama.cpp build** — codebook-free `IQ3_A770` type + a multi-precision loader, not in upstream. Stock llama.cpp can't load it. It's one patch on a pinned base; the quickstart builds it for you.

## Quickstart — Docker (needs only Docker + an Intel Arc GPU)

```bash
hf download Frosty40/Nex-N2-mini-Turbo-Phase-Twin --local-dir nexn2 && cd nexn2
bash build/docker-build.sh
docker run --rm -it --device /dev/dri -v "$PWD":/models -p 8090:8080 nexn2-turbo -m /models/Nex-N2-mini-Turbo-Phase-Twin.gguf -ngl 99 --jinja --cache-ram 0 --ctx-checkpoints 0
```

OpenAI-compatible API on `http://localhost:8090`. No host oneAPI needed — the compiler + Arc runtime live in the image.

- **`--cache-ram 0 --ctx-checkpoints 0` are required** — this hybrid linear-attention model goes incoherent on turn 2+ if prompt-cache / context-checkpoint restore is left on.
- **Two cards (Q4 phase):** add `-sm layer -ts 1,1 --override-kv general.tensor_variant.default=int:1`.
- **Bare-metal build, flags, troubleshooting:** [`build/BUILD.md`](build/BUILD.md).

## Running

- **Reasoning model:** emits a `<think>` trace — allow generous `max_tokens`. Sampling (Nex-AGI): `temperature 0.7, top_p 0.95, top_k 40`. Vision: add `--mmproj mmproj-f16.gguf`.
- **Launcher:** `./run-nx2.sh --sm auto` auto-detects GPU count and sets the right flags.
- **Elastic fit:** `./install-nx2.sh` detects VRAM and prunes the dual to a right-sized single file (or load-time `--override-kv general.tensor_variant.budget_mb=int:15900`).

## Credits & license

This is **[Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini) by [Nex-AGI](https://nex-agi.com)** (Apache-2.0), post-trained on **[Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)** (Qwen/Alibaba, Apache-2.0) — all the intelligence is theirs; this project only quantizes it to run fast on Arc.

- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** © ggml authors (MIT) — engine, quant framework, SYCL backend; the build patch is a derivative work.
- **k-quants** (ikawrakow & ggml authors, MIT) — `IQ3_A770` reuses Q3_K superblock packing; `Q6_K` covers non-expert tensors. NF4/QLoRA influenced the code-point design.
- **imatrix:** Bartowski's `calibration_datav3`. **Eval:** WikiText-2.

Released **Apache-2.0** (inherited from the base model); this project's own code is MIT. Full attribution in [`NOTICE`](NOTICE). Methodology & per-feature patches: **[github.com/newjordan/NexN2-B70-Turbo](https://github.com/newjordan/NexN2-B70-Turbo)**.
