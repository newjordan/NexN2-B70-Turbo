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
---

<p align="center">
  <img src="https://huggingface.co/Frosty40/Nex-N2-mini-B70-Turbo-GGUF/resolve/main/cover.jpg" alt="NexN2 B70 Turbo" width="440">
</p>

# NexN2 B70 Turbo — Nex-N2-mini GGUF

GGUF + importance-matrix quantizations of **[Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini)** — the Qwen3.5-35B-A3B MoE (~3B active, multimodal reasoning) **by the [Nex-AGI](https://nex-agi.com) team** — tuned and measured for fast **local** inference on the **Intel Arc Pro B70** (Battlemage) via llama.cpp's SYCL backend.

**The model is [Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini), built by the [Nex-AGI](https://nex-agi.com) team** — the reasoning, multimodal capability, and training are entirely theirs; this is an independent community quantization that only repackages their model to run fast on Intel Arc, with sincere thanks to Nex-AGI for releasing it openly. The base model is **© Nex-AGI** (Apache-2.0), post-trained on **Qwen3.5-35B-A3B-Base © Qwen Team** (Apache-2.0); these quants are redistributed under Apache-2.0 and the names identify the upstream work.

Code, kernel patch, full methodology, and reproducible benchmarks: **https://github.com/newjordan/NexN2-B70-Turbo**

## Files

| file | quant | size | bpw | mean KLD ↓ vs Q6_K | top-1 | decode t/s (B70) |
|---|---|---:|---:|---:|---:|---:|
| `Nex-N2-mini-B70-Turbo-Q5_K_M.gguf` ⭐ | Q5_K_M | 23.0 GB | 5.71 | **0.0201** | 94.0% | **88.1** |
| `Nex-N2-mini-B70-Turbo-Q4_K_M.gguf` ⚡ | Q4_K_M | 19.7 GB | 4.88 | 0.0389 | 91.6% | **93.5** |
| `Nex-N2-mini-B70-Turbo-Q4_K_dyn.gguf` | Q4_K_dyn | 21.3 GB | 5.27 | 0.0277 | 93.1% | 78.3 |
| `Nex-N2-mini-B70-Turbo-IQ4_XS.gguf` | IQ4_XS | 17.4 GB | 4.32 | 0.0466 | 90.8% | 52.8 |
| `Nex-N2-mini-B70-Turbo-Q3_K_dyn.gguf` | Q3_K_dyn | 17.1 GB | 4.24 | 0.0848 | 88.0% | 64.6 |
| `Nex-N2-mini-B70-Turbo-Q3_K_M.gguf` | Q3_K_M | 15.6 GB | 3.87 | 0.1048 | 86.3% | 62.1 |
| `Nex-N2-mini-B70-Turbo-Q3_K_S.gguf` | Q3_K_S | 14.1 GB | 3.50 | 0.1479 | 83.9% | 52.2 |
| `mmproj-f16.gguf` | vision projector | 0.8 GB | — | — | — | — |

**Recommendation: `Q5_K_M`** (best accuracy under the Q6_K reference, near-fastest) or **`Q4_K_M`** for max speed. On SYCL, Q4_K / Q5_K carry the optimized kernels and lead the decode column — they're the speed picks. Accuracy is KL-divergence + top-1 agreement vs a Q6_K reference (PPL 6.572, wikitext-2, 100 chunks). For vision, add `mmproj-f16.gguf`.

## Benchmarks (Intel Arc Pro B70, real measurements)

With the project's Turbo package on Arc Pro B70:

| config | decode @ ctx0 | decode @ 131k |
|---|---:|---:|
| fresh stock control: reorder off / FA off | 68.8 t/s | 20.0 t/s |
| **NexN2 B70 Turbo Q5_K_M** (accuracy pick; MoE reorder + FA + retained NX2 fusions) | **88.1 t/s** | **42.3 t/s** |
| **NexN2 B70 Turbo Q4_K_M** (speed pick; MoE reorder + retained NX2 fusions) | **93.5 t/s** | not remeasured |

Q5_K_M/Q4_K_M decode updated 2026-06-12 with the Turbo patch chain's Q6_K MoE reorder
(`patches/0003` in the GitHub repo). `patches/0004` adds default-on exact NX2
fused gate/up SwiGLU, post-down weighted-sum fusion, and fused F32 MoE tail add
(Q5_K_M ctx0 reaches 88.1 t/s in the retained default; Q4_K_M reaches 93.5 t/s
on the release-card evidence). The 131k Q5_K_M table entry is
a full-package measurement, not a fused-MoE-only attribution: the audited release
gate measured `0004` at 42.0952 -> 42.3148 t/s (+0.52%) for Q5_K_M 131k. The
large 20.0 -> 42.3 t/s 131k change is the overall Turbo deployment comparison
against the fresh stock control.

Broader generic gate/up fusion experiments remain opt-in profiling modes. Shared-expert
dense gate/up, activation-Q8 cache, gate/up XOR reduction, vec4 weighted-sum,
local-weight weighted-sum, down-weighted-sum, atomic down-weighted-sum, SwiGLU
activation variants, gate/up rowpack scheduling, gate/up local-Q8 activation
caching, gate/up expert-pack scheduling, gate/up dual-dot vecdot, weighted-tail, dispatch guard, gate/up Q8 handoff, exact Q6 down specialization, shared-gate-tail,
shared-gate-tail local-gate broadcast, shared-gate-sigmoid-tail, tail-add vec4,
tail-add+RMS_NORM, RMS_NORM+MUL, post-norm Q8 handoff, and selective post-norm
Q8 handoff probes are also opt-in only after failing to beat the retained default.

Accuracy guardrail for the retained kernel path: Q5_K_M 30-chunk WikiText PPL
showed no measured loss in the fresh release-gate run (5.5682 +/- 0.15248
baseline vs 5.5676 +/- 0.15244 candidate), and targeted `MUL_MAT_ID` backend
ops passed 690/690.

## Run it (llama.cpp)

```bash
# build llama.cpp with the SYCL backend (oneAPI/icpx); see the GitHub repo for the Turbo kernel patch
llama-server -m Nex-N2-mini-B70-Turbo-Q5_K_M.gguf -ngl 99 -fa on -c 131072 \
  -ctk f16 -ctv f16 --jinja --host 127.0.0.1 --port 8090
```

OpenAI-compatible endpoint at `http://127.0.0.1:8090/v1`. **Nex-N2-mini is a reasoning model** — it emits a `<think>` trace, so give it generous `max_tokens`. Native recommended sampling (per Nex-AGI): `temperature 0.7, top_p 0.95, top_k 40`.

## Provenance & changes (Apache-2.0 §4(b))

- **Base model:** [nex-agi/Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini) (Apache-2.0), post-trained on [Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) (Apache-2.0).
- **Changes from the original:** converted to GGUF; quantized with a llama.cpp importance matrix (imatrix calibrated on Bartowski `calibration_datav3`); set GGUF metadata `qwen35moe.block_count=40` and `qwen35moe.nextn_predict_layers=0` so the model loads in llama.cpp (the MTP/NextN head is speculative-only and absent from the checkpoint — lossless for standard inference).
- The weights are quantizations of the original; no other modifications.

## License & attribution

⭐ **Built on [Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini) by the [Nex-AGI](https://nex-agi.com) team** — their model, their intelligence; this project only quantizes it for Intel Arc. → [nex-agi.com](https://nex-agi.com) · [GitHub](https://github.com/nex-agi/Nex-N2) · [HuggingFace](https://huggingface.co/nex-agi)

Released under the **Apache License 2.0**, inherited from the base model. Retain the attribution above. See [`NOTICE`](https://github.com/newjordan/NexN2-B70-Turbo/blob/main/NOTICE) for the full attribution chain (Qwen → Nex-AGI → these quants; plus llama.cpp / MIT).

```bibtex
@misc{qwen3.5,
  title  = {{Qwen3.5}: Towards Native Multimodal Agents},
  author = {{Qwen Team}},
  month  = {February},
  year   = {2026},
  url    = {https://qwen.ai/blog?id=qwen3.5}
}
```
Please also credit **Nex-AGI's Nex-N2-mini** (https://github.com/nex-agi/Nex-N2) as the base model.
