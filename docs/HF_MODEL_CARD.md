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
  <img src="https://github.com/user-attachments/assets/0670218d-63e6-4fa1-94e0-ffc4f36c53e4" alt="NexN2 B70 Turbo" width="440">
</p>

# NexN2 B70 Turbo — Nex-N2-mini GGUF

GGUF + importance-matrix quantizations of **[nex-agi/Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini)** (Qwen3.5-35B-A3B MoE, ~3B active, multimodal reasoning), tuned and measured for fast **local** inference on the **Intel Arc Pro B70** (Battlemage) via llama.cpp's SYCL backend.

> ⚠️ **Unofficial community quantization.** Not affiliated with, endorsed by, or sponsored by Nex-AGI, the Qwen Team / Alibaba, or Intel. The base model is **© Nex-AGI** (Apache-2.0), post-trained on **Qwen3.5-35B-A3B-Base © Qwen Team** (Apache-2.0). These quants are redistributed under Apache-2.0.

Code, kernel patch, full methodology, and reproducible benchmarks: **https://github.com/newjordan/NexN2-B70-Turbo**

## Files

| file | quant | size | bpw | mean KLD ↓ vs Q6_K | top-1 | decode t/s (B70) |
|---|---|---:|---:|---:|---:|---:|
| `Nex-N2-mini-B70-Turbo-Q5_K_M.gguf` ⭐ | Q5_K_M | 23.0 GB | 5.71 | **0.0201** | 94.0% | **81.7** |
| `Nex-N2-mini-B70-Turbo-Q4_K_M.gguf` ⚡ | Q4_K_M | 19.7 GB | 4.88 | 0.0389 | 91.6% | **85.7** |
| `Nex-N2-mini-B70-Turbo-Q4_K_dyn.gguf` | Q4_K_dyn | 21.3 GB | 5.27 | 0.0277 | 93.1% | 78.3 |
| `Nex-N2-mini-B70-Turbo-IQ4_XS.gguf` | IQ4_XS | 17.4 GB | 4.32 | 0.0466 | 90.8% | 52.8 |
| `Nex-N2-mini-B70-Turbo-Q3_K_dyn.gguf` | Q3_K_dyn | 17.1 GB | 4.24 | 0.0848 | 88.0% | 64.6 |
| `Nex-N2-mini-B70-Turbo-Q3_K_M.gguf` | Q3_K_M | 15.6 GB | 3.87 | 0.1048 | 86.3% | 62.1 |
| `Nex-N2-mini-B70-Turbo-Q3_K_S.gguf` | Q3_K_S | 14.1 GB | 3.50 | 0.1479 | 83.9% | 52.2 |
| `mmproj-f16.gguf` | vision projector | 0.8 GB | — | — | — | — |

**Recommendation: `Q5_K_M`** (best accuracy under the Q6_K reference, near-fastest) or **`Q4_K_M`** for max speed / more context headroom. On SYCL, smaller is *not* faster — IQ4_XS/Q3_K use unoptimized kernels; Q4_K/Q5_K win. Accuracy is KL-divergence + top-1 agreement vs a Q6_K reference (PPL 6.572, wikitext-2, 100 chunks). For vision, add `mmproj-f16.gguf`.

## Benchmarks (Intel Arc Pro B70, real measurements)

With the project's reorder-on-MoE "Turbo" kernel + Flash-Attention:

| config | decode @ ctx0 | decode @ 131k |
|---|---:|---:|
| stock llama.cpp SYCL | 55.4 t/s | 20.0 t/s |
| **NexN2 B70 Turbo** (reorder + FA) | **81.3 t/s** | **41.0 t/s** |

Long context is *usable*, not just allocatable: needle-in-haystack 8/8 up to 120k tokens at every depth. Full tables (reorder × depth, FA × depth, NIAH) and the kernel patch are in the [GitHub repo](https://github.com/newjordan/NexN2-B70-Turbo).

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
- Weights are not otherwise altered beyond quantization.

## License & attribution

Released under the **Apache License 2.0**, inherited from the base model. You must retain the attribution above. See [`NOTICE`](https://github.com/newjordan/NexN2-B70-Turbo/blob/main/NOTICE) for the full attribution chain (Qwen → Nex-AGI → these quants; plus llama.cpp/MIT).

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
