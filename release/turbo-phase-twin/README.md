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

**One file. Two precision phases. Flip `--sm`.**

A single GGUF of **[Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini)** — the
Qwen3.5-35B-A3B MoE (~3B active, multimodal reasoning) **by the
[Nex-AGI](https://nex-agi.com) team** — that carries **two expert precisions at once** and
hot-swaps between them at load with one flag:

| `--sm` | phase | GPUs | experts | resident | decode (B70) | accuracy |
|---|---|---|---|---|---:|---|
| **off** | **IQ3** (default) | 1× 16 GB | IQ3_A770 **3.19 bpw** (codebook-free) | ~15.0 GiB | **~82 t/s** | KLD 0.0547 / top-1 89.9% |
| **on** | **Q4** | 2× 16 GB | Q4_K **4.5 bpw**, split `-sm layer` | ~18.8 GiB | bandwidth-positive¹ | **KLD 0.0245 / top-1 93.2% (near-lossless)** |

The two phases are *twins* in one ~34 GiB file: the loader keeps only the phase you select
and reads only its bytes, so single-card users pay for 15 GiB and two-card users pay for the
Q4 split — never both. **`--sm off` runs the whole model on one 16 GB Arc A770 at ~82 tok/s.**

> ⚠️ **This model requires a custom llama.cpp build** (codebook-free `IQ3_A770` type +
> the multi-precision loader). **Stock llama.cpp cannot load it** — it will report an unknown
> tensor type. The build is **one patch on a pinned base**; see **[Install](#install)** below.
> Everything you need is in this repo.

---

## What it is

`Nex-N2-mini` is a 256-expert MoE (8 active/token, ~34.7 B total / ~3 B active, hybrid
linear-attention, multimodal). Experts are **92.9%** of the weights, so the whole speed/size/
accuracy story is about how the experts are quantized.

This "Phase Twin" packages two answers in one file:

- **IQ3 phase** — a bespoke **codebook-free 3-bit** expert quant (`IQ3_A770`, 3.19 bpw) built
  to fit the entire model on a **single Arc A770 16 GB** and decode fast through a fused,
  reordered dp4a MoE path. This is the headline: a 35B-class MoE, one consumer card, ~82 t/s.
- **Q4 phase** — the same experts at **Q4_K 4.5 bpw**, meant to be split across **two** cards
  (`-sm layer`). It is *near-lossless* (within 0.24% perplexity of the Q6_K reference) **and**
  faster per card, because splitting halves the bytes each GPU reads per token.

You pick the phase with `--sm`; nothing is re-downloaded or re-quantized.

¹ **Why two cards is also *faster*, not just more accurate.** Decode reads the active experts
(8 of 256) each token; split across two cards the per-card bytes ≈ `(active/2) × bpw`. The
break-even vs the one-card 3.19 bpw base is ~6.4 bpw — so **Q4_K (4.5) reads fewer bytes per
card than the 1-card 3-bit base**: the split buys accuracy *and* relieves decode bandwidth.
(Single-card B70 decode is measured; two-card throughput is bandwidth-*projected* — validate
on real 2-GPU hardware.)

## Quality (KLD vs a Q6_K reference, wikitext-2, 100×512 tok)

| phase | bpw | mean KLD ↓ | top-1 ↑ | PPL(Q)/PPL(ref) |
|---|---:|---:|---:|---:|
| IQ3 (1-card) | 3.19 | 0.0547 | 89.91% | 1.0163 |
| **Q4 (2-card)** | 4.5 | **0.0245** | **93.24%** | **1.0024** |

The Q4 phase **more than halves KLD (−55%)** and sits **within 0.24% PPL of the full Q6_K
base — effectively lossless** — while top-1 agreement rises 89.9% → 93.2%.

## Speed (Intel Arc Pro B70, A770-proxy — measured, single-card)

The 1-card IQ3 decode rests on a fused, reordered dp4a MoE path. Measured back-to-back,
same binary, prefill-matched (so it's the kernel, not scheduler contention):

| reorder path | pp512 (control) | **tg128** |
|---|---:|---:|
| dormant (fallback) | 599.7 | 43.71 ± 0.04 |
| **live (this build)** | 600.2 | **82.3 ± 0.05** |

**+88% decode** from the fused reorder stack, reproducible to 0.2%.

---

## Install

You need llama.cpp built with the **SYCL** backend (Intel oneAPI) and this repo's single
patch. One base commit, one patch — the result is the exact validated code.

```bash
# 0) Intel oneAPI (icpx + MKL/DPCPP) installed and on PATH:
source /opt/intel/oneapi/setvars.sh

# 1) llama.cpp at the pinned base + this repo's patch
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout f0156d1401500512ad85042ccf38970568b12253
git apply /path/to/llama.cpp-turbo-phase-twin.patch     # from this repo's build/

# 2) build (SYCL)
cmake -B build -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx
cmake --build build -j --target llama-server llama-cli llama-bench
```

`build/build.sh` does all of the above in one shot. Full methodology, the per-feature patch
stack (0001–0010), and reproducible benchmarks live at
**https://github.com/newjordan/NexN2-B70-Turbo**.

## Operation

Use the included launcher (auto-detects GPU count) — it picks the phase and adds the right
flags for you:

```bash
./run-nx2.sh --sm auto                  # 2+ GPUs → Q4 split; else 1-card IQ3
./run-nx2.sh --sm off -- -p "Hello" -n 64     # force 1-card IQ3
NX2_TOOL=llama-server ./run-nx2.sh --sm on --port 8090   # force 2-card Q4
```

Or drive llama.cpp directly:

```bash
# IQ3 phase (default, 1 card)
llama-server -m Nex-N2-mini-Turbo-Phase-Twin.gguf -ngl 99 --port 8090 --jinja

# Q4 phase (2 cards) — select variant 1 + split layer-wise
llama-server -m Nex-N2-mini-Turbo-Phase-Twin.gguf -ngl 99 -sm layer -ts 1,1 \
  --override-kv general.tensor_variant.default=int:1 --port 8090 --jinja
```

OpenAI-compatible endpoint at `http://127.0.0.1:8090/v1`. **Nex-N2-mini is a reasoning
model** — it emits a `<think>` trace, so allow generous `max_tokens`. Native recommended
sampling (per Nex-AGI): `temperature 0.7, top_p 0.95, top_k 40`. For vision, add
`--mmproj mmproj-f16.gguf`.

### How the twin works (one file, one selected phase)

The GGUF stores each expert tensor twice: the canonical IQ3 tensor and a `<name>.v1` Q4_K
sibling. The patched loader reads `general.tensor_variant.default` (default `0` = IQ3;
override to `1` for Q4_K), keeps that variant, **drops the other before allocation**, and
renames the survivor to the canonical name — so the rest of llama.cpp sees an ordinary
single-precision model and only the selected phase's bytes are loaded. No-op on normal
single-precision GGUFs.

### Elastic: auto-fit *any* VRAM (one command)

`--sm` picks one of two global phases. The same file is also **elastic**: it carries a
per-tensor importance ranking (`general.tensor_variant.promote_order`), and the loader can
solve a **VRAM-budget knapsack** — start every expert at IQ3 and promote the most important
ones to Q4_K, by imatrix-importance-per-byte, until the footprint fits your card. The mix it
picks sits on a *convex* quality curve: spending an A770's spare ~0.9 GiB of single-card
headroom closes ~34 % of the IQ3→Q4_K KLD gap, on one card, same kernels, **no re-download**.

```bash
# elastic at load time — precision dialed to a budget (MiB of weights)
llama-cli -m Nex-N2-mini-Turbo-Phase-Twin.gguf \
  --override-kv general.tensor_variant.budget_mb=int:15900 -p "Hello" -n 64

# OR one command: detect VRAM → pick budget → AUTOPRUNE to a single right-sized file
./install-nx2.sh                 # autodetect
./install-nx2.sh --vram-gib 16   # force A770 16 GB → ~15.4 GiB fitted model, excess dropped
./install-nx2.sh --keep-dual     # keep the elastic dual; tune per-run instead
```

`install-nx2.sh` calls `prune-dual.py`, which runs the **same selection as the loader** and
writes a plain single-precision GGUF — so you can shrink the 34 GiB file to exactly what your
card needs and drop the unused bytes. `budget_mb` absent/`0` → ordinary `--sm` behavior.

## Files

| file | what |
|---|---|
| `Nex-N2-mini-Turbo-Phase-Twin.gguf` | the model — both phases, ~34 GiB (loads only the selected phase) |
| `mmproj-f16.gguf` | vision projector (optional, for multimodal) |
| `run-nx2.sh` | `--sm` launcher (off=1-card IQ3, on=2-card Q4) |
| `install-nx2.sh` | one-command: detect VRAM → pick budget → autoprune → fitted launcher |
| `prune-dual.py` | autoprune the dual to a single right-sized GGUF for a VRAM budget |
| `build/llama.cpp-turbo-phase-twin.patch` | the single build patch (applies on base `f0156d140`) |
| `build/build.sh` | clone + checkout + apply + build, one command |
| `build/BUILD.md` | build notes & troubleshooting |

## Provenance & changes (Apache-2.0 §4(b))

- **Base model:** [nex-agi/Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini)
  (Apache-2.0), post-trained on
  [Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) (Apache-2.0).
- **Changes:** converted to GGUF; experts quantized two ways — a codebook-free 3-bit type
  (`IQ3_A770`) and `Q4_K` — and packed as a dual-variant ("twin") GGUF; non-expert tensors at
  `Q6_K`. Both quant sets use a llama.cpp importance matrix (imatrix) computed from Bartowski's
  `calibration_datav3`. GGUF metadata `qwen35moe.block_count=40` and
  `qwen35moe.nextn_predict_layers=0` set so the model loads in llama.cpp (the MTP/NextN head
  is speculative-only and absent from the checkpoint — lossless for standard inference).
- The weights are quantizations of the original; no other modifications.

## Credits & dependencies

**This model is [Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini), built by the
[Nex-AGI](https://nex-agi.com) team.** The architecture, training, reasoning, and multimodal
capability are entirely theirs — this project only quantizes their model to run fast on Intel
Arc. All the intelligence here is Nex-AGI's, and sincere thanks to them for releasing a
genuinely capable open MoE. Full attribution in [`NOTICE`](NOTICE); the chain in brief:

- ⭐ **[Nex-N2-mini](https://huggingface.co/nex-agi/Nex-N2-mini)** © **Nex-AGI** (Apache-2.0) —
  **the model these quants are built from**, including the multimodal vision projector shipped
  here. → [nex-agi.com](https://nex-agi.com) · [GitHub](https://github.com/nex-agi/Nex-N2) ·
  [HuggingFace](https://huggingface.co/nex-agi)
- **[Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)** © Qwen Team /
  Alibaba (Apache-2.0) — the foundation model Nex-N2-mini was post-trained on.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** © the ggml authors (MIT) — the
  inference engine, quant framework, and SYCL backend. The included build patch is a
  derivative work of it (kept under MIT).
- **k-quants** by Iwan Kawrakow (ikawrakow) & the ggml authors (in llama.cpp, MIT) —
  `IQ3_A770` reuses Q3_K's superblock bit-packing; `Q6_K` covers non-expert tensors.
- **NormalFloat / NF4** (Dettmers et al., *QLoRA*, 2023) — design influence for the
  non-uniform, zero-centered code points; the table & codebook-free packing are this project's.
- **`calibration_datav3`** by Bartowski — the imatrix calibration text.
- **WikiText-2** (Merity et al., 2016; CC BY-SA 3.0) — used only to measure PPL/KLD.

## License & attribution

Released under the **Apache License 2.0**, inherited from the base model; this project's own
code (patch, launcher, build scripts) is MIT. Retain the attribution above. See
[`NOTICE`](NOTICE) for the complete chain.

```bibtex
@misc{qwen3.5,
  title  = {{Qwen3.5}: Towards Native Multimodal Agents},
  author = {{Qwen Team}},
  month  = {February},
  year   = {2026},
  url    = {https://qwen.ai/blog?id=qwen3.5}
}
```
Please also credit **Nex-AGI's Nex-N2-mini** (https://github.com/nex-agi/Nex-N2) as the base.
