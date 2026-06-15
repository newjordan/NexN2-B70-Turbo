# NX2-IQ3_A770 — single-card & two-card deployment

Nex-N2-mini (`qwen35moe`: ~34.7 B total / ~A3B active, 256 experts, hybrid
linear-attention) packaged as **one model, two residency points** selected by `--sm`.
Launcher: [`../serving/run-nx2.sh`](../serving/run-nx2.sh).

## TL;DR

| `--sm` | GPUs | file | experts | size | path | decode |
|---|---|---|---|---|---|---|
| **off** | 1× 16 GB | `NX2-IQ3_A770` | 3.19 bpw (codebook-free LUT) | ~15.0 GiB | fused reorder dp4a (0007/0008) | **~82 t/s** (B70, measured) |
| **on** | 2× 16 GB | `NX2-Q4_K` | 4.5 bpw | ~18.8 GiB | Q4_K reorder, `-sm layer -ts 1,1` | bandwidth-projected (see below) |

```bash
serving/run-nx2.sh --sm auto    # detects GPU count; off=1-card IQ3, on=2-card Q4_K
serving/run-nx2.sh --sm off -- -p "The capital of France is" -n 16   # force 1-card
NX2_TOOL=llama-server serving/run-nx2.sh --sm on --port 8080         # force 2-card
```

## Why two precisions — the bits-vs-bandwidth lever

Decode reads the **active** experts (8 of 256) each token. Per-card bytes ≈
`(active/2) × bpw` once split across two cards:

| 2-card fill | bpw | per-card read (4 experts) | vs 1-card 3-bit base (8×3.19 = 25.5) |
|---|---|---|---|
| **Q4_K** | 4.5 | 18.0 | **−30% → splitting also speeds decode** |
| Q6_K | 6.4 | 25.6 | ≈ break-even |
| Q8_0 | 8.5 | 34.0 | +33% → split can't pay for the bytes |

Break-even is **~6.4 bpw**. Below it, two-card split reads *fewer* bytes/card than the
one-card 3-bit base, so `--sm on` buys **accuracy *and* relieves decode bandwidth** at
once. **Q4_K (4.5) sits in the win zone** — that's why the 2-card fill is 4-bit, not 8.

## What runs (no new kernels for the 2-card path)

- **1-card IQ3:** patch `0007` (per-expert SoA reorder + one fused dp4a expert-indexed
  GEMV per matmul) + `0008` (fused gate/up + SwiGLU). The `IQ3_A770` codebook-free 3-bit
  type (`0005`) + dedicated `LLAMA_FTYPE_MOSTLY_IQ3_A770` (`0009`).
- **2-card Q4_K:** the existing Q4_K MoE reorder (the B70-Turbo chain, `0001`/`0004`) under
  `-sm layer`, which runs whole layers per device so the per-device reorder/fusion stays
  intact. **Zero IQ3-specific kernel work** — it's a standard Q4_K_M-class build.

## Quality (final eval — KLD vs `NX2-Q6K.kld`, wikitext-2, 100×512 tok)

| model | bpw (experts) | mean KLD | top-1 | PPL(Q)/PPL(base) |
|---|---|---|---|---|
| `NX2-IQ3_A770` (1-card, GPU int8) | 3.19 | 0.0547 | 89.91% | 1.0163 |
| **`NX2-Q4_K` (2-card)** | 4.5 | **0.0245** | **93.24%** | **1.0024** |

Same build, same corpus (`wiki.test.raw`, 100×512 tok), same `NX2-Q6K.kld` base. The 2-card
mode **more than halves KLD (−55%)** and sits **within 0.24% PPL of the full Q6_K base —
effectively lossless** — while top-1 rises 89.9% → 93.2%. So `--sm on` is *more accurate*,
and by the per-card bandwidth table above also *decode-positive*: accuracy and speed, same
switch. Raw logs: [`../results/nx2-final-eval/`](../results/nx2-final-eval/).

## Testing status (honest)

- **1-card path: fully validated on Intel Arc Pro B70 (Battlemage, A770-proxy).** Every
  throughput claim is a back-to-back same-build A/B (e.g. 0007: 43.7→78.6 t/s at matched pp;
  0008: 79.0→82.0). End-to-end coherent generation. KLD as above.
- **2-card path: accuracy is real** (measured on CPU — device-count-independent), and the
  kernels are the production-tested Q4_K reorder. **Throughput is bandwidth-*projected*, not
  measured** — only a single dev card is available here. Validate on real 2-GPU hardware
  with the handoff harness ([`../eval/nx2/run_tensor_split_claim.sh`](../eval/nx2/run_tensor_split_claim.sh)
  pattern); the bandwidth table above predicts the split is decode-positive at 4.5 bpw.

## HuggingFace layout (one repo = "one upload")

```
NX2-IQ3_A770.gguf     # 1-card, 15 GiB   (rename of NX2-IQ3_A770-mixed-LUT.gguf)
NX2-Q4_K.gguf         # 2-card, 18.8 GiB (rename of NX2-IQ3_A770-Q4fill.gguf)
run-nx2.sh            # the --sm switch
mmproj-f16.gguf       # vision tower (optional, separate from the text budget)
```

A true single-file in-engine `--sm` toggle (one GGUF carrying both precisions, loader
swaps experts by device count) is a deeper llama.cpp change; the launcher + two-file repo
delivers the same one-command convenience today.
