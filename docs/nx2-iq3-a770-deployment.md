# NX2-IQ3_A770 — single-card & two-card deployment

Nex-N2-mini (`qwen35moe`: ~34.7 B total / ~A3B active, 256 experts, hybrid
linear-attention) packaged as **one GGUF, two residency points** selected by `--sm`.
A single multi-precision file (`NX2-IQ3_A770-dual.gguf`) carries both expert sets; the
loader (patch `0010`) resolves it to one variant at load and reads only that variant's
tensor data. Launcher: [`../serving/run-nx2.sh`](../serving/run-nx2.sh).

## TL;DR

| `--sm` | GPUs | variant | experts | resident | path | decode |
|---|---|---|---|---|---|---|
| **off** | 1× 16 GB | 0 (default) | IQ3_A770 3.19 bpw (codebook-free LUT) | ~15.0 GiB | fused reorder dp4a (0007/0008) | **~82 t/s** (B70, measured) |
| **on** | 2× 16 GB | 1 (`--override-kv`) | Q4_K 4.5 bpw | ~18.8 GiB | Q4_K reorder, `-sm layer -ts 1,1` | bandwidth-projected (see below) |

One file on disk (~34 GiB); each mode loads only its variant's experts (the loader drops
the sibling set before allocation), so VRAM is the "resident" column, not the file size.

```bash
serving/run-nx2.sh --sm auto    # detects GPU count; off=1-card IQ3, on=2-card Q4_K
serving/run-nx2.sh --sm off -- -p "The capital of France is" -n 16   # force 1-card
NX2_TOOL=llama-server serving/run-nx2.sh --sm on --port 8080         # force 2-card
# under the hood, --sm on adds:  --override-kv general.tensor_variant.default=int:1
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
- **Single-file hot-swap:** patch `0010` (`select_tensor_variant`) lets one GGUF ship both
  expert sets — canonical IQ3 plus a `<name>.v1` Q4_K sibling per expert tensor. At load the
  loader keeps the variant named by `general.tensor_variant.default` (overridable with
  `--override-kv …=int:1`), renames it to the canonical name, and drops the rest **before**
  tensor creation, so the rest of llama.cpp sees an ordinary single-precision model and only
  the selected variant's bytes are read. No-op when the key is absent → upstream models are
  unaffected. Build the file with [`../scripts/merge-tensor-variants.py`](../scripts/merge-tensor-variants.py).

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
  0008: 79.0→82.0). End-to-end coherent generation. KLD as above. The whole reorder stack vs
  its dormant fallback was re-measured **same-binary** (env-gated, no rebuild between arms):
  **tg 43.71 → 82.3 t/s, +88%**, at matched pp512 (~600, spread 0.09%), reproducible to 0.2%,
  ±0.05 error bars — [`../results/init-tensor-ab/`](../results/init-tensor-ab/).
- **2-card path: accuracy is real** (measured on CPU — device-count-independent), and the
  kernels are the production-tested Q4_K reorder. **Throughput is bandwidth-*projected*, not
  measured** — only a single dev card is available here. Validate on real 2-GPU hardware
  with the handoff harness ([`../eval/nx2/run_tensor_split_claim.sh`](../eval/nx2/run_tensor_split_claim.sh)
  pattern); the bandwidth table above predicts the split is decode-positive at 4.5 bpw.
- **Single-file loader (`0010`): both variants load+generate from the one `dual.gguf`.**
  Variant 0 (IQ3) on GPU and variant 1 (Q4_K, via `--override-kv …=int:1`) on CPU each
  produced coherent output with **no integrity error** — confirming the loader drops the
  unselected sibling set (854 → 734 tensors) before allocation rather than loading both. The
  `select_tensor_variant` log shows `selected tensor variant N; 120 groups, dropped 120
  sibling tensors`, and the override path logs `validate_override … = 1` first.

## HuggingFace layout (one repo = "one upload")

```
NX2-IQ3_A770-dual.gguf  # ONE file, ~34 GiB — both precisions; --sm picks the variant
run-nx2.sh              # the --sm switch (adds --override-kv for variant 1)
mmproj-f16.gguf         # vision tower (optional, separate from the text budget)
```

The single-file in-engine `--sm` toggle is real (patch `0010`): one GGUF carries both
precisions and the loader swaps variants by a metadata key, so a user downloads one file and
flips `--sm` to trade card-count for accuracy. No second download, no re-quantize.
(Prefer two separate files — 15 GiB IQ3 + 18.8 GiB Q4_K — to save the ~15 GiB of unused bytes
per mode on disk? `merge-tensor-variants.py`'s inputs are exactly those two files; ship them
instead and drop `--override-kv`. The single file is the convenience default.)

## Elastic precision (`0011`) — one file auto-fits ANY VRAM

`--sm` picks one of two *global* points (all-IQ3 or all-Q4_K). Patch `0011` makes the same
dual file **continuous**: it embeds a per-tensor importance ranking
(`general.tensor_variant.promote_order`, from [`../scripts/rank-experts.py`](../scripts/rank-experts.py)),
and the loader solves a load-time **budget knapsack** — start every expert at IQ3, greedily
promote to Q4_K by imatrix-importance-per-byte until the weight footprint fits a target.

```bash
# elastic at load time: one file, precision dialed to the budget (no re-download, no re-quantize)
llama-cli -m dual.gguf --override-kv general.tensor_variant.budget_mb=int:15900 ...
```

The per-tensor mix it selects is exactly a point on the **convex KLD frontier** measured in
[`../results/elastic-precision/`](../results/elastic-precision/): e.g. ~+0.9 GiB over the
all-IQ3 footprint (a 16 GB A770's spare headroom) closes ~34 % of the IQ3→Q4_K quality gap, on
**one card, the same `0007`/`0008` kernels, zero new bytes downloaded**. `budget_mb` absent or
`0` → unchanged `0010` behavior (upstream models untouched).

### One-command install + autoprune

[`../serving/install-nx2.sh`](../serving/install-nx2.sh) detects VRAM, picks the budget, and
**autoprunes** the 34 GiB dual down to a single right-sized model (drops the unused variant
bytes — "the excess"), then writes a launcher:

```bash
serving/install-nx2.sh                 # autodetect VRAM -> autoprune to fit -> run-nx2-fitted.sh
serving/install-nx2.sh --vram-gib 16   # force A770 16 GB: fits a ~15.4 GiB model, 9 experts at Q4_K
serving/install-nx2.sh --keep-dual     # keep the elastic dual, tune precision per-run instead
```

Under the hood it calls [`../scripts/prune-dual.py`](../scripts/prune-dual.py), which runs the
**same greedy selection as the loader** (verified: the loader's `select_tensor_variant` and
`prune-dual.py` produce identical per-tensor mixes at matched budgets). The pruned file is a
plain single-precision GGUF — no variant machinery, loads anywhere.
