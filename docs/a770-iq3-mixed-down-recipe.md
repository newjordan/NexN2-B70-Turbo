# A770 IQ3 — Mixed Down-Proj Recipe (the single-card 16 GB winner)

Companion to [`iq3-a770-block-layout.md`](iq3-a770-block-layout.md). That doc
froze the codebook-free `IQ3_A770` block format; this one records the **quant
mix** that gives the best quality we can land on **one Intel Arc A770 16 GB**
while staying codebook-free (dp4a-aligned).

**Status:** CPU-measured **quality** (KLD vs `NX2-Q6K.kld`, wikitext-2, 100×512
tok). The `IQ3_A770` SYCL kernel (`0006`/`0007`) is still a follow-up, so A770
*throughput* is not yet measured — these are correctness/quality results that fix
the recipe to build the kernel against. Target model: Nex-N2-mini (`qwen35moe`,
256 experts, experts = 92.9% of weights).

---

## 1. Winner: `q3floor_q4top24` (2-level down-proj)

| field | value |
|---|---|
| gate / up experts | `IQ3_A770` (codebook-free, 3.19 bpw) — **all 40 layers** |
| down experts — floor | `Q3_K` on all 40 down layers |
| down experts — top | `Q4_K` on the **24 highest-importance layers** (override over the floor) |
| non-expert tensors | `Q6_K` (embeddings, output, attn, ssm, shared expert) |
| **size** | **15.01 GiB** |
| **mean KLD** | **0.062362** |
| top-1 agreement | **89.55 %** |
| PPL(Q)/PPL(base) | **1.0167** |
| 16 GB fit | ✅ ~1 GiB free for KV/scratch (hybrid linear attn → KV is tiny) |

Down layers at `Q4_K` (top-24 importance):
`{10,11,12,14,15,19,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}`.
The other 16 (`{0,1,2,3,4,5,6,7,8,9,13,16,17,18,20,22}`) stay at the `Q3_K` floor.

**Why it wins:** beats the best *stock* option `iq3_s` on **every axis** at a
comparable footprint, and it is codebook-free (stock `iq3_s` is a 512-entry grid
that serializes on shared memory and does not compose with the
reorder/`mmvq`/`dp4a` path this repo is built on):

| | size | KLD | top-1 | PPL ratio | codebook-free |
|---|---|---|---|---|---|
| stock `iq3_s` (best stock ≤16 GB) | 14.84 | 0.0653 | 89.25 % | 1.0234 | ❌ |
| **`q3floor_q4top24` (ours)** | 15.01 | **0.0624** | **89.55 %** | **1.0167** | ✅ |

**Aggressive variant** `q3floor_q4top32` — `Q4_K` on the top-32 layers — reaches
**0.0596 KLD @ 15.27 GiB** (89.68 % top-1), nearly the all-`Q4_K` ceiling
(0.0567). Use it for short-context / max-quality; at 15.27 GiB it's tight against
usable A770 VRAM (~15.5–15.9 GiB after driver reservation), so it is *not* the
safe default.

---

## 2. Full single-card frontier (codebook-free unless noted)

All rows: gate/up = `IQ3_A770`, non-expert = `Q6_K`, imatrix = `NX2.imatrix`.
The **2-level** series (Q3_K floor + Q4_K on top-K importance layers) dominates
the earlier pure-`topk` series (IQ3 floor) at every equal-size point.

| recipe | down-proj | size GiB | mean KLD | top-1 % | PPL ratio | note |
|---|---|---|---|---|---|---|
| uniform `IQ3_A770` | all iq3 | 13.90 | 0.0754 | 88.5 | 1.030 | baseline |
| `IQ3_NL` (8-entry LUT) | all iq3 | 13.90 | 0.0727 | 88.5 | 1.027 | LUT ≈ +3 % only |
| `down_q3k` | all Q3_K | 14.21 | 0.0697 | 88.8 | 1.019 | 2-level K=0 |
| `q3floor_q4top8` | Q3_K + 8×Q4_K | 14.48 | 0.0664 | 89.08 | 1.0181 | |
| `topk20` (IQ3 floor) | 20×Q4_K else iq3 | 14.72 | 0.0663 | 89.07 | 1.023 | superseded |
| `q3floor_q4top16` | Q3_K + 16×Q4_K | 14.74 | 0.0642 | 89.25 | 1.0178 | beats topk20 |
| stock `iq3_s` | all iq3_s | 14.84 | 0.0653 | 89.3 | 1.023 | codebook, bar to beat |
| `topk28` (IQ3 floor) | 28×Q4_K else iq3 | 15.05 | 0.0635 | 89.49 | 1.0186 | superseded |
| **`q3floor_q4top24`** ⭐ | Q3_K + 24×Q4_K | 15.01 | **0.0624** | 89.55 | 1.0167 | **winner** |
| `q3floor_q4top32` | Q3_K + 32×Q4_K | 15.27 | 0.0596 | 89.68 | 1.0183 | max-quality / tight |
| `down_q4k` | all Q4_K | 15.54 | 0.0567 | 90.0 | 1.020 | ceiling, no KV room |

**Deployment picks** (all codebook-free, all fit 16 GB):
- **max context headroom** → `down_q3k` (0.0697 @ 14.21, ~1.6 GiB free)
- **best balanced (default)** → `q3floor_q4top24` (0.0624 @ 15.01, ~1 GiB free)
- **max quality, short context** → `q3floor_q4top32` (0.0596 @ 15.27, tight)

---

## 3. Key findings

1. **The down-proj is the whole lever.** gate/up quality barely moves KLD;
   moving down-proj above 3.19 bpw is what buys quality (`down_q4k`: 0.0754 →
   0.0567 by touching only down).
2. **Uniform floor > concentrated bump.** A *uniform* Q3_K floor across all 40
   down layers is more bit-efficient than concentrating Q4_K on the "most
   important" layers over an IQ3 floor. At every equal-size point the 2-level
   series (Q3_K floor) beats the pure-`topk` series (IQ3 floor): 14.74 GiB
   0.0642 vs 0.0663; ~15.0 GiB 0.0624 vs 0.0635. High imatrix activation energy
   ≠ high quantization sensitivity — the benefit of extra bits is spread across
   layers, so raise the floor first, then spend the remainder on the top layers.
3. **Importance is late-layer-concentrated.** imatrix mean-act² ranking
   ([`scripts/rank-down-layers.py`](../scripts/rank-down-layers.py)): `blk.39` is
   ~40× any other layer (it feeds the output head); early layers are negligible.
   This is why the Q4_K overrides fill late→early.
4. **The LUT (IQ3_NL) is a dead end on its own** — only ~3 % KLD over uniform
   `IQ3_A770` and it doesn't touch the budget; spend bits on down instead.

---

## 4. Reproduce

Ranking (writes the importance order used below):
```bash
nx2-venv/bin/python scripts/rank-down-layers.py
```

Frontier sweeps (memory-gated, CPU KLD):
```bash
bash scripts/iq3-down-sweep.sh          # uniform-down points (q3_k/iq3_s/q4_k)
bash scripts/iq3-partial-down-sweep.sh  # topk{6,12,20,28}, IQ3 floor + Q4_K top-K
bash scripts/iq3-2level-down-sweep.sh   # q3floor_q4top{8,16,24,32}, the winning family
```

Winner directly (`--tensor-type` is first-match-wins, so the per-layer `Q4_K`
patterns must precede the generic `Q3_K` floor fallback):
```bash
llama-quantize --imatrix NX2.imatrix \
  --tensor-type ffn_gate_exps=iq3_a770 --tensor-type ffn_up_exps=iq3_a770 \
  $(for L in 10 11 12 14 15 19 21 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39; \
      do printf -- '--tensor-type blk\\.%s\\.ffn_down_exps=q4_k ' "$L"; done) \
  --tensor-type ffn_down_exps=q3_k \
  NX2-bf16.gguf NX2-IQ3_A770-mixed.gguf q6_k
```

Winner artifact kept at
`models/nex-n2-mini/iq3-2level-down/NX2.q3floor_q4top24.gguf` (hardlinked as
`NX2-IQ3_A770-mixed.gguf`). Raw CSVs in [`../results/iq3-mixed-down/`](../results/iq3-mixed-down/).

---

## 5. Open / next

- **SYCL kernel** (`0006`/`0007`) to make `IQ3_A770` run on the A770 GPU and
  measure real throughput — the recipe above is the build target. Until then the
  mix runs CPU-only; quality is fixed, speed is the open question.
- Frontier is effectively closed for this format: `down_q4k` (0.0567) is the
  all-Q4_K ceiling and needs 15.54 GiB; nothing codebook-free beats it without
  going over the card. Further KLD gains require a richer down format (e.g. the
  unbuilt L2 asym-IQ3 3.5 bpw) or accepting the codebook `iq3_s`/`down_q4k` cost.
