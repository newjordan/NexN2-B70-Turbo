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

## 1. Winner: `topk28` (mixed down-proj)

| field | value |
|---|---|
| gate / up experts | `IQ3_A770` (codebook-free, 3.19 bpw) — **all 40 layers** |
| down experts | `Q4_K` on the **28 highest-importance layers**, `IQ3_A770` on the rest |
| non-expert tensors | `Q6_K` (embeddings, output, attn, ssm, shared expert) |
| **size** | **15.05 GiB** |
| **mean KLD** | **0.063528** |
| top-1 agreement | **89.49 %** |
| PPL(Q)/PPL(base) | **1.0186** |
| 16 GB fit | ✅ ~1 GiB free for KV/scratch (hybrid linear attn → KV is tiny) |

Down-proj layers kept at `IQ3_A770` (the 12 lowest-importance, all early):
`{0,1,2,3,4,8,9,13,16,17,20,22}`. Every other down layer → `Q4_K`.

**Why it matters:** `topk28` beats the best *stock* option `iq3_s` on **every
axis** at essentially the same footprint, and it is codebook-free (the stock
`iq3_s` is a 512-entry grid that serializes on shared memory and does not compose
with the reorder/`mmvq`/`dp4a` path this repo is built on):

| | size | KLD | top-1 | PPL ratio | codebook-free |
|---|---|---|---|---|---|
| stock `iq3_s` (best stock ≤16 GB) | 14.84 | 0.0653 | 89.25 % | 1.0234 | ❌ |
| **`topk28` (ours)** | 15.05 | **0.0635** | **89.49 %** | **1.0186** | ✅ |

---

## 2. Full single-card frontier (codebook-free unless noted)

All rows: gate/up = `IQ3_A770`, non-expert = `Q6_K`, imatrix = `NX2.imatrix`.

| recipe | down-proj | size GiB | mean KLD | top-1 % | PPL ratio | note |
|---|---|---|---|---|---|---|
| uniform `IQ3_A770` | all iq3 | 13.90 | 0.0754 | 88.5 | 1.030 | baseline |
| `IQ3_NL` (8-entry LUT) | all iq3 | 13.90 | 0.0727 | 88.5 | 1.027 | LUT ≈ +3 % only |
| `topk6` | 6 late → Q4_K | 14.15 | 0.0717 | 88.8 | 1.025 | dominated |
| `down_q3k` | **all → Q3_K** | 14.21 | 0.0697 | 88.8 | 1.019 | max-headroom pick |
| `down_iq3s` | all → iq3_s | 14.21 | 0.0679 | 89.3 | 1.023 | codebook |
| `topk12` | 12 → Q4_K | 14.39 | 0.0686 | 89.1 | 1.023 | |
| `topk20` | 20 → Q4_K | 14.72 | 0.0663 | 89.1 | 1.023 | |
| stock `iq3_s` | all iq3_s | 14.84 | 0.0653 | 89.3 | 1.023 | codebook, bar to beat |
| **`topk28`** ⭐ | 28 → Q4_K | 15.05 | **0.0635** | 89.5 | 1.019 | **winner** |
| `down_q4k` | all → Q4_K | 15.54 | 0.0567 | 90.0 | 1.020 | best KLD but too tight |

**Deployment picks** (all codebook-free, all fit 16 GB):
- **max context headroom** → `down_q3k` (0.0697 @ 14.21, ~1.6 GiB free)
- **best quality that comfortably fits** → `topk28` (0.0635 @ 15.05, ~1 GiB free)
- `down_q4k` (0.0567) is the quality ceiling but 15.54 GiB leaves no real KV room.

---

## 3. Key findings

1. **The down-proj is the whole lever.** gate/up quality barely moves KLD;
   moving down-proj above 3.19 bpw is what buys quality. (`down_q4k`: 0.0754 →
   0.0567 by touching only down.)
2. **Uniform bump > concentrated bump at the low end.** `down_q3k` (a *uniform*
   +0.25 bpw across all 40 down layers) beats `topk6` (a *big* Q4_K bump on the 6
   "most important" layers) at the same size. High imatrix activation energy ≠
   high quantization sensitivity — the benefit of extra bits is spread across
   layers, so spreading a modest bump is more bit-efficient than concentrating.
3. **Importance is late-layer-concentrated.** imatrix mean-act² ranking
   ([`scripts/rank-down-layers.py`](../scripts/rank-down-layers.py)): `blk.39` is
   ~40× any other layer (it feeds the output head); early layers are negligible.
   This is why the `topk` series fills late→early.
4. **The LUT (IQ3_NL) is a dead end on its own** — only ~3 % KLD over uniform
   `IQ3_A770` and it doesn't touch the budget; spend bits on down instead.

---

## 4. Reproduce

Ranking (writes the importance order used below):
```bash
nx2-venv/bin/python scripts/rank-down-layers.py
```

Full frontier sweeps (memory-gated, CPU KLD):
```bash
bash scripts/iq3-down-sweep.sh          # uniform-down points (q3_k/iq3_s/q4_k)
bash scripts/iq3-partial-down-sweep.sh  # topk{6,12,20,28} partial-q4_k points
```

Winner directly (per-layer overrides; `--tensor-type` is first-match-wins, so the
per-layer `Q4_K` patterns must precede the generic `IQ3_A770` fallback):
```bash
llama-quantize --imatrix NX2.imatrix \
  --tensor-type ffn_gate_exps=iq3_a770 --tensor-type ffn_up_exps=iq3_a770 \
  $(for L in 5 6 7 10 11 12 14 15 18 19 21 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39; \
      do printf -- '--tensor-type blk\\.%s\\.ffn_down_exps=q4_k ' "$L"; done) \
  --tensor-type ffn_down_exps=iq3_a770 \
  NX2-bf16.gguf NX2-IQ3_A770-mixed.gguf q6_k
```

Winner artifact kept at
`models/nex-n2-mini/iq3-partial-down/NX2.topk28.gguf` (hardlinked as
`NX2-IQ3_A770-mixed.gguf`). Raw CSVs: `iq3-down-sweep/down-sweep.csv`,
`iq3-partial-down/partial-down.csv`.

---

## 5. Open / next

- **2-level down (in progress):** uniform `Q3_K` down as the base + `Q4_K` on the
  top-K importance layers. Since uniform bumping proved the more efficient lever
  (finding #2), layering it under selective `Q4_K` should push the frontier left
  of `topk28` at a comfortable fit — chasing `down_q4k`'s 0.0567 into 16 GB.
- **SYCL kernel** (`0006`/`0007`) to make `IQ3_A770` run on the A770 GPU and
  measure real throughput — the recipe above is the build target.
