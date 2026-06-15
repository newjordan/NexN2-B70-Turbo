# IQ3_A770 Block Layout — Design Draft (for review before code)

Companion to [`a770-iq3-expert-residency.md`](a770-iq3-expert-residency.md). This
fixes the on-disk block layout, bit packing, bpw budget, and CPU-oracle surface
for `GGML_TYPE_IQ3_A770` so patch `0005` (serialize + CPU dequant + CPU dot) can
be written against a frozen spec. Nothing here is in code yet — it is the
artifact to sign off on first.

**Sign-offs:** primary format **L3** (lean, symmetric, 3.1875 bpw); target model
**Nex-N2-mini**; single-card track **Policy C** (all-IQ3) + **Policy D**
(asym L2 down) with **A'** kept as a multi-card/headroom comparison.

Status: **FROZEN as the `0005` spec.** Implemented in
[`../patches/0005-iq3-a770-plumbing.patch`](../patches/0005-iq3-a770-plumbing.patch)
(CPU oracle). A0 correctness validated — see [§10](#10-implementation--a0-results).
L2 (asym down for Policy D) and the SYCL kernel (`0006`/`0007`) are follow-ups.

---

## 0. Target model: Nex-N2-mini (`qwen35moe`)

Confirmed from `NX2-Q6_K.gguf` (arch `qwen35moe`, supported in the `fabler` tree
as `LLM_ARCH_QWEN35MOE`). This is **not** a vanilla MoE — it is a hybrid
linear-attention multimodal MoE:

- 40 layers, `d_model=2048`, **256 experts, 8 active**, expert `d_ff=512`,
  shared expert `d_ff=512`.
- **Attention is hybrid:** `full_attention` every 4th layer (10 layers, 16 heads
  / **2 KV heads** / `head_dim=256`, partial RoPE 0.25), `linear_attention`
  (gated delta-net + causal conv, `ssm.state_size=128`, `inner_size=4096`,
  `group_count=16`) in the other 30 layers.
- `vocab=248320`, untied embeddings, `context_length=262144`.
- Multimodal: separate vision tower (`mmproj-f16.gguf`, ~0.9 GB) — **out of the
  text 16GB budget**; loaded separately.

**Exact parameter / byte breakdown** (from the Q6_K GGUF tensor table):

| group | tensors | params | Q6_K GB | share |
|---|---|---|---|---|
| expert gate | `ffn_gate_exps [2048,512,256]` ×40 | 10.74B | 8.81 | 30.9% |
| expert up | `ffn_up_exps   [2048,512,256]` ×40 | 10.74B | 8.81 | 30.9% |
| expert down | `ffn_down_exps [512,2048,256]` ×40 | 10.74B | 8.81 | 30.9% |
| attention | `attn_*` (full + linear proj) | 1.03B | 0.84 | 3.0% |
| token_embd | | 0.51B | 0.42 | 1.5% |
| output_head | | 0.51B | 0.42 | 1.5% |
| linear-attn state proj | `ssm_*`, conv, `beta_alpha` | 0.26B | 0.21 | 0.7% |
| shared expert | `*_shexp` | 0.13B | 0.10 | 0.4% |
| router | `ffn_gate_inp` (F32) | 0.02B | 0.08 | 0.3% |
| **total** | | **≈34.66B** | **28.50** | |

**The decisive fact: experts are 92.9% of the model.** The IQ3_A770 format bpw
is therefore almost the entire VRAM budget, and the fragile `down` projection is
a full third of it (10.74B). Assets already on disk that the campaign uses:
`NX2.imatrix` (importance matrix — required for L3 quality), `NX2-Q6_K.gguf` +
`NX2-Q6K.kld` (KLD/top-1 oracle), `Nex-N2-mini.Q4_K_M.gguf` (B-row baseline).

---

## 1. Core design decision: codebook-free, K-quant-style

llama.cpp's existing 3-bit types are **grid/codebook** formats (`iq3_xxs`:
256-entry grid; `iq3_s`: 512-entry grid + signs). Codebook lookups serialize on a
constant/shared-memory table, block the cheap unpack→int8→`dp4a` flow, and don't
compose with this repo's reorder + `mmvq` strategy, which is built entirely on
**direct bit-unpack K-quants**.

**`IQ3_A770` is "IQ3" in bit budget but structurally a K-quant** — super-block of
256, per-sub-block scales, direct 3-bit unpack (2 low bits + 1 high bit), no
grid. Defining a new GGML type (vs reusing `Q3_K`) lets us pick the scale
encoding, the bit-split for coalesced reads, and — where quality demands it — a
per-sub-block **min** (asymmetric form), which `block_q8_K`'s `bsums` already
supports for free in the dot.

---

## 2. Layout variants

`QK_K=256`, `K_SCALE_SIZE=12`, `ggml_half=2 B`. All variants split each 3-bit
weight as **2 low bits (`qs`, 4 w/byte) + 1 high bit (`qh`, 8 w/byte)** so unpack
and reorder code is shared and mirrors `Q3_K`/`Q5_K`.

| Var | sym/asym | sub-blocks | scale bits | bytes | **bpw** | role |
|----|----|----|----|----|----|----|
| **L3** ⭐ primary | symmetric | 8×32 | 4-bit | **102** | **3.1875** | residency-true gate/up & all-IQ3 |
| **L2** | asymmetric (min) | 8×32 | 6-bit sc + 6-bit min | 112 | 3.5000 | quality option for fragile `down` |
| **L1** | symmetric | 16×16 | 6-bit | 110 | 3.4375 | Q3_K-size symmetric ablation |

L3 byte budget: `qs`64 + `qh`32 + `scales`4 + `d`2 = 102. The 4-byte scale plane
forces **4-bit per-sub-block scales** (8×32) — coarse, so imatrix weighting and
keeping `down` richer (Policy D) matter. L2/L1 are documented for the down-proj
hedge and ablation; they are not the first build target.

---

## 3. Primary format: L3 (symmetric, 3.1875 bpw)

### 3.1 AoS struct (serialization + CPU oracle)

The implemented struct uses the **exact Q3_K field order** (`hmask, qs, scales, d`)
so the dequant / vec_dot / reorder code is a direct Q3_K adaptation — only the
scale plane shrinks (`12 → 4` bytes) and the sub-block size grows (`16 → 32`):

```c
#define QK_K 256
// 3-bit symmetric, codebook-free. Q3_K qs/hmask packing; 8 sub-blocks of 32
// with 4-bit signed scales (stored [0,15], used as sc-8). x = d*sc*(q-4).
// Effectively 3.1875 bpw.
typedef struct {
    uint8_t   hmask[QK_K/8]; // 32 B  high bit of each 3-bit quant (the doc's "qh")
    uint8_t   qs[QK_K/4];    // 64 B  low 2 bits of each 3-bit quant
    uint8_t   scales[4];     //  4 B  8 sub-blocks × 4-bit signed scale
    ggml_half d;             //  2 B  super-block scale
} block_iq3_a770;            // 102 B
static_assert(sizeof(block_iq3_a770)
    == sizeof(ggml_half) + QK_K/4 + QK_K/8 + 4,
    "wrong iq3_a770 block size/padding");
```

### 3.2 Per-weight reconstruction

For weight `l` in `[0,256)`, sub-block `is = l/32`:

```
q3 = (low 2 bits from qs) | (high bit from hmask)        // Q3_K packing, q3 in [0,7]
sc = ((scales[is>>1] >> (4*(is&1))) & 0xF) - 8           // 4-bit signed, [-8,7]
x  = d * sc * (q3 - 4)
```

Symmetric (zero-centered at `q3=4`): **no `dmin`, no min plane, and the dot needs
no `bsums` term** — simpler than the asym K-quants. Scales are signed 4-bit
(Q3_K's scheme at lower resolution); the quantizer reuses `make_q3_quants`
(ref) / `make_qx_quants` (imatrix-weighted), so scale search is the proven
K-quant path.

### 3.3 Dot against `block_q8_K` (the oracle)

```
dot = Σ_sub  d_act * d * sc[sub] * Σ_{l in sub} ((q3[l]-4) * q8[l])
```

Pure unpack + multiply-accumulate; no min/`bsums` correction. The SYCL port
(`0006`/`0007`) is a small delta over the existing reordered `Q3_K` `mmvq`.

### 3.4 Asymmetric down-proj variant (L2, for Policy D)

If down-proj fails KLD at L3, the `down` tensors use L2 (112 B, 6-bit scale +
6-bit min, `d`+`dmin`, `x = d·sc·q − dmin·m`, dot reuses `bsums` exactly like
`ggml_vec_dot_q4_K_q8_K`). Same `qs/qh` bit-split, so the unpack path is shared.

---

## 4. Reordered (planar / SoA) layout for SYCL

Mirrors the verified `dequantize_block_q3_K_reorder` convention — fields scattered
into contiguous planes across the whole tensor (per-expert sub-matrix for
`MUL_MAT_ID`). For `n` blocks, L3 plane offsets:

```
qs      : base + 0
hmask   : base + n*(QK_K/4)
scales  : base + n*(QK_K/4) + n*(QK_K/8)
d       : base + n*(QK_K/4) + n*(QK_K/8) + n*4
```

`MUL_MAT_ID` prepends `expert_id * n_blocks_per_expert * 102`. Plane order
`qs | qh | scales | d` matches the existing reorder tiling so `0007` reuses it.

---

## 5. CPU-oracle surface (patch 0005)

Correctness-first, no SYCL. Add + register:

- **Enums/ftype:** `GGML_TYPE_IQ3_A770` (ggml.h); `GGML_FTYPE_MOSTLY_IQ3_A770`;
  `LLAMA_FTYPE_MOSTLY_IQ3_A770` (llama.h); wire `ggml_ftype_to_ggml_type` + the
  llama quant switch.
- **type_traits** (ggml.c): `blck_size=256`, `type_size=102`, `is_quantized=true`,
  `to_float`, `from_float_ref`, `vec_dot`, `vec_dot_type=GGML_TYPE_Q8_K`.
- **`quantize_row_iq3_a770_ref` / `quantize_iq3_a770`** — imatrix-aware (accept
  `const float * quant_weights`); use `NX2.imatrix`.
- **`dequantize_row_iq3_a770`**, **`ggml_vec_dot_iq3_a770_q8_K`** (oracle, §3.3).
- **Quant policy** (`llama_tensor_get_type`): map `ffn_gate_exps`/`ffn_up_exps`
  (and, per policy, `ffn_down_exps`) → `IQ3_A770`; keep `ffn_gate_inp`,
  `token_embd`, `output`, `attn_*`, `*_shexp`, `ssm_*` at the policy's higher
  precision.
- **Coverage:** `test-backend-ops -o MUL_MAT` and `-o MUL_MAT_ID` (CPU). SYCL
  parity rows arrive with `0006`/`0007`.

This is the A0 ablation row (CPU ref only): serialization, dequant, dot
correctness, sane KLD vs `NX2-Q6K.kld`.

---

## 6. Residency ledger (the 16 GB gate)

Bytes/param: L3 0.3984, L2 0.4375, Q4_K 0.5625, Q5_K 0.6875, Q6_K 0.8203.
Non-expert kept at Q6_K = 2.08 GB (trim embeds/output/attn → Q5 saves ~0.3 GB).
KV: 10 full-attn layers only → ~20 KiB/tok f16 (0.31 GB @16k, 0.63 GB @32k,
2.5 GB @128k; `q8_0` halves it). Linear state ~0.07 GB constant. Scratch ~0.5 GB.

| Policy | gate/up | down | weights | + KV/state @32k (q8_0) | single-card 16 GB |
|---|---|---|---|---|---|
| **C** all-IQ3 | L3 | L3 | 12.83 + 2.08 = **14.9** | ~15.8 | ✅ comfortable; scales to 128k |
| **D** lean+asym down | L3 | L2 | 8.56 + 4.70 + 2.08 = **15.3** | ~15.9 (with embed trim) | ✅ tight; hedges down-proj |
| **A'** Q4_K down | L3 | Q4_K | 8.56 + 6.04 + 2.08 = **16.7** | over before KV | ❌ multi-card / headroom only |

**Conclusions for this model:**
1. Only all-IQ3 families fit one A770. **Policy A' does not** — down-proj is a
   third of 32B of experts, so Q4_K down alone is 6 GB. A' is retained as a
   *quality/throughput comparison* and a multi-card (M-row) baseline, not a
   single-card candidate.
2. The binding quality risk is **down-proj at 3-bit**. The within-budget hedge is
   **Policy D** (down → L2 asym 3.5).
3. Hybrid linear attention makes KV nearly free → **128k context is plausible
   single-card** under Policy C, which is a stronger story than a dense 30B-A3B.

---

## 7. Composition with later patches

- `0006` SYCL dequant + vecdot: direct port of §3.2/§3.3, no grid.
- `0007` reordered `MUL_MAT_ID`: reuses §4 offsets + existing reorder tiling.
- `0008` fused gate/up SwiGLU: gate & up both `IQ3_A770` → two `dp4a` accums over
  one shared `q8_K` activation, then SwiGLU. (`down` may be L2 under Policy D —
  the fusion only spans gate/up, so this is fine.)
- `0009` tensor policy + ftype: encodes C / D / A' as the quant mix.

---

## 8. Build / test plan for 0005

1. Implement struct + ref quant/dequant/dot (CPU). Unit: round-trip a random
   tensor, check max abs error vs symmetric 3-bit bound.
2. `test-backend-ops -o MUL_MAT -o MUL_MAT_ID` CPU pass.
3. Quantize NX2 with `--imatrix NX2.imatrix` under Policy C and Policy D.
4. KLD/top-1 vs `NX2-Q6K.kld`; PPL spot vs `Nex-N2-mini.Q4_K_M`.
5. Record GGUF size + computed residency ledger per policy.

Gate to `0006` (SYCL): C and/or D pass MUL_MAT_ID parity, beat stock Q3 KLD, and
fit the audited 16 GB ledger.

---

## 9. Open decisions

1. **Single-card policy set.** Earlier pick was "build C and A' and compare," but
   the real ledger shows **A' does not fit one A770**. Recommend the single-card
   track be **C (primary) + D (down-proj hedge)**, and keep **A' as the
   multi-card / headroom comparison row**. Confirm or override.
2. **L3 4-bit sub-scales are coarse.** Accept for gate/up (imatrix-mitigated);
   the hedge is L2 for down. OK, or widen L3 scales (would push >102 B)?
3. **Non-expert precision:** keep Q6 (2.08 GB) or trim embeds/output/attn to Q5
   (~1.78 GB) to buy ledger headroom for Policy D / 128k context?

**Resolved:** (1) single-card track = **C + D**, A' = comparison only; (2) L3
4-bit scales accepted, L2 down is the hedge; (3) non-expert kept **Q6** for
Policy C (fits comfortably — see §10). Branch `a770-iq3` off `fabler` created;
spec frozen; CPU oracle landed in `0005`.

---

## 10. Implementation & A0 results

Patch [`0005-iq3-a770-plumbing.patch`](../patches/0005-iq3-a770-plumbing.patch)
(branch `a770-iq3` off `fabler`, llama.cpp commit `d730e7745`). CPU-only; disjoint
from the `0004` SYCL WIP. Touches 11 files:

- `ggml.h` (type 42 `GGML_TYPE_IQ3_A770`, ftype 28), `ggml-common.h`
  (`block_iq3_a770`, 102 B).
- `ggml-quants.{h,c}` (ref quant/dequant, imatrix-aware `quantize_iq3_a770`,
  `ggml_validate_row_data`), `ggml.c` (type_traits, ftype map, `quantize_chunk`).
- `ggml-cpu/{quants.h,quants.c,ggml-cpu.c,arch-fallback.h}` (`vec_dot_iq3_a770_q8_K`
  generic + traits, `vec_dot_type=Q8_K`).
- `tests/test-backend-ops.cpp` (type added to `all_types`/`other_types`),
  `tests/iq3_a770_oracle_check.c` (standalone correctness check).

**Build:** clean CPU build (`-march=native`); only a pre-existing `-Wswitch`
warning in `ggml_compute_forward_clamp` (clamp never runs on quant types).

**A0 correctness:**

| check | result |
|---|---|
| `vec_dot` vs dequantize-then-dot (oracle micro-test) | rel diff **9.3e-08** (bit-exact) |
| weight round-trip NRMSE (random gaussian, ref path) | **0.142** |
| Policy C quantize (NX2-bf16 → iq3_a770 experts, imatrix) | **14.92 GB** (3.44 BPW overall; experts 3.19 bpw, 512→102 MiB/tensor) |
| end-to-end CPU generation (`"The capital of France is"`) | **"Paris."** — coherent, 7.9 t/s decode |
| KLD vs `NX2-Q6K.kld` (wikitext-2, 100×512 tok) | **mean 0.0754**, median 0.0307, 99.9% 2.456, top-1 **88.5%**, PPL ratio **1.030** |

**Quality gate — beats stock Q3:** mean KLD **0.0754** sits between Q4_K (0.039)
and Q3_K_M (0.105) on the project frontier, at **14.9 GB** (smaller than Q3_K_M's
16.8 GB) — i.e. better quality than stock Q3 at smaller size. ✅

**Residency confirmation:** Policy C measured **14.92 GB** vs the ~14.9 GB ledger
prediction (§6) — fits one A770 16 GB with the hybrid-attention KV headroom.

**Usage (no dedicated ftype needed):**

```bash
llama-quantize --imatrix NX2.imatrix \
  --tensor-type ffn_gate_exps=iq3_a770 \
  --tensor-type ffn_up_exps=iq3_a770 \
  --tensor-type ffn_down_exps=iq3_a770 \
  NX2-bf16.gguf NX2-IQ3_A770-C.gguf q6_k
```

**Follow-ups:** L2 asym-down type (makes Policy D real; today D falls back to a
Q4_K-down "A'-style" point); dedicated `LLAMA_FTYPE_MOSTLY_IQ3_A770` +
`llama_tensor_get_type` policy; `0006`/`0007` SYCL dequant + reordered
`MUL_MAT_ID` (enables true CPU-vs-SYCL `MUL_MAT_ID` parity and A770 throughput).
