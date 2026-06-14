# llama.cpp SYCL backend patches

Reproducible patches generated against ggml-org/llama.cpp base commit `ac4cddeb0`
(`vendor : update LibreSSL to 4.3.2`, ggml-org/llama.cpp #24397).
The backend changes were developed and validated locally for the Intel Arc Pro B70.
Upstream PR for the reusable SYCL MoE reorder subset (`0001` plus the Q6_K
coverage from `0003`): https://github.com/ggml-org/llama.cpp/pull/24452
Patch `0004` is the project-local NX2 fusion package and is not part of that
minimal upstream PR.

## A770 IQ3 expert residency branch

Branch `a770-iq3-expert-residency` keeps the B70 Turbo patch chain intact and
uses it as the base for an A770 16GB residency experiment. The branch target is
**Nex-N2-mini** (`qwen35moe`: ~34.7B total / ~A3B active, 256 experts, hybrid
linear-attention with full attention every 4th layer), used as a single-card
A770 residency testbed rather than the B70 Turbo release model. The success
criterion is single-card residency plus usable fused-SYCL decode rather than
beating the B70 Q4/Q5 frontier.

Early calibration may run on the B70, but those runs are valid for this branch
only when they keep an audited 16GB residency ledger for model weights, KV,
scratch, and lazy-reorder buffers. Single-A770 throughput remains the promotion
hardware. Active expert residency means switching which MoE expert groups are
hot when the route pattern demands it; multi-card and symlink/layout simulation
work comes after the single-card IQ3 path is real.

A770-only patches start after the retained Turbo chain:

```text
0005 GGML_TYPE_IQ3_A770 plumbing            (LANDED - CPU oracle)
0006 IQ3 SYCL dequant + vecdot reference kernel (M1 dequant + M2 dp4a mmvq - landed in wip/, B70-validated)
0007 IQ3 MoE MUL_MAT_ID reordered decode path  (LANDED in wip/ - tg 43.6->78.7 t/s, beats stock Q3_K_M 62; needs init_tensor extra fix)
0008 exact MoE fused gate/up path
0009 tensor policy + quant ftype (L2 asym down, dedicated LLAMA_FTYPE)
```

**`0005` (landed):** `block_iq3_a770` — a codebook-free 3-bit quant using Q3_K's
`qs`/`hmask` packing with 8 sub-blocks of 32 and 4-bit signed scales (102 B,
3.1875 bpw). CPU quantize/dequant/`vec_dot` (`vec_dot_type = Q8_K`) + full type
registration; no SYCL (disjoint from the `0004` SYCL WIP). A0 validation: oracle
`vec_dot` is bit-exact vs dequantize-then-dot (rel diff ~1e-7), round-trip NRMSE
0.142, **Policy C quantize = 14.92 GB** (fits one A770 16 GB), and CPU inference
generates coherent text. Use it now via `--tensor-type ffn_*_exps=iq3_a770` on a
`q6_k` base (Policy C) — no dedicated ftype required yet. See
[`../docs/iq3-a770-block-layout.md`](../docs/iq3-a770-block-layout.md) §10.

See [`../docs/a770-iq3-expert-residency.md`](../docs/a770-iq3-expert-residency.md)
for the tensor policy, benchmark matrix, and promotion bar, and
[`../docs/iq3-a770-block-layout.md`](../docs/iq3-a770-block-layout.md) for the
frozen block layout. `0006+` (SYCL) remain to be built.

Apply on a clean checkout at the pinned base, in order:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout ac4cddeb0
git am /path/to/NexN2-B70-Turbo/patches/000*.patch
```

- **0001 — sycl: support reordered Q4_K and Q5_K MoE `MUL_MAT_ID`** — extends the SYCL
  weight-reorder optimization to the fused MoE expert GEMV (`mul_mat_id`), with
  reorder-aware reads across every decode and prefill sub-path (decode GEMV, dense
  mmvq, DMMV, and the dequant-to-fp16/fp32 GEMM path). Adds a Q5_K reorder DMMV
  kernel + per-expert reorder converters. Correctness-clean in retained targeted
  checks (714/714 MUL_MAT_ID op tests, PPL unchanged).
  **2026-06-10 re-validation** ([`../results/upstream-pr/SUMMARY.md`](../results/upstream-pr/SUMMARY.md)):
  treat this as a llama.cpp contribution-readiness check for one code patch, separate from
  the project-level NX2 GGUF benchmark results.
- **0002 — sycl: make concat submit-only** — drops two redundant host waits in
  `ggml_sycl_op_concat` (the queue is in-order); removes 17 host syncs per decoded
  token on NexN2 and makes CONCAT recordable into SYCL graphs.
- **0003 — sycl: extend MoE reorder to Q6_K `mul_mat_id`** — per-expert SoA reorder +
  fused reordered GEMV for Q6_K expert weights (`ffn_down_exps` in the NX2
  Q4_K_M/Q5_K_M mixes — about a third of the expert bytes), reusing the existing dense
  Q6_K reorder traits/readers. Also admits graph-safe `MUL_MAT_ID` nodes in
  `check_graph_compatibility` (SYCL graphs remain default-off; see the negative replay
  result in [`../results/fabler-q6k-reorder/`](../results/fabler-q6k-reorder/)).
  **2026-06-12 validation:** 714/714 MUL_MAT_ID op tests on the pinned-base chain,
  PPL statistically unchanged (5.5723 ± 0.153 vs 5.5643 ± 0.152), ctx0 decode
  Q5_K_M 81.1 → 85.8 t/s, Q4_K_M 85.7 → 91.0 t/s.
- **0004 — sycl: specialize NX2 MoE gate/up, weighted sum, and tail add** —
  default-on exact NX2 Q4_K/Q5_K fused gate/up SwiGLU specialization
  (`2048x512`, top-k 8), default-on post-down weighted expert sum fusion, plus
  a default-on F32 MoE tail-add fusion. The default mode `3` clears the Q5_K_M
  ctx0 gate (85.04 → 88.08 t/s, +3.57%), keeps 30-chunk PPL flat
  (5.5642 ± 0.15223), and has no retained 131k regression over noise
  (42.29 t/s). Q4_K_M ctx0 reaches 93.49 t/s. Raw results:
  [`../results/fabler-gate-up-fusion/20260612T083707Z-nx2-weighted-sum-fusion/`](../results/fabler-gate-up-fusion/20260612T083707Z-nx2-weighted-sum-fusion/).
  [`../results/fabler-gate-up-fusion/20260612T091359Z-nx2-tail-add-fusion/`](../results/fabler-gate-up-fusion/20260612T091359Z-nx2-tail-add-fusion/).
  A shared-expert dense gate/up fusion probe is present but disabled by default
  after measuring slower Q5_K_M ctx0 decode (87.27 off vs 86.90 on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T090325Z-nx2-shared-gate-up-fusion/`](../results/fabler-gate-up-fusion/20260612T090325Z-nx2-shared-gate-up-fusion/).
  An activation-Q8 cache probe is present as
  `GGML_SYCL_ENABLE_MOE_ACT_Q8_CACHE=1`; it lets the shared-expert dense gate/up
  fusion reuse the MoE gate/up activation quantization, but remains default-off
  after losing to local requantization (88.14 cache-off vs 87.76 cache-on, then
  87.85 cache-off vs 87.43 cache-on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T114939Z-nx2-activation-q8-cache/`](../results/fabler-gate-up-fusion/20260612T114939Z-nx2-activation-q8-cache/).
  A gate/up XOR reducer probe is present as
  `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=2`, but mode `1` remains default after
  measuring faster (87.96 vs 87.81 t/s); raw result:
  [`../results/fabler-gate-up-fusion/20260612T095919Z-nx2-gate-up-xor-reduce/`](../results/fabler-gate-up-fusion/20260612T095919Z-nx2-gate-up-xor-reduce/).
  Gate/up SwiGLU activation probes are present as
  `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=3` (`exp2` + reciprocal) and mode `4`
  (`exp` + reciprocal), but mode `1` remains default after both measured
  slower; raw result:
  [`../results/fabler-gate-up-fusion/20260612T102545Z-nx2-swiglu-exp2-recip/`](../results/fabler-gate-up-fusion/20260612T102545Z-nx2-swiglu-exp2-recip/).
  A gate/up rowpack scheduling probe is present as
  `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=5`, but mode `1` remains default after
  measuring faster (88.10 vs 87.70 t/s); raw result:
  [`../results/fabler-gate-up-fusion/20260612T103921Z-nx2-gate-up-rowpack4/`](../results/fabler-gate-up-fusion/20260612T103921Z-nx2-gate-up-rowpack4/).
  Rowpack modes `6` and `7` test two and eight output rows per workgroup; both
  passed targeted tests, but mode `1` remains default after the reversed Q5_K_M
  repeat and Q4_K_M guard favored mode `1` over mode `6` (87.72 vs 87.46 t/s
  on Q5, 93.80 vs 93.53 t/s on Q4); raw result:
  [`../results/fabler-gate-up-fusion/20260612T113923Z-nx2-gate-up-rowpack2-8/`](../results/fabler-gate-up-fusion/20260612T113923Z-nx2-gate-up-rowpack2-8/).
  Mode `8` copies the shared 2048-wide reordered Q8_1 activation row into
  workgroup-local memory for a four-row gate/up workgroup. It passed targeted
  testing but remains default-off after tying retained mode on Q5_K_M across a
  reversed repeat (88.20 vs 88.17, then 87.67 vs 87.67 t/s) and regressing
  Q4_K_M (90.08 vs 93.91 t/s); raw result:
  [`../results/fabler-gate-up-fusion/20260612T130548Z-nx2-gate-up-local-q8/`](../results/fabler-gate-up-fusion/20260612T130548Z-nx2-gate-up-local-q8/).
  Modes `9` and `10` test an expert-pack scheduler, with one selected expert
  per workgroup and eight subgroups covering the top-k experts for a row; mode
  `10` also uses the local-Q8 activation cache. Both passed targeted tests, but
  remain default-off after mode `9` stayed below the Q5_K_M +2% promotion bar
  across a reversed repeat (86.83 vs 88.09 t/s, then 87.00 vs 87.13 t/s) and
  mode `10` trailed mode `9` (87.39 t/s). Q4_K_M mode `9` measured 92.75 vs
  93.87 t/s; raw result:
  [`../results/fabler-gate-up-fusion/20260612T131422Z-nx2-gate-up-expert-pack/`](../results/fabler-gate-up-fusion/20260612T131422Z-nx2-gate-up-expert-pack/).
  Mode `11` fuses the reordered Q4_K/Q5_K vecdot helper itself, sharing Q8_1
  activation lane loads while accumulating up and gate dots in one inner loop.
  It remains default-off after passing targeted tests but tying retained Q5_K_M
  after the reversed repeat (87.07 vs 87.59 t/s, then 87.29 vs 87.30 t/s) and
  regressing Q4_K_M (94.15 vs 92.76 t/s); raw result:
  [`../results/fabler-gate-up-fusion/20260612T133008Z-nx2-gate-up-dual-dot/`](../results/fabler-gate-up-fusion/20260612T133008Z-nx2-gate-up-dual-dot/).
  A gate/up Q8 handoff probe is present as
  `GGML_SYCL_ENABLE_MOE_GATE_UP_Q8_HANDOFF=1`; it writes a down-ready Q8_1 side
  buffer from the fused gate/up kernel and lets the following Q6_K down
  projection skip quantization, but remains default-off after mixed/slower
  repeats (88.02 on vs 87.93 off, then 87.34 on vs 87.56 off); raw result:
  [`../results/fabler-gate-up-fusion/20260612T104952Z-nx2-gate-up-q8-handoff/`](../results/fabler-gate-up-fusion/20260612T104952Z-nx2-gate-up-q8-handoff/).
  A current-stack exact Q6_K down specialization is present as
  `GGML_SYCL_ENABLE_MOE_Q6_DOWN_NX2_SPECIALIZE=1`, but remains default-off
  after measuring slower (87.84 off vs 87.62 on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T110216Z-nx2-q6-down-exact-current/`](../results/fabler-gate-up-fusion/20260612T110216Z-nx2-q6-down-exact-current/).
  Combining Q8 handoff with exact Q6_K down also remains default-off after
  measuring slower (87.68 default vs 87.51 combo); raw result:
  [`../results/fabler-gate-up-fusion/20260612T110640Z-nx2-q8-handoff-q6-down-exact-combo/`](../results/fabler-gate-up-fusion/20260612T110640Z-nx2-q8-handoff-q6-down-exact-combo/).
  A shared-gate-tail probe is present as
  `GGML_SYCL_ENABLE_MOE_SHARED_GATE_TAIL_FUSION=1`; it fuses the scalar shared
  expert gate multiply and the two F32 tail adds, but remains default-off
  because Q5_K_M ctx0 improved only below the +2% promotion bar across noisy
  repeats (87.81 off vs 88.16 on, then 87.67 off vs 87.83 on). Q4_K_M ctx0
  measured 92.64 off vs 94.22 on; raw result:
  [`../results/fabler-gate-up-fusion/20260612T111328Z-nx2-shared-gate-tail-fusion/`](../results/fabler-gate-up-fusion/20260612T111328Z-nx2-shared-gate-tail-fusion/).
  Mode `2` vectorizes the same shared-gate-tail kernel over four rows per
  work-item and is also default-off after passing targeted tests but missing
  the Q5_K_M +2% promotion bar (87.07 off vs 87.88 on, then 87.22 off vs
  87.80 on); Q4_K_M measured 92.00 off vs 94.43 on; raw result:
  [`../results/fabler-gate-up-fusion/20260612T112008Z-nx2-shared-gate-tail-vec4/`](../results/fabler-gate-up-fusion/20260612T112008Z-nx2-shared-gate-tail-vec4/).
  Modes `5` and `6` broadcast the scalar shared-expert gate through
  workgroup-local memory for scalar and vec4 tails, but also remain default-off
  after passing targeted tests and measuring slower than the same-run retained
  default on Q5_K_M (87.81 default vs 87.65 mode `5` and 87.55 mode `6`);
  raw result:
  [`../results/fabler-gate-up-fusion/20260612T125854Z-nx2-shared-gate-tail-local-gate/`](../results/fabler-gate-up-fusion/20260612T125854Z-nx2-shared-gate-tail-local-gate/).
  Modes `3` and `4` additionally consume the preceding scalar sigmoid; mode
  `4` is the vec4 local-broadcast variant. They remain default-off after mode
  `4` measured mixed Q5_K_M repeats (87.18 off vs 88.15 on, then 87.82 off vs
  87.47 on), while Q4_K_M measured 92.75 off vs 94.52 on; raw result:
  [`../results/fabler-gate-up-fusion/20260612T112753Z-nx2-shared-gate-sigmoid-tail/`](../results/fabler-gate-up-fusion/20260612T112753Z-nx2-shared-gate-sigmoid-tail/).
  A vec4 weighted-sum probe is also present as opt-in mode
  `GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=2` only after measuring slower than
  scalar/default (88.08 vs 87.91 t/s), with PPL flat; raw result:
  [`../results/fabler-gate-up-fusion/20260612T092946Z-nx2-weighted-sum-vec4/`](../results/fabler-gate-up-fusion/20260612T092946Z-nx2-weighted-sum-vec4/).
  A local-weight weighted-sum probe is present as opt-in mode
  `GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=3`, but measured slower than
  scalar/default (88.06 vs 87.60 t/s); raw result:
  [`../results/fabler-gate-up-fusion/20260612T103409Z-nx2-weighted-sum-local-weights/`](../results/fabler-gate-up-fusion/20260612T103409Z-nx2-weighted-sum-local-weights/).
  A Q6_K down+weighted-sum probe
  `GGML_SYCL_ENABLE_MOE_DOWN_WEIGHTED_SUM_FUSION=1` is present but default-off
  after measuring slower (87.98 off vs 87.85 on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T100913Z-nx2-down-weighted-sum/`](../results/fabler-gate-up-fusion/20260612T100913Z-nx2-down-weighted-sum/).
  Mode `GGML_SYCL_ENABLE_MOE_DOWN_WEIGHTED_SUM_FUSION=2` keeps per-expert
  parallelism and atomically accumulates weighted Q6_K down contributions, but
  also remains default-off after passing targeted `MUL_MAT_ID` tests and
  measuring slower (88.17 off vs 87.68 on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T101912Z-nx2-down-weighted-atomic/`](../results/fabler-gate-up-fusion/20260612T101912Z-nx2-down-weighted-atomic/).
  Mode `GGML_SYCL_ENABLE_MOE_DOWN_WEIGHTED_SUM_FUSION=3` keeps per-expert
  parallelism with one workgroup per output row and local-memory reduction
  instead of global atomics, but also remains default-off after passing targeted
  tests and measuring slower (87.99 off vs 87.68 on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T115730Z-nx2-down-weighted-local-reduce/`](../results/fabler-gate-up-fusion/20260612T115730Z-nx2-down-weighted-local-reduce/).
  A combined weighted-tail probe
  `GGML_SYCL_ENABLE_MOE_WEIGHTED_TAIL_FUSION=1` is present but default-off after
  measuring slower (87.98 off vs 87.74 on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T094531Z-nx2-weighted-tail-fusion/`](../results/fabler-gate-up-fusion/20260612T094531Z-nx2-weighted-tail-fusion/).
  A graph dispatch guard
  `GGML_SYCL_ENABLE_MOE_DISPATCH_GUARD=1` is present but default-off after
  measuring slower (88.09 off vs 87.78 on); raw result:
  [`../results/fabler-gate-up-fusion/20260612T101554Z-nx2-dispatch-guard/`](../results/fabler-gate-up-fusion/20260612T101554Z-nx2-dispatch-guard/).
  A tail-add vec4 probe is also present as opt-in
  `GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=2`; it was too noisy to promote, so
  scalar mode `1` remains default. Raw result:
  [`../results/fabler-gate-up-fusion/20260612T094758Z-nx2-tail-add-vec4/`](../results/fabler-gate-up-fusion/20260612T094758Z-nx2-tail-add-vec4/).
  A tail-add+RMS_NORM probe is present as
  `GGML_SYCL_ENABLE_MOE_TAIL_RMS_FUSION=1`; it writes both the final residual
  vector and RMS_NORM output from one kernel, but remains default-off after the
  primary Q5_K_M ctx0 path regressed (87.95/87.57 off vs 86.37 on). Q4_K_M
  improved in the single guard pair (91.78 off vs 92.77 on), but Q5_K_M is the
  promotion gate. Raw result:
  [`../results/fabler-gate-up-fusion/20260612T120710Z-nx2-tail-rms-fusion/`](../results/fabler-gate-up-fusion/20260612T120710Z-nx2-tail-rms-fusion/).
  A standalone RMS_NORM+MUL probe is present as
  `GGML_SYCL_ENABLE_RMS_NORM_MUL_FUSION=1` or `2`; mode `2` uses 1024
  work-items and was faster than the 512-work-item mode. It remains
  default-off because Q5_K_M ctx0 improved but stayed below the +2% promotion
  bar (88.05/87.84 off vs 89.15/88.97 on), while Q4_K_M improved in the guard
  pair (91.28 off vs 95.77 on). `RMS_NORM` testing passed 21/21,
  `MUL_MAT_ID` passed 690/690, and a 5-chunk Q5_K_M WikiText spot was flat
  (5.1289 off vs 5.1329 on). Raw result:
  [`../results/fabler-gate-up-fusion/20260612T122324Z-nx2-rms-norm-mul-fusion/`](../results/fabler-gate-up-fusion/20260612T122324Z-nx2-rms-norm-mul-fusion/).
  Mode `3` also writes a reordered Q8_1 side buffer from the RMS_NORM+MUL
  kernel and lets the following exact-NX2 MoE gate/up fusion skip activation
  quantization. A subgroup-parallel Q8 writer improved over the first serial
  in-kernel Q8 attempt (88.77 vs 88.04 t/s on Q5_K_M), but still trails
  standalone mode `2`; it remains default-off after Q5_K_M stayed below the
  +2% bar, while Q4_K_M improved in the guard pair (92.37 off vs 95.80 on).
  `RMS_NORM` passed 21/21, `MUL_MAT_ID` passed 690/690, and a 5-chunk Q5_K_M
  WikiText spot measured 5.1402. Raw result:
  [`../results/fabler-gate-up-fusion/20260612T123830Z-nx2-rms-mul-q8-handoff/`](../results/fabler-gate-up-fusion/20260612T123830Z-nx2-rms-mul-q8-handoff/).
  Mode `4` narrows that handoff with graph lookahead, writing Q8 only when the
  later exact-NX2 MoE gate/up fusion consumes the RMS_NORM+MUL output. It
  remains default-off after passing `RMS_NORM` 21/21 and `MUL_MAT_ID` 690/690
  but regressing Q5_K_M ctx0 versus same-run mode `2` (87.58/87.48 vs
  89.01 t/s); Q4_K_M mode `4` measured 94.32 t/s. Raw result:
  [`../results/fabler-gate-up-fusion/20260612T125005Z-nx2-rms-mul-selective-q8-handoff/`](../results/fabler-gate-up-fusion/20260612T125005Z-nx2-rms-mul-selective-q8-handoff/).

For a shorter campaign index, see
[`../results/fabler-gate-up-fusion/README.md`](../results/fabler-gate-up-fusion/README.md).

Build: oneAPI icpx 2026.0, SYCL backend, Intel Arc Pro B70 (Battlemage). WARP_SIZE 16.
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```
