# A770 IQ3 Expert Residency Variant Plan

This branch tracks an A770-specific residency experiment, not a replacement for
the B70 Turbo release path. The working target is a 30B-A3B MoE that can live on
one Intel Arc A770 16GB while retaining enough quality to beat the practical
alternatives: split-card Q4, stock IQ4/Q3 paths, or CPU spill.

`IQ3_A770` is the branch-local working name for the quant/kernel path. The
runtime idea is active expert residency: keep the current MoE expert working set
resident, switch expert groups when the route pattern demands it, and avoid
pretending every expert has to be hot at once. Until a concrete block layout and
SYCL kernel land, treat `IQ3_A770` as a design target, not a shipped quant type.

## Success Criterion

The A770 variant is successful if it proves this claim on real hardware:

```text
30B-A3B IQ3_A770 fits on one A770 16GB,
keeps KLD/top-1 closer to IQ4/Q4 than stock Q3,
and decodes faster than split-card Q4 or any spill path.
```

That is deliberately different from the B70 criterion. B70 Turbo is a speed and
quality frontier for a 35B-A3B model with enough VRAM for Q4/Q5. A770 IQ3_A770
is a residency frontier for a smaller 30B-A3B model where VRAM is the primary
constraint.

## Initial Scope

Start narrow:

- one 30B-A3B target model
- one block layout
- one tensor policy
- one-card residency discipline
- `MUL_MAT_ID` decode path first
- CPU reference path only as a correctness oracle

Do not start by generalizing across dense matmul, every quant mix, prefill, and
multi-card split behavior. Those come after single-card decode is real.

## Calibration Phases

Use the B70 first, but make it obey the A770 problem. The B70 has enough VRAM to
hide mistakes, so every B70 calibration run for this branch must report whether
the model would fit inside a 16GB residency envelope:

```text
phase 1: B70, 16GB-constrained discipline
  validate IQ3_A770 serialization and CPU reference behavior
  measure IQ3 decode throughput with A770-style tensor policy
  keep reported model residency, KV, scratch, and lazy-reorder buffers <= 16GB
  fail any run that needs B70-only memory comfort to look good

phase 2: single A770
  measure IQ3_A770 single-card throughput first
  retune subgroup/workgroup/register choices on Alchemist
  rerun the same IQ3_A770 rows and believe these numbers for promotion

phase 3: multi-card concept
  test split-card Q4/Q5 and IQ3_A770 variants after single-card decode is real
  include symlink/layout simulation only as a follow-up plumbing experiment
```

The B70 result is allowed to guide kernel shape and quant policy. It is not
allowed to prove the final A770 claim by itself.

## Patch Shape

Base from the retained Turbo chain:

```text
0001 Q4/Q5 MoE reorder
0002 concat submit-only
0003 Q6_K MoE reorder, if still useful for the chosen tensor policy
selected 0004 fusions, gated to the exact model shape being tested
```

Planned A770 additions:

```text
0005 GGML_TYPE_IQ3_A770 plumbing
0006 IQ3 SYCL dequant + vecdot reference kernel
0007 IQ3 MoE MUL_MAT_ID reordered decode path
0008 exact 30B MoE fused gate/up path
0009 tensor policy + quant ftype
```

Patch `0005` should be correctness-first: serialization, CPU dequant, CPU dot,
and backend-op coverage. Patch `0007` is the first performance gate because it
proves whether the layout belongs in the MoE expert GEMV path.

## Tensor Policy

Default fit-first candidates:

```text
A. gate/up experts: IQ3_A770, down experts: Q5_K or Q4_K
B. gate/up experts: IQ3_A770, down experts: IQ4_XS or Q4_K
C. all expert projections: IQ3_A770, only if KLD/top-1 survives
```

Keep these higher precision until the data says otherwise:

- router and expert gate tensors
- token embeddings
- output head
- attention-sensitive tensors
- shared expert tensors, if present
- down projections, if they dominate quality loss

The B70 Q6_K-down policy should not be copied blindly. It was a good B70
quality/performance tradeoff, but on A770 it may consume the margin needed for
KV cache, scratch, and comfortable residency.

## Kernel Rules

The A770 path should avoid standalone dequant writes in the hot path. Smaller
weights only matter if the consumer kernel reads the packed form directly and
does useful work before writing results back to memory.

Priority order:

1. `MUL_MAT_ID` decode correctness against CPU for IQ3_A770.
2. Reordered per-expert layout in the SYCL expert GEMV path.
3. Fused gate/up SwiGLU for the exact 30B-A3B shape.
4. Tail, weighted-sum, and handoff fusions only when measured positive on A770.
5. Active expert residency controls only after the base IQ3_A770 path is
   correct and measurable.

B70 remains useful for compile, correctness, and rough roofline sanity. Final
promotion numbers must come from A770 because Alchemist scheduling, subgroup
shape, register pressure, and memory behavior can diverge from Battlemage.

For B70 calibration, throughput is useful only when paired with the 16GB
residency ledger. A fast IQ3 path that relies on extra B70 VRAM for reordered
copies, scratch, or large KV comfort is not a valid A770 candidate.

## Benchmark Matrix

Run every row on the same 30B-A3B target where possible.

| config | purpose |
|---|---|
| IQ3_A770 on B70 under 16GB ledger | first throughput calibration |
| IQ3_A770 on one A770 | promotion hardware |
| Q4_K_M split across 2x A770 | quality/performance baseline |
| IQ4_XS single or split | existing low-bit baseline |
| Q3_K_M / Q3_K_S stock | lower-bit quality floor |
| IQ3_A770 CPU-ref only | serialization and correctness gate |
| IQ3_A770 SYCL no fusion | isolate quant/kernel cost |
| IQ3_A770 + MoE reorder | first real candidate |
| IQ3_A770 + fused gate/up | primary candidate |
| IQ3_A770 + active expert residency | residency candidate |
| IQ3_A770 + tail/weighted-sum fusions | keep only if measured positive |

Required metrics:

- GGUF size and model GiB resident
- peak VRAM
- ctx0 decode
- 4k, 16k, and 32k decode
- prefill
- PPL
- mean KLD versus Q6 or f16 reference
- top-1 agreement
- `MUL_MAT_ID` backend-op parity
- first-token timing after lazy reorder

## Promotion Bar

A candidate can move from B70 calibration to A770 hardware testing when all of
these are true:

- it fits a B70 run with an audited 16GB residency ledger, target context, and
  no CPU spill
- `MUL_MAT_ID` backend-op tests pass against CPU reference
- PPL does not show a material regression versus the selected low-bit baseline
- KLD/top-1 lands closer to IQ4/Q4 than stock Q3
- the first-token lazy-reorder cost is measured and documented

It can move from A770 testing to a retained candidate only when it also:

- fits one A770 16GB with the target context and no CPU spill
- measures IQ3_A770 single-card throughput before any multi-card path
- decodes faster than the split-card Q4 or spill-path baseline for the target
  use case

Final release claims require the single-A770 phase. The B70 16GB run can
promote a kernel or tensor policy to A770 testing, not publish the A770 result.
