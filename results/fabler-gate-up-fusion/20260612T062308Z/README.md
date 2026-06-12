# Fabler campaign: SYCL MoE gate/up fusion and NX2 specialization

Date: 2026-06-12. Hardware: Intel Arc Pro B70, oneAPI icpx 2026.0, Level Zero 1.15.38308.
Branch: llama.cpp `fabler` (`0001..0003` at `e0bfc65c2`) plus exported
[`patches/0004-sycl-fuse-moe-gate-up-glu.patch`](../../../patches/0004-sycl-fuse-moe-gate-up-glu.patch).

## What changed

`0004` adds two SYCL MoE decode experiments.

The first is an experimental execution fusion for the decode-shape
`MUL_MAT_ID, MUL_MAT_ID, GLU` MoE gate/up pattern:

- same activation tensor and ids tensor for gate and up
- same expert weight type, shape, and stride
- F32 activation/output, contiguous decode/TG input
- unswapped `SWIGLU`
- reordered Q4_K/Q5_K/Q6_K expert weights only

The fused kernel quantizes the shared activation row once into Q8_1 SoA, reads
gate and up expert tensors in the same launch, computes both dot products, and
writes `silu(gate) * up` directly to the GLU destination.

The second is a default-on exact-shape specialization for NX2 gate/up reordered
MoE GEMV: Q4_K/Q5_K expert weights, `ncols=2048`, `nrows=512`, and top-k
`n_experts_used=8`. It unrolls the eight K-blocks per row and removes the generic
runtime block math. It can be disabled with:

```bash
GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=0
```

## Verdict

The direct gate+up fusion path is correct in targeted backend-op testing but
**does not clear the Q5_K_M acceptance gate**. It is therefore **disabled by
default** and only runs with:

```bash
GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=1
```

`GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=2` runs a shared-Q8 two-output experiment
that quantizes once but leaves the normal GLU node in place; it was also slower
on Q5.

The NX2 exact-shape specialization **does** clear the Q5_K_M ctx0 acceptance
gate and stays default-on:

| model | mode | ctx0 decode |
|---|---|---:|
| NX2-Q5_K_M | baseline default, no 0004 specialization | 85.04 t/s |
| NX2-Q5_K_M | direct gate+up fusion opt-in | 84.73 t/s |
| NX2-Q5_K_M | shared-Q8 experiment opt-in | 84.73 t/s |
| NX2-Q5_K_M | NX2 specialization default-on | 86.74 t/s |

Guardrails:

| model | mode | ctx0 decode |
|---|---|---:|
| NX2-Q4_K_M | baseline default, no 0004 specialization | 90.66 t/s |
| NX2-Q4_K_M | direct gate+up fusion opt-in | 91.70 t/s |
| NX2-Q4_K_M | NX2 specialization default-on | 90.46 t/s |

The Q4 delta is within run noise. The long-context guardrail does not regress:

| model | mode | 131k decode |
|---|---|---:|
| NX2-Q5_K_M | baseline default, no 0004 specialization | 42.0707 t/s |
| NX2-Q5_K_M | NX2 specialization default-on | 42.1837 t/s |
| retained prior deployed control | default off | 42.0656 t/s |

## Correctness

- Build: `cmake --build build --target llama-bench test-backend-ops -j 8`
- `test-backend-ops test -o MUL_MAT_ID`: **690/690** passed with the default-on
  NX2 specialization path.
- `GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=1 test-backend-ops test -o MUL_MAT_ID`:
  **690/690** passed with the opt-in path.
- `GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=2 test-backend-ops test -o MUL_MAT_ID`:
  **690/690** passed with the shared-Q8 experiment.
- Full `test-backend-ops test`: **11477/11489** passed; the 12 failures are the
  known quantized `GET_ROWS` tolerance failures already present in retained upstream
  logs and unrelated to this MoE path.

## Raw artifacts

- `test-backend-ops-mulmatid.txt` — targeted op test, default off
- `test-backend-ops-mulmatid-fusion-on.txt` — targeted op test, opt-in fusion
- `test-backend-ops-mulmatid-shared-q8.txt` — targeted op test, shared-Q8 mode
- `test-backend-ops-mulmatid-nx2-specialize-default.txt` — targeted op test,
  default-on NX2 specialization
- `test-backend-ops-full.txt` — full backend-op run with known GET_ROWS failures
- `q5km-ctx0-default-off.json` / `.log` — baseline behavior before default-on NX2 specialization
- `q5km-ctx0-fusion-on.json` / `.log` — primary Q5 opt-in fusion
- `q5km-ctx0-shared-q8.json` / `.log` — primary Q5 shared-Q8 experiment
- `q5km-ctx0-nx2-specialize-default.json` / `.log` — primary Q5 packaged behavior
- `q4km-ctx0-default-off.json` / `.log` — Q4 guardrail before default-on NX2 specialization
- `q4km-ctx0-fusion-on.json` / `.log` — Q4 opt-in fusion
- `q4km-ctx0-nx2-specialize-default.json` / `.log` — Q4 packaged behavior
- `q5km-131k-default-off.json` / `.log` — long-context baseline
- `q5km-131k-nx2-specialize.json` / `.log` — long-context packaged behavior
