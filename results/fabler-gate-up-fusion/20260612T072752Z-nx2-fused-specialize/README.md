# Fabler continuation: default NX2 fused gate/up specialization

Date: 2026-06-12. Hardware: Intel Arc Pro B70, oneAPI icpx 2026.0, Level Zero 1.15.38308.
Branch: llama.cpp `fabler` (`0001..0003` at `e0bfc65c2`) plus updated
[`patches/0004-sycl-fuse-moe-gate-up-glu.patch`](../../../patches/0004-sycl-fuse-moe-gate-up-glu.patch).

## What changed

The previous `0004` default specialized the separate NX2 Q4_K/Q5_K gate and up
expert GEMVs. This continuation specializes the direct fused SwiGLU kernel for
the same exact NX2 shape:

- Q4_K/Q5_K expert weights
- `ncols=2048`, `nrows=512`, top-k 8
- shared activation row quantized once to Q8_1 SoA
- gate and up dots computed in one kernel
- `silu(gate) * up` written directly to the GLU destination

`GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION` now defaults to mode `3`, which fuses only
this exact NX2 Q4_K/Q5_K shape. Mode `1` remains the broader direct-fusion
experiment and mode `2` remains the shared-Q8 two-output experiment.

## Results

Q5_K_M ctx0:

| mode | ctx0 decode |
|---|---:|
| prior pre-0004 baseline | 85.04 t/s |
| separate NX2 GEMV specialization, same build | 85.65 t/s |
| opt-in fused NX2 specialization | 86.78 t/s |
| default mode `3` fused NX2 specialization | **87.19 t/s** |

Guardrails:

| model | mode | decode |
|---|---|---:|
| NX2-Q4_K_M ctx0 | separate NX2 GEMV specialization | 90.77 t/s |
| NX2-Q4_K_M ctx0 | default mode `3` fused NX2 specialization | **91.58 t/s** |
| NX2-Q5_K_M 131k | fusion disabled repeat | 42.1585 t/s |
| NX2-Q5_K_M 131k | default mode `3` repeat | 42.0981 t/s |

An earlier single default-mode 131k run produced 40.90 t/s, but the immediate
disabled-vs-default repeat returned to the normal 42.1 t/s band. Treat the repeat
pair as the retained guardrail.

## Correctness

- Build: `cmake --build build --target llama-bench test-backend-ops -j 8`
- `GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=1 test-backend-ops test -o MUL_MAT_ID`:
  **690/690** passed after adding the NX2-specialized fused kernel.
- Default mode `3` `test-backend-ops test -o MUL_MAT_ID`: **690/690** passed.
- Q5_K_M 30-chunk WikiText PPL:
  - fusion disabled: **5.5669 +/- 0.15246**
  - default mode `3`: **5.5692 +/- 0.15251**

## Raw artifacts

- `test-backend-ops-mulmatid-fusion-nx2-specialize.txt`
- `test-backend-ops-mulmatid-default-mode3.txt`
- `q5km-ctx0-default-nx2-specialize.json` / `.log`
- `q5km-ctx0-fusion-nx2-specialize.json` / `.log`
- `q5km-ctx0-default-mode3.json` / `.log`
- `q4km-ctx0-default-nx2-specialize.json` / `.log`
- `q4km-ctx0-fusion-nx2-specialize.json` / `.log`
- `q4km-ctx0-default-mode3.json` / `.log`
- `q5km-131k-default-nx2-specialize.json` / `.log`
- `q5km-131k-fusion-nx2-specialize.json` / `.log`
- `q5km-131k-default-mode3.json` / `.log`
- `q5km-131k-fusion-disabled-repeat.json` / `.log`
- `q5km-131k-default-mode3-repeat.json` / `.log`
- `ppl-q5km-30chunks-fusion-disabled.txt`
- `ppl-q5km-30chunks-default-mode3.txt`
