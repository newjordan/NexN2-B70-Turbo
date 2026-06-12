# Fabler micro-experiment: NX2 no-row-guard and dispatch cleanup

Date: 2026-06-12. Hardware: Intel Arc Pro B70, oneAPI icpx 2026.0.
Branch: llama.cpp `fabler` with the retained default mode `3` NX2 fused gate/up
SwiGLU path from `0004`.

## Experiment

Two small hot-path cleanups were tested on top of the retained exact NX2 fused
gate/up path:

1. Remove the row bounds guard from the exact-shape NX2 kernels. For this launch
   shape, `nrows=512` and `GGML_SYCL_MMV_Y=1`, so the launch emits exactly 512
   row groups.
2. Reuse the mode `3` exact-shape eligibility result and skip a second full
   fusion eligibility check before launching the fused kernel.

## Result

The row-guard removal alone was correctness-clean but did not improve the primary
Q5_K_M scoreboard:

| model | mode | ctx0 decode |
|---|---|---:|
| NX2-Q5_K_M | no row guard | 87.06 t/s |
| NX2-Q4_K_M | no row guard | 91.94 t/s |

Adding the dispatch cleanup improved Q4 but regressed Q5 repeatably:

| model | mode | ctx0 decode |
|---|---|---:|
| NX2-Q5_K_M | dispatch cleanup | 85.18 t/s |
| NX2-Q5_K_M | dispatch cleanup repeat | 85.06 t/s |
| NX2-Q4_K_M | dispatch cleanup | 92.26 t/s |

Targeted `MUL_MAT_ID` testing passed for both builds. Because Q5_K_M ctx0 is the
primary scoreboard, neither cleanup is retained in `0004`.

## Raw artifacts

- `test-backend-ops-mulmatid-default-no-row-guard.txt`
- `q5km-ctx0-default-no-row-guard.json` / `.log`
- `q4km-ctx0-default-no-row-guard.json` / `.log`
- `test-backend-ops-mulmatid-default-dispatch-cleanup.txt`
- `q5km-ctx0-default-dispatch-cleanup.json` / `.log`
- `q5km-ctx0-default-dispatch-cleanup-repeat.json` / `.log`
- `q4km-ctx0-default-dispatch-cleanup.json` / `.log`
