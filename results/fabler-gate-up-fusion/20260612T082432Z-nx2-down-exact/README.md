# Fabler micro-experiment: exact NX2 down-projection GEMV

Date: 2026-06-12. Hardware: Intel Arc Pro B70, oneAPI icpx 2026.0.
Branch: llama.cpp `fabler` with retained `0004` mode `3` exact NX2 fused gate/up
SwiGLU path.

## Experiment

After gate/up fusion cleared the primary Q5_K_M gate, this tested an exact-shape
MoE GEMV specialization for the NX2 down projection:

- `ncols=512`, `nrows=2048`, top-k 8
- compile-time block count and `nblocks`
- initial variant covered Q4_K/Q5_K/Q6_K down tensors
- narrowed variant covered Q6_K down tensors only

The motivation was to attack `ffn_down_exps`, which is mostly Q6_K in the flagship
Q4_K_M/Q5_K_M mixes.

## Result

Correctness passed, but the primary Q5_K_M ctx0 scoreboard did not improve.

All down types exact:

| model | mode | ctx0 decode |
|---|---|---:|
| NX2-Q5_K_M | Q4/Q5/Q6 down exact | 85.37 t/s |
| NX2-Q5_K_M | Q4/Q5/Q6 down exact repeat | 86.87 t/s |
| NX2-Q4_K_M | Q4/Q5/Q6 down exact | 93.53 t/s |

Q6_K-only exact:

| model | mode | ctx0 decode |
|---|---|---:|
| NX2-Q5_K_M | Q6 down exact only | 86.77 t/s |
| NX2-Q4_K_M | Q6 down exact only | 91.55 t/s |

The all-down variant was a strong Q4 result, but Q5_K_M ctx0 is the primary
scoreboard and remained below the retained 87.19 t/s mode `3` result. The
experiment is not retained in `0004`.

## Raw artifacts

- `test-backend-ops-mulmatid-down-exact.txt`
- `q5km-ctx0-down-exact.json` / `.log`
- `q5km-ctx0-down-exact-repeat.json` / `.log`
- `q4km-ctx0-down-exact.json` / `.log`
- `test-backend-ops-mulmatid-q6-down-exact-only.txt`
- `q5km-ctx0-q6-down-exact-only.json` / `.log`
- `q4km-ctx0-q6-down-exact-only.json` / `.log`
