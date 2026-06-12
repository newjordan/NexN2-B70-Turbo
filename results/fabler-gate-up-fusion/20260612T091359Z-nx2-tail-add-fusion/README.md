# NX2 Tail Add Fusion

Date: 2026-06-12

Experiment: fuse a narrow F32 `[2048]` two-add chain into one SYCL kernel. This
targets the NX2 MoE tail pattern:

1. `ffn_moe_out_merged = ffn_moe_out + ffn_moe_shexp`
2. `ffn_out = ffn_moe_out_merged + ffn_inp`

The matcher is shape- and use-count-gated, requires contiguous F32 vectors, and
is controlled by `GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION` (default on).

## Results

| check | result |
|---|---:|
| Q5_K_M ctx0, tail add off | 87.3238 t/s |
| Q5_K_M ctx0, tail add on | 87.9634 t/s |
| Q4_K_M ctx0, tail add on | 93.4917 t/s |
| Q5_K_M 131k, tail add on | 42.2928 t/s |
| Q5_K_M PPL 30 chunks, tail add on | 5.5642 +/- 0.15223 |
| `test-backend-ops test -o MUL_MAT_ID` | 690/690 passed |

The retained default stays faster than the prior weighted-sum default and keeps
the Q4/131k/PPL guardrails in band.

## Raw Files

- `q5_ctx0_tail_add_off.json`
- `q5_ctx0_tail_add_on.json`
- `q4_ctx0_tail_add_on.json`
- `q5_131k_tail_add_on.json`
- `ppl-q5km-30chunks-tail-add-on.txt`
- `test-backend-ops-mul-mat-id-tail-add.txt`
