# NX2 Weighted-Sum Fusion

Date: 2026-06-12

Experiment: keep patch `0004` mode `3` exact NX2 gate/up SwiGLU fusion and add
a default-on post-down weighted expert sum fusion for the F32 `[2048,8]` MoE
down output. The new graph fusion replaces the following weighted multiply plus
seven add kernels with one direct write to `ffn_moe_out`.

## Results

| check | result |
|---|---:|
| Q5_K_M ctx0, weighted sum off | 85.5535 t/s |
| Q5_K_M ctx0, weighted sum on | 87.3219 t/s |
| Q4_K_M ctx0, weighted sum on | 92.3487 t/s |
| Q5_K_M 131k, weighted sum on | 42.0761 t/s |
| Q5_K_M PPL 30 chunks, weighted sum on | 5.5722 +/- 0.15256 |
| `test-backend-ops test -o MUL_MAT_ID` | 690/690 passed |
| full `test-backend-ops test` | 11477/11489 passed |

The full backend-op failures are the same known 12 `GET_ROWS` q2_K/q4_K/q5_K
cases already present in the retained SYCL validation trail.

## Raw Files

- `q5_ctx0_weighted_sum_off.json`
- `q5_ctx0_weighted_sum_on.json`
- `q4_ctx0_weighted_sum_on.json`
- `q5_131k_weighted_sum_on.json`
- `ppl-q5km-30chunks-weighted-sum-on.txt`
- `test-backend-ops-mul-mat-id-weighted-sum.txt`
- `test-backend-ops-full-weighted-sum.txt`
