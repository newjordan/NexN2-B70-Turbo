# NX2 Weighted-Sum Vec4 Probe

Date: 2026-06-12

Experiment: add `GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=2`, a vec4 row-pack
variant of the exact NX2 post-down weighted expert sum. Mode `1` remains the
default scalar retained path.

## Results

| check | result |
|---|---:|
| Q5_K_M ctx0, scalar weighted sum | 88.0789 t/s |
| Q5_K_M ctx0, vec4 weighted sum | 87.9073 t/s |
| Q5_K_M PPL 30 chunks, vec4 weighted sum | 5.5685 +/- 0.15255 |
| full backend-op suite, retained default | 11477/11489, known 12 `GET_ROWS` failures |

The vec4 path is correct in the PPL spot check but slower on the Q5_K_M ctx0
scoreboard, so it remains opt-in only.

## Raw Files

- `q5_ctx0_weighted_sum_scalar.json`
- `q5_ctx0_weighted_sum_vec4.json`
- `ppl-q5km-30chunks-weighted-sum-vec4.txt`
- `test-backend-ops-full-retained-default.txt`
