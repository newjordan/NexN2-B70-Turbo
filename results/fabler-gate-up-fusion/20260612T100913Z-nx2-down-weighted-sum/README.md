# NX2 Down Weighted-Sum Fusion Probe

Date: 2026-06-12

Experiment: add `GGML_SYCL_ENABLE_MOE_DOWN_WEIGHTED_SUM_FUSION=1`, an exact
NX2 Q6_K down-projection kernel that computes the eight selected expert down
dots for each output row, applies top-k weights in-kernel, and writes the
reduced `[2048]` MoE output directly. This avoids materializing the `[2048,8]`
down output and skips the separate weighted-sum kernel, but reduces parallelism
inside the down projection.

## Results

| check | result |
|---|---:|
| Q5_K_M ctx0, down-weighted off | 87.9828 t/s |
| Q5_K_M ctx0, down-weighted on | 87.8523 t/s |
| targeted `MUL_MAT_ID`, retained default | 690/690 |

The larger down+weighted fusion is slower and noisier on the Q5_K_M ctx0
scoreboard, so it remains default-off and opt-in only.

## Raw Files

- `q5_ctx0_down_weighted_off.json`
- `q5_ctx0_down_weighted_on.json`
- `test-backend-ops-mul-mat-id-default.txt`
