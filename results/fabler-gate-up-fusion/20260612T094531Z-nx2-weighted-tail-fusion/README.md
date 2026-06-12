# NX2 Weighted-Tail Fusion Probe

Date: 2026-06-12

Experiment: add `GGML_SYCL_ENABLE_MOE_WEIGHTED_TAIL_FUSION=1`, a combined kernel
that fuses the exact NX2 post-down weighted expert sum with the following two
F32 tail adds. The kernel writes the final residual output directly instead of
writing `ffn_moe_out` and launching a separate tail-add kernel.

## Results

| check | result |
|---|---:|
| Q5_K_M ctx0, weighted-tail off | 87.9788 t/s |
| Q5_K_M ctx0, weighted-tail on | 87.7427 t/s |

The larger fusion is correct enough to run the primary model bench, but it is
slower on the Q5_K_M ctx0 scoreboard. It remains default-off and opt-in only.

## Raw Files

- `q5_ctx0_weighted_tail_off.json`
- `q5_ctx0_weighted_tail_on.json`
