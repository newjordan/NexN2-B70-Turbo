# NX2 Gate/Up XOR Reducer Probe

Date: 2026-06-12

Experiment: add `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=2`, an alternate manual
XOR subgroup reducer for the exact NX2 fused MoE gate/up SwiGLU kernel. Mode `1`
remains the retained `sycl::reduce_over_group` implementation.

## Results

| check | result |
|---|---:|
| Q5_K_M ctx0, reduce_over_group | 87.9569 t/s |
| Q5_K_M ctx0, XOR reducer | 87.8054 t/s |
| targeted `MUL_MAT_ID`, XOR reducer | 690/690 |

The XOR reducer is correct in targeted backend-op testing but slower and noisier
on the Q5_K_M ctx0 scoreboard, so it remains opt-in only.

## Raw Files

- `q5_ctx0_reduce_over_group.json`
- `q5_ctx0_xor_reduce.json`
- `test-backend-ops-mul-mat-id-xor-reduce.txt`
