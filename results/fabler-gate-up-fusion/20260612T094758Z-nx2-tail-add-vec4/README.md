# NX2 Tail-Add Vec4 Probe

Date: 2026-06-12

Experiment: add `GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=2`, a vec4 row-pack
variant of the retained F32 MoE tail-add kernel. Mode `1` remains the default
scalar retained path.

## Results

| check | result |
|---|---:|
| Q5_K_M ctx0, scalar tail add | 87.6672 t/s |
| Q5_K_M ctx0, vec4 tail add | 87.7949 t/s |
| Q5_K_M ctx0, vec4 repeat | 87.6466 t/s |
| Q5_K_M ctx0, scalar repeat | 86.9449 t/s |
| targeted `MUL_MAT_ID`, retained default | 690/690 |
| full backend-op suite, retained default | 11477/11489, known 12 `GET_ROWS` failures |

The vec4 path did not produce a stable enough win to promote. It remains
opt-in only; scalar mode `1` stays the retained default.

## Raw Files

- `q5_ctx0_tail_scalar.json`
- `q5_ctx0_tail_vec4.json`
- `q5_ctx0_tail_vec4_repeat.json`
- `q5_ctx0_tail_scalar_repeat.json`
- `test-backend-ops-mul-mat-id-default.txt`
- `test-backend-ops-full-default.txt`
