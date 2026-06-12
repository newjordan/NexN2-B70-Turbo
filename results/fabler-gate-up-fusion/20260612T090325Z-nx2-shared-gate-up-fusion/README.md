# NX2 Shared Gate/Up Fusion Probe

Date: 2026-06-12

Experiment: add an opt-in shared-expert dense `MUL_MAT, MUL_MAT, GLU`
SwiGLU fusion for the NX2 `[2048,512]` Q4_K/Q5_K decode shape. The kernel
quantizes the shared activation once, reads the shared-expert gate and up
weights in one reordered MMVQ kernel, and writes the SwiGLU output directly.

## Result

The path is correct in the targeted backend-op gate but slower on the primary
Q5_K_M ctx0 scoreboard, so it remains disabled by default behind
`GGML_SYCL_ENABLE_MOE_SHARED_GATE_UP_FUSION=1`.

| check | result |
|---|---:|
| Q5_K_M ctx0, shared gate/up off | 87.2700 t/s |
| Q5_K_M ctx0, shared gate/up on, first reducer | 86.8808 t/s |
| Q5_K_M ctx0, shared gate/up on, xor reducer | 86.9050 t/s |
| `test-backend-ops test -o MUL_MAT_ID`, final default | 690/690 passed |

## Raw Files

- `q5_ctx0_shared_gate_up_off.json`
- `q5_ctx0_shared_gate_up_on.json`
- `q5_ctx0_shared_gate_up_on_xor_reduce.json`
- `test-backend-ops-mul-mat-id-default.txt`
