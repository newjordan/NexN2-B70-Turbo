# NX2 Tail-Add + RMS_NORM Fusion Probe

Branch: llama.cpp `fabler` with retained `0004` defaults, plus an opt-in
`GGML_SYCL_ENABLE_MOE_TAIL_RMS_FUSION=1` experiment.

Experiment: match the exact NX2 final MoE tail pattern
`ADD, ADD, RMS_NORM`, write the residual vector and RMS_NORM output from one
SYCL kernel, and skip the standalone two-ADD tail kernel plus standalone
RMS_NORM. The probe is conservative: F32, contiguous `[2048]` vectors only,
and it preserves the second ADD operand order used by the scalar tail-add path.

Results:

| run | avg tok/s |
| --- | ---: |
| Q5_K_M ctx0, tail-RMS off | 87.9524 |
| Q5_K_M ctx0, tail-RMS off repeat | 87.5693 |
| Q5_K_M ctx0, tail-RMS on | 86.3660 |
| Q4_K_M ctx0, tail-RMS off | 91.7811 |
| Q4_K_M ctx0, tail-RMS on | 92.7675 |

Targeted `MUL_MAT_ID` backend-op testing still passed (`690/690`) with the
env flag enabled. The Q5_K_M primary scoreboard regressed, so this remains
default-off and opt-in only.
