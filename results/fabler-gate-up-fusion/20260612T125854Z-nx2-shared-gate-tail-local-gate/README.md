# NX2 shared-gate-tail local-gate broadcast

Date: 2026-06-12

Build: `/home/frosty40/llama.cpp/build`, Release SYCL (`icx/icpx`,
`GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, `GGML_SYCL_GRAPH=ON`).

This tests modes `5` and `6` of
`GGML_SYCL_ENABLE_MOE_SHARED_GATE_TAIL_FUSION`. Both target the exact NX2 F32
shared-expert tail:

```text
ffn_moe_shexp * ffn_shexp_gate + ffn_moe_out + ffn_inp
```

Mode `5` loads the scalar shared-expert gate once per workgroup into local
memory before the scalar tail loop. Mode `6` combines the same local-gate
broadcast with the vec4 tail loop.

Command shape: `llama-bench -ngl 99 -fa on -p 0 -n 128 -r 5`.

| run | avg t/s | stddev |
|---|---:|---:|
| Q5_K_M ctx0, retained default | 87.8138 | 0.3769 |
| Q5_K_M ctx0, mode `5` local gate | 87.6483 | 0.3116 |
| Q5_K_M ctx0, mode `6` vec4 local gate | 87.5545 | 0.4399 |

Correctness smoke checks:

| check | result |
|---|---:|
| `test-backend-ops test -o MUL_MAT_ID`, mode `5` | 690/690 |
| `test-backend-ops test -o MUL_MAT_ID`, mode `6` | 690/690 |

Conclusion: reducing scalar gate global reads does not pay for the local-memory
barrier/nd-range shape on the primary Q5_K_M ctx0 scoreboard. Modes `5` and `6`
remain default-off.
