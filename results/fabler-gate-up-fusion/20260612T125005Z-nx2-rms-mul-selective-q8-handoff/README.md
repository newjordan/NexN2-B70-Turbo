# NX2 RMS_NORM+MUL selective Q8 handoff

Date: 2026-06-12

Build: `/home/frosty40/llama.cpp/build`, Release SYCL (`icx/icpx`,
`GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, `GGML_SYCL_GRAPH=ON`).

This tests mode `4` of `GGML_SYCL_ENABLE_RMS_NORM_MUL_FUSION`: keep the
1024-work-item RMS_NORM+MUL fusion, but only emit the reordered Q8_1 side
buffer when graph lookahead sees the later fused MoE gate/up SwiGLU consume the
MUL output. The intent was to avoid mode `3`'s unused Q8 writes on attention
norms while preserving the post-norm handoff.

Command shape: `llama-bench -ngl 99 -fa on -p 0 -n 128 -r 5`.

| run | avg t/s | stddev |
|---|---:|---:|
| Q5_K_M ctx0, mode `4` | 87.5791 | 0.3610 |
| Q5_K_M ctx0, mode `4` repeat | 87.4778 | 0.3216 |
| Q5_K_M ctx0, mode `2` same-run repeat | 89.0116 | 0.4189 |
| Q5_K_M ctx0, mode `0` same-run baseline | 88.0206 | 0.0774 |
| Q4_K_M ctx0, mode `4` | 94.3220 | 0.6140 |

Correctness smoke checks with mode `4`:

| check | result |
|---|---:|
| `test-backend-ops test -o RMS_NORM` | 21/21 |
| `test-backend-ops test -o MUL_MAT_ID` | 690/690 |

Conclusion: mode `4` is correct but slower than standalone mode `2` on the
primary Q5_K_M ctx0 scoreboard, and it also trails the earlier Q4 mode `2`/`3`
guard results. Mode `2` remains below the +2% promotion gate against the
same-run mode `0` baseline. Both remain default-off.
