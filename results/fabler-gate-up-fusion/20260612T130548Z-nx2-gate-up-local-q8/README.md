# NX2 gate/up local-Q8 activation cache

Date: 2026-06-12

Build: `/home/frosty40/llama.cpp/build`, Release SYCL (`icx/icpx`,
`GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, `GGML_SYCL_GRAPH=ON`).

This tests mode `8` of `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE`. It changes the
exact NX2 fused gate/up SwiGLU kernel from the retained one-row workgroup to a
four-row workgroup that first copies the shared 2048-wide reordered Q8_1
activation row into workgroup-local memory. The goal is to make row grouping
reuse the activation vector instead of having each row subgroup reload the same
Q8 data from global/cache.

Command shape: `llama-bench -ngl 99 -fa on -p 0 -n 128 -r 5`.

| run | avg t/s | stddev |
|---|---:|---:|
| Q5_K_M ctx0, retained mode `1` | 88.1727 | 0.0480 |
| Q5_K_M ctx0, mode `8` local Q8 | 88.1966 | 0.0527 |
| Q5_K_M ctx0, mode `8` repeat | 87.6660 | 0.4300 |
| Q5_K_M ctx0, retained mode `1` repeat | 87.6672 | 0.3380 |
| Q4_K_M ctx0, mode `8` local Q8 | 90.0796 | 0.7096 |
| Q4_K_M ctx0, retained mode `1` | 93.9052 | 0.1061 |

Correctness smoke check:

| check | result |
|---|---:|
| `test-backend-ops test -o MUL_MAT_ID`, mode `8` | 690/690 |

Conclusion: local-Q8 caching is correct but not promotable. Q5_K_M ties retained
mode across the reversed repeat, while Q4_K_M regresses hard. Mode `8` remains
default-off.
