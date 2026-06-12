# NX2 gate/up dual-dot vecdot

Date: 2026-06-12

Build: `/home/frosty40/llama.cpp/build`, Release SYCL (`icx/icpx`,
`GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, `GGML_SYCL_GRAPH=ON`).

This tests mode `11` of `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE`. It keeps the
retained one-row-per-workgroup exact NX2 gate/up scheduler, but replaces the
two independent reordered Q4_K/Q5_K vecdot helper calls with a dual-dot helper
that loads the shared Q8_1 activation lane data once and accumulates up and gate
dot products in the same inner loop.

Command shape: `llama-bench -ngl 99 -fa on -p 0 -n 128 -r 5`.

| run | avg t/s | stddev |
|---|---:|---:|
| Q5_K_M ctx0, retained mode `1` | 87.0736 | 0.6281 |
| Q5_K_M ctx0, mode `11` dual-dot | 87.5911 | 0.3818 |
| Q5_K_M ctx0, mode `11` repeat | 87.2951 | 0.1998 |
| Q5_K_M ctx0, retained mode `1` repeat | 87.2913 | 0.3669 |
| Q4_K_M ctx0, retained mode `1` | 94.1547 | 0.0663 |
| Q4_K_M ctx0, mode `11` dual-dot | 92.7612 | 0.0940 |

Correctness smoke check:

| check | result |
|---|---:|
| `test-backend-ops test -o MUL_MAT_ID`, mode `11` | 690/690 |

Conclusion: the dual-dot inner loop is correct but not promotable. Q5_K_M only
ties the retained path after the reversed repeat, and Q4_K_M regresses. Mode
`11` remains default-off.
