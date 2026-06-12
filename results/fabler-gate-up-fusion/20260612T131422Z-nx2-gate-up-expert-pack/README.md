# NX2 gate/up expert-pack scheduling

Date: 2026-06-12

Build: `/home/frosty40/llama.cpp/build`, Release SYCL (`icx/icpx`,
`GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, `GGML_SYCL_GRAPH=ON`).

This tests modes `9` and `10` of `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE`. Mode
`9` changes the exact NX2 fused gate/up SwiGLU scheduler from one output row
per workgroup to one selected expert per workgroup, with eight subgroups
covering the eight selected experts for a row. Mode `10` adds the mode `8`
local-Q8 activation cache to that expert-pack scheduler.

Command shape: `llama-bench -ngl 99 -fa on -p 0 -n 128 -r 5`.

| run | avg t/s | stddev |
|---|---:|---:|
| Q5_K_M ctx0, retained mode `1` | 86.8296 | 0.6946 |
| Q5_K_M ctx0, mode `9` expert-pack | 88.0915 | 0.3262 |
| Q5_K_M ctx0, mode `10` expert-pack + local Q8 | 87.3904 | 0.4002 |
| Q5_K_M ctx0, mode `9` repeat | 87.1350 | 0.0648 |
| Q5_K_M ctx0, retained mode `1` repeat | 86.9981 | 0.0646 |
| Q4_K_M ctx0, retained mode `1` | 92.7483 | 0.5510 |
| Q4_K_M ctx0, mode `9` expert-pack | 93.8672 | 0.1364 |

Correctness smoke check:

| check | result |
|---|---:|
| `test-backend-ops test -o MUL_MAT_ID`, mode `9` | 690/690 |
| `test-backend-ops test -o MUL_MAT_ID`, mode `10` | 690/690 |

Conclusion: mode `9` is correct and mildly positive, but it does not clear the
Q5_K_M +2% promotion gate across the reversed repeat (+1.45%, then +0.16%).
Mode `10` loses to mode `9`, so local-Q8 caching is still not useful for this
gate/up kernel family. Both modes remain default-off.
