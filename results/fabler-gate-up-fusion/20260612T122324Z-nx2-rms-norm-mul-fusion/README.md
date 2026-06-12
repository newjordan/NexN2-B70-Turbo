# NX2 RMS_NORM + MUL Fusion Probe

Branch: llama.cpp `fabler` with retained `0004` defaults, plus a default-off
`GGML_SYCL_ENABLE_RMS_NORM_MUL_FUSION` experiment.

Experiment: add the SYCL equivalent of the CUDA/OpenCL/WebGPU/Vulkan
`RMS_NORM + MUL` graph fusion for the NX2 decode hot shape. The first probe is
conservative: F32 contiguous `[2048]` RMS_NORM input, F32 contiguous `[2048]`
norm-weight vector, and F32 contiguous output. It fuses the RMS reduction and
norm-weight multiply into one kernel while still leaving the multiplied vector
materialized for downstream graph consumers.

Modes:

| mode | launch shape |
| --- | --- |
| `1` | 512 work-items |
| `2` | 1024 work-items, matching the existing large-row RMS_NORM launch shape |

Results:

| run | avg tok/s |
| --- | ---: |
| Q5_K_M ctx0, fusion off | 88.0455 |
| Q5_K_M ctx0, fusion off repeat | 87.8436 |
| Q5_K_M ctx0, mode `1` | 88.4860 |
| Q5_K_M ctx0, mode `1` repeat | 88.3056 |
| Q5_K_M ctx0, mode `2` | 89.1515 |
| Q5_K_M ctx0, mode `2` repeat | 88.9745 |
| Q4_K_M ctx0, fusion off | 91.2825 |
| Q4_K_M ctx0, mode `2` | 95.7701 |

Correctness / smoke:

| check | result |
| --- | --- |
| `test-backend-ops test -o RMS_NORM`, mode `2` | 21/21 passed |
| `test-backend-ops test -o MUL_MAT_ID`, mode `2` | 690/690 passed |
| Q5_K_M WikiText, 5 chunks, off | 5.1289 |
| Q5_K_M WikiText, 5 chunks, mode `2` | 5.1329 |

Mode `2` is the better launch shape and improves both Q5_K_M and Q4_K_M in
these measurements, but the primary Q5_K_M ctx0 gain is still below the +2%
promotion bar. It remains default-off and opt-in only.
