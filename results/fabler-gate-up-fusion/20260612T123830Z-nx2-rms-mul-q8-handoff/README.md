# NX2 RMS_NORM + MUL Q8 Handoff Probe

Branch: llama.cpp `fabler` with retained `0004` defaults, plus
`GGML_SYCL_ENABLE_RMS_NORM_MUL_FUSION=3`.

Experiment: extend the standalone SYCL `RMS_NORM + MUL` fusion so the same
kernel also emits a reordered Q8_1 SoA side buffer for the later exact-NX2 MoE
gate/up fusion. The graph shape is:

```text
RMS_NORM -> MUL(attn_post_norm) -> ... routing ops ... -> MUL_MAT_ID gate/up
```

The MoE gate/up fused kernel recognizes the reshape of `attn_post_norm` and
consumes the side buffer, skipping its normal activation quantization launch
while still writing the F32 `attn_post_norm` tensor for other graph consumers.

Two mode-3 quantizers were tested:

| variant | avg tok/s |
| --- | ---: |
| Q5_K_M ctx0, initial serial in-kernel Q8 | 88.0404 |
| Q5_K_M ctx0, subgroup-parallel Q8 | 88.7678 |

The subgroup-parallel version maps one 16-lane subgroup to each Q8_1 block,
with each lane quantizing two values. It is much better than the initial serial
in-workgroup quantizer, but still trails standalone RMS_NORM+MUL mode `2`
(`89.15` and `88.97` t/s in the previous probe).

Final subgroup-parallel results:

| run | avg tok/s |
| --- | ---: |
| Q5_K_M ctx0, fusion off | 87.5647 |
| Q5_K_M ctx0, mode `3` | 88.7678 |
| Q4_K_M ctx0, fusion off | 92.3680 |
| Q4_K_M ctx0, mode `3` | 95.8022 |

Correctness / smoke:

| check | result |
| --- | --- |
| `test-backend-ops test -o RMS_NORM`, mode `3` | 21/21 passed |
| `test-backend-ops test -o MUL_MAT_ID`, mode `3` | 690/690 passed |
| Q5_K_M WikiText, 5 chunks, mode `3` | 5.1402 |

Mode `3` proves the post-norm Q8 handoff architecture works and Q4_K_M likes
it, but Q5_K_M remains below the +2% promotion bar and below standalone mode
`2`. It remains default-off and opt-in only.
