# IQ3_A770 winner — B70-proxy throughput (NOT A770; dev card only)

Model: NX2-IQ3_A770-mixed-LUT.gguf (q3floor_q4top24 recipe, 15.01 GiB).
Device: Intel Arc Pro B70 (Battlemage), Level-Zero, oneAPI 2026.0. A770 unavailable;
B70 is the same SYCL backend, used as a proxy. Numbers taken with a competing
llama-server (Q5) resident on the GPU, so treat as a conservative floor.

| path | test | t/s |
|---|---|---|
| M1 (dequant->fp16->GEMM, correct-but-slow) | pp512 | 707.6 ± 0.1 |
| M1 (dequant->fp16->GEMM, correct-but-slow) | tg64  | 31.8 ± 0.02 |

Correctness: test-backend-ops MUL_MAT + MUL_MAT_ID all OK on SYCL/B70 (0 fail);
end-to-end "The capital of France is" -> "Paris." on GPU.

M2 (dp4a mul_mat_vec_q, reads 3.19bpw directly, no fp16 detour) targets the tg gap
vs stock Q3_K_M (62 t/s) -- pending.
