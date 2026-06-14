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

## M2 — dp4a `mul_mat_vec_q` reference kernel (landed)

`vec_dot_iq3_a770_q8_1` reads the packed 3.19bpw block directly (LUT `kvalues_iq3nl`,
Q3_K-style 2+1 bit unpack, 8×32 4-bit signed scales, `dp4a`) — no fp16 dequant
detour. Bit-matches the CPU oracle `ggml_vec_dot_iq3_a770_q8_K`. Patch:
[`../../patches/wip/0006-iq3-a770-sycl-vecdot-M2.partial.patch`](../../patches/wip/0006-iq3-a770-sycl-vecdot-M2.partial.patch).

Correctness (`test-backend-ops`, SYCL/B70): MUL_MAT (dp4a) **11/11 OK**; MUL_MAT_ID
(unchanged dequant+GEMM fallback) **2/2 OK**.

Kernel A/B — `test-backend-ops perf -o MUL_MAT`, m=4096, k=14336, same build/device
(dp4a vs the M1 dequant+fp16+GEMM path, toggled via `can_use_mul_mat_vec_q`):

| n (cols) | M1 dequant+GEMM | M2 dp4a | speedup |
|---|---|---|---|
| **1 (GEMV / decode)** | 1576.7 µs | **682.6 µs** | **2.31×** |
| 2 | 1559.2 µs | 1289.9 µs | 1.21× |
| 8 | 1559.6 µs | 5448.6 µs | 0.29× |

dp4a wins decisively at n=1 — the decode shape — by reading 3.19bpw instead of
materializing the whole weight tile to fp16. Above n≈2 the simple per-column mmvq
loses to amortized dequant+GEMM (the standard mmvq/GEMM crossover; matches stock
`iq3_s`, which uses the same per-column launcher). A batched ncols kernel is a
later optimization.

Scope: every IQ3_A770 tensor in the model is a MoE expert (`ffn_*_exps`,
**MUL_MAT_ID**), so the M1 model pp/tg figures above are unchanged by M2, which only
touches dense MUL_MAT. M2 is the validated dp4a reference kernel; carrying it into
the expert decode (reordered MUL_MAT_ID) is `0007` — that is where the model tg gap
vs stock Q3_K_M (62 t/s) closes.

## 0007 — IQ3 MoE MUL_MAT_ID reordered dp4a decode (landed) ⭐

Lazily reorders each IQ3_A770 expert tensor into a per-expert SoA
(`[qs][hmask][scales][d]`) on first decode, then runs a **single fused expert GEMV**
(`reorder_vec_dot_q_sycl<IQ3_A770>` = the M2 LUT dp4a reading SoA planes) for all 8
active experts per matmul — vs M2's 8 separate per-expert mmvq launches. Patch:
[`../../patches/wip/0007-iq3-a770-sycl-moe-reorder.partial.patch`](../../patches/wip/0007-iq3-a770-sycl-moe-reorder.partial.patch)
(7 components across 6 SYCL files).

**Root-cause note:** the first 6 components compiled and the model stayed coherent, but
instrumentation showed the reorder path was **dormant** — `ggml_backend_sycl_buffer_init_tensor`
only allocated the reorder `ggml_tensor_extra_gpu` for `Q4_0/Q8_0/Q4_K/Q6_K`, so IQ3
tensors had `extra==nullptr` and `opt_for_reorder_id` silently skipped them
(`use_reorder=0` → fell back to the M2 per-expert path). Adding `GGML_TYPE_IQ3_A770`
to that switch is the fix that actually engages 0007.

Correctness (SYCL/B70): `test-backend-ops MUL_MAT_ID` **2/2 OK with reorder active**
(`use_reorder=1` confirmed by trace — the reorder vec_dot is exercised and matches the
CPU reference); end-to-end `llama-completion` "The capital of France is" → **"Paris."**

Model throughput (`llama-bench`, NX2-IQ3_A770-mixed-LUT.gguf, -ngl 99, B70, competing
Q5 server resident). **Back-to-back `init_tensor` on/off toggle, same build session,
same resident server, `-r 5`** — the rigorous A/B:

| config | pp512 | tg64 |
|---|---|---|
| reorder **OFF** — M2 per-expert mmvq | 600.65 ± 1.63 | 43.69 ± 0.03 |
| reorder **ON** — 0007 fused reorder dp4a | 600.09 ± 0.74 | **78.56 ± 0.10** |

**pp512 is identical across the toggle (600.65 vs 600.09) — the contention control** — so the
**tg jump 43.69 → 78.56 t/s = +79.8% (1.80×) is unambiguously the 0007 algorithm, not GPU
contention.** Error bars ±0.1 over 5 reps. **Clears stock Q3_K_M's 62 t/s.** The win is
launch-overhead elimination: one expert-indexed GEMV per matmul for all 8 active experts,
vs M2's 8 separate per-expert launches (×40 layers ×3 projections). pp is unchanged because
prefill is batch-GEMM, not the decode reorder path. (M1 dequant→fp16→GEMM was pp707/tg32 in
an earlier session — indicative, different contention.)
