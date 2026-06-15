# init_tensor reorder A/B — the dormancy switch, quantified

**Claim validated:** the one-line `case GGML_TYPE_IQ3_A770:` in
`ggml_backend_sycl_buffer_init_tensor` (which allocates the per-tensor
`ggml_tensor_extra_gpu` that holds `optimized_feature.reorder`) is worth **+88% decode
throughput**. Without it, `src0->extra == nullptr`, `opt_for_reorder_id` silently returns,
and every IQ3 MoE expert falls back to the per-expert path — **no error, coherent output**,
just ~half the speed. This is the bug that cost a full debug cycle; here it is measured.

## Method (build-variance-free)

Same binary, same model, same machine, back-to-back. The only variable is an env gate
temporarily wrapped around that one `case` (`GGML_SYCL_NO_IQ3_REORDER`; reverted after the
run, never committed) so OFF and ON are the *identical* executable — no rebuild between
arms, so build/link nondeterminism cannot leak in. `pp512` is the contention control: if it
moves, the GPU wasn't dedicated. B70 dedicated (the resident Q5_K_M server was stopped for
the run, restarted after). `llama-bench -p 512 -n 128 -r 5`.

What OFF disables: the missing `extra` kills the reorder path wholesale, so **both** the
0007 fused expert-indexed decode **and** the 0008 fused gate/up+SwiGLU revert to the
per-expert fallback. So this A/B measures the entire IQ3 reorder stack vs. the dormant path.

## Result

| arm | pp512 (t/s, contention control) | **tg128 (t/s)** |
|---|---|---|
| **OFF** — reorder extra skipped (dormant) | 599.66 ± 2.04 | **43.71 ± 0.04** |
| **ON** — reorder live (0007+0008)         | 600.21 ± 1.38 | **82.26 ± 0.04** |
| **ON** — repeat (reproducibility)         | 600.17 ± 1.58 | **82.43 ± 0.05** |

- **tg: 43.71 → 82.3 t/s = +88.3%** (1.883×).
- **pp matched**: 599.66 / 600.21 / 600.17 — spread 0.09%, well inside error. No contention
  confound; the tg delta is the kernel, not the scheduler.
- **Reproducible**: the two ON arms agree to 0.2% (82.26 vs 82.43), error bars ±0.05.

Raw: [`ab-76709523.log`](ab-76709523.log). Model `NX2-IQ3_A770-mixed-LUT.gguf` (15.0 GiB,
experts IQ3_A770 3.19 bpw). Tool `llama-bench` (not `llama-cli` — the latter's qwen `>`
continuation quirk pollutes tg). Lesson for any new SYCL reorder type: **confirm the
`init_tensor` extra-allocation switch lists it**, and measure tg back-to-back, not "it
compiles and generates text."
