# SYCL reorder-on-MoE patch — full validation summary (2026-06-10)

Due-diligence matrix run on the B70 (`eval/upstream/test_matrix.sh` + follow-ups).
Builds: `base` = upstream master d2462f8 (unpatched) · `pr` = master + the patch
(cherry-picked cleanly, branch `sycl-moe-reorder`) · `old` = f0156d1 (the patch's
original base). No upstream ggml-sycl commits between f0156d1 and master.

## Correctness — PASSES

- `test-backend-ops test -o MUL_MAT_ID` (pr): **716/716 OK** vs CPU reference.
- Full suite (pr): 14 FAILs, **all GET_ROWS tolerance misses (~3e-7 vs 1e-7)
  that fail identically on unpatched base** — pre-existing upstream SYCL
  behavior, untouched by the patch.
- Perplexity (Q5_K_M, wikitext-2, 30 chunks): base **5.5643 ±0.152** vs
  pr **5.5662 ±0.152** — statistically identical.
- Greedy determinism: **FA-off → token-identical** reorder on/off (48 tok).
  FA-on → outputs diverge after ~40 tokens (both coherent; ulp-level
  summation-order differences amplified by greedy argmax). This also
  reproduces on the original build — the old "byte-identical" claim held
  only under the conditions originally tested. Documented in methodology.

## Performance — the patch's gain does not survive scrutiny

tg128 (Q4_K_M / Q5_K_M t/s, llama-bench, fa 1, r=4):

|              | unpatched      | + patch        |
|--------------|----------------|----------------|
| old f0156d1  | 85.35 / 80.62  | 85.53 / 81.30  |
| master       | 84.89 / 79.08  | 85.75 / 79.59  |

- Patch over its own base: **+0.2% / +0.8%** — within noise.
- Server-flow A/B (lazy reorder provably triggered): base 81.07 vs pr
  81.06 t/s — **no gain**.
- Kernel-level (`test-backend-ops perf MUL_MAT_ID`, decode shape n=1):
  14.83 → 14.40 us (**~+3%**), noise at other batch sizes.
- The historical "+16.3% / +17.7% decode" claim reproduces **only as the
  `GGML_SYCL_DISABLE_OPT` on/off toggle** (+17.9% / +17.3% measured) — but
  that toggle also disables upstream's pre-existing dense reorder, which is
  where essentially all of the win lives. The MoE addition contributes ~1%.
- pp512: 1046 → 1095 t/s (+4.7%) in a single run — unverified, treat as
  indicative only.

## Verdict

The patch is **numerically sound and does not regress anything**, but its
incremental performance contribution over upstream master is ~0–1% end-to-end
on NX2 (likely because decode is bottlenecked by the CPU-resident Delta-Net
layers, and upstream's dense reorder already covers the hot GPU paths).
The "Turbo" +17% is real but belongs to upstream's existing reorder
infrastructure, not to this patch. No PR (user decision) — and on this
evidence the patch would not have cleared upstream's "verify performance is
not affected / improved" bar anyway.

Artifacts: backend-ops logs, greedy outputs, bench JSONs, PPL logs in this
directory. Branch `sycl-moe-reorder` retained in ~/llama.cpp for reference.
