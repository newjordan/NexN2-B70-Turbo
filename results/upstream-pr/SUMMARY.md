# SYCL reorder-on-MoE patch — full validation summary (2026-06-10)

Due-diligence matrix run on the B70 (`eval/upstream/test_matrix.sh` + follow-ups).
The retained full-matrix logs compare unpatched upstream and the validated
branch. The current PR branch is `sycl-moe-reorder-ready` at
`a7597d733 sycl: support reordered Q4_K and Q5_K MoE MUL_MAT_ID`, rebased on
`ac4cddeb0`.

## Correctness — targeted checks pass

- Fresh rebased PR preflight: `test-backend-ops test -o MUL_MAT_ID` reports
  **714/714 tests passed** vs CPU reference on Intel Arc Pro B70.
- Full suite: unpatched upstream and the validated branch both report
  **11502/11514 tests passed**. The 12 actual failing tests are the same
  GET_ROWS tolerance misses (~3e-7 vs 1e-7) in q2_K/q4_K/q5_K, plus
  backend/final summary FAIL lines.
- Perplexity (Q5_K_M, wikitext-2, 30 chunks): unpatched upstream
  **5.5643 ±0.152** vs validated branch **5.5662 ±0.152** — statistically
  identical.
- Retained greedy artifacts are FA-on and are **not token-identical** across
  reorder variants. Greedy text is a smoke test only, not correctness evidence.

## Performance Scope

This directory is a contribution-readiness audit for one llama.cpp code patch. It
is not the project-level benchmark story for the NX2 GGUF variants. The saved
`llama-bench` artifacts use the local NX2 model files:

- `/home/frosty40/models/nex-n2-mini/sweep/NX2-Q4_K_M.gguf`
- `/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf`

Those artifacts show the NX2 variants running around 80-84 tok/s token
generation on Arc Pro B70 under the retained `llama-bench` settings. Raw
comparison data remains in `bench-base.json` and `bench-pr.json` for anyone
reviewing the isolated code patch.

For the fresh reproducible project-level control comparison, use
`results/nx2-controls/20260610T220943Z/`:

| config | ctx0 decode |
|---|---:|
| stock control: reorder off / FA off | 69.84 t/s |
| deployed: Turbo + FA | 81.69 t/s |

- Earlier server-flow and kernel-level numbers were not retained in this
  artifact directory; rerun before using them in upstream-facing material.
- First-token timing is retained separately in
  `results/nx2-first-token/20260610T230504Z/`; the measured Q5_K_M control did
  not show a first-token lazy-reorder penalty.

## Verdict

The patch is **numerically sound in the retained targeted checks**. It covers
SYCL reorder correctness for Q4_K/Q5_K MoE `mul_mat_id`, tested on the local NX2
GGUF variants. The broader Turbo package also includes model artifacts, runtime
configuration, quant selection, and validation records.

Artifacts: backend-ops logs, greedy outputs, bench JSONs, PPL logs in this
directory. Branch `sycl-moe-reorder-ready` is retained in
`/home/frosty40/llama.cpp-sycl-moe-ready` for reference.
