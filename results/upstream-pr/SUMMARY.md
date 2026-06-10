# SYCL reorder-on-MoE patch — full validation summary (2026-06-10)

Due-diligence matrix run on the B70 (`eval/upstream/test_matrix.sh` + follow-ups).
Builds: `base` = upstream master `d2462f8` (unpatched) · `candidate` =
`d2462f8` + the patch exported from branch `sycl-moe-reorder-ready`.

## Correctness — targeted checks pass

- `test-backend-ops test -o MUL_MAT_ID` (pr): **714/714 tests passed** vs CPU reference.
- Full suite: base and candidate both report **11502/11514 tests passed**. The
  12 actual failing tests are the same GET_ROWS tolerance misses (~3e-7 vs
  1e-7) in q2_K/q4_K/q5_K, plus backend/final summary FAIL lines.
- Perplexity (Q5_K_M, wikitext-2, 30 chunks): base **5.5643 ±0.152** vs
  pr **5.5662 ±0.152** — statistically identical.
- Retained greedy artifacts are FA-on and are **not token-identical** across
  reorder/base variants. Treat greedy text as a smoke test only, not correctness
  evidence.

## Performance Scope

This directory is a contribution-readiness audit for one llama.cpp code patch. It
is not the project-level benchmark story for the NX2 GGUF variants. The saved
`llama-bench` artifacts use the local NX2 model files:

- `/home/frosty40/models/nex-n2-mini/sweep/NX2-Q4_K_M.gguf`
- `/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf`

Those artifacts show the NX2 variants running around 80-84 tok/s token
generation on Arc Pro B70 under the retained `llama-bench` settings. Raw base
vs patch data remains in `bench-base.json` and `bench-pr.json` for anyone
reviewing the isolated code patch.

For the fresh reproducible project-level control comparison, use
`results/nx2-controls/20260610T220943Z/`:

| config | ctx0 decode | 131k decode |
|---|---:|---:|
| stock control: reorder off / FA off | 69.84 t/s | 20.10 t/s |
| deployed: Turbo + FA | 81.69 t/s | 40.88 t/s |

The older `results/longctx-fa.csv` historical control records 55.43 t/s at
ctx0, but the original raw command/log was not retained and current reruns do
not reproduce that low ctx0 value. The deep-context stock control is stable
around 20 t/s, and deployed 131k remains around 41 t/s.

- Earlier server-flow and kernel-level numbers were not retained in this
  artifact directory; rerun before using them in upstream-facing material.
- The historical "+16.3% / +17.7% decode" claim reproduces **only as the
  `GGML_SYCL_DISABLE_OPT` on/off toggle** (+17.9% / +17.3% measured) — but
  that toggle also disables upstream's pre-existing dense reorder, which is
  where most of that specific on/off delta lives. Do not use that toggle as the
  project-level benchmark.
- First-token timing is retained separately in
  `results/nx2-first-token/20260610T230504Z/`; the measured Q5_K_M control did
  not show a first-token lazy-reorder penalty.

## Verdict

The patch is **numerically sound in the retained targeted checks**. Keep any
llama.cpp contribution claim narrow: this is a SYCL reorder coverage/correctness patch for
Q4_K/Q5_K MoE `mul_mat_id`, tested on the local NX2 GGUF variants. Do not use
this audit to minimize or summarize the broader NX2 model-artifact work.

Artifacts: backend-ops logs, greedy outputs, bench JSONs, PPL logs in this
directory. Branch `sycl-moe-reorder-ready` is retained in
`/home/frosty40/llama.cpp-sycl-moe-ready` for reference.
