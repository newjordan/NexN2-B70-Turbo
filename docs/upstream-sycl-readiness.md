# Upstream SYCL PR Readiness Notes

Internal audit notes only. Do not paste this into a llama.cpp PR. llama.cpp
requires that PR descriptions, commit messages, and reviewer replies be written
by the human contributor.

Date: 2026-06-10

## Candidate

- Local upstream checkout: `/home/frosty40/llama.cpp-sycl-moe-ready`
- Candidate branch: `sycl-moe-reorder-ready`
- Candidate commit: `b5994f6 sycl: support reordered Q4_K and Q5_K MoE MUL_MAT_ID`
- Base used for validation: `origin/master` at `d2462f8`
- Patch copy in this repo: `patches/0001-sycl-reorder-on-MoE-for-Q4_K-and-Q5_K-mul_mat_id.patch`
- Main `/home/frosty40/llama.cpp` checkout remains on `iq3-b70`. Keep
  contribution prep isolated to the dedicated ready/clean worktrees.

## Upstream Constraints

- llama.cpp has a strict AI policy. AI may assist with exploration and review,
  but the submitter must understand and be able to explain every line.
- llama.cpp prohibits AI-written PR descriptions, commit messages, and reviewer
  responses.
- The PR template requires an AI usage disclosure.
- `ggml/src/ggml-sycl/` is owned by `@ggml-org/ggml-sycl`.
- SYCL CI appears disabled/commented in the workflow, so local hardware
  validation matters more than usual.

## What The Patch Does

- Extends SYCL reorder handling to fused MoE expert `mul_mat_id` for Q4_K/Q5_K.
- Adds per-expert SoA reorder for 3D expert tensors.
- Adds a Q5_K reordered DMMV path.
- Allows the fused MoE MMVQ path to use reordered Q4_K/Q5_K expert weights.
- Uses reorder-aware Q8_1 quantization for `src1` when the expert weights are
  already reordered.
- Unsupported 3D reorder cases return `false` and fall back to existing reads
  instead of aborting.

## Defensible Validation Facts

- `git diff --check d2462f8..sycl-moe-reorder-ready` is clean.
- Build logs exist for base and PR builds in `results/upstream-pr/`.
- A smaller MoE smoke fixture is now available:
  `Phi-mini-MoE-instruct-Q4_K_M.gguf` from
  `gabriellarson/Phi-mini-MoE-instruct-GGUF`, stored at
  `/home/frosty40/models/micro-moe/`. On the cleaned branch, it loads on SYCL,
  generates, and
  `GGML_SYCL_DEBUG=1` shows 3D Q4_K expert `MUL_MAT_ID` calls such as
  `blk.0.ffn_gate_exps.weight: type=q4_K; ne=[4096, 960, 16, 1]`. Retained
  smoke outputs are in `results/micro-moe/`.
- Targeted op test: `test-backend-ops test -o MUL_MAT_ID` on the candidate build is
  714/714 tests passed. A subagent reran this on 2026-06-10 after loading the
  oneAPI runtime; the first attempt failed only because `libsvml.so` was missing
  before `setvars.sh`.
- Full backend-ops logs are retained for both unpatched base and candidate:
  both report 11502/11514 tests passed with the same 12 `GET_ROWS` tolerance
  failures in q2_K/q4_K/q5_K.
- Perplexity spot check is statistically unchanged:
  - base Q5_K_M: 5.5643 +/- 0.15232
  - PR Q5_K_M: 5.5662 +/- 0.15242
- `llama-bench` artifacts exist for the NX2 Q4_K_M and Q5_K_M model variants.
  Use the raw JSON files for contribution review only; do not reframe them as
  the project-level benchmark of the NX2 model work.
- First-token timing is retained in `results/nx2-first-token/20260610T230504Z/`.
  Under the measured Q5_K_M control, median prompt/first-token eval was 1.12 s
  with reorder off and 1.05 s with reorder on; no first-token lazy-reorder
  penalty was observed.

## Claims To Avoid

- Do not claim the patch alone gives +16% to +18% end-to-end decode improvement.
  That is a patch-attribution question, not the project-level NX2 throughput
  result.
- Do not claim the full backend suite passes; both base and candidate retain the
  same 12 full-suite failures.
- Do not frame the change as "Turbo" or NexN2-specific in upstream material.

## Review Risks

- The first-token timing is a whole optimization-path control, not isolated
  attribution to one line.
- The upstream performance case should be kept separate from the project
  benchmark. The stronger upstream argument is correctness/coverage of the
  reorder path, possible prefill improvement, and avoiding fallback once expert
  weights are reordered.

## Conservative Recommendation

Do not open a PR yet.

First prepare a smaller, cleaner local branch:

1. Run a focused A/B that proves the patch path is actually exercised and does
   not regress common Q4_K/Q5_K MoE decode/prefill shapes.
2. Ask the SYCL maintainers whether they want this as a small correctness and
   coverage PR.
