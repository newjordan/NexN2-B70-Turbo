# Upstream SYCL PR Readiness Notes

Contributor prep notes only. llama.cpp requires PR descriptions, commit
messages, and reviewer replies to be written by the human contributor.

Date: 2026-06-10

## PR Branch

- Local upstream checkout: `/home/frosty40/llama.cpp-sycl-moe-ready`
- Upstream PR: https://github.com/ggml-org/llama.cpp/pull/24452
- Branch: `sycl-moe-reorder-ready`
- Commit: `a7597d733 sycl: support reordered Q4_K and Q5_K MoE MUL_MAT_ID`
- Current PR base: `origin/master` at `ac4cddeb0`
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

- `git diff --check origin/master..sycl-moe-reorder-ready` is clean.
- Build logs exist for unpatched upstream and validated branch builds in
  `results/upstream-pr/`.
- A smaller MoE smoke fixture is now available:
  `Phi-mini-MoE-instruct-Q4_K_M.gguf` from
  `gabriellarson/Phi-mini-MoE-instruct-GGUF`, stored at
  `/home/frosty40/models/micro-moe/`. On the cleaned branch, it loads on SYCL,
  generates, and
  `GGML_SYCL_DEBUG=1` shows 3D Q4_K expert `MUL_MAT_ID` calls such as
  `blk.0.ffn_gate_exps.weight: type=q4_K; ne=[4096, 960, 16, 1]`. Retained
  smoke outputs are in `results/micro-moe/`.
- Patch packaging preflight: the patch applies and commits cleanly on a fresh
  temporary checkout at `ac4cddeb0`.
- Fresh PR-side build preflight: the `sycl-moe-reorder-ready` branch builds
  `test-backend-ops` with oneAPI/icpx 2026.0 and `GGML_SYCL=ON`.
- Fresh targeted op preflight: `test-backend-ops test -o MUL_MAT_ID` reports
  714/714 tests passed on Intel Arc Pro B70.
- Full backend-ops logs are retained for both unpatched upstream and validated branch:
  both report 11502/11514 tests passed with the same 12 `GET_ROWS` tolerance
  failures in q2_K/q4_K/q5_K.
- Perplexity spot check is statistically unchanged:
  - unpatched upstream Q5_K_M: 5.5643 +/- 0.15232
  - validated branch Q5_K_M: 5.5662 +/- 0.15242
- `llama-bench` artifacts exist for the NX2 Q4_K_M and Q5_K_M model variants.
  Use the raw JSON files for contribution review only; do not reframe them as
  the project-level benchmark of the NX2 model work.
- First-token timing is retained in `results/nx2-first-token/20260610T230504Z/`.
  Under the measured Q5_K_M control, median prompt/first-token eval was 1.12 s
  with reorder off and 1.05 s with reorder on; no first-token lazy-reorder
  penalty was observed.

## Claim Boundaries

- Keep the +16% to +18% end-to-end decode improvement out of the isolated
  llama.cpp patch claim. That is project-level NX2 throughput evidence, not a
  standalone patch-attribution result.
- State the full backend-suite result exactly: unpatched upstream and validated
  branch both retain the same 12 full-suite failures.
- Frame upstream material around the reusable SYCL kernel path, not the broader
  NexN2 package.

## Review Risks

- The first-token timing is a whole optimization-path control, not isolated
  attribution to one line.
- The upstream performance case should be kept separate from the project
  benchmark. The stronger upstream argument is correctness/coverage of the
  reorder path, possible prefill improvement, and avoiding fallback once expert
  weights are reordered.

## Follow-up Patches (2026-06-12, fabler branch — private, not for upstream)

Two further commits exist on local `fabler` (exported as `patches/0002`/`0003`,
cherry-picked and re-validated on a temp branch over the pinned base —
714/714 MUL_MAT_ID op tests). These stay in this repository as part of the
Turbo deployment package; no upstream submission is planned for them.

- `0002` concat submit-only.
- `0003` Q6_K MoE reorder + graph-safe `MUL_MAT_ID` compatibility. Recorded
  negative result: SYCL graph replay is a 5x slowdown on B70 / Level Zero 1.15
  because per-token graph update throws on a topology mismatch — graphs stay
  default-off; never claim a graph speedup on this stack.

## PR Readiness

Ready to open as a narrow SYCL backend PR.

The PR should be framed as Turbo SYCL kernel coverage for reordered Q4_K/Q5_K
MoE `mul_mat_id`: per-expert reorder, reordered MMVQ/GEMV dispatch, Q5_K
reordered DMMV, and safe fallback for unsupported 3D reorder cases. Keep the
PR description focused on correctness and coverage; use performance numbers only
as local supporting evidence if reviewers ask.
