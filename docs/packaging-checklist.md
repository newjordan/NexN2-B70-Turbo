# Packaging Checklist

Use this as the local include/exclude list before any report or release commit.

## Include In Turbo Report Package

- `README.md`
- `docs/README.md`
- `docs/HF_MODEL_CARD.md`
- `docs/kernel-release-gate.md`
- `docs/lab-report.md`
- `docs/methodology.md`
- `docs/turbo-understanding.md`
- `docs/turbo-understanding.html`
- `docs/upstream-sycl-readiness.md`
- `docs/sycl-pr-cross-device-validation.md`
- `docs/packaging-checklist.md`
- `patches/0001-sycl-reorder-on-MoE-for-Q4_K-and-Q5_K-mul_mat_id.patch`
- `patches/0002-sycl-make-concat-submit-only-drop-host-waits-on-in-o.patch`
- `patches/0003-sycl-extend-MoE-reorder-to-Q6_K-mul_mat_id-allow-gra.patch`
- `patches/0004-sycl-fuse-moe-gate-up-glu.patch`
- `patches/README.md`
- `eval/micro-moe/`
- `eval/nx2/`
- `eval/upstream/test_matrix.sh`
- `results/README.md`
- `results/model-checksums.sha256`
- `results/micro-moe/`
- `results/fabler-q6k-reorder/`
- `results/fabler-gate-up-fusion/`
- `results/nx2-kernel-release-gate/`
- `results/nx2-controls/`
- `results/nx2-first-token/`
- `results/nx2-path-debug/`
- `results/upstream-pr/`

Note: `.gitignore` excludes new `*.log` files. Commit summaries and JSON with a
normal `git add`; use `git add -f results/.../*.log` only when a raw log is part
of the evidence trail being preserved.

## Leave Local Unless Explicitly Needed

- `eval/memory/contention_probe.py`
- `eval/memory/tag_bench.py`
- `eval/memory/tag_corpus.json`
- `results/tag-bench-raw/`
- `results/tag-recall.csv`
- `docs/session-handoff-20260612-kernel-campaign.md`
- `results/atlas-3090/`
- `results/sycl-pr-cross-device/20260612T161733Z/`

Those files appear to belong to a separate memory/tagging experiment, not the
Turbo lab-report cleanup. The Atlas and local cross-device sanity outputs are
also not release evidence: Atlas is CUDA, and the local cross-device run used a
small Q4_K fixture rather than a non-B70 Intel SYCL Q6_K model.

## External Worktrees

- `/home/frosty40/llama.cpp-sycl-moe-clean`: traceable cleanup branch with
  intermediate commits.
- `/home/frosty40/llama.cpp-sycl-moe-ready`: contribution-ready PR branch at
  `6b856575`.
- `/home/frosty40/llama.cpp`: left on `iq3-b70`; do not treat it as the clean
  contribution branch.
