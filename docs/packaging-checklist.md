# Packaging Checklist

Use this as the local include/exclude list before any report or release commit.

## Include In Turbo Report Package

- `README.md`
- `docs/HF_MODEL_CARD.md`
- `docs/lab-report.md`
- `docs/methodology.md`
- `docs/turbo-understanding.md`
- `docs/turbo-understanding.html`
- `docs/upstream-sycl-readiness.md`
- `docs/packaging-checklist.md`
- `patches/0001-sycl-reorder-on-MoE-for-Q4_K-and-Q5_K-mul_mat_id.patch`
- `patches/README.md`
- `eval/micro-moe/`
- `eval/nx2/`
- `eval/upstream/test_matrix.sh`
- `results/model-checksums.sha256`
- `results/micro-moe/`
- `results/nx2-controls/`
- `results/nx2-first-token/`
- `results/nx2-path-debug/`
- `results/upstream-pr/`

## Leave Local Unless Explicitly Needed

- `eval/memory/contention_probe.py`
- `eval/memory/tag_bench.py`
- `eval/memory/tag_corpus.json`
- `results/tag-bench-raw/`
- `results/tag-recall.csv`

Those files appear to belong to a separate memory/tagging experiment, not the
Turbo lab-report cleanup.

## External Worktrees

- `/home/frosty40/llama.cpp-sycl-moe-clean`: traceable cleanup branch with
  intermediate commits.
- `/home/frosty40/llama.cpp-sycl-moe-ready`: single-commit contribution-ready
  patch branch at `b5994f6`.
- `/home/frosty40/llama.cpp`: left on `iq3-b70`; do not treat it as the clean
  contribution branch.
