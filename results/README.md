# Results Index

This tree separates retained release evidence from exploratory measurements.

## Release Evidence

- `frontier.csv` - original quant frontier table.
- `model-checksums.sha256` - retained model artifact checksums.
- `nx2-controls/` - stock-vs-Turbo B70 control runs.
- `fabler-q6k-reorder/` - Q6_K MoE reorder campaign.
- `fabler-gate-up-fusion/` - patch `0004` NX2 fusion campaign and rejected probes.
- `nx2-kernel-release-gate/` - release-gate comparisons used to decide whether the HF card should move.
- `upstream-pr/` - llama.cpp PR validation artifacts for the reusable SYCL MoE reorder component.

## Supporting Evidence

- `nx2-first-token/` - first-token timing checks.
- `nx2-path-debug/` - debug traces proving NX2 MoE expert tensors reach the SYCL path.
- `micro-moe/` - small MoE fixture smoke tests.
- `sycl-pr-cross-device/` - local runner sanity output for cross-device validation flow.

## Not Release Evidence

- `atlas-3090/` - CUDA/RTX 3090 sanity data. Useful as machine notes only; it does not exercise the SYCL Turbo path.
- `tag-*` and `tag-bench-raw/` - separate memory/tagging experiments.
