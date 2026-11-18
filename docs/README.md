# Documentation Index

## Release-Facing

- `HF_MODEL_CARD.md` - source for the Hugging Face model card.
- `lab-report.md` - project-level report and retained benchmark narrative.
- `methodology.md` - measurement setup, controls, and correctness criteria.
- `kernel-release-gate.md` - policy for deciding whether new kernel results justify updating the HF card.
- `upstream-sycl-readiness.md` - notes for the llama.cpp SYCL PR.
- `a770-iq3-expert-residency.md` - A770-specific IQ3 expert residency experiment plan.
- `iq3-a770-block-layout.md` - frozen IQ3_A770 block layout/bpw spec + patch 0005 (CPU oracle) results.
- `a770-iq3-mixed-down-recipe.md` - mixed-precision down recipe (q3floor_q4top24) behind the shipped A770 IQ3 winner.
- `nx2-iq3-a770-deployment.md` - NX2-IQ3_A770 deployment: 1-card IQ3 vs 2-card Q4_K (`--sm`).

## Operational

- `packaging-checklist.md` - include/exclude list for release cleanup.
- `sycl-pr-cross-device-validation.md` - instructions for non-B70 Intel SYCL validation.
- `tensor-split-test-handoff.md` - handoff for the `--sm tensor` multi-GPU benchmark leg.

## Retained Experiments

- `reap-expert-pruning-plan.md` - REAP router-weighted expert pruning plan.
- `reap-results.md` - REAP evaluated and dropped; the documented negative result.

## Generated/Rendered

- `turbo-understanding.md` / `turbo-understanding.html`
- `turbo-web-report.html`
