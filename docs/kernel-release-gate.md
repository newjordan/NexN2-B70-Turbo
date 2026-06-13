# Kernel Release Gate

This is the decision path for whether a new SYCL kernel campaign result is
worth updating the Hugging Face model card.

## Gate

Primary promotion metric:

- Q5_K_M ctx0 decode must improve by at least 2.0%.
- The Q5_K_M gain must also be larger than 2x combined standard error of the
  two llama-bench sample means.

Guardrails:

- Q4_K_M ctx0 must not regress by more than 1.0%.
- Q5_K_M 131k decode must not regress by more than 3.0%.
- Q5_K_M 30-chunk WikiText PPL must not regress by more than 0.5%.
- Targeted `test-backend-ops test -o MUL_MAT_ID` must pass.

## Reproduce

Run a fresh release gate against the current candidate binary:

```bash
eval/nx2/run_kernel_release_gate.sh
```

By default, the runner compares the same binary with the retained candidate
features enabled versus disabled. To compare against a separate deployed or
published baseline build:

```bash
BASELINE_BIN=/path/to/baseline/build/bin \
BIN=/path/to/candidate/build/bin \
eval/nx2/run_kernel_release_gate.sh
```

The runner writes JSON, logs, and `SUMMARY.md` under:

```text
results/nx2-kernel-release-gate/<timestamp>/
```

To add an informational 2-device tensor-split row for multi-GPU claims:

```bash
RUN_TENSOR_SPLIT=1 TENSOR_SPLIT=1/1 eval/nx2/run_kernel_release_gate.sh
```

That emits `q5-ctx0-sm-tensor` baseline/candidate rows using
`llama-bench -sm tensor -ts "$TENSOR_SPLIT"`. The row is reported in
`SUMMARY.md`, but it is not part of the HF-update promotion decision because the
single-B70 package remains the release target.

To rescore an existing run:

```bash
eval/nx2/summarize_kernel_release_gate.py --write results/nx2-kernel-release-gate/<timestamp>
```

## Current Decisions

Current retained artifact summary:

```text
results/nx2-kernel-release-gate/20260612-existing-artifacts/SUMMARY.md
```

Result: **HF_UPDATE_WORTHY** for patch `0004` versus the pre-`0004` retained
default. The measured deltas are:

| case | baseline t/s | candidate t/s | delta |
|---|---:|---:|---:|
| Q5_K_M ctx0 | 85.6525 | 87.9634 | +2.70% |
| Q4_K_M ctx0 | 90.7697 | 93.4917 | +3.00% |
| Q5_K_M 131k | 42.0656 | 42.2928 | +0.54% |

Accuracy/correctness guardrails:

| check | baseline | candidate | result |
|---|---:|---:|---|
| Q5_K_M 30-chunk PPL | 5.5669 +/- 0.15246 | 5.5642 +/- 0.15223 | no measured loss |
| `MUL_MAT_ID` backend ops | - | 690/690 | pass |

That is enough to update the HF model card if the published card still reports
the pre-`0004` performance package. It is not enough to claim a stable Q5_K_M
90+ t/s result.

Fresh same-binary run:

```text
results/nx2-kernel-release-gate/20260612T140842Z/SUMMARY.md
```

Result: **NO_HF_UPDATE** for the current same-binary incremental comparison,
because Q5_K_M ctx0 improves by only +0.97% against that run's baseline:

| case | baseline t/s | candidate t/s | delta |
|---|---:|---:|---:|
| Q5_K_M ctx0 | 87.3168 | 88.1645 | +0.97% |
| Q4_K_M ctx0 | 91.2511 | 94.1822 | +3.21% |
| Q5_K_M 131k | 42.0952 | 42.3148 | +0.52% |

Fresh accuracy/correctness guardrails still pass:

| check | baseline | candidate | result |
|---|---:|---:|---|
| Q5_K_M 30-chunk PPL | 5.5682 +/- 0.15248 | 5.5676 +/- 0.15244 | no measured loss |
| `MUL_MAT_ID` backend ops | - | 690/690 | pass |

The overnight vec8 weighted-sum/tail-add probe is **not** HF-update-worthy yet.
It is not exported in `patches/0004`, lacks reversed-repeat and Q4 guardrail
validation, and did not beat the retained scalar default in the combined mode.
