# NX2 Kernel Release Gate: 20260612-existing-artifacts

Decision: **HF_UPDATE_WORTHY**

## Why

- q5 ctx0 clears promotion bar and guardrails pass

## Throughput

| case | baseline t/s | candidate t/s | delta | combined SEM | samples |
|---|---:|---:|---:|---:|---:|
| q5-ctx0 | 85.6525 | 87.9634 | +2.70% | 0.15% | 5/5 |
| q4-ctx0 | 90.7697 | 93.4917 | +3.00% | 0.24% | 5/5 |
| q5-131k | 42.0656 | 42.2928 | +0.54% | 0.06% | 5/5 |

## Accuracy

| case | baseline PPL | candidate PPL | delta | stderr baseline/candidate |
|---|---:|---:|---:|---:|
| q5-30chunk-ppl | 5.5669 | 5.5642 | -0.05% | 0.15246/0.15223 |

## Correctness

- candidate `MUL_MAT_ID` backend-op check: 690/690

## Inputs

- q4-ctx0: `baseline-q4-ctx0.json` (e0bfc65c2) vs `candidate-q4-ctx0.json` (e0bfc65c2)
- q5-131k: `baseline-q5-131k.json` (e0bfc65c2) vs `candidate-q5-131k.json` (e0bfc65c2)
- q5-ctx0: `baseline-q5-ctx0.json` (e0bfc65c2) vs `candidate-q5-ctx0.json` (e0bfc65c2)
- q5-ppl: `baseline-q5-ppl.log` vs `candidate-q5-ppl.log`
- backend-ops: `candidate-test-backend-ops-mul-mat-id.log`
