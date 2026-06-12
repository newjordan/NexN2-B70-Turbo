# NX2 Kernel Release Gate: 20260612T140842Z

Decision: **NO_HF_UPDATE**

## Why

- q5 ctx0 gain 0.97% is below 2.0% promotion bar

## Throughput

| case | baseline t/s | candidate t/s | delta | combined SEM | samples |
|---|---:|---:|---:|---:|---:|
| q5-ctx0 | 87.3168 | 88.1645 | +0.97% | 0.02% | 7/7 |
| q4-ctx0 | 91.2511 | 94.1822 | +3.21% | 0.13% | 7/7 |
| q5-131k | 42.0952 | 42.3148 | +0.52% | 0.02% | 3/3 |

## Accuracy

| case | baseline PPL | candidate PPL | delta | stderr baseline/candidate |
|---|---:|---:|---:|---:|
| q5-30chunk-ppl | 5.5682 | 5.5676 | -0.01% | 0.15248/0.15244 |

## Correctness

- candidate `MUL_MAT_ID` backend-op check: 690/690

## Inputs

- q4-ctx0: `baseline-q4-ctx0.json` (e0bfc65c2) vs `candidate-q4-ctx0.json` (e0bfc65c2)
- q5-131k: `baseline-q5-131k.json` (e0bfc65c2) vs `candidate-q5-131k.json` (e0bfc65c2)
- q5-ctx0: `baseline-q5-ctx0.json` (e0bfc65c2) vs `candidate-q5-ctx0.json` (e0bfc65c2)
- q5-ppl: `baseline-q5-ppl.log` vs `candidate-q5-ppl.log`
- backend-ops: `candidate-test-backend-ops-mul-mat-id.log`
