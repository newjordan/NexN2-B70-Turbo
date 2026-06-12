# NX2 Weighted/Tail Vec8 Probe

Date: 2026-06-12

Experiment: overnight vec8 follow-up around the retained weighted-sum and tail-add family.

## Results

| run | avg t/s | stddev |
|---|---:|---:|
| Q5_K_M ctx0, weighted vec8 | 88.1075 | 0.1354 |
| Q5_K_M ctx0, tail vec8 | 87.7895 | 0.2787 |
| Q5_K_M ctx0, weighted-tail vec8 | 86.8605 | 0.0608 |

Correctness smoke:

| check | result |
|---|---:|
| `test-backend-ops test -o MUL_MAT_ID`, retained default | 690/690 |

Conclusion: not release-worthy. The probe lacks the full reversed-repeat and Q4/131k/PPL guardrail package, and the combined weighted-tail variant loses clearly. The retained scalar weighted-sum plus scalar tail-add default remains the release path.
