# Ctx0 Harness Check

Purpose: confirm that the current stock ctx0 control is stable across short
`llama-bench` generation settings.

Current rerun on the same NX2 Q5_K_M artifact and current build:

| setting | avg t/s |
|---|---:|
| stock ctx0, `-n 32 -r 2` | 69.49 |
| stock ctx0, `-n 64 -r 2` | 69.76 |
| stock ctx0, `-n 128 -r 5` | 69.59 |

Conclusion: stock ctx0 is stable around 69.5-69.8 t/s under the tested
generation settings. Use the fresh `eval/nx2/run_controls.sh` result for the
reported stock ctx0 control.
