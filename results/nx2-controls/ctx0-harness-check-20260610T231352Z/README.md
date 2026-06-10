# Ctx0 Harness Check

Purpose: check whether the historical `55.43 t/s` stock ctx0 value was caused
by the older short `llama-bench` settings.

Source of historical settings:

```text
/home/frosty40/nx2-turbo/results/longctx-fa.csv
llama-bench -p 0 -n <32|64> -r <1|2>
```

Current rerun on the same NX2 Q5_K_M artifact and current build:

| setting | avg t/s |
|---|---:|
| stock ctx0, `-n 32 -r 2` | 69.49 |
| stock ctx0, `-n 64 -r 2` | 69.76 |
| stock ctx0, `-n 128 -r 5` | 69.59 |

Conclusion: the old short `n_gen`/repetition settings do not reproduce the
historical `55.43 t/s` value. Treat `55.43` as historical retained CSV data with
insufficient raw provenance. Use the fresh `eval/nx2/run_controls.sh` result for
reproducible stock ctx0 control.
