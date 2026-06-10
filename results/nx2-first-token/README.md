# NX2 First-Token Timing

Retained run:

```text
results/nx2-first-token/20260610T230504Z/
```

Command source:

```text
eval/nx2/measure_first_token.sh
```

The run uses the real NX2 Q5_K_M artifact, the cleaned SYCL branch build, `-fa
on`, `-fit off`, `--no-warmup`, and fresh processes for each repetition.

Median timing from `timeline.csv`:

| case | model setup | prompt/first-token eval | steady eval |
|---|---:|---:|---:|
| stock first token, reorder off | 22.02 s | 1.12 s | n/a |
| Turbo first token, reorder on | 22.06 s | 1.05 s | n/a |
| stock steady, reorder off | 22.79 s | 1.07 s | 1.95 s / 128 |
| Turbo steady, reorder on | 22.59 s | 0.94 s | 1.87 s / 128 |

Interpretation: this control did not show a first-token lazy-reorder penalty on
Q5_K_M under the measured settings. Treat it as a whole optimization-path
control (`GGML_SYCL_DISABLE_OPT` off/on), not isolated attribution to one code
line.
