# Tensor Split Test Handoff

This branch packages the `--sm tensor` benchmark leg so a multi-GPU tester can
answer whether Turbo's current Q5 path helps or falls back under tensor split.

## One-command test

From this repo:

```bash
eval/nx2/run_tensor_split_claim.sh
```

Common overrides:

```bash
BIN=/path/to/llama.cpp/build/bin \
Q5_MODEL=/path/to/NX2-Q5_K_M.gguf \
TENSOR_SPLIT=1/1 \
REPS=5 \
eval/nx2/run_tensor_split_claim.sh
```

The script sources oneAPI `setvars.sh`, runs `llama-bench`, and writes artifacts
under:

```text
results/nx2-tensor-split-claim/<timestamp>/
```

## What It Runs

It benchmarks Q5_K_M ctx0 decode four ways:

| row | split mode | Turbo features |
|---|---|---|
| `baseline-q5-ctx0` | `-sm layer` | `0004` feature toggles off |
| `candidate-q5-ctx0` | `-sm layer` | retained Turbo defaults on |
| `baseline-q5-ctx0-sm-tensor` | `-sm tensor -ts "$TENSOR_SPLIT"` | `0004` feature toggles off |
| `candidate-q5-ctx0-sm-tensor` | `-sm tensor -ts "$TENSOR_SPLIT"` | retained Turbo defaults on |

The important line for the multi-GPU claim is `q5-ctx0-sm-tensor` in
`SUMMARY.md`. The normal `q5-ctx0` row is included as a sanity/control row.

## Send Back

Please share these files from the output directory:

- `SUMMARY.md`
- `summary.txt`
- `MANIFEST.txt`
- `baseline-q5-ctx0-sm-tensor.json`
- `candidate-q5-ctx0-sm-tensor.json`
- matching `.log` files if the run errors or looks suspicious

The expected interpretation is simple: if `q5-ctx0-sm-tensor` is flat or slower,
tensor split is likely bypassing the newest single-device Turbo fusion paths. If
it improves, we have evidence that enough of the retained path survives under
tensor split to make a multi-GPU claim worth deeper validation.
