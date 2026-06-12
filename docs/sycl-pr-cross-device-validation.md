# SYCL PR Cross-Device Validation

Use this when asking someone to validate the llama.cpp SYCL MoE reorder PR on a
non-B70 Intel GPU.

Build the candidate with at least:

```bash
cmake --build build --target test-backend-ops llama-bench llama-perplexity -j 2
```

## Goal

This is portability validation, not a new headline benchmark:

- targeted `MUL_MAT_ID` correctness must pass,
- optimized decode must not lose quality versus optimization disabled,
- any speedup should be reported with the device name and model file.

## Runner

```bash
BIN=/path/to/candidate/llama.cpp/build/bin \
MODEL=/path/to/mixed-k-moe.gguf \
eval/nx2/run_sycl_pr_cross_device.sh
```

Optional baseline build:

```bash
BASELINE_BIN=/path/to/pre-pr/build/bin \
BIN=/path/to/candidate/build/bin \
MODEL=/path/to/mixed-k-moe.gguf \
eval/nx2/run_sycl_pr_cross_device.sh
```

The runner writes:

```text
results/sycl-pr-cross-device/<timestamp>/
```

## Model Choice

For Q6_K PR coverage, prefer a mixed K-quant MoE model whose down-projection
experts are Q6_K. The known local fixture is:

```text
/home/frosty40/models/nex-n2-mini/sweep/NX2-Q4_K_M.gguf
```

That file is about 20 GB, so it needs a GPU with enough VRAM. Smaller MoE
fixtures such as `Phi-mini-MoE-instruct-Q4_K_M.gguf` are still useful for
cross-device Q4_K MoE reorder smoke tests, but they do not prove the new Q6_K
path.

## Report Format

Ask validators to paste:

- GPU name and driver from `device.txt`,
- `SUMMARY.md`,
- whether the model fully offloaded with `-ngl 99`,
- any failure logs.

For the PR thread, avoid broad claims. A good summary is:

```text
Validated on <GPU>: MUL_MAT_ID passed, PPL unchanged within noise, decode opt-on
was <x>% vs opt-off on <model>.
```
