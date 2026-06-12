# NexN2 B70 Turbo Lab Report

Date: 2026-06-10

> **Addendum 2026-06-12 (fabler campaign):** Turbo's MoE reorder now also covers the
> Q6_K `ffn_down_exps` expert weights (patches `0002`/`0003`). Fresh same-harness
> controls: stock 68.84 → Turbo **85.48 t/s** at ctx0 (**+24%**); Q4_K_M reaches
> **91.0 t/s**. PPL and `MUL_MAT_ID` op tests unchanged (714/714 on the pinned-base
> chain). SYCL graph replay was evaluated and rejected (5× slowdown on this driver;
> graphs remain default-off). Evidence:
> [`../results/fabler-q6k-reorder/`](../results/fabler-q6k-reorder/). The numbers
> below are the retained 2026-06-10 measurements.

## Summary

NexN2 B70 Turbo is a local deployment package for Nex-N2-mini on Intel Arc Pro
B70. It combines corrected GGUF artifacts, imatrix quant selection, the SYCL MoE
reorder path, Flash-Attention serving configuration, validation artifacts, and
OpenAI-compatible serving.

The current retained project control is Q5_K_M on the real NX2 GGUF artifact,
rerun with `eval/nx2/run_controls.sh` on 2026-06-10.

| config | decode @ ctx0 |
|---|---:|
| stock control: reorder off / FA off | 69.84 t/s |
| NexN2 B70 Turbo: reorder + FA | 81.69 t/s |
| gain | +17% |

## Package

| artifact | role |
|---|---|
| `NX2-Q5_K_M.gguf` | recommended all-rounder |
| `NX2-Q4_K_M.gguf` | fastest retained high-quality variant |
| `NX2-Q6_K.gguf` | accuracy reference |
| `NX2.imatrix` | calibration artifact |

The practical recommendation is Q5_K_M: 23.0 GB, 5.71 bpw, mean KLD 0.0201,
top-1 agreement 94.0%, and 81.7 t/s decode on the retained B70 measurement.
Q4_K_M is the speed-focused option at 85.7 t/s decode.

The GGUFs include the NexN2 llama.cpp load repair:
`qwen35moe.block_count=40` and `qwen35moe.nextn_predict_layers=0`.

## Validation

| check | retained result |
|---|---|
| `MUL_MAT_ID` backend op | 714/714 passed against CPU reference |
| full backend-op comparison | unpatched upstream and validated branch both report 11502/11514 passed |
| PPL spot check | 5.5643 +/- 0.15232 vs 5.5662 +/- 0.15242 |
| first-token timing | no measured lazy-reorder penalty under the retained Q5_K_M settings |

Correctness evidence is backend-op and perplexity based. Greedy token identity is
not used as the correctness criterion for the reorder path.

## Turbo SYCL Kernel Path

The reusable backend contribution is the Turbo SYCL kernel path for Q4_K/Q5_K
MoE `mul_mat_id`. The patch file is the reviewable delivery format:

```text
patches/0001-sycl-reorder-on-MoE-for-Q4_K-and-Q5_K-mul_mat_id.patch
```

The Turbo kernel path extends the existing SYCL reordered weight layout to MoE
expert `mul_mat_id` for Q4_K and Q5_K tensors. It covers the decode GEMV path,
dense MMVQ path, DMMV path, dequant-to-fp16/fp32 GEMM path, Q5_K reorder DMMV,
and per-expert reorder conversion.

Primary SYCL files touched:

- `ggml/src/ggml-sycl/mmvq.cpp`: reordered MoE MMVQ/GEMV launch path.
- `ggml/src/ggml-sycl/mmvq.hpp`: exported reordered MoE entry point.
- `ggml/src/ggml-sycl/dmmv.cpp`: Q5_K reordered DMMV/dequant path.
- `ggml/src/ggml-sycl/ggml-sycl.cpp`: lazy per-expert reorder and
  `mul_mat_id` dispatch.

| item | value |
|---|---|
| llama.cpp anchor | `ac4cddeb0` |
| upstream PR | https://github.com/ggml-org/llama.cpp/pull/24452 |
| retained branch | `sycl-moe-reorder-ready` |
| retained commit | `a7597d733 sycl: support reordered Q4_K and Q5_K MoE MUL_MAT_ID` |
| hardware used | Intel Arc Pro B70, oneAPI/icpx 2026.0, SYCL backend |
| targeted op result | 714/714 `MUL_MAT_ID` tests passed against CPU reference |
| full op comparison | unpatched upstream and validated branch both report 11502/11514 passed |
| PPL comparison | 5.5643 +/- 0.15232 vs 5.5662 +/- 0.15242 |

The project benchmark above is the NX2 package result. The SYCL component is the
Turbo kernel path for Q4_K/Q5_K MoE `mul_mat_id`, reviewed separately from the
model package.

## What Turbo Is

Turbo is the complete B70 package: model artifacts, quant selection, runtime
configuration, SYCL MoE reorder work, validation records, and serving setup. The
SYCL MoE reorder work is also retained as a reusable backend component for
Q4_K/Q5_K `mul_mat_id`.

## Reproduce

```bash
bash eval/nx2/run_controls.sh
bash serving/llama-server.sh
```

## Key Files

- `README.md`: project overview.
- `docs/HF_MODEL_CARD.md`: model package presentation.
- `docs/methodology.md`: measurement and conversion notes.
- `results/frontier.csv`: quant frontier.
- `results/nx2-controls/20260610T220943Z/`: retained NX2 controls.
- `results/model-checksums.sha256`: model artifact checksums.
