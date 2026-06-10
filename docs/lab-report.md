# NexN2 B70 Turbo Lab Report

Date: 2026-06-10

## Executive Summary

NexN2 B70 Turbo is a full local deployment package for Nex-N2-mini on Intel Arc
Pro B70. It combines GGUF conversion, MTP/NextN metadata repair, imatrix
calibration, quant selection, SYCL runtime/kernel work, Flash-Attention serving
configuration, long-context validation, and OpenAI-compatible serving.

The fresh reproducible project control is Q5_K_M on the real NX2 GGUF artifact.
Rerun via `eval/nx2/run_controls.sh` on 2026-06-10:

| config | ctx0 decode | 131k decode |
|---|---:|---:|
| stock control: reorder off / FA off | 69.84 t/s | 20.10 t/s |
| deployed: Turbo + FA | 81.69 t/s | 40.88 t/s |
| reproducible gain | +17% | +103% |

Historical retained CSV data in `results/longctx-fa.csv` records:

| config | ctx0 decode | 131k decode |
|---|---:|---:|
| stock control: reorder off / FA off | 55.43 t/s | 19.99 t/s |
| deployed: Turbo + FA | 81.26 t/s | 40.95 t/s |

The `40.95 t/s` value is the deployed 131k-context result, not an old baseline.
The fresh rerun confirms the deep-context result. A follow-up harness check in
`results/nx2-controls/ctx0-harness-check-20260610T231352Z/` reran stock ctx0
with `-n 32`, `-n 64`, and `-n 128`; all were about 69.5-69.8 t/s. The older
55.43 t/s value comes from historical CSV-only data in `/home/frosty40/nx2-turbo`
and is not reproduced by the current build/model/harness.

## Deliverables

**A. Model package**

- Nex-N2-mini-derived GGUF variants in `/home/frosty40/models/nex-n2-mini/`.
- Published/model-card material in `docs/HF_MODEL_CARD.md`.
- GGUF metadata repair: `qwen35moe.block_count=40` and
  `qwen35moe.nextn_predict_layers=0`.
- Imatrix calibration and quant frontier in `results/frontier.csv`.
- SHA256 checksums in `results/model-checksums.sha256`.

**B. Runtime and validation package**

- Serving launcher: `serving/llama-server.sh`.
- Long-context controls: `results/longctx-fa.csv`.
- Reorder-depth controls: `results/longctx-reorder.csv`.
- Needle-in-haystack results: `results/niah.csv` and
  `results/niah-pareto.csv`.
- Methodology: `docs/methodology.md`.

**C. llama.cpp backend contribution candidate**

- Clean worktree: `/home/frosty40/llama.cpp-sycl-moe-clean`.
- Ready worktree: `/home/frosty40/llama.cpp-sycl-moe-ready`.
- Clean branch: `sycl-moe-reorder-clean`.
- Ready branch: `sycl-moe-reorder-ready`.
- Ready commit: `b5994f6 sycl: support reordered Q4_K and Q5_K MoE MUL_MAT_ID`.
- Audit notes: `docs/upstream-sycl-readiness.md`.
- Micro-MoE fixture results: `results/micro-moe/`.

## Model Artifacts

The key retained NX2 artifacts are:

| artifact | role |
|---|---|
| `NX2-Q5_K_M.gguf` | recommended all-rounder |
| `NX2-Q4_K_M.gguf` | fastest retained high-quality variant |
| `NX2-Q6_K.gguf` | accuracy/reference quant |
| `NX2.imatrix` | calibration artifact |

The quant frontier shows Q5_K_M as the practical winner: 23.0 GB, 5.71 bpw,
mean KLD 0.0201, top-1 agreement 94.0%, and 81.7 t/s decode. Q4_K_M is the
fastest retained high-quality variant at 85.7 t/s decode.

## Controls

Use explicit control names:

| label | meaning |
|---|---|
| stock control | reorder off / FA off |
| deployed Turbo + FA | reorder/runtime path enabled, Flash-Attention enabled |
| micro-MoE opt off | `GGML_SYCL_DISABLE_OPT=1` |
| micro-MoE opt on | `GGML_SYCL_DISABLE_OPT=0` |
| backend candidate audit | base llama.cpp code vs candidate code |

Avoid saying “base” without naming the base.

## Throughput Results

Fresh reproducible control from `results/nx2-controls/20260610T220943Z/`:

| config | ctx0 | 131k |
|---|---:|---:|
| stock control | 69.84 t/s | 20.10 t/s |
| deployed Turbo + FA | 81.69 t/s | 40.88 t/s |

Historical control from `results/longctx-fa.csv`:

| config | ctx0 | 131k |
|---|---:|---:|
| stock control | 55.43 t/s | 19.99 t/s |
| deployed Turbo + FA | 81.26 t/s | 40.95 t/s |

Ctx0 harness check from
`results/nx2-controls/ctx0-harness-check-20260610T231352Z/`:

| stock ctx0 setting | avg t/s |
|---|---:|
| `-n 32 -r 2` | 69.49 |
| `-n 64 -r 2` | 69.76 |
| `-n 128 -r 5` | 69.59 |

Use the fresh 69.x stock ctx0 number for reproducible claims. Keep 55.43 only
as historical retained CSV data unless its original raw command/log is recovered.

Reorder-depth control from `results/longctx-reorder.csv`:

| depth | Q5_K_M off -> on | gain | Q4_K_M off -> on | gain |
|---|---:|---:|---:|---:|
| 0 | 70.07 -> 82.84 | +18.2% | 74.53 -> 87.45 | +17.3% |
| 4096 | 65.47 -> 76.00 | +16.1% | 69.33 -> 79.61 | +14.8% |
| 16384 | 55.65 -> 62.59 | +12.5% | 58.28 -> 65.26 | +12.0% |
| 32768 | 46.21 -> 51.01 | +10.4% | 47.94 -> 52.63 | +9.8% |

Flash-Attention control from `results/longctx-fa.csv`:

| depth | FA off | FA on | gain |
|---|---:|---:|---:|
| 0 | 82.71 | 81.26 | -1.8% |
| 32768 | 50.97 | 64.57 | +26.7% |
| 131072 | 21.07 | 40.95 | +94.3% |
| 262144 | n/a | 26.80 | fits in 32 GB |

## Accuracy And Correctness

- Q6_K is the retained accuracy reference.
- Frontier metrics include KLD and top-1 agreement against Q6_K.
- Reorder path correctness uses `test-backend-ops` and perplexity, not greedy
  token identity.
- Retained targeted `MUL_MAT_ID` check: 714/714 passed against CPU reference.
- Retained full-suite backend-op comparison: base and candidate both report
  11502/11514 passed with the same 12 `GET_ROWS` failures.
- Retained PPL spot check: Q5_K_M base 5.5643 +/- 0.15232, patched 5.5662 +/-
  0.15242.

## Long Context

NexN2 is a hybrid model. Ten of forty layers are full attention and the rest are
Gated Delta Net layers. KV grows at about 20 KiB/token, so full 262144 context
is about 5 GiB of KV.

Retained long-context evidence:

- Q5_K_M f16 KV reaches native 262144 context.
- 131k deployed decode is 40.95 t/s.
- Needle-in-haystack retrieval passes 8/8 up to 120k in retained results.
- The long-context Pareto campaign reports 155/155 pass across 32k-520k
  configurations.

## Micro-MoE Reproducer

The smaller reproducible MoE fixture is:

```text
/home/frosty40/models/micro-moe/Phi-mini-MoE-instruct-Q4_K_M.gguf
```

Recorded by llama.cpp as:

```text
phimoe 16x3.8B Q4_K - Medium
7,647,632,704 params
```

It loads on the clean SYCL branch and reaches `MUL_MAT_ID` with 3D Q4_K expert
tensors:

```text
blk.0.ffn_gate_exps.weight: type=q4_K; ne=[4096, 960, 16, 1]
```

Micro-MoE control:

| setting | avg t/s | delta |
|---|---:|---:|
| optimization disabled | 101.50 | baseline |
| optimization enabled | 106.30 | +4.73% |

Artifacts are in `results/micro-moe/`.

Known-good script dry-run after the oneAPI strict-mode fix:

```text
results/micro-moe/20260610T223353Z/
opt off: 100.77 t/s
opt on: 107.36 t/s
```

The dry-run includes `path-debug.log` and `path-debug-filtered.log`.

## First-Token Cost

Retained first-token timing is in:

```text
results/nx2-first-token/20260610T230504Z/
```

Median timing from `timeline.csv`:

| case | model setup | prompt/first-token eval | steady eval |
|---|---:|---:|---:|
| stock first token, reorder off | 22.02 s | 1.12 s | n/a |
| Turbo first token, reorder on | 22.06 s | 1.05 s | n/a |
| stock steady, reorder off | 22.79 s | 1.07 s | 1.95 s / 128 |
| Turbo steady, reorder on | 22.59 s | 0.94 s | 1.87 s / 128 |

This did not show a first-token lazy-reorder penalty on Q5_K_M under the
measured settings. Treat it as a whole optimization-path control, not isolated
attribution to one source line.

## NX2 Path Evidence

NX2 Q4_K_M and Q5_K_M path-debug logs were captured with `GGML_SYCL_DEBUG=1`
and one-token `llama-bench` runs:

```text
results/nx2-path-debug/20260610T223353Z/
```

Representative Q4_K_M expert path:

```text
ggml_sycl_mul_mat_id ... src0='blk.0.ffn_gate_exps.weight':type=q4_K;ne=[2048, 512, 256, 1]
```

Representative Q5_K_M expert path:

```text
ggml_sycl_mul_mat_id ... src0='blk.0.ffn_gate_exps.weight':type=q5_K;ne=[2048, 512, 256, 1]
```

This proves the actual NX2 model artifacts exercise 3D expert `MUL_MAT_ID`
paths on SYCL.

## Why The Speedup Happens

The system improves decode by aligning the model artifact, quant choice, and
runtime path with the B70:

- Q4_K/Q5_K are the fast retained quant families for this model on SYCL.
- Reordered weight layout improves the expert GEMV/MMVQ path.
- Flash-Attention is the large depth win at 131k context.
- f16 KV keeps long-context decode fast on this backend.
- Metadata repair removes the speculative MTP/NextN block from standard
  inference loading.

The micro-MoE result shows the same class of 3D expert routing is reproducible
on a smaller model, while its smaller gain also shows architecture and quant
distribution matter.

## Not Claimed

- Do not claim a dense model exercises the MoE expert path.
- Do not claim greedy token identity as primary correctness evidence.
- Do not use `40.95 t/s` as an old baseline; it is the deployed 131k result.
- Do not use the historical 55.43 ctx0 stock result as reproducible unless the
  original raw command/log is recovered.
- Do not describe the whole Turbo project as only one llama.cpp code patch.
- Do not use llama.cpp PR language as the project-level benchmark story.

## Repro Commands

NX2 controls:

```bash
bash eval/nx2/run_controls.sh
```

Micro-MoE smoke and control:

```bash
bash eval/micro-moe/run_phi_moe_smoke.sh
```

Serving:

```bash
bash serving/llama-server.sh
```

Backend candidate matrix:

```bash
bash eval/upstream/test_matrix.sh
```

## Artifact Index

- `README.md`: project-facing overview.
- `docs/methodology.md`: hardware, conversion, quant, context methodology.
- `docs/turbo-understanding.md`: engineering study notes.
- `docs/upstream-sycl-readiness.md`: backend contribution audit notes.
- `results/frontier.csv`: quant frontier.
- `results/longctx-reorder.csv`: reorder-depth controls.
- `results/longctx-fa.csv`: FA-depth and deployed control.
- `results/micro-moe/`: smaller MoE fixture evidence.
- `results/nx2-controls/20260610T220943Z/`: fresh NX2 stock/deployed controls.
- `results/nx2-path-debug/20260610T223353Z/`: NX2 Q4/Q5 path-debug evidence.
- `results/model-checksums.sha256`: model artifact checksums.
- `results/upstream-pr/`: backend contribution audit artifacts.
- `eval/nx2/run_controls.sh`: NX2 control runner.
- `eval/micro-moe/run_phi_moe_smoke.sh`: micro-MoE runner.
