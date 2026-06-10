# Micro MoE Fixture Smoke Test

Date: 2026-06-10

Purpose: identify a smaller reproducible MoE GGUF that can exercise the SYCL
`MUL_MAT_ID` expert path without using the full NX2 model artifacts.

## Fixture

- Repo: `gabriellarson/Phi-mini-MoE-instruct-GGUF`
- File: `Phi-mini-MoE-instruct-Q4_K_M.gguf`
- Local path: `/home/frosty40/models/micro-moe/Phi-mini-MoE-instruct-Q4_K_M.gguf`
- Recorded type: `phimoe 16x3.8B Q4_K - Medium`
- Recorded params: `7,647,632,704`
- Recorded size: `4,992,376,064` bytes

## Code Under Test

- Worktree: `/home/frosty40/llama.cpp-sycl-moe-clean`
- Branch: `sycl-moe-reorder-clean`
- Clean branch head: `885e3e3 sycl: trim MoE reorder comments`
- Ready branch: `/home/frosty40/llama.cpp-sycl-moe-ready`,
  `b5994f6 sycl: support reordered Q4_K and Q5_K MoE MUL_MAT_ID`
- Backend: SYCL
- GPU: Intel Arc Pro B70

## Smoke Results

Interactive load/generate succeeded with the clean branch:

```text
model: Phi-mini-MoE-instruct-Q4_K_M.gguf
prompt: 2+2=
generation: 68.9 t/s
```

Initial short retained `llama-bench` runs:

| setting | command shape | avg t/s |
|---|---|---:|
| optimize enabled | `GGML_SYCL_DISABLE_OPT=0`, `-p 0 -n 32 -r 3 -fa 0` | 104.94 |
| optimize disabled | `GGML_SYCL_DISABLE_OPT=1`, `-p 0 -n 32 -r 3 -fa 0` | 101.64 |

Files:

- `phi-mini-moe-q4km-opt-on.json`
- `phi-mini-moe-q4km-opt-off.json`

Follow-up controlled run with longer generation:

| setting | command shape | avg t/s | stddev | delta |
|---|---|---:|---:|---:|
| optimize disabled | `GGML_SYCL_DISABLE_OPT=1`, `-p 0 -n 128 -r 5 -fa 0` | 101.50 | 0.05 | baseline |
| optimize enabled | `GGML_SYCL_DISABLE_OPT=0`, `-p 0 -n 128 -r 5 -fa 0` | 106.30 | 0.27 | +4.73% |

Files:

- `phi-mini-moe-q4km-control-opt-off-n128-r5.json`
- `phi-mini-moe-q4km-control-opt-on-n128-r5.json`

Known-good script dry-run after fixing oneAPI `setvars.sh` ordering:

```text
results/micro-moe/20260610T223353Z/
opt off: 100.77 t/s
opt on: 107.36 t/s
```

## Path Evidence

`GGML_SYCL_DEBUG=1` confirms the model reaches SYCL `MUL_MAT_ID` with 3D
Q4_K expert tensors. Representative line:

```text
ggml_sycl_mul_mat_id ... src0='blk.0.ffn_gate_exps.weight':type=q4_K;ne=[4096, 960, 16, 1]
```

This makes Phi-mini-MoE Q4_K_M a useful smaller MoE smoke fixture for the Turbo
SYCL optimization path. It is not a replacement for NX2 validation, but it
answers the reproducibility question: yes, there is a smaller MoE GGUF that
loads and exercises the relevant `MUL_MAT_ID` expert shape on SYCL.
