# NexN2 B70 Turbo Lab Report

Date: 2026-06-10

> **Addendum 2026-06-12 (fabler campaign):** Turbo's MoE reorder now also covers the
> Q6_K `ffn_down_exps` expert weights (patches `0002`/`0003`). Fresh same-harness
> intermediate controls: stock 68.84 -> Q6 reorder path **85.48 t/s** at ctx0 (**+24%**); Q4_K_M reaches
> **91.0 t/s**. PPL and `MUL_MAT_ID` op tests unchanged (714/714 on the pinned-base
> chain). SYCL graph replay was evaluated and rejected (5× slowdown on this driver;
> graphs remain default-off). Evidence:
> [`../results/fabler-q6k-reorder/`](../results/fabler-q6k-reorder/). The numbers
> below are the retained 2026-06-10 measurements.
>
> **Addendum 2026-06-12 (NX2 MoE specialization):** `patches/0004` now defaults
> on exact NX2 Q4_K/Q5_K gate/up SwiGLU fusion (`2048x512`, top-k 8), the
> post-down weighted expert sum, and a fused F32 MoE tail add while keeping
> broader generic fusion variants opt-in. The retained default raises Q5_K_M
> ctx0 decode to **88.08 t/s** (tail-add A/B: 87.32 off vs 87.96 on), keeps
> Q5_K_M 30-chunk PPL flat (5.5642 +/- 0.15223), and the retained 131k guardrail
> improves to 42.29 t/s. Q4_K_M ctx0 reaches **93.49 t/s** under the same
> default. Targeted `MUL_MAT_ID` testing passed 690/690 with the default path;
> the full backend-op suite has the same known 12 `GET_ROWS` failures. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T083707Z-nx2-weighted-sum-fusion/`](../results/fabler-gate-up-fusion/20260612T083707Z-nx2-weighted-sum-fusion/).
> Tail-add follow-up evidence:
> [`../results/fabler-gate-up-fusion/20260612T091359Z-nx2-tail-add-fusion/`](../results/fabler-gate-up-fusion/20260612T091359Z-nx2-tail-add-fusion/).
> A shared-expert dense gate/up fusion follow-up is implemented but default-off:
> it measured Q5_K_M ctx0 at 87.27 t/s off vs 86.90 t/s on, so it remains an
> opt-in profiling path. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T090325Z-nx2-shared-gate-up-fusion/`](../results/fabler-gate-up-fusion/20260612T090325Z-nx2-shared-gate-up-fusion/).
> A structural activation-Q8 cache probe attempted to reuse the MoE gate/up
> activation quantization for that shared-expert dense gate/up path. It passed
> targeted `MUL_MAT_ID` testing, but cache-on lost to simply requantizing in the
> shared path: 87.76 t/s with cache on versus 88.14 t/s with cache off, then
> 87.43 t/s versus 87.85 t/s in a reversed repeat. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T114939Z-nx2-activation-q8-cache/`](../results/fabler-gate-up-fusion/20260612T114939Z-nx2-activation-q8-cache/).
> A vec4 weighted-sum follow-up is also implemented as an opt-in mode only:
> scalar/default measured 88.08 t/s versus 87.91 t/s for vec4, with PPL flat.
> Evidence:
> [`../results/fabler-gate-up-fusion/20260612T092946Z-nx2-weighted-sum-vec4/`](../results/fabler-gate-up-fusion/20260612T092946Z-nx2-weighted-sum-vec4/).
> A local-weight weighted-sum mode is also opt-in only: caching the eight top-k
> weights in workgroup-local memory measured 87.60 t/s versus 88.06 t/s for the
> retained scalar path. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T103409Z-nx2-weighted-sum-local-weights/`](../results/fabler-gate-up-fusion/20260612T103409Z-nx2-weighted-sum-local-weights/).
> A larger combined weighted-tail fusion is also present but default-off after
> measuring 87.98 t/s off versus 87.74 t/s on. A tail-add vec4 mode is present
> as opt-in mode `2`, but was too noisy to promote. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T094531Z-nx2-weighted-tail-fusion/`](../results/fabler-gate-up-fusion/20260612T094531Z-nx2-weighted-tail-fusion/)
> and
> [`../results/fabler-gate-up-fusion/20260612T094758Z-nx2-tail-add-vec4/`](../results/fabler-gate-up-fusion/20260612T094758Z-nx2-tail-add-vec4/).
> A tail-add+RMS_NORM fusion probe is also present but default-off: it writes
> both the final residual vector and the RMS_NORM output from one kernel, but
> Q5_K_M ctx0 regressed to 86.37 t/s versus 87.95/87.57 t/s off. Q4_K_M
> improved in the single guard pair (92.77 t/s on versus 91.78 t/s off), but
> Q5_K_M is the promotion gate. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T120710Z-nx2-tail-rms-fusion/`](../results/fabler-gate-up-fusion/20260612T120710Z-nx2-tail-rms-fusion/).
> A standalone RMS_NORM+MUL fusion probe is also implemented as
> `GGML_SYCL_ENABLE_RMS_NORM_MUL_FUSION`. Mode `2` uses 1024 work-items and was
> the best launch shape: Q5_K_M ctx0 measured 89.15 and 88.97 t/s versus
> 88.05/87.84 off, and Q4_K_M measured 95.77 t/s versus 91.28 off. It passed
> `RMS_NORM` 21/21 and `MUL_MAT_ID` 690/690 smoke tests, and a 5-chunk Q5_K_M
> WikiText spot was flat (5.1289 off versus 5.1329 on), but Q5_K_M remains
> below the +2% promotion bar, so it is default-off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T122324Z-nx2-rms-norm-mul-fusion/`](../results/fabler-gate-up-fusion/20260612T122324Z-nx2-rms-norm-mul-fusion/).
> Mode `3` pushes RMS_NORM+MUL further by also writing a reordered Q8_1 side
> buffer for the later MoE gate/up fusion, which then skips its normal
> activation quantization launch. The initial serial in-kernel Q8 path measured
> 88.04 t/s on Q5_K_M; the subgroup-parallel Q8 path improved to 88.77 t/s and
> Q4_K_M reached 95.80 t/s versus 92.37 t/s off. It passed `RMS_NORM` 21/21 and
> `MUL_MAT_ID` 690/690 smoke tests, with a 5-chunk Q5_K_M WikiText spot at
> 5.1402, but it still trails standalone mode `2` on Q5_K_M and remains
> default-off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T123830Z-nx2-rms-mul-q8-handoff/`](../results/fabler-gate-up-fusion/20260612T123830Z-nx2-rms-mul-q8-handoff/).
> Mode `4` narrows the same Q8 handoff with graph lookahead: it keeps mode `2`
> for RMS_NORM+MUL nodes whose output is not consumed by the later MoE gate/up
> fusion, and writes Q8 only for the post-norm activation that is consumed. It
> passed `RMS_NORM` 21/21 and `MUL_MAT_ID` 690/690, but Q5_K_M ctx0 regressed
> to 87.58 and 87.48 t/s versus 89.01 t/s for same-run mode `2`; Q4_K_M mode
> `4` measured 94.32 t/s. It remains default-off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T125005Z-nx2-rms-mul-selective-q8-handoff/`](../results/fabler-gate-up-fusion/20260612T125005Z-nx2-rms-mul-selective-q8-handoff/).
> A gate/up XOR subgroup reducer mode is also present but opt-in only:
> `reduce_over_group` measured 87.96 t/s versus 87.81 t/s for XOR, with
> targeted `MUL_MAT_ID` still passing 690/690. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T095919Z-nx2-gate-up-xor-reduce/`](../results/fabler-gate-up-fusion/20260612T095919Z-nx2-gate-up-xor-reduce/).
> Two SwiGLU activation variants are also opt-in only. Mode `3` (`exp2` plus
> native reciprocal) measured 87.74 t/s versus 88.06 t/s for retained mode `1`;
> mode `4` (native `exp` plus native reciprocal) measured 87.76 t/s versus a
> noisy 87.88 t/s retained-mode repeat. Both passed targeted `MUL_MAT_ID`
> testing. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T102545Z-nx2-swiglu-exp2-recip/`](../results/fabler-gate-up-fusion/20260612T102545Z-nx2-swiglu-exp2-recip/).
> A four-row gate/up rowpack scheduling mode is also opt-in only: it passed
> targeted `MUL_MAT_ID` testing but measured 87.70 t/s versus 88.10 t/s for
> retained mode `1`. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T103921Z-nx2-gate-up-rowpack4/`](../results/fabler-gate-up-fusion/20260612T103921Z-nx2-gate-up-rowpack4/).
> Two-row and eight-row variants were also tested as modes `6` and `7`; both
> passed targeted `MUL_MAT_ID` testing. The first Q5_K_M pass was noisy
> (85.97 t/s for mode `1`, 87.86 for mode `6`, 87.64 for mode `7`), but the
> reversed repeat restored mode `1` ahead of mode `6` (87.72 vs 87.46 t/s), and
> Q4_K_M also favored mode `1` (93.80 vs 93.53 t/s). Evidence:
> [`../results/fabler-gate-up-fusion/20260612T113923Z-nx2-gate-up-rowpack2-8/`](../results/fabler-gate-up-fusion/20260612T113923Z-nx2-gate-up-rowpack2-8/).
> Mode `8` uses a four-row gate/up workgroup that first copies the shared Q8_1
> activation row into local memory. It passed targeted `MUL_MAT_ID` testing,
> but Q5_K_M only tied retained mode in the reversed repeat (88.20 vs 88.17,
> then 87.67 vs 87.67 t/s), while Q4_K_M regressed to 90.08 t/s versus
> 93.91 t/s retained. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T130548Z-nx2-gate-up-local-q8/`](../results/fabler-gate-up-fusion/20260612T130548Z-nx2-gate-up-local-q8/).
> Modes `9` and `10` test an expert-pack gate/up scheduler: one selected expert
> per workgroup with eight subgroups covering the top-k experts for one output
> row; mode `10` additionally caches the shared Q8_1 activation row in
> workgroup-local memory. Both passed targeted `MUL_MAT_ID` testing, but Q5_K_M
> mode `9` stayed below the +2% promotion bar across the reversed repeat
> (88.09 vs 86.83 t/s, then 87.13 vs 87.00 t/s), and mode `10` trailed at
> 87.39 t/s. Q4_K_M mode `9` measured 93.87 t/s versus 92.75 t/s retained in
> the single guard pair. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T131422Z-nx2-gate-up-expert-pack/`](../results/fabler-gate-up-fusion/20260612T131422Z-nx2-gate-up-expert-pack/).
> Mode `11` fuses the inner reordered Q4_K/Q5_K vecdot helper itself: it loads
> each shared Q8_1 activation lane once and accumulates up and gate dot products
> in the same loop while keeping the retained one-row scheduler. It passed
> targeted `MUL_MAT_ID` testing, but Q5_K_M only tied retained mode after the
> reversed repeat (87.59 vs 87.07 t/s, then 87.30 vs 87.29 t/s) and Q4_K_M
> regressed to 92.76 t/s versus 94.15 t/s retained. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T133008Z-nx2-gate-up-dual-dot/`](../results/fabler-gate-up-fusion/20260612T133008Z-nx2-gate-up-dual-dot/).
> A gate/up Q8 handoff mode is also opt-in only: it writes a down-ready Q8_1
> side buffer from the fused gate/up kernel so the following Q6_K down projection
> can skip its quantization launch. It passed targeted `MUL_MAT_ID` smoke
> testing and model benchmark execution, but measured 88.02 t/s on vs
> 87.93 t/s off in one pair and 87.34 t/s on vs 87.56 t/s off in a reversed
> repeat. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T104952Z-nx2-gate-up-q8-handoff/`](../results/fabler-gate-up-fusion/20260612T104952Z-nx2-gate-up-q8-handoff/).
> A current-stack exact Q6_K down specialization is also opt-in only: it
> preserves expert parallelism but specializes the reordered down GEMV to
> `512x2048`, top-k 8. It passed targeted `MUL_MAT_ID` testing but measured
> 87.62 t/s versus 87.84 t/s with the specialization off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T110216Z-nx2-q6-down-exact-current/`](../results/fabler-gate-up-fusion/20260612T110216Z-nx2-q6-down-exact-current/).
> Combining Q8 handoff with exact Q6_K down also remains default-off after
> measuring 87.51 t/s versus 87.68 t/s for the retained default. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T110640Z-nx2-q8-handoff-q6-down-exact-combo/`](../results/fabler-gate-up-fusion/20260612T110640Z-nx2-q8-handoff-q6-down-exact-combo/).
> A shared-gate-tail fusion is also opt-in only: it fuses the scalar shared
> expert gate multiply and the two F32 tail adds into one kernel. It passed
> targeted `MUL_MAT_ID` smoke testing, but Q5_K_M ctx0 stayed below the +2%
> promotion bar across noisy repeats (88.16 on vs 87.81 off, then 87.83 on vs
> 87.67 off). Q4_K_M ctx0 measured 94.22 t/s on versus 92.64 t/s off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T111328Z-nx2-shared-gate-tail-fusion/`](../results/fabler-gate-up-fusion/20260612T111328Z-nx2-shared-gate-tail-fusion/).
> A vec4 shared-gate-tail mode `2` is also present and passed targeted testing;
> it measured Q5_K_M ctx0 at 87.88 t/s versus 87.07 t/s off, then 87.80 t/s
> versus 87.22 t/s off in a reversed repeat, while Q4_K_M measured 94.43 t/s
> versus 92.00 t/s off. Because Q5 remains below the +2% promotion gate, it is
> kept opt-in. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T112008Z-nx2-shared-gate-tail-vec4/`](../results/fabler-gate-up-fusion/20260612T112008Z-nx2-shared-gate-tail-vec4/).
> Shared-gate-tail modes `5` and `6` additionally broadcast the scalar gate
> through workgroup-local memory for scalar and vec4 tails. Both passed targeted
> `MUL_MAT_ID` testing, but Q5_K_M ctx0 measured 87.65 and 87.55 t/s versus
> 87.81 t/s for the same-run retained default, so the local-gate broadcast is
> also default-off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T125854Z-nx2-shared-gate-tail-local-gate/`](../results/fabler-gate-up-fusion/20260612T125854Z-nx2-shared-gate-tail-local-gate/).
> A sigmoid+shared-gate-tail mode `4` additionally folds the preceding scalar
> sigmoid into the vec4 tail kernel and broadcasts the gate through workgroup
> local memory. It passed targeted testing and measured 88.15 t/s versus
> 87.18 t/s off in one Q5_K_M pair, but reversed to 87.47 t/s versus 87.82 t/s
> off in the repeat. Q4_K_M still improved to 94.52 t/s versus 92.75 t/s off.
> It remains default-off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T112753Z-nx2-shared-gate-sigmoid-tail/`](../results/fabler-gate-up-fusion/20260612T112753Z-nx2-shared-gate-sigmoid-tail/).
> A Q6_K down+weighted-sum fusion is present but default-off after measuring
> 87.98 t/s off versus 87.85 t/s on. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T100913Z-nx2-down-weighted-sum/`](../results/fabler-gate-up-fusion/20260612T100913Z-nx2-down-weighted-sum/).
> A local-reduction down+weighted mode `3` is also present: it assigns one
> workgroup per output row, one subgroup per selected expert, and reduces the
> eight weighted expert sums through local memory instead of serializing or
> using global atomics. It passed targeted `MUL_MAT_ID` testing but measured
> 87.68 t/s versus 87.99 t/s off, so it remains default-off. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T115730Z-nx2-down-weighted-local-reduce/`](../results/fabler-gate-up-fusion/20260612T115730Z-nx2-down-weighted-local-reduce/).
> The atomic down+weighted variant preserves expert parallelism and passes
> targeted `MUL_MAT_ID` testing (690/690), but stays default-off after measuring
> 88.17 t/s off versus 87.68 t/s on. The dispatch guard probe also stays
> default-off after measuring 88.09 t/s off versus 87.78 t/s on. Evidence:
> [`../results/fabler-gate-up-fusion/20260612T101912Z-nx2-down-weighted-atomic/`](../results/fabler-gate-up-fusion/20260612T101912Z-nx2-down-weighted-atomic/)
> and
> [`../results/fabler-gate-up-fusion/20260612T101554Z-nx2-dispatch-guard/`](../results/fabler-gate-up-fusion/20260612T101554Z-nx2-dispatch-guard/).

## Summary

NexN2 B70 Turbo is a local deployment package for Nex-N2-mini on Intel Arc Pro
B70. It combines corrected GGUF artifacts, imatrix quant selection, the SYCL MoE
reorder path, Flash-Attention serving configuration, validation artifacts, and
OpenAI-compatible serving.

The current retained project control is Q5_K_M on the real NX2 GGUF artifact,
with the 2026-06-12 release-gate results retained under
`results/nx2-kernel-release-gate/`.

| config | decode @ ctx0 |
|---|---:|
| fresh stock control: reorder off / FA off | 68.8 t/s |
| NexN2 B70 Turbo: reorder + FA + NX2 fused MoE | 88.1 t/s |
| gain | +28% |

## Package

| artifact | role |
|---|---|
| `NX2-Q5_K_M.gguf` | recommended all-rounder |
| `NX2-Q4_K_M.gguf` | fastest retained high-quality variant |
| `NX2-Q6_K.gguf` | accuracy reference |
| `NX2.imatrix` | calibration artifact |

The practical recommendation is Q5_K_M: 23.0 GB, 5.71 bpw, mean KLD 0.0201,
top-1 agreement 94.0%, and 88.1 t/s decode on the retained B70 measurement.
Q4_K_M is the speed-focused option at 93.5 t/s decode.

The GGUFs include the NexN2 llama.cpp load repair:
`qwen35moe.block_count=40` and `qwen35moe.nextn_predict_layers=0`.

## Validation

| check | retained result |
|---|---|
| `MUL_MAT_ID` backend op | 690/690 passed for retained `0004` path; 714/714 for the Q6_K reorder chain |
| full backend-op comparison | same known SYCL `GET_ROWS` failures as the retained validation trail |
| PPL spot check | 5.5682 +/- 0.15248 baseline vs 5.5676 +/- 0.15244 candidate in the fresh release gate |
| first-token timing | no measured lazy-reorder penalty under the retained Q5_K_M settings |

Correctness evidence is backend-op and perplexity based. Greedy token identity is
not used as the correctness criterion for the reorder path.

## Turbo SYCL Kernel Path

The reusable backend contribution is the Turbo SYCL reorder path for MoE
`mul_mat_id`. The project patch chain is the reviewable delivery format:

```text
patches/0001-sycl-reorder-on-MoE-for-Q4_K-and-Q5_K-mul_mat_id.patch
patches/0003-sycl-extend-MoE-reorder-to-Q6_K-mul_mat_id-allow-gra.patch
patches/0004-sycl-fuse-moe-gate-up-glu.patch
```

The Turbo kernel path extends the existing SYCL reordered weight layout to MoE
expert `mul_mat_id` for Q4_K, Q5_K, and Q6_K tensors. It covers the decode GEMV
path, dense MMVQ path, DMMV path, dequant-to-fp16/fp32 GEMM path, Q5_K reorder
DMMV, Q6_K reordered MoE dispatch, and per-expert reorder conversion. Patch
`0004` is NX2-specific and remains separate from the minimal llama.cpp PR.

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
| retained PR commit | `6b856575 sycl: extend MoE reorder to Q6_K mul_mat_id` |
| hardware used | Intel Arc Pro B70, oneAPI/icpx 2026.0, SYCL backend |
| targeted op result | 714/714 `MUL_MAT_ID` tests passed against CPU reference for the PR chain |
| project release gate | retained `0004` path passed 690/690 targeted `MUL_MAT_ID` and PPL guardrails |

The project benchmark above is the NX2 package result. The SYCL component is the
Turbo kernel path for Q4_K/Q5_K/Q6_K MoE `mul_mat_id`, reviewed separately from
the model package. The exact NX2 fusions in `0004` are project-local package
work, not part of the minimal upstream PR.

## What Turbo Is

Turbo is the complete B70 package: model artifacts, quant selection, runtime
configuration, SYCL MoE reorder work, validation records, and serving setup. The
SYCL MoE reorder work is also retained as a reusable backend component for
Q4_K/Q5_K/Q6_K `mul_mat_id`.

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
- `results/nx2-kernel-release-gate/`: retained release-gate summaries.
- `results/fabler-gate-up-fusion/`: patch `0004` campaign index and raw artifacts.
- `results/nx2-controls/`: retained NX2 controls.
- `results/model-checksums.sha256`: model artifact checksums.
