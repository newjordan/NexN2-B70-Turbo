# Methodology

All measurements are on the real Nex-N2-mini model and the real Intel Arc Pro B70.

## Hardware
Intel Arc Pro B70 (Battlemage, PCI `8086:e223`), 32 GB GDDR6 ECC, 256-bit, ~600 GB/s (measured ~598 read / ~525 copy). 32 Xe2 cores, 256 XMX engines @ 2.8 GHz. XMX peak ~367 TOPS int8 / ~183 TFLOPS fp16. Ubuntu 24.04, oneAPI (icpx 2026.0), llama.cpp SYCL backend. Subgroup / WARP_SIZE = **16** on Intel. The retained Turbo path validates reordered MoE expert GEMV for Q4_K/Q5_K/Q6_K and the NX2 gate/up, weighted-sum, and tail-add fusions on this B70 stack.

## Loading NexN2 in llama.cpp (MTP / NextN metadata)
NexN2 carries an MTP (multi-token-prediction) block in its metadata (`qwen35moe.block_count=41`, `qwen35moe.nextn_predict_layers=1`); that head is speculative-only and is absent from the published checkpoint. Point llama.cpp at the 40 standard layers by setting:

```bash
python gguf-py/gguf/scripts/gguf_set_metadata.py NexN2.gguf qwen35moe.block_count 40 --force
python gguf-py/gguf/scripts/gguf_set_metadata.py NexN2.gguf qwen35moe.nextn_predict_layers 0 --force
```
(or convert with `--no-mtp`). This is lossless for standard inference and derived quants inherit it — **the published GGUFs already have it applied**, so they load out of the box.

## Speed comes from the kernel
Decode is kernel-limited on this stack: Q4_K_M (4.88 bpw) decodes at 85.7 t/s while the lower-bit IQ4_XS (4.32 bpw) runs at 52.8 — Q4_K / Q5_K carry the optimized, reorder-capable kernels. The lever for more speed is a better kernel (reorder-on-MoE), which is what Turbo delivers.

## Project control data
The current package-level control comparison is the Q5_K_M release-gate record in
`results/nx2-kernel-release-gate/20260612-existing-artifacts/`, measured on the
NX2 GGUF artifact and Arc Pro B70 with the retained patch chain (`0001`-`0004`,
including the Q6_K MoE reorder and NX2 fused gate/up, weighted-sum, and tail-add
path):

| config | ctx0 decode |
|---|---:|
| fresh stock control: reorder off / FA off | 68.8 t/s |
| deployed: Turbo + FA + NX2 fused MoE | 88.1 t/s |

This is the reproducible project-level before/after: +28% at ctx0. The
intermediate Q6_K reorder control (+24%) is retained in
`results/nx2-controls/20260612T052635Z/`; the prior 0001-only control (+17%) is
retained in `results/nx2-controls/20260610T220943Z/`.

NX2 path-debug evidence was retained in
`results/nx2-path-debug/20260610T223353Z/`. Q4_K_M and Q5_K_M both reach SYCL
`ggml_sycl_mul_mat_id` with 3D expert tensors:

```text
Q4_K_M: blk.0.ffn_gate_exps.weight type=q4_K ne=[2048, 512, 256, 1]
Q5_K_M: blk.0.ffn_gate_exps.weight type=q5_K ne=[2048, 512, 256, 1]
```

Model checksums are retained in `results/model-checksums.sha256`.

First-token timing is retained in `results/nx2-first-token/20260610T230504Z/`.
The cleaned-branch Q5_K_M control did not show a first-token lazy-reorder
penalty under `-fa on`, `-fit off`, `--no-warmup`; median prompt/first-token
eval was 1.12 s with reorder off and 1.05 s with reorder on. This is a whole
optimization-path control, not isolated single-line attribution.

## Reorder correctness checks
Greedy (`--temp 0`) decode is not primary correctness evidence for the SYCL reorder path. The retained FA-on greedy artifacts in `results/upstream-pr/` diverge across reorder variants, which is expected for small numerical differences amplified by argmax. Correctness evidence here relies on `test-backend-ops` and perplexity.

## Reorder-on-MoE: correctness and scope
Turbo extends the dense SYCL reorder to the fused MoE expert GEMV (`mul_mat_id`) for Q4_K, Q5_K, and Q6_K. Correctness holds under the retained validation artifacts: `test-backend-ops` MUL_MAT_ID passes vs CPU reference, the earlier unpatched upstream and validated branch full-suite backend-op logs had the same 12 `GET_ROWS` failures, and PPL is statistically unchanged on the retained NX2 checks. The earlier "byte-identical" / token-identical wording exceeded the retained evidence.

The project-level throughput numbers in this repo are measured on the NX2 GGUF variants, not on an abstract upstream model. The backend contribution audit in `results/upstream-pr/` is validation detail for one reusable SYCL component, separate from the NX2 model artifact benchmark.

## Gate+up fusion and NX2 specialization
`patches/0004` contains two generations of MoE decode work. The first measured
the generic `MUL_MAT_ID, MUL_MAT_ID, GLU` gate/up SwiGLU fusion and shared-Q8
experiments; those remain opt-in profiling modes because the generic path did
not beat the Q5_K_M ctx0 scoreboard.

The retained default is mode `3`: exact NX2 Q4_K/Q5_K gate/up SwiGLU fusion
(`ncols=2048`, `nrows=512`, top-k 8), a post-down weighted-sum fusion for the
eight selected experts, and a fused F32 MoE tail add. The gate/up kernel
quantizes the shared activation row once, computes gate and up dots in one
specialized kernel, applies SwiGLU in-kernel, and writes the GLU destination
directly. The weighted-sum kernel replaces the following F32 multiply plus seven
add kernels with one direct write to the MoE output. The tail-add kernel fuses
`ffn_moe_out + ffn_moe_shexp + ffn_inp` into one small F32 kernel. In
`results/fabler-gate-up-fusion/20260612T083707Z-nx2-weighted-sum-fusion/`,
Q5_K_M ctx0 improves from 85.04 to 87.32 t/s (+2.69%). In
`results/fabler-gate-up-fusion/20260612T091359Z-nx2-tail-add-fusion/`, the
tail-add step lifts Q5_K_M ctx0 further to 87.96 t/s. The latest retained
scalar/default rerun in
`results/fabler-gate-up-fusion/20260612T092946Z-nx2-weighted-sum-vec4/` measures
88.08 t/s. Guardrails: Q4_K_M ctx0 is 93.49 t/s, Q5_K_M 131k is 42.29 t/s, and
Q5_K_M 30-chunk WikiText PPL is flat (5.5642 +/- 0.15223).

The follow-up shared-expert dense gate/up fusion probe is kept opt-in only:
`results/fabler-gate-up-fusion/20260612T090325Z-nx2-shared-gate-up-fusion/`
measured Q5_K_M ctx0 at 87.27 t/s with the path off versus 86.90 t/s with the
best fused reducer on.

The activation-Q8 cache probe (`GGML_SYCL_ENABLE_MOE_ACT_Q8_CACHE=1`) stores
the MoE gate/up producer's reordered Q8_1 activation row in the backend context
and lets the later shared-expert dense gate/up fusion consume it when the source
activation tensor and Q8 layout match. This tested whether reusing the first
MoE activation quantization could make
`GGML_SYCL_ENABLE_MOE_SHARED_GATE_UP_FUSION=1` viable. It passed targeted
`MUL_MAT_ID` testing, but lost to local requantization inside the shared path:
87.76 t/s cache-on versus 88.14 t/s cache-off, then 87.43 t/s cache-on versus
87.85 t/s cache-off in a reversed repeat. Raw results:
`results/fabler-gate-up-fusion/20260612T114939Z-nx2-activation-q8-cache/`.
It remains default-off.

The vec4 weighted-sum follow-up is also retained only as an opt-in profiling
mode (`GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=2`): it measured 87.91 t/s
versus 88.08 t/s for the scalar/default weighted-sum path, with the 30-chunk PPL
spot check still flat at 5.5685 +/- 0.15255.

Mode `3` of the weighted-sum fusion caches the eight normalized top-k weights
in workgroup-local memory before summing rows. It passed the targeted
`MUL_MAT_ID` smoke test, but the barrier/local-memory overhead lost to the
retained scalar kernel: 87.60 t/s versus 88.06 t/s in
`results/fabler-gate-up-fusion/20260612T103409Z-nx2-weighted-sum-local-weights/`.
It remains opt-in only.

Two follow-ups pushed the post-down/tail side harder but are not promoted. The
combined weighted-tail kernel (`GGML_SYCL_ENABLE_MOE_WEIGHTED_TAIL_FUSION=1`)
writes the final residual output directly, but measured 87.74 t/s versus
87.98 t/s with the path off. The tail-add vec4 mode
(`GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=2`) measured 87.79 t/s in one pair and
87.65 t/s in a reversed repeat, with enough noise that scalar mode `1` remains
the retained default.

The tail-add+RMS_NORM probe
(`GGML_SYCL_ENABLE_MOE_TAIL_RMS_FUSION=1`) extends the final MoE tail match from
`ADD, ADD` to `ADD, ADD, RMS_NORM`, writing both the residual vector and the
RMS_NORM output from one kernel. It passed targeted `MUL_MAT_ID` testing with
the flag enabled, but the primary Q5_K_M ctx0 path regressed to 86.37 t/s
versus 87.95 t/s and 87.57 t/s off. Q4_K_M improved in the single guard pair
(92.77 t/s on versus 91.78 t/s off), but Q5_K_M is the promotion gate. Raw
results:
`results/fabler-gate-up-fusion/20260612T120710Z-nx2-tail-rms-fusion/`.
It remains default-off.

The standalone RMS_NORM+MUL probe
(`GGML_SYCL_ENABLE_RMS_NORM_MUL_FUSION`) adds the SYCL version of the
CUDA/OpenCL/WebGPU/Vulkan norm-weight fusion for the exact NX2 decode vector:
F32 contiguous `[2048]` RMS_NORM input, F32 contiguous `[2048]` norm weight,
and F32 output. Mode `1` uses 512 work-items; mode `2` uses 1024 work-items to
match the existing large-row RMS_NORM launch shape. Mode `2` was faster:
Q5_K_M ctx0 measured 89.15 and 88.97 t/s versus 88.05 and 87.84 t/s off, while
Q4_K_M measured 95.77 t/s versus 91.28 t/s off in the guard pair. Correctness
smoke checks passed (`RMS_NORM` 21/21, `MUL_MAT_ID` 690/690), and a 5-chunk
Q5_K_M WikiText spot was flat (5.1289 off versus 5.1329 on). Because the
primary Q5_K_M gain is still below the +2% promotion bar, the probe remains
default-off. Raw results:
`results/fabler-gate-up-fusion/20260612T122324Z-nx2-rms-norm-mul-fusion/`.

Mode `3` of `GGML_SYCL_ENABLE_RMS_NORM_MUL_FUSION` pushes the same idea further:
the RMS_NORM+MUL kernel also emits a reordered Q8_1 SoA side buffer keyed by
the F32 `attn_post_norm` tensor, and the later exact-NX2 MoE gate/up fusion
recognizes the reshape of that tensor and skips its normal activation
quantization launch. The first in-kernel Q8 implementation serialized each
Q8_1 block inside the workgroup and measured 88.04 t/s on Q5_K_M. Replacing it
with a subgroup-parallel quantizer, one 16-lane subgroup per Q8_1 block and two
values per lane, improved Q5_K_M to 88.77 t/s and Q4_K_M to 95.80 t/s versus
92.37 t/s off in the local guard pair. It passed `RMS_NORM` 21/21 and
`MUL_MAT_ID` 690/690 smoke tests, with a 5-chunk Q5_K_M WikiText spot at
5.1402. It proves the post-norm Q8 handoff architecture works, but still trails
standalone mode `2` on Q5_K_M and remains below the +2% promotion bar. Raw
results:
`results/fabler-gate-up-fusion/20260612T123830Z-nx2-rms-mul-q8-handoff/`.

Mode `4` keeps the same Q8 handoff default-off and makes it graph-selective:
the RMS_NORM+MUL fusion writes Q8 only when lookahead finds the later exact-NX2
MoE gate/up fusion consuming that MUL output. It passed `RMS_NORM` 21/21 and
`MUL_MAT_ID` 690/690, but Q5_K_M ctx0 measured 87.58 and 87.48 t/s versus
89.01 t/s for a same-run mode `2` repeat; Q4_K_M mode `4` measured 94.32 t/s.
Raw results:
`results/fabler-gate-up-fusion/20260612T125005Z-nx2-rms-mul-selective-q8-handoff/`.

The gate/up reducer follow-up (`GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=2`) swaps
the retained `sycl::reduce_over_group` reduction for a manual XOR subgroup
reduction in the exact NX2 fused gate/up kernel. It passed targeted
`MUL_MAT_ID` testing but measured 87.81 t/s versus 87.96 t/s for the retained
reducer, so mode `1` remains default.

Two additional exact-NX2 gate/up activation modes are present as profiling
variants under `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE`. Mode `3` changes SwiGLU
from native `exp` plus division to `exp2` plus native reciprocal; mode `4`
keeps native `exp` but uses native reciprocal. Both passed targeted
`MUL_MAT_ID` testing (690/690), but neither beat retained mode `1`: mode `3`
measured 87.74 t/s versus 88.06 t/s, and mode `4` measured 87.76 t/s versus a
noisy 87.88 t/s repeat. Raw results:
`results/fabler-gate-up-fusion/20260612T102545Z-nx2-swiglu-exp2-recip/`.

Mode `5` of `GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE` changes the exact-NX2 gate/up
kernel schedule from one output row per workgroup to four rows per workgroup
(one subgroup per row). It passed targeted `MUL_MAT_ID` testing, but measured
87.70 t/s versus 88.10 t/s for retained mode `1` in
`results/fabler-gate-up-fusion/20260612T103921Z-nx2-gate-up-rowpack4/`.
It remains opt-in only.
Modes `6` and `7` test the same scheduler family at two and eight output rows
per workgroup. Both passed targeted `MUL_MAT_ID` testing. The first Q5_K_M pass
was noisy, with mode `1` at 85.97 t/s, mode `6` at 87.86 t/s, and mode `7` at
87.64 t/s. The reversed repeat put retained mode `1` back ahead of mode `6`
(87.72 versus 87.46 t/s), and Q4_K_M also favored mode `1` (93.80 versus
93.53 t/s). Raw results:
`results/fabler-gate-up-fusion/20260612T113923Z-nx2-gate-up-rowpack2-8/`.
Modes `5`, `6`, and `7` therefore remain opt-in only.
Mode `8` tries to make row grouping reuse the shared activation vector: a
four-row workgroup copies the 2048-wide reordered Q8_1 activation row into
workgroup-local memory before the gate/up dot products. It passed targeted
`MUL_MAT_ID` testing, but Q5_K_M only tied retained mode across the reversed
repeat (88.20 versus 88.17 t/s, then 87.67 versus 87.67 t/s), and Q4_K_M
regressed to 90.08 t/s versus 93.91 t/s retained. Raw results:
`results/fabler-gate-up-fusion/20260612T130548Z-nx2-gate-up-local-q8/`.
It remains opt-in only.
Modes `9` and `10` test an expert-pack scheduler for the same exact-NX2 gate/up
kernel family: one selected expert per workgroup, with eight subgroups covering
the top-k experts for one output row. Mode `10` additionally uses the mode `8`
local-Q8 activation cache. Both modes passed targeted `MUL_MAT_ID` testing, but
mode `9` stayed below the Q5_K_M +2% promotion gate across a reversed repeat
(88.09 versus 86.83 t/s, then 87.13 versus 87.00 t/s), while mode `10` measured
87.39 t/s. Q4_K_M mode `9` measured 93.87 t/s versus 92.75 t/s retained in the
single guard pair. Raw results:
`results/fabler-gate-up-fusion/20260612T131422Z-nx2-gate-up-expert-pack/`.
They remain opt-in only.
Mode `11` keeps the retained one-row scheduler but fuses the reordered vecdot
helper itself for Q4_K/Q5_K gate/up: the shared Q8_1 activation lane data is
loaded once and the up/gate dot products are accumulated in the same inner
loop. It passed targeted `MUL_MAT_ID` testing, but Q5_K_M only tied retained
mode in the reversed repeat (87.59 versus 87.07 t/s, then 87.30 versus
87.29 t/s), and Q4_K_M regressed to 92.76 t/s versus 94.15 t/s retained. Raw
results:
`results/fabler-gate-up-fusion/20260612T133008Z-nx2-gate-up-dual-dot/`.
It remains opt-in only.

The gate/up Q8 handoff probe (`GGML_SYCL_ENABLE_MOE_GATE_UP_Q8_HANDOFF=1`)
adds a backend-context side buffer keyed by the GLU tensor. The exact NX2
gate/up producer writes both the F32 SwiGLU output and a reordered Q8_1 view of
the `[512,8]` GLU activation for the following Q6_K down projection, allowing
that down `MUL_MAT_ID` to skip its normal quantization launch. It passed the
targeted `MUL_MAT_ID` smoke test and ran the model benchmark, but the heavier
32-row producer workgroup did not beat the retained path across repeats:
88.02 t/s on versus 87.93 t/s off in the first pair, then 87.34 t/s on versus
87.56 t/s off in the reversed pair. Raw results:
`results/fabler-gate-up-fusion/20260612T104952Z-nx2-gate-up-q8-handoff/`.
It remains default-off.

The exact Q6_K down GEMV retest
(`GGML_SYCL_ENABLE_MOE_Q6_DOWN_NX2_SPECIALIZE=1`) specializes the reordered
Q6_K MoE down projection for the NX2 shape (`ncols=512`, `nrows=2048`, top-k 8)
while preserving one output/expert subgroup per row. It passed targeted
`MUL_MAT_ID` testing but did not beat the generic reordered Q6_K path in the
current fused stack: 87.62 t/s versus 87.84 t/s with the specialization off in
`results/fabler-gate-up-fusion/20260612T110216Z-nx2-q6-down-exact-current/`.
It remains default-off.

The combined Q8-handoff plus exact-Q6-down interaction was also measured:
`GGML_SYCL_ENABLE_MOE_GATE_UP_Q8_HANDOFF=1` together with
`GGML_SYCL_ENABLE_MOE_Q6_DOWN_NX2_SPECIALIZE=1` produced 87.51 t/s versus
87.68 t/s for the retained default in
`results/fabler-gate-up-fusion/20260612T110640Z-nx2-q8-handoff-q6-down-exact-combo/`.
The two structural probes therefore remain independently opt-in and default-off.

The shared-gate-tail probe (`GGML_SYCL_ENABLE_MOE_SHARED_GATE_TAIL_FUSION=1`)
matches the Qwen3.5 MoE scalar shared-expert gate multiply followed by the two
F32 tail adds, and writes `moe_out + shexp * gate + residual` directly. It
passed targeted `MUL_MAT_ID` smoke testing, but remains opt-in because Q5_K_M
ctx0 improved only below the +2% promotion gate across noisy repeats: 88.16 t/s
on versus 87.81 t/s off, then 87.83 t/s on versus 87.67 t/s off. Q4_K_M ctx0
measured 94.22 t/s on versus 92.64 t/s off. Raw results:
`results/fabler-gate-up-fusion/20260612T111328Z-nx2-shared-gate-tail-fusion/`.
Mode `2` vectorizes the same kernel over four rows per work-item. It also
passed targeted testing and was the better shared-gate-tail variant, but still
missed the Q5_K_M +2% promotion bar: 87.88 t/s versus 87.07 t/s off, then
87.80 t/s versus 87.22 t/s off in a reversed repeat. Q4_K_M measured 94.43 t/s
versus 92.00 t/s off. Raw results:
`results/fabler-gate-up-fusion/20260612T112008Z-nx2-shared-gate-tail-vec4/`.
Modes `5` and `6` keep the match at the post-sigmoid multiply but broadcast the
scalar shared-expert gate through workgroup-local memory for scalar and vec4
tails. They passed targeted `MUL_MAT_ID` testing, but did not beat the retained
default on Q5_K_M ctx0: 87.65 and 87.55 t/s versus 87.81 t/s for the same-run
default. Raw results:
`results/fabler-gate-up-fusion/20260612T125854Z-nx2-shared-gate-tail-local-gate/`.
Modes `3` and `4` move the match start one node earlier and also consume the
scalar `SIGMOID` before the shared-expert gate multiply. Mode `4` is the vec4
variant and broadcasts the sigmoid through workgroup-local memory. It passed
targeted testing but was order-sensitive on Q5_K_M: 88.15 t/s versus 87.18 t/s
off in the first post-local-broadcast pair, then 87.47 t/s versus 87.82 t/s off
in the reversed repeat. Q4_K_M measured 94.52 t/s versus 92.75 t/s off. Raw
results:
`results/fabler-gate-up-fusion/20260612T112753Z-nx2-shared-gate-sigmoid-tail/`.
Modes `3` and `4` therefore remain opt-in only.

The down-projection fusion follow-up
(`GGML_SYCL_ENABLE_MOE_DOWN_WEIGHTED_SUM_FUSION=1`) computes Q6_K down dots for
all eight selected experts inside one output-row kernel, applies top-k weights
in-kernel, and writes `ffn_moe_out` directly. It avoids the `[2048,8]` down
materialization and the separate weighted-sum launch, but the reduced
parallelism lost on the scoreboard: 87.85 t/s with the path on versus 87.98 t/s
with it off. It remains default-off.

The atomic down-projection variant
(`GGML_SYCL_ENABLE_MOE_DOWN_WEIGHTED_SUM_FUSION=2`) keeps per-expert row
parallelism and atomically accumulates weighted Q6_K down contributions into
the reduced `[2048]` output. It passed targeted `MUL_MAT_ID` backend-op testing
(690/690), but the extra zero-fill and atomics lost to the retained path:
87.68 t/s versus 88.17 t/s with the path off in
`results/fabler-gate-up-fusion/20260612T101912Z-nx2-down-weighted-atomic/`.
It remains default-off.

Mode `3` of the down-projection fusion keeps expert parallelism without global
atomics: one workgroup owns each output row, eight subgroups compute the eight
selected expert dot products, subgroup leaders write weighted sums to local
memory, and one subgroup reduces the eight values into `ffn_moe_out`. It passed
targeted `MUL_MAT_ID` testing but still lost to the retained materialize-plus-
weighted-sum path: 87.68 t/s versus 87.99 t/s off in
`results/fabler-gate-up-fusion/20260612T115730Z-nx2-down-weighted-local-reduce/`.
It remains default-off.

The graph dispatch guard probe (`GGML_SYCL_ENABLE_MOE_DISPATCH_GUARD=1`) limits
MoE fusion matcher calls to plausible starting ops (`MUL_MAT_ID`, `MUL_MAT`,
`MUL`, and `ADD`). It did not improve Q5_K_M ctx0 timing: 87.78 t/s with the
guard on versus 88.09 t/s with it off in
`results/fabler-gate-up-fusion/20260612T101554Z-nx2-dispatch-guard/`. The guard
therefore remains default-off.

## Accuracy reference
The KLD / PPL reference is **Q6_K** (27 GB, fits VRAM in full at `-ngl 99`), which is near-lossless. All tested quants (Q5_K and below) are lower precision, so the comparison is valid.

## Notes
- Qwen3.5 **Gated Delta Net** (chunked linear attention) runs on the **CPU** in this backend; the MoE experts stay on the GPU — so decode benefits from CPU headroom.
- Convert with **transformers ≥ 5.x** (NexN2's tokenizer is `TokenizersBackend`).
- `GGML_SYCL_F16=ON` ≈ 2.4× prefill.
- imatrix: full GPU on Q6_K, 129 chunks, Bartowski `calibration_datav3` (covers all 256 experts).
