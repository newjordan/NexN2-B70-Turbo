# Turbo Understanding Notes

Internal study notes for the Turbo SYCL reorder-on-MoE work. This is not PR
text.

## Short Version

"Turbo" has two related deliverables that should be described separately:

1. The project-level NX2 work: GGUF conversion, imatrix quantization, metadata
   fixes, serving profile, validation, and B70 measurements for the
   Nex-N2-mini-derived `NX2-Q4_K_M.gguf` and `NX2-Q5_K_M.gguf` variants.
2. A reusable llama.cpp SYCL backend contribution branch for quantized MoE
   expert matmuls. This code extends an existing SYCL optimization called
   weight reorder so it can also be used by the fused MoE `MUL_MAT_ID` fast
   path for Q4_K, Q5_K, and Q6_K expert weights.

The project-level benchmark is the important headline: the retained NX2 GGUF
variants reach 88.1 tok/s for Q5_K_M and 93.5 tok/s for Q4_K_M on the Intel Arc
Pro B70. The backend contribution audit is only for deciding what can be claimed
about the reusable llama.cpp code contribution.

The backend contribution branch changes four llama.cpp files:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`
- `ggml/src/ggml-sycl/mmvq.cpp`
- `ggml/src/ggml-sycl/mmvq.hpp`
- `ggml/src/ggml-sycl/dmmv.cpp`

The ready branch is `sycl-moe-reorder-ready` in
`/home/frosty40/llama.cpp-sycl-moe-ready`. The cleanup history remains in
`/home/frosty40/llama.cpp-sycl-moe-clean`.

## Baseline Concept

In normal quantized matmul, each quantized block stores its fields together:

```text
block 0: qs, scales, dm
block 1: qs, scales, dm
block 2: qs, scales, dm
...
```

SYCL already had a faster reordered layout for dense matmul. It groups each
field across all blocks:

```text
all qs | all qh if Q5_K | all scales | all dm
```

That is a structure-of-arrays layout. The dense reordered MMVQ code already knew
how to read this layout through:

- `ggml_sycl_reordered::block_q_t<T>` in `quants.hpp`
- `reorder_vec_dot_q_sycl<T>` in `vecdotq.hpp`
- `quantize_and_reorder_q8_1_soa` in `quantize.hpp`

Before Turbo, the fused MoE `MUL_MAT_ID` path explicitly rejected reordered
weights and fell back.

## MoE Shape

For MoE expert weights, `src0` is a 3D tensor:

```text
src0: [ne0, ne1, n_expert]
```

Each expert is one slice. `src0->nb[2]` is the byte stride from expert N to
expert N+1. For a routed token, `ids` selects which expert slices to multiply.

The fused single-token path is:

```text
ggml_sycl_mul_mat_id()
  if ne12 == 1:
    try ggml_sycl_mul_mat_id_mmvq_fused()
  else:
    use slower grouped/fallback path
```

Turbo changes only this fused path and related reorder coverage.

## What Was Added

### 1. Per-expert reorder converters

File: `ggml-sycl.cpp`

New functions:

- `reorder_qw_q4_k_moe`
- `reorder_qw_q5_k_moe`
- `reorder_qw_q6_k_moe`

These copy the whole expert tensor to a temporary buffer, then scatter every
block of every expert into a per-expert reordered layout:

```text
expert 0: all qs | all qh | all scales | all dm
expert 1: all qs | all qh | all scales | all dm
expert 2: all qs | all qh | all scales | all dm
...
```

The key invariant is that each expert remains self-contained. That matters
because the MoE kernel jumps to an expert with:

```cpp
vx = vx_base + expert_id * expert_weight_stride;
```

where `expert_weight_stride == src0->nb[2]`.

### 2. `reorder_qw()` learns about 3D expert tensors

File: `ggml-sycl.cpp`

Before Turbo, `reorder_qw()` handled dense 2D weights. Turbo adds:

```cpp
if (src0->ne[2] > 1) {
    switch (src0->type) {
        case GGML_TYPE_Q4_K: ...
        case GGML_TYPE_Q5_K: ...
        case GGML_TYPE_Q6_K: ...
    }
}
```

Unsupported 3D reorder cases now return `false`, leaving those tensors on the
existing non-reordered read path instead of aborting.

### 3. Lazy reorder for `MUL_MAT_ID`

File: `ggml-sycl.cpp`

New function:

- `opt_for_reorder_id`

This is the MoE counterpart to the existing dense `opt_for_reorder`. It checks:

- optimization is enabled
- device allows reorder
- `src0` is Q4_K, Q5_K, or Q6_K
- tensor has not already been reordered

Then it calls `reorder_qw()` and marks:

```cpp
extra->optimized_feature.reorder = true;
```

### 4. Fused MoE fast path no longer rejects reordered weights

File: `ggml-sycl.cpp`

Before Turbo, the fused path had:

```cpp
if (src0_extra && src0_extra->optimized_feature.reorder) return false;
```

That meant: if expert weights were reordered, the fused MoE path could not run.

Turbo removes that rejection. It lazily reorders eligible expert weights, then
chooses between:

```text
standard weights -> quantize_q8_1 -> ggml_sycl_mul_mat_vec_q_id
reordered weights -> quantize_and_reorder_q8_1_soa -> ggml_sycl_mul_mat_vec_q_id_reorder
```

The activation side (`src1`) must be quantized into the matching Q8_1 layout.
That is why the patch switches to `quantize_and_reorder_q8_1_soa` when
`use_reorder` is true.

### 5. Reordered MoE MMVQ kernel

Files: `mmvq.cpp`, `mmvq.hpp`

New public helper:

- `ggml_sycl_mul_mat_vec_q_id_reorder`

New kernel path:

- `mul_mat_vec_q_moe_reorder`
- `launch_mul_mat_vec_q_moe_reorder`

This takes the existing dense reordered vector-dot machinery and applies MoE
indexing to it.

The dense reordered kernel computes one output row for one weight matrix. The
MoE reordered kernel computes:

```text
for each routed expert slot:
  expert_id = ids[expert_slot]
  vx = expert slice selected by expert_id
  vy = activation row for that expert slot
  dst = output row for that expert slot
```

Then it uses the same reordered block offset logic:

```cpp
bx_offset = block_type::get_block_offset(ibx, nblocks);
d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
partial_sum += reorder_vec_dot_q_sycl<T>()(...);
```

### 6. Q5_K reordered DMMV

File: `dmmv.cpp`

Turbo adds:

- `dequantize_mul_mat_vec_q5_k_reorder`
- `dequantize_mul_mat_vec_q5_K_sycl_reorder`

And changes the Q5_K DMMV dispatch to choose the reordered kernel when
`optimized_feature.reorder` is set.

This is coverage, not just speed. Once a tensor is physically reordered in
memory, every path that may read it must understand the reordered layout or must
avoid using that tensor. Q4_K already had a reordered DMMV path; Q5_K did not.

## Runtime Flow

For an eligible Q4_K/Q5_K/Q6_K MoE single-token decode:

```text
ggml_sycl_mul_mat_id
  ggml_sycl_mul_mat_id_mmvq_fused
    shape/type bail checks
    opt_for_reorder_id
      reorder_qw
        reorder_qw_q4_k_moe, reorder_qw_q5_k_moe, or reorder_qw_q6_k_moe
      mark tensor reordered
    quantize src1
      reordered weights: quantize_and_reorder_q8_1_soa
      normal weights: quantize_q8_1
    run kernel
      reordered weights: ggml_sycl_mul_mat_vec_q_id_reorder
      normal weights: ggml_sycl_mul_mat_vec_q_id
```

## What "I Changed"

If you are explaining this as the patch author, the honest claim is:

1. I extended existing SYCL dense reorder infrastructure to MoE expert
   `MUL_MAT_ID` for Q4_K, Q5_K, and Q6_K.
2. I added per-expert reorder conversion so each expert slice remains a
   self-contained reordered tensor.
3. I added a reordered fused MoE MMVQ kernel by combining existing MoE expert
   indexing with existing dense reordered vector-dot reads.
4. I added Q5_K reordered DMMV coverage so reordered Q5_K tensors are not read
   by an incompatible fallback path.
5. I changed the fused `MUL_MAT_ID` dispatcher to use reordered activation
   quantization and the reordered MoE kernel when the expert weights have been
   reordered.

The retained evidence does not attribute +16% to +18% decode to the patch
alone as an incremental patch result.

## Retained Evidence

Supported by retained artifacts:

- The benchmarked model artifacts are the NX2 variants:
  - `/home/frosty40/models/nex-n2-mini/sweep/NX2-Q4_K_M.gguf`
  - `/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf`
- Project-level throughput is 88.1 tok/s for Q5_K_M and 93.5 tok/s for Q4_K_M
  on Arc Pro B70 in the retained package numbers.
- Current release-gate control data is in
  `results/nx2-kernel-release-gate/20260612-existing-artifacts/` for Q5_K_M:
  - fresh stock control, reorder off / FA off: 68.8 tok/s at ctx0
  - deployed Turbo + FA + NX2 fused MoE: 88.1 tok/s at ctx0
  - this reproducible comparison is +28% at ctx0
- A smaller MoE smoke fixture has been identified for upstream-style repro:
  `Phi-mini-MoE-instruct-Q4_K_M.gguf` (`phimoe 16x3.8B Q4_K - Medium`). It
  loads on the clean SYCL branch and reaches 3D Q4_K expert `MUL_MAT_ID` calls;
  retained outputs are in `results/micro-moe/`.
- NX2 Q4_K_M and Q5_K_M path-debug evidence is retained in
  `results/nx2-path-debug/20260610T223353Z/`; both actual NX2 artifacts reach
  3D expert `MUL_MAT_ID` calls on SYCL.
- Model package checksums are retained in `results/model-checksums.sha256`.
- Ready PR branch is at `6b856575` over the current upstream base.
- `git diff --check` is clean.
- Targeted `test-backend-ops test -o MUL_MAT_ID`: 714/714 passed.
- Full backend-op logs are retained for both unpatched base and validated branch:
  11502/11514 passed in both, with the same 12 `GET_ROWS` failures.
- First-token timing is retained in `results/nx2-first-token/20260610T230504Z/`;
  this Q5_K_M control did not show a first-token lazy-reorder penalty under the
  measured settings.
- Perplexity is unchanged in a 30-chunk Q5_K_M spot check:
  - base: 5.5643 +/- 0.15232
  - patched: 5.5662 +/- 0.15242

Open audit items, kept separate from the project benchmark:

- Use `test-backend-ops` and PPL as correctness evidence for the reorder path;
  retained FA-on greedy text is secondary context.
- Rerun server-flow or kernel-level perf numbers before using them in any
  public-facing material.

## Things You Should Be Able To Explain

1. Why reorder helps: it changes memory layout, not math.
2. Why MoE is different: expert weights are 3D, and each expert slice must be
   independently reordered.
3. Why `src1` quantization changes: reordered weight kernels expect Q8_1 in SoA
   form too.
4. Why Q5_K is more complex than Q4_K: Q5_K has low 4-bit data plus a separate
   high-bit plane (`qh`).
5. Why every fallback matters: once weights are reordered in-place, any reader
   that assumes original `block_q*_K` layout will compute garbage.
6. Why project-level and backend-contribution claims are separate: the NX2 GGUF
   variants are the benchmarked product, while the SYCL patch audit answers only
   what that one code contribution adds on top of a control build.

## Likely Maintainer Questions

- Is returning `false` for unsupported 3D reorder cases acceptable to SYCL
  maintainers?
- Can this be tested on a smaller/reproducible MoE model?
- Is the retained first-token optimization-path timing acceptable to SYCL
  maintainers?
- Does this affect non-MoE Q4_K/Q5_K dense paths?
- Is this worth upstreaming as a focused correctness/coverage change even if the
  main NX2 project result comes from the model artifacts and serving stack?

## Fabler Extension (2026-06-12)

Two follow-up backend changes, exported as `patches/0002` and `patches/0003`
(validated in `results/fabler-q6k-reorder/`):

1. **Q6_K MoE reorder.** The NX2 Q4_K_M/Q5_K_M mixes store `ffn_down_exps` as
   Q6_K (`ne=[512, 2048, 256]`) — roughly a third of the expert bytes — and it
   previously ran the non-reordered fused MMVQ. `reorder_qw_q6_k_moe` scatters
   every expert block into a per-expert `[ql][qh][scales][d]` SoA, and the fused
   dispatch gains a `Q6_K` reorder case. All dense readers for reordered Q6_K
   (mmvq, DMMV, dequant-GEMM) already existed upstream, so coverage is complete.
   ctx0 decode: Q5_K_M 81.1 → 85.8 t/s, Q4_K_M 85.7 → 91.0 t/s; PPL and
   MUL_MAT_ID op tests unchanged.
2. **Concat submit-only + graph-safe MUL_MAT_ID.** `ggml_sycl_op_concat` dropped
   two host waits that the in-order queue makes redundant (17 host syncs/token on
   NexN2). `check_graph_compatibility` now admits `MUL_MAT_ID` nodes that will
   take the fused no-host-sync path — the ids stay on device, and the
   reorder-pending check keeps the one-time lazy reorder out of recordings (the
   first eager pass triggers it).

**Negative result, recorded on purpose:** with `GGML_SYCL_DISABLE_GRAPH=0` the
full NexN2 decode graph records cleanly, but the per-token executable-graph
update throws `Cannot update using a graph with a different topology. Mismatch
found in the number of nodes`, so the backend re-finalizes ~3.7k nodes every
token and decode collapses to 15.5 t/s (vs 81.3 graphs-off). SYCL graphs stay
default-off on this stack; the graph-safety work is kept because it is correct
and free when graphs are off.

## Next Best Learning Exercise

Read these in order:

1. Dense reordered MMVQ: `mul_mat_vec_q_reorder` in `mmvq.cpp`.
2. Reordered block layout: `block_q_t<GGML_TYPE_Q4_K>` and
   `block_q_t<GGML_TYPE_Q5_K>` in `quants.hpp`.
3. Reordered dot readers: `reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>` and
   `<GGML_TYPE_Q5_K>` in `vecdotq.hpp`.
4. Standard MoE MMVQ: `ggml_sycl_mul_mat_vec_q_id` in `mmvq.cpp`.
5. Turbo MoE MMVQ: `ggml_sycl_mul_mat_vec_q_id_reorder` in `mmvq.cpp`.
6. Dispatcher: `ggml_sycl_mul_mat_id_mmvq_fused` in `ggml-sycl.cpp`.
