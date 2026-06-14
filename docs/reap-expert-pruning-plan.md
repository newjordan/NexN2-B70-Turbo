# REAP Router-Driven Expert Pruning — Plan (Stage 3)

## Context

Patch `0005` proved the *bits* work: `IQ3_A770` quantizes all 256 Nex-N2-mini
experts uniformly to 3.19 bpw → 14.92 GB, KLD 0.0754 (beats stock Q3). But
uniform quantization is not the breakout. **REAP** (Router-weighted Expert
Activation Pruning, Cerebras, ICLR 2026 — arXiv 2510.13999, `CerebrasResearch/reap`)
is: score each expert by router-weighted activation, *remove* the low-value ones
(pruning beats merging — merging causes "functional subspace collapse"), and let
the router drive a smaller, sharper expert set. REAP has published checkpoints on
exactly this architecture class (**Kimi-Linear-REAP-35B-A3B**, a hybrid
linear-attention MoE like Nex-N2-mini). Pruning frees budget; we then spend it on
the Pareto sweet spot of {fewer experts × more bits}.

**Chosen approach (signed off):** prune-then-residency; sweep
{25%,50%} × {IQ3_A770, Q4_K, Q5_K} for the (size, KLD) Pareto; build a **true REAP
saliency collector first** (`Sⱼ = mean over routed tokens of gⱼ·‖fⱼ(x)‖₂`).

Saliency: `Sⱼ = (1/|Xⱼ|) Σ_{x∈Xⱼ} gⱼ(x)·‖fⱼ(x)‖₂` — `gⱼ` = router gate weight,
`fⱼ(x)` = expert j's (pre-gate) down-projection output. Prune lowest-`S` experts
**per layer** (keeps every layer ≥ `n_expert_used=8` and uniform tensor shapes).

## Verified design facts (from code, qwen35moe / Nex-N2-mini)

- MoE nodes (`src/llama-graph.cpp`): `ffn_moe_topk` (ids, :1549), `ffn_moe_down`
  = **pre-gate** expert output `fⱼ(x)` `[n_embd,8,n_tok]` (:1741), `ffn_moe_weighted`
  = `gⱼ·fⱼ` (:1759). Node names carry layer as **suffix** `-{il}` (`llama-context.cpp:2357`).
- NX2-bf16 uses the **separate** gate/up path (no `ffn_gate_up_exps`); **no router
  bias** (`exp_probs_b=nullptr`); softmax gating, normalized weights.
- Instrumentation: `params.cb_eval` (same hook as `tools/imatrix`); marking a node
  "needed" forces materialization and suppresses fusion across that cut — so taps
  are readable even on the fused SYCL path.
- Pruning = **offline GGUF surgery** (gguf-py): expert dim is `ne[-1]` = **numpy
  axis 0** after the reader's axis reversal; slice `ffn_{gate,up,down}_exps` and
  router `ffn_gate_inp` to the kept ids (ascending → implicit 0..N−1 remap, no
  runtime table); leave `*_shexp` untouched; set `qwen35moe.expert_count=N`,
  `expert_used_count` stays 8. Graph adapts via `hparams.n_expert`
  (`LLAMA_MAX_EXPERTS=512`). Prune on **bf16** then quantize (no block slicing).
- **imatrix must be regenerated per pruned model** (per-expert rows are keyed by
  index; pruning remaps indices).
- **MTP block** `blk.40.*` shares `expert_count` — prune it with the same policy
  (or mirror an adjacent layer's keep-set if the collector never sees it).

## Phases

**Step 0 — DONE:** `0005` committed + pushed.

1. **Collector (C++):** new `tools/reap/reap-collect.cpp` (clone `tools/imatrix`
   scaffold). `cb_eval` taps two nodes/layer: at `ffn_moe_down-{il}` cache the
   expert ids (`t->src[2]`, like imatrix `:264-281`); at `ffn_moe_weighted-{il}`
   compute per used-slot the L2 norm over `n_embd` (= `gⱼ·‖fⱼ‖₂` since `gⱼ≥0`),
   attribute to the cached id. Accumulate per-(layer,expert) `s_sum` and `count`
   (count = router-frequency histogram, reused in Phase 5). Save `NX2.reap.gguf`.
   Run on **NX2-Q6_K** (`-ngl 99`, fast; router is F32 in every quant so selection
   is identical) over `wiki.test.raw`, `--override-kv qwen35moe.block_count=40`.
2. **Scorer (Python):** `scripts/reap-score.py` → `Sᵐᵉᵃⁿ = s_sum/count` →
   per-layer top-(1−r) keep sets for r∈{0.25,0.50} → `NX2.reap.keep.r{25,50}.json`.
3. **Surgery (Python, gguf-py):** `scripts/reap-prune-gguf.py` → slice bf16 →
   `NX2-bf16.reap-r{25,50}.gguf`; validate load (`expert_count`, `llama-cli -n 1`).
4. **Imatrix:** regenerate `NX2.reap-r{25,50}.imatrix` on each pruned model.
5. **Sweep (bash):** `scripts/reap-sweep.sh` (clone `scripts/nx2-phase1-sweep.sh`,
   parameterize source+imatrix). 7 cells: `full×IQ3` baseline + `{r25,r50}×{IQ3,Q4_K,Q5_K}`.
   IQ3 via `--tensor-type ffn_*_exps=iq3_a770` on a q6_k base. Metrics: KLD vs
   `NX2-Q6K.kld`, PPL, top-1, size, VRAM, pp/tg → **(size, KLD) Pareto** + 16 GB flag.
6. **Phase 5 — dynamic residency** (later): reuse collector `count` for hot/cold
   sets; tiers — (A) precision tiering, (B) `-ot` CPU offload of cold experts,
   (C) swap-on-miss; gates R0–R3 (KLD within budget, fit 16 GB, tg regression).

## Gates

Per-cell: KLD/top-1 vs `NX2-Q6K.kld`, fits 16 GB at `-ngl 99`, beats stock Q3.
**Headline question:** does `r50×Q5_K` (fewer experts, more bits) beat
`full×IQ3_A770` (all experts, fewer bits) at equal size on the (size, KLD) Pareto?

## Risks

imatrix index misalignment (→ regenerate); GGUF reader/writer axis reversal (slice
numpy axis 0, pass `raw_shape` with `ne[-1]=N`, assert byte counts); MTP block
expert_count coupling; disk (pruned bf16 ~35–52 GB each — quantize then delete
intermediates; 293 GB free). Collect-on-Q6 is ranking-robust (router is F32).

## Reuse

`tools/imatrix/imatrix.cpp` (cb_eval + MUL_MAT_ID id-reading), `gguf-py`
reader/writer, `scripts/nx2-phase1-sweep.sh` (quantize/KLD/PPL/bench),
`NX2.imatrix`, `NX2-Q6K.kld`, `nx2-eval/wikitext-2-raw/wiki.test.raw`,
`results/frontier.csv` (Q5 0.020 / Q4 0.039 / Q3_K_M 0.105 baselines).
