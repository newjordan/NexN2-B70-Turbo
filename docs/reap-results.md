# REAP Result — Dropped (documented negative result)

REAP (router-weighted expert pruning) was evaluated as a way to fit a
higher-quality Nex-N2-mini on one A770 16 GiB. **Conclusion: dropped — no pruned
variant beats uniform full-256×IQ3_A770 within the 16 GiB budget.**

## Pareto (size GiB vs KLD-to-Q6, wikitext-2, 50-chunk)

| cell | experts | quant | size GiB | mean KLD | top-1 | fits 16 GiB |
|---|---|---|---|---|---|---|
| r50×IQ3 | 128 | IQ3_A770 | 7.88 | 0.165 | 84.7% | ✅ |
| r50×Q4 | 128 | Q4_K | 10.59 | 0.141 | 86.1% | ✅ |
| r25×IQ3 | 192 | IQ3_A770 | 10.89 | 0.102 | 87.2% | ✅ |
| r50×Q5 | 128 | Q5_K | 12.35 | 0.131 | 86.9% | ✅ |
| **full×IQ3 (kept)** | 256 | IQ3_A770 | **13.90** | **0.075** | 88.5% | ✅ |
| r25×Q4 | 192 | Q4_K | 15.15 | 0.074 | 89.5% | ⚠️ edge |
| r25×Q5 | 192 | Q5_K | 17.69 | 0.062 | 90.8% | ❌ over |

## Why it failed

- **full-256×IQ3 (13.9 GiB / 0.075) dominates every fitting REAP cell.** The best
  fitting prune (r25×IQ3) is smaller but +36% KLD; r25×Q4 ties quality but is
  bigger.
- REAP's only quality *improvement*, r25×Q5 (0.062), needs **17.7 GiB — outside
  the single-card budget.**
- Routing concentration (top-128 experts = 75% of routed-token mass) was
  misleading: the tail 25% of experts still carry distributional fidelity, which
  KLD penalizes heavily when removed. Pruned capacity is capacity this model uses.

REAP would pay off on a larger card (~18–20 GiB → r25×Q5) or for an ultra-small
build (r50×IQ3 at 7.9 GiB). Neither is this goal.

## Kept / reusable

`scripts/reap-{score,prune-gguf,prune-imatrix}.py`, `scripts/reap-sweep.sh`,
`patches/reap-collect-tool.patch` (the `tools/reap` saliency collector),
`NX2.reap.gguf` (saliency), keep-maps, `results/reap-pareto.csv`. The pruned bf16
+ sweep GGUFs (~164 GiB) were reclaimed.

## Next

`full-256×IQ3_A770` stays the single-card model. Quality work shifts to improving
the quant itself at ≤16 GiB (the bpw–quality wall: ~3.5 bpw experts ≈ 0.06–0.08
KLD; Q4-level 0.039 needs ~4.5 bpw ≈ over budget). EAGLE (speculative decoding,
throughput) deferred — needs a trained draft head and is hard on a hybrid
linear-attention MoE.
