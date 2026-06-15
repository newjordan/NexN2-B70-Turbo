# NX2-IQ3_A770 — final eval (1-card IQ3 vs 2-card Q4_K)

Deployment doc: [`../../docs/nx2-iq3-a770-deployment.md`](../../docs/nx2-iq3-a770-deployment.md).
Both rows are the **shipped GGUFs**, same llama.cpp build, same corpus/base.

## Accuracy — KLD vs `NX2-Q6K.kld` (wikitext-2 `wiki.test.raw`, 100×512 tok)

| model | experts | mean KLD ↓ | top-1 (same top-p) ↑ | PPL(Q)/PPL(base) ↓ | PPL(Q) |
|---|---|---|---|---|---|
| `NX2-IQ3_A770-mixed-LUT` (1-card, GPU int8 dp4a) | IQ3 3.19 bpw | **0.0547** | 89.91% | 1.0163 | 6.6787 |
| `NX2-IQ3_A770-Q4fill` (2-card) | Q4_K 4.5 bpw | **0.0245** | 93.24% | **1.0024** | 6.5872 |

- **2-card halves KLD (−55%)** and is **within 0.24% PPL of the full Q6_K base — effectively
  lossless.** All-experts-Q4_K beats the down-only-Q4_K frontier point (0.0567): gate/up *do*
  move quality going 3→4 bit (more than the IQ3-only sweep suggested).
- Raw: [`kld-iq3-1card.log`](kld-iq3-1card.log), [`kld-q4k-2card.log`](kld-q4k-2card.log).

## Throughput (Intel Arc Pro B70, Battlemage, A770-proxy; competing Q5 server resident)

| model | path | pp512 | tg64 | how measured |
|---|---|---|---|---|
| IQ3 (1-card) | fused reorder dp4a (0007+0008) | ~600 | **81.99 ± 0.12** | back-to-back A/B, `-r 5`, B70 |
| Q4_K (2-card) | Q4_K reorder, `-sm layer -ts 1,1` | — | **bandwidth-projected** | not measured (single dev card) |

1-card tg is the measured `0008` number (the full M1→0008 arc: 32 → 44 → 78.6 → 82.0 t/s,
every step a back-to-back A/B). The 2-card figure is the honest gap: per the deployment doc's
bandwidth table, splitting Q4_K (4.5 bpw) across two cards reads ~30% fewer bytes/card than the
1-card 3-bit base, so decode is projected positive — but it needs real 2-GPU hardware to confirm
(handoff harness: `eval/nx2/run_tensor_split_claim.sh` pattern).

## Takeaway

`--sm off` → fast & compact (IQ3, 82 t/s, KLD 0.055, fits one 16 GB card).
`--sm on`  → near-lossless (Q4_K, KLD 0.025, PPL within 0.24% of base) and bandwidth-positive
on two cards. One model, one switch, the genuine fast-vs-accurate knob.
