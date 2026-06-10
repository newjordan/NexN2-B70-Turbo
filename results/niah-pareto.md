# NIAH Pareto sweep — NexN2 on the B70

155 probes total, 155 PASS, 0 FAIL. Multi-needle RULER-style harness (eval/niah/niah_sweep.py), temp 0, fixed seed, substring grading, max_tokens 400.

Timing caveat: external load (load avg 21-29) on 2026-06-10 ~01:50-02:35 skewed `prefill_s`/`decode_ts` for the `yarn2-iq4xs/f16` rows at 32k-196k (verdicts unaffected). Its 260k/520k timings are from the earlier quiet-box smoke and are clean.

## linear2/q8_0

| tokens | pass | probes | depths×samples | prefill_s | decode t/s |
|-------:|-----:|-------:|---------------:|----------:|-----------:|
| 32,688 | 3/3 | 3 | 3×1 | 51 | 29.2 |
| 65,155 | 3/3 | 3 | 3×1 | 124 | 18.0 |
| 130,648 | 3/3 | 3 | 3×1 | 352 | 10.3 |
| 196,192 | 3/3 | 3 | 3×1 | 683 | 7.2 |
| 259,609 | 3/3 | 3 | 3×1 | 1105 | 5.6 |
| 327,736 | 3/3 | 3 | 3×1 | 1668 | 4.5 |
| 393,342 | 3/3 | 3 | 3×1 | 2324 | 3.7 |
| 458,876 | 3/3 | 3 | 3×1 | 3084 | 3.3 |
| 520,093 | 3/3 | 3 | 3×1 | 3981 | 2.8 |

## native-131k/f16

| tokens | pass | probes | depths×samples | prefill_s | decode t/s |
|-------:|-----:|-------:|---------------:|----------:|-----------:|
| 32,688 | 3/3 | 3 | 3×1 | 50 | 65.8 |
| 65,155 | 3/3 | 3 | 3×1 | 124 | 54.8 |
| 126,589 | 3/3 | 3 | 3×1 | 334 | 42.4 |

## native/f16

| tokens | pass | probes | depths×samples | prefill_s | decode t/s |
|-------:|-----:|-------:|---------------:|----------:|-----------:|
| 32,688 | 3/3 | 3 | 3×1 | 51 | 65.7 |
| 65,155 | 3/3 | 3 | 3×1 | 124 | 54.7 |
| 130,648 | 3/3 | 3 | 3×1 | 350 | 41.7 |
| 196,192 | 3/3 | 3 | 3×1 | 681 | 33.5 |
| 257,660 | 3/3 | 3 | 3×1 | 1087 | 28.2 |

## native/q8_0

| tokens | pass | probes | depths×samples | prefill_s | decode t/s |
|-------:|-----:|-------:|---------------:|----------:|-----------:|
| 32,688 | 3/3 | 3 | 3×1 | 50 | 29.2 |
| 65,155 | 3/3 | 3 | 3×1 | 123 | 18.0 |
| 130,648 | 3/3 | 3 | 3×1 | 351 | 10.3 |
| 196,192 | 3/3 | 3 | 3×1 | 683 | 7.1 |
| 257,660 | 3/3 | 3 | 3×1 | 1091 | 5.6 |

## yarn1.5/q8_0

| tokens | pass | probes | depths×samples | prefill_s | decode t/s |
|-------:|-----:|-------:|---------------:|----------:|-----------:|
| 32,688 | 3/3 | 3 | 3×1 | 50 | 29.3 |
| 65,155 | 3/3 | 3 | 3×1 | 124 | 18.1 |
| 130,648 | 3/3 | 3 | 3×1 | 352 | 10.3 |
| 196,192 | 3/3 | 3 | 3×1 | 684 | 7.1 |
| 259,609 | 3/3 | 3 | 3×1 | 1104 | 5.5 |
| 327,736 | 3/3 | 3 | 3×1 | 1668 | 4.4 |
| 389,204 | 3/3 | 3 | 3×1 | 2279 | 3.8 |

## yarn2-iq4xs/f16

| tokens | pass | probes | depths×samples | prefill_s | decode t/s |
|-------:|-----:|-------:|---------------:|----------:|-----------:|
| 32,688 | 3/3 | 3 | 3×1 | 73 | 30.0 |
| 65,155 | 3/3 | 3 | 3×1 | 169 | 27.5 |
| 130,648 | 3/3 | 3 | 3×1 | 389 | 34.8 |
| 196,192 | 3/3 | 3 | 3×1 | 683 | 29.2 |
| 259,609 | 3/3 | 3 | 3×1 | 1106 | 24.6 |
| 327,736 | 3/3 | 3 | 3×1 | 1675 | 22.2 |
| 393,342 | 3/3 | 3 | 3×1 | 2319 | 19.7 |
| 458,876 | 3/3 | 3 | 3×1 | 3079 | 17.8 |
| 519,851 | 7/7 | 7 | 7×1 | 3877 | 16.3 |
| 520,093 | 3/3 | 3 | 3×1 | 3868 | 16.3 |
| 520,104 | 7/7 | 7 | 7×1 | 3881 | 16.3 |

## yarn2/q8_0

| tokens | pass | probes | depths×samples | prefill_s | decode t/s |
|-------:|-----:|-------:|---------------:|----------:|-----------:|
| 32,688 | 3/3 | 3 | 3×1 | 50 | 29.1 |
| 65,155 | 3/3 | 3 | 3×1 | 124 | 18.1 |
| 130,648 | 3/3 | 3 | 3×1 | 351 | 10.3 |
| 196,192 | 3/3 | 3 | 3×1 | 685 | 7.1 |
| 259,609 | 3/3 | 3 | 3×1 | 1112 | 5.4 |
| 327,736 | 3/3 | 3 | 3×1 | 1671 | 4.4 |
| 393,342 | 3/3 | 3 | 3×1 | 2327 | 3.7 |
| 458,876 | 3/3 | 3 | 3×1 | 3085 | 3.3 |
| 520,093 | 3/3 | 3 | 3×1 | 3886 | 2.8 |

## Conclusion (T6, 2026-06-10)

155 probes, 155 PASS, 0 FAIL across 7 server configs and lengths 32k-520k.
No retrieval cliff was found anywhere on the grid.

- **Reliable native ceiling: 262,144 tokens** (Q5_K_M + f16 KV, no rope tricks):
  100% retrieval through 257,660 tokens, decode 28.2 t/s, prefill 18 min cold.
  This is the recommended serving config and Track B's hot-window ceiling.
- **512k is real but special-purpose: IQ4_XS + YaRN x2 + f16 KV**: 100% retrieval
  at 520,104 tokens across all 7 depths x 2 independent haystacks (17/17), decode
  16.3 t/s, VRAM 27.9/31.9 GiB, cold prefill ~65 min. The original "can we do
  500k?" question: **yes** — with an hour of prefill per cold haystack.
- **q8_0 KV costs nothing in accuracy but 5-10x in decode** (2.8 vs 16.3 t/s at
  520k) on this SYCL stack — measurement-only, never production.
- **yarn2 vs linear2: no accuracy difference** at coarse depths on this model.

Evidence strength: 512k claim = 7 depths x 2 full samples + 3 coarse probes;
native-256k claim = 3 depths x 2 configs (f16/q8_0). A third 512k sample, a
3-sample native densification, and a dense q8_0 cross-check were cut by user
call after 155/155 — diminishing returns.
