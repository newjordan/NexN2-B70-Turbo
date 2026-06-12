# NX2 First-Token Timing — 2026-06-12 (full patch chain 0001–0003)

Rerun of the first-token control on the fabler-branch build, which now lazily
reorders Q4_K, Q5_K **and Q6_K** MoE expert weights on the first decode token.
Same protocol as the retained 2026-06-10 run (fresh process per rep, `-fa on`,
`-fit off`, `--no-warmup`, NX2 Q5_K_M); the harness awk summary was rebuilt
inline because the system awk is mawk (raw logs + `summary.csv` retained here).

Medians of 3 reps:

| case | prompt/first-token eval | steady eval (128 tok) |
|---|---:|---:|
| stock, reorder off | 0.772 s | 1.847 s |
| Turbo, reorder on | 0.803 s | 1.655 s |

Interpretation: the one-time lazy reorder — now covering ~2× the expert bytes it
did before (Q6_K `ffn_down_exps` included) — costs ~31 ms on the first token.
No meaningful first-token penalty; steady decode improves as expected. Load
times here (~0.8 s) are warm-cache and not comparable with the cold-cache
2026-06-10 run; the off-vs-on comparison within this run is like-for-like.
