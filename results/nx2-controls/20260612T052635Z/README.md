# NX2 Controls — 2026-06-12 (full patch chain 0001–0003)

Fresh `eval/nx2/run_controls.sh` rerun on the fabler-branch build (Q4_K/Q5_K **and
Q6_K** MoE reorder + concat submit-only). Same NX2 Q5_K_M artifact and harness as the
retained 2026-06-10 control (`results/nx2-controls/20260610T220943Z/`).

| config | depth | avg t/s |
|---|---:|---:|
| stock control: reorder off / FA off | 0 | 68.84 |
| **deployed: Turbo + FA** | 0 | **85.48** |
| stock control: reorder off / FA off | 131072 | 20.02 |
| **deployed: Turbo + FA** | 131072 | **42.07** |

Project-level before/after at ctx0: **+24.2%** (was +17% with the 0001-only chain).
The 131k comparison is the full deployed configuration vs stock (reorder and FA both
change); FA dominates the gain at depth.
