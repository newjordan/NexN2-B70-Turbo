# Fabler micro-experiment: NX2 fused row packing

Date: 2026-06-12. Hardware: Intel Arc Pro B70, oneAPI icpx 2026.0.
Branch: llama.cpp `fabler` with the retained default mode `3` exact NX2 fused
gate/up SwiGLU path from `0004`.

## Experiment

The retained fused NX2 launch maps one output row to one 16-lane subgroup and one
workgroup. This experiment packed two output rows into each workgroup by using a
local Y dimension of 2 while preserving one subgroup per row. The intent was to
halve workgroup count for the 512-row expert output while keeping dot-product
work and reduction structure unchanged.

## Result

Correctness passed, but throughput did not improve the primary Q5_K_M scoreboard:

| model | mode | ctx0 decode |
|---|---|---:|
| NX2-Q5_K_M | rowpack2 | 86.998 t/s |
| NX2-Q4_K_M | rowpack2 | 91.516 t/s |

Retained `0004` mode `3` remains the one-row-per-workgroup fused NX2 launch
(Q5_K_M ctx0 retained at 87.19 t/s).

## Raw artifacts

- `test-backend-ops-mulmatid-rowpack2.txt`
- `q5km-ctx0-rowpack2.json` / `.log`
- `q4km-ctx0-rowpack2.json` / `.log`
