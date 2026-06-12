# Fabler Gate/Up Fusion Campaign

Date: 2026-06-12. Hardware: Intel Arc Pro B70, oneAPI icpx 2026.0, llama.cpp SYCL backend.

This directory records the patch `0004` research campaign after Q6_K MoE reorder landed.

## Retained Default

The retained `0004` default is:

- exact NX2 Q4_K/Q5_K fused gate/up SwiGLU specialization,
- post-down weighted expert sum fusion,
- F32 MoE tail-add fusion.

Retained release-gate evidence:

| check | result |
|---|---:|
| Q5_K_M ctx0, pre-`0004` retained default | 85.6525 t/s |
| Q5_K_M ctx0, retained `0004` path | 87.9634 t/s |
| Q4_K_M ctx0, retained `0004` path | 93.4917 t/s |
| Q5_K_M 131k, retained `0004` path | 42.2928 t/s |
| Q5_K_M 30-chunk PPL, retained `0004` path | 5.5642 +/- 0.15223 |
| targeted `MUL_MAT_ID` | 690/690 |

See:

- `20260612T083707Z-nx2-weighted-sum-fusion/`
- `20260612T091359Z-nx2-tail-add-fusion/`
- `../nx2-kernel-release-gate/20260612-existing-artifacts/SUMMARY.md`

## Default-Off Probes

The following probes are preserved as negative or inconclusive evidence. They should not be cited as retained default behavior:

| directory | outcome |
|---|---|
| `20260612T090325Z-nx2-shared-gate-up-fusion/` | shared dense gate/up slower on Q5_K_M |
| `20260612T092946Z-nx2-weighted-sum-vec4/` | vec4 weighted sum slower than scalar |
| `20260612T094531Z-nx2-weighted-tail-fusion/` | combined weighted-tail slower |
| `20260612T094758Z-nx2-tail-add-vec4/` | too noisy to promote |
| `20260612T095919Z-nx2-gate-up-xor-reduce/` | XOR reducer slower than `reduce_over_group` |
| `20260612T100913Z-nx2-down-weighted-sum/` | down+weighted fusion slower |
| `20260612T101554Z-nx2-dispatch-guard/` | dispatch guard slower |
| `20260612T101912Z-nx2-down-weighted-atomic/` | atomic down+weighted slower |
| `20260612T102545Z-nx2-swiglu-exp2-recip/` | activation variants slower |
| `20260612T103409Z-nx2-weighted-sum-local-weights/` | local top-k weights slower |
| `20260612T103921Z-nx2-gate-up-rowpack4/` | rowpack4 slower |
| `20260612T104952Z-nx2-gate-up-q8-handoff/` | Q8 handoff mixed/slower |
| `20260612T110216Z-nx2-q6-down-exact-current/` | exact Q6_K down slower than generic reordered path |
| `20260612T110640Z-nx2-q8-handoff-q6-down-exact-combo/` | combined Q8 handoff + exact Q6 down slower |
| `20260612T111328Z-nx2-shared-gate-tail-fusion/` | below Q5_K_M promotion gate |
| `20260612T112008Z-nx2-shared-gate-tail-vec4/` | below Q5_K_M promotion gate |
| `20260612T112753Z-nx2-shared-gate-sigmoid-tail/` | mixed repeats |
| `20260612T113923Z-nx2-gate-up-rowpack2-8/` | retained mode won reversed repeat |
| `20260612T114939Z-nx2-activation-q8-cache/` | cache-on lost to local requantization |
| `20260612T115730Z-nx2-down-weighted-local-reduce/` | local-reduce down+weighted slower |
| `20260612T120710Z-nx2-tail-rms-fusion/` | Q5_K_M regressed |
| `20260612T122324Z-nx2-rms-norm-mul-fusion/` | positive but below +2% Q5_K_M promotion gate |
| `20260612T123830Z-nx2-rms-mul-q8-handoff/` | Q8 handoff trails RMS_NORM+MUL mode 2 on Q5_K_M |
| `20260612T125005Z-nx2-rms-mul-selective-q8-handoff/` | selective Q8 handoff regressed Q5_K_M |
| `20260612T125854Z-nx2-shared-gate-tail-local-gate/` | local gate broadcast slower |
| `20260612T130548Z-nx2-gate-up-local-q8/` | Q4_K_M regressed |
| `20260612T131422Z-nx2-gate-up-expert-pack/` | below Q5_K_M promotion gate |
| `20260612T133008Z-nx2-gate-up-dual-dot/` | Q5_K_M tied, Q4_K_M regressed |
| `20260612T133740Z-nx2-weighted-tail-vec8/` | not release-worthy; see local README |
