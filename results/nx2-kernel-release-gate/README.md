# NX2 Kernel Release Gate Results

This directory stores release-gate inputs and summaries produced by:

```bash
eval/nx2/run_kernel_release_gate.sh
```

Decision policy is documented in `docs/kernel-release-gate.md`.

## Retained Decisions

| directory | decision | note |
|---|---|---|
| `20260612-existing-artifacts/` | `HF_UPDATE_WORTHY` | patch `0004` versus pre-`0004` retained default |
| `20260612T140842Z/` | `NO_HF_UPDATE` | same-binary incremental run; Q5_K_M ctx0 delta below +2% gate |
