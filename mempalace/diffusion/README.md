# Mempalace tagger — verbatim compression-tagging layer

The in-between layer between Hermes and the Letta archive. It does NOT summarize:
compression lives in the **search space** (a tag index), while injected recall stays
the verbatim archived text. Every tag is mechanically validated as a literal substring
of its source item; tags that fail are dropped, so the model cannot put paraphrase or
hallucination into the index. Feasibility numbers: `results/tag-recall.md`.

```
hermes plugin                      this directory
  _archive() ──letta POST──► passage id
       └──queue file──► ~/.hermes/mempalace-tag-queue/<id>.json
                              │ tagger.py (watcher, pinned CPU)
                              │   slot-infill extraction (DiffusionGemma, EB steps=8)
                              │   substring validation  ◄── the invariant
                              ▼
                  ~/.hermes/mempalace-tags.db (items verbatim + tags + FTS5)
                              ▲
  _search() leg A: FTS tag match (exact identifiers) ── verbatim items
  _search() leg B: letta embedding search ───────────── verbatim items
```

## Components

- `tagger.py` — queue-watching service. Backends: `dgemma` (default; resident
  `llama-diffusion-cli`, ~52 s/item, zero GPU/server contention) and `nexn2`
  (the :8090 server, ~5 s/item, but serializes with live agent turns on the
  single slot). Env: `TAG_BACKEND`, `TAG_STEPS`, `TAG_THREADS`, `TAG_QUEUE`,
  `TAG_DB`, `TAG_BIN`, `TAG_MODEL`, `TAG_NEXN2_URL`.
- `serve_diffusion.sh` — launcher: `taskset -c 0-11 nice -n 10`, pid file, health
  check. **Pinning is mandatory**: an unpinned 16-thread run while the GPU decoded
  caused Xe GT0 engine resets → `UR_RESULT_ERROR_DEVICE_LOST` → llama-server abort
  (2026-06-10 18:04, see results/tag-recall.md). Pinned impact: −9.2% decode.

## Runtime

- Binary: `~/llama.cpp/build-diffusion-cpu/bin/llama-diffusion-cli` — CPU-only build
  of llama.cpp PR #24423 (DiffusionGemma support, pre-merge) plus two local patches
  on worktree branch `pr-diffusiongemma-tagger` (`/tmp/wt-diffusion`):
  `LLAMA_DIFFUSION_NO_THINK` (template thought channel off — otherwise the 256-token
  canvas is consumed by reasoning) and `LLAMA_DIFFUSION_STDIO` (`/reset`, `\n`
  escapes, `<<DONE>>` sentinels → resident-model request loop).
- Model: `~/models/diffusion/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (16.8 GB,
  system RAM — the B70 is never touched).
- Dream-7B was evaluated and eliminated (0–25% extraction validity, 3–6 min/item).

## Operating

```bash
bash mempalace/diffusion/serve_diffusion.sh          # start (idempotent)
tail -f ~/.hermes/logs/tagger.log                    # watch
TAG_BACKEND=nexn2 bash .../serve_diffusion.sh        # fast backend (slot-sharing!)
kill $(cat ~/.hermes/tagger.pid)                     # stop — recall degrades to
                                                     # embedding-only, nothing breaks
```

Failed queue items land in `<queue>/failed/` and are never retried implicitly.
The DB is safe to delete: it is a derived index; rebuilding = re-queueing items.
