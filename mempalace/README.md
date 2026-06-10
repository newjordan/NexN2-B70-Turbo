# MemPalace — rolling research memory for the local Hermes agent

Two substrates, head-to-head (eval under `../eval/memory/`):

## `lean/` — local-first substrate

Files + sqlite FTS5 BM25 + local embeddings + `[[link]]` graph +
MemGPT-style controller against the NexN2 server on `:8090`.

- `store.py` — palace store: markdown rooms on disk, sqlite index
  (FTS5, embeddings, link graph). Files are the source of truth.
- `embed.py` / `serve_embed.sh` — nomic-embed-text-v1.5 Q8_0 on `:8091`
  (CPU, `-ngl 0`; preserves B70 VRAM).
- `retriever.py` — BM25 + vector RRF fusion, 1-hop `[[link]]` expansion,
  recency·importance·relevance scoring.
- `llm.py` — chat client for `:8090`; strips the NexN2 `<think>` trace.
- `controller.py` — rolling loop: retrieve → step → self-edited core
  memory (```core-update```) + eager room saving (```save-room```) →
  compaction of evicted turns into rooms → periodic reflection.
- `cli.py` — `stats` / `add` / `search` / `step`.

Run the self-tests (store, retriever offline; embed needs `:8091`):

    nx2-venv/bin/python mempalace/lean/store.py
    nx2-venv/bin/python mempalace/lean/retriever.py
    bash mempalace/lean/serve_embed.sh && nx2-venv/bin/python mempalace/lean/embed.py

## `letta/` — Letta/MemGPT substrate

Letta 0.16.8 in `~/letta-venv` (988 MB), pointed at the `:8090`
OpenAI-compatible endpoint + the `:8091` embedding endpoint. See
`letta/README.md` for the pre-flight findings and agent setup.
