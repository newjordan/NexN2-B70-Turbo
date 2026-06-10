# Letta/MemGPT substrate — pre-flight findings

- **Install:** Letta 0.16.8 in `~/letta-venv` (988 MB; includes `letta_client`
  SDK and the `letta server` CLI). Server is sqlite-backed by default.
- **Local LLM:** per-agent `LLMConfig` accepts
  `model_endpoint_type="openai"`, `model_endpoint="http://127.0.0.1:8090/v1"`,
  `context_window=262144` — drives llama.cpp's OpenAI-compatible endpoint.
- **Local embeddings:** per-agent `EmbeddingConfig` accepts
  `embedding_endpoint_type="openai"`,
  `embedding_endpoint="http://127.0.0.1:8091/v1"`, `embedding_dim=768`
  (nomic-embed-text-v1.5 Q8_0 on CPU via `../lean/serve_embed.sh`).

## Gotchas found during bring-up (Letta 0.16.8 from pip)

- Missing deps in the wheel: `asyncpg`, `pgvector`, `pg8000` must be pip-installed.
- The server is **Postgres-only** — the sqlite settings remnants never reach the
  engine (`letta/server/db.py` builds an asyncpg engine from `letta_pg_uri`,
  default `localhost:5432`). We use the `pgserver` pip package (embedded no-root
  Postgres with pgvector) under `~/nx2-palace-run/pgdata`.
- `pgserver` refcounts handles and stops Postgres when its Python process exits —
  `serve_letta.sh` keeps a long-lived holder process (`pg-holder.pid`).
- Letta's URI re-builder drops the username — create a superuser role matching
  the OS user (`CREATE ROLE frosty40 SUPERUSER LOGIN`).
- The wheel ships **no alembic migrations**; create the schema once via Letta's
  own async engine: `Base.metadata.create_all` (48 tables; see session notes).

## Bring-up

    bash mempalace/letta/serve_letta.sh          # Letta server on :8283
    ~/letta-venv/bin/python mempalace/letta/setup_agent.py   # create + smoke-test agent

The smoke test sends one message and expects `LETTA-LOCAL-OK` back through
the local model. Run only when :8090 is idle (single slot — requests queue).

## Hermes integration (T9, done 2026-06-10)

Hermes keeps its own brain (`:8090`) and gains durable memory via MCP:
`mcp_memory.py` is a stdlib stdio MCP server exposing `memory_save` /
`memory_search`, backed directly by the Letta agent's archival memory REST
API (semantic search via pgvector + local nomic embeddings — the Letta agent
loop is bypassed for latency).

Registered with:

    hermes mcp add mempalace \
      --command /home/frosty40/nx2-venv/bin/python \
      --args /home/frosty40/nx2-b70-turbo/mempalace/letta/mcp_memory.py \
      --env LETTA_AGENT_ID=agent-4b7d7066-3f13-4b8e-98f4-fde35b05c6e5

End-to-end validated: facts saved in one `hermes -z` session were recalled
verbatim by a brand-new session (no shared history) via `memory_search`.
Requires letta server (:8283, `serve_letta.sh`) + embeddings (:8091,
`../lean/serve_embed.sh`) to be up.

## Automatic memory (supersedes the MCP bridge for Hermes)

`../hermes-plugin/mempalace/` is a Hermes **MemoryProvider plugin** (symlinked
to `~/.hermes/plugins/mempalace`, activated via `memory.provider: mempalace`
in `~/.hermes/config.yaml`, endpoint config in `~/.hermes/mempalace.json`).
Retention is structural — no model cooperation needed:

- `sync_turn` — every completed exchange is archived the moment it ends
  (dedup-guarded, `<think>` stripped, tagged with session + timestamp)
- `on_pre_compress` — the full conversation span is archived before Hermes
  compression summarizes it away (belt-and-braces; turns are already saved)
- `on_delegation` — subagent task/result pairs are archived
- `prefetch` — top-4 relevant memories are auto-injected before every turn
  as a [MEMPALACE RECALL] block
- `memory_search` / `memory_save` tools remain for explicit access

Validated: a fact stated casually in one session (zero memory instructions)
was auto-archived and recalled verbatim by a fresh session that simply asked
about it. The MCP bridge (`mcp_memory.py`) was deregistered from Hermes to
avoid duplicate tools — it remains useful for non-Hermes MCP clients.
