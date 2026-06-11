ENTITIES: Letta ; MemGPT ; llama.cpp ; pgserver ; Postgres ; pgvector
METRICS: 988 MB ; 262144 ; 768 ; 48 tables
FILES: ~/letta-venv ; ../lean/serve_embed.sh ; letta/server/db.py ; ~/nx2-palace-run/pgdata ; serve_letta.sh ; mempalace/letta/serve_letta.sh ; mempalace/letta/setup_agent.py ; mcp_memory.py ; /home/frosty40/nx2-venv/bin/python ; /home/frosty40/nx2-b70-turbo/mempalace/letta/mcp_memory.py
NUMBERS: 2026-06-10 ; 12:00 ; 0.16.8 ; 8090 ; 8091 ; 8283 ; 5432
ERRORS: Missing deps in the wheel ; The wheel ships no alembic migrations
DECISIONS: Letta 0.16.8 in ~/letta-venv ; Server is sqlite-backed by default ; The server is Postgres-only ; create a superuser role matching the OS user ; create the schema once via Letta's own async engine
TOPICS: Install ; Local LLM ; Local embeddings ; Gotchas ; Bring-up ; Hermes integration