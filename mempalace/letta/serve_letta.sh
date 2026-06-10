#!/usr/bin/env bash
# Launch the Letta server (sqlite-backed) on :8283, detached.
set -o pipefail
set -u

RUN="/home/frosty40/nx2-palace-run"
LOG="$RUN/letta-server.log"
PORT=8283
mkdir -p "$RUN"

if ss -tlnH "sport = :${PORT}" 2>/dev/null | grep -q .; then
  echo "[letta] already listening on :${PORT}"
  exit 0
fi

# Letta 0.16.8's server is Postgres-only (its sqlite paths never reach the
# engine) — use the embedded no-root postgres from the pgserver pip package
# (bundles pgvector), socket-only under nx2-palace-run/pgdata.
# pgserver refcounts handles and stops postgres when its python process
# exits, so a long-lived holder process must stay up alongside letta.
if [ ! -S "$RUN/pgdata/.s.PGSQL.5432" ]; then
  nohup /home/frosty40/letta-venv/bin/python -c "
import pgserver, time
pg = pgserver.get_server('$RUN/pgdata')
pg.psql('CREATE EXTENSION IF NOT EXISTS vector;')
print('[pg-holder] embedded postgres up', flush=True)
time.sleep(10**9)
" >> "$RUN/pg-holder.log" 2>&1 &
  echo $! > "$RUN/pg-holder.pid"
  disown 2>/dev/null || true
fi
for _ in $(seq 1 60); do
  [ -S "$RUN/pgdata/.s.PGSQL.5432" ] && break
  sleep 1
done
[ -S "$RUN/pgdata/.s.PGSQL.5432" ] || { echo "[letta] FATAL: embedded postgres socket never appeared" >&2; exit 1; }
export LETTA_PG_URI="postgresql://postgres:@/postgres?host=$RUN/pgdata"

nohup /home/frosty40/letta-venv/bin/letta server --port "$PORT" \
      >> "$LOG" 2>&1 &
SRV=$!
disown "$SRV" 2>/dev/null || true
echo "$SRV" > "$RUN/letta-server.pid"

for _ in $(seq 1 90); do
  kill -0 "$SRV" 2>/dev/null || { echo "[letta] FATAL: died; tail:" >&2; tail -n 20 "$LOG" >&2; exit 1; }
  if curl -s --max-time 3 "http://127.0.0.1:${PORT}/v1/health/" | grep -qi 'ok\|healthy'; then
    echo "[letta] healthy: pid=$SRV port=$PORT"
    exit 0
  fi
  sleep 2
done
echo "[letta] FATAL: health timeout; tail:" >&2; tail -n 20 "$LOG" >&2
exit 1
