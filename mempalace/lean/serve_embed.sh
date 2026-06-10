#!/usr/bin/env bash
# Embedding endpoint for MemPalace: nomic-embed-text-v1.5 Q8_0 on :8091.
# Runs on CPU (-ngl 0) to preserve B70 VRAM for the main model.
# Same detach/health pattern as eval/niah/serve.sh.
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u

BIN="${BIN:-/home/frosty40/llama.cpp/build/bin}"
MODEL="${MODEL:-/home/frosty40/models/embeddings/nomic-embed-text-v1.5.Q8_0.gguf}"
PORT="${PORT:-8091}"
RUN="/home/frosty40/nx2-palace-run"
LOG="$RUN/embed-server.log"
mkdir -p "$RUN"

pat="llama-server.*--port ${PORT}"
port_owner() { ss -tlnpH "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u; }

OLD="$( { pgrep -f "$pat" || true; port_owner; } | sort -u )"
[ -n "$OLD" ] && { echo "[embed] stopping old: $OLD" >&2; kill -TERM $OLD 2>/dev/null || true; sleep 2; }
if [ -n "$(port_owner)" ]; then
  kill -KILL $(port_owner) 2>/dev/null || true; sleep 1
fi
if [ -n "$(port_owner)" ]; then
  echo "[embed] FATAL: :${PORT} still bound" >&2; exit 1
fi

nohup "$BIN/llama-server" -m "$MODEL" --embedding --pooling mean -ngl 0 \
      -c 8192 -b 8192 -ub 8192 --host 127.0.0.1 --port "$PORT" \
      >> "$LOG" 2>&1 &
SRV=$!
disown "$SRV" 2>/dev/null || true
echo "$SRV" > "$RUN/embed-server.pid"

for _ in $(seq 1 60); do
  kill -0 "$SRV" 2>/dev/null || { echo "[embed] FATAL: died; tail:" >&2; tail -n 10 "$LOG" >&2; exit 1; }
  if curl -s --max-time 3 "http://127.0.0.1:${PORT}/health" | grep -q '"ok"\|ok'; then
    echo "[embed] healthy: pid=$SRV port=$PORT"
    exit 0
  fi
  sleep 1
done
echo "[embed] FATAL: health timeout; tail:" >&2; tail -n 10 "$LOG" >&2
exit 1
