#!/usr/bin/env bash
# Campaign serve helper for the NIAH context sweep.
# Re-launches the NexN2 llama.cpp server on :8090 with configurable CTX / KV / RoPE.
# Backward-compatible defaults reproduce the live production config exactly.
#
# Behavior: kills the current PORT owner (by socket via ss AND by pgrep pattern),
# waits until the port is actually free, launches the new server detached
# (nohup+disown; pid -> nx2-niah-run/server.pid), then blocks until /props
# reports n_ctx == CTX (exit 0) or fails loudly (exit 1).
# GOTCHA: GPU model-load is SIGKILLed by the agent sandbox (exit 144) — run this
# script via Bash with dangerouslyDisableSandbox=true. Model load at big CTX can
# take minutes; call with run_in_background or a generous timeout.
#
# Env knobs:
#   MODEL           gguf path (default: Q5_K_M sweep winner)
#   CTX             context size           (default 262144)
#   KV              KV cache type f16|q8_0 (default f16) -> -ctk/-ctv
#   FA              flash attn on|off      (default on)
#   NGL             gpu layers             (default 99)
#   PORT            (default 8090)
#   ROPE_SCALING    none|linear|yarn       (default: unset -> model native)
#   ROPE_SCALE      context scale factor N (linear/yarn)
#   YARN_ORIG_CTX   YaRN original ctx      (e.g. 262144)
#   YARN_EXT_FACTOR YaRN extrapolation mix (0.0 = full interpolation)
#   LOG             server log path        (default nx2-niah-run/server.log)
#   HEALTH_TIMEOUT  seconds to wait for healthy n_ctx (default 1800)
set -o pipefail

# oneAPI's vars.sh references unbound vars (OCL_ICD_FILENAMES) — a fatal abort
# under `set -u` that even `|| true` can't catch, so enable -u only afterwards.
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

BIN="${BIN:-/home/frosty40/llama.cpp/build/bin}"
MODEL="${MODEL:-/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf}"
CTX="${CTX:-262144}"
KV="${KV:-f16}"
FA="${FA:-on}"
NGL="${NGL:-99}"
PORT="${PORT:-8090}"
RUN="/home/frosty40/nx2-niah-run"
LOG="${LOG:-$RUN/server.log}"
mkdir -p "$RUN"

ROPE_ARGS=()
[ -n "${ROPE_SCALING:-}" ]    && ROPE_ARGS+=(--rope-scaling "$ROPE_SCALING")
[ -n "${ROPE_SCALE:-}" ]      && ROPE_ARGS+=(--rope-scale "$ROPE_SCALE")
[ -n "${YARN_ORIG_CTX:-}" ]   && ROPE_ARGS+=(--yarn-orig-ctx "$YARN_ORIG_CTX")
[ -n "${YARN_EXT_FACTOR:-}" ] && ROPE_ARGS+=(--yarn-ext-factor "$YARN_EXT_FACTOR")

# Stop any server currently on PORT and wait for the SOCKET to free, not just
# the pgrep pattern: during the 2026-06-09 handoff bug the old process kept
# :8090 bound after pgrep went quiet and the new server failed to bind.
pat="llama-server.*--port ${PORT}"
port_owner() { ss -tlnpH "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u; }

OLD="$( { pgrep -f "$pat" || true; port_owner; } | sort -u )"
if [ -n "$OLD" ]; then
  echo "[serve] stopping old server pid(s): $OLD" >&2
  kill -TERM $OLD 2>/dev/null || true
fi
for _ in $(seq 1 240); do
  if ! pgrep -f "$pat" >/dev/null 2>&1 && [ -z "$(port_owner)" ]; then break; fi
  sleep 0.5
done
LEFT="$( { pgrep -f "$pat" || true; port_owner; } | sort -u )"
if [ -n "$LEFT" ]; then
  echo "[serve] force-killing stragglers: $LEFT" >&2
  kill -KILL $LEFT 2>/dev/null || true
  sleep 1
fi
if [ -n "$(port_owner)" ]; then
  echo "[serve] FATAL: :${PORT} still has a listener after kill; aborting launch" >&2
  exit 1
fi

ARGS=(-m "$MODEL" --alias nex-n2-mini -ngl "$NGL" -fa "$FA" -ctk "$KV" -ctv "$KV"
      -c "$CTX" -np 1 --host 127.0.0.1 --port "$PORT" --jinja "${ROPE_ARGS[@]}")

# llama-server unconditionally caps the slot context to the model's trained
# context (qwen35moe.context_length = 262144); for CTX beyond it, lift the cap
# via metadata override. YaRN still scales from the true trained window because
# --yarn-orig-ctx is passed explicitly above.
N_CTX_TRAIN="${N_CTX_TRAIN:-262144}"
if [ "$CTX" -gt "$N_CTX_TRAIN" ]; then
  ARGS+=(--override-kv "qwen35moe.context_length=int:${CTX}")
fi

{
  echo "=========================================================="
  echo "[serve] $(cat /proc/uptime | awk '{print $1}')s uptime  launching:"
  printf '[serve] '; printf '%q ' "$BIN/llama-server" "${ARGS[@]}"; echo
  echo "[serve] MODEL=$MODEL CTX=$CTX KV=$KV FA=$FA rope=[${ROPE_ARGS[*]:-native}]"
  echo "=========================================================="
} >> "$LOG"

nohup "$BIN/llama-server" "${ARGS[@]}" >> "$LOG" 2>&1 &
SRV=$!
disown "$SRV" 2>/dev/null || true
echo "$SRV" > "$RUN/server.pid"

# Health gate: /props must answer AND report the requested n_ctx. A dying old
# server can still answer status:ok during handoff — n_ctx is the real signal.
TRIES=$(( ${HEALTH_TIMEOUT:-1800} / 5 ))
for _ in $(seq 1 "$TRIES"); do
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "[serve] FATAL: server pid $SRV died during load; last log lines:" >&2
    tail -n 20 "$LOG" >&2
    exit 1
  fi
  N_CTX="$(curl -s --max-time 5 "http://127.0.0.1:${PORT}/props" 2>/dev/null \
           | grep -oP '"n_ctx":\s*\K[0-9]+' | head -n1 || true)"
  if [ "$N_CTX" = "$CTX" ]; then
    echo "[serve] healthy: pid=$SRV n_ctx=$N_CTX port=$PORT"
    exit 0
  fi
  if [ -n "$N_CTX" ] && [ "$(port_owner)" = "$SRV" ]; then
    echo "[serve] FATAL: our server bound :${PORT} but n_ctx=$N_CTX != requested $CTX" >&2
    exit 1
  fi
  sleep 5
done
echo "[serve] FATAL: health timeout after ${HEALTH_TIMEOUT:-1800}s; tail of $LOG:" >&2
tail -n 20 "$LOG" >&2
exit 1
