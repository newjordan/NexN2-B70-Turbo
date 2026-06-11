#!/usr/bin/env bash
# Launch the mempalace tagger service, CPU-pinned and niced.
#
# The B70 incident of 2026-06-10 18:04 (Xe GT0 engine reset -> DEVICE_LOST ->
# llama-server abort) happened with an UNPINNED 16-thread CPU model saturating
# the host while the GPU decoded. Pinning + nice is mandatory, not tuning:
# cores 0-11 for the tagger leaves 20 cores for the server's Delta-Net decode
# and SYCL host threads (measured impact: -9.2% decode while tagging).
#
# Env knobs (defaults in brackets):
#   TAG_BACKEND  dgemma | nexn2            [dgemma]
#   TAG_STEPS    EB denoising step cap     [8]
#   TAG_THREADS  CPU threads               [12]
#   CORES        taskset core list         [0-11]
#   NICENESS     nice level                [10]
#   TAG_QUEUE    queue dir                 [~/.hermes/mempalace-tag-queue]
#   TAG_DB       sqlite index              [~/.hermes/mempalace-tags.db]
#   LOG          log file                  [~/.hermes/logs/tagger.log]
set -o pipefail
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORES="${CORES:-0-11}"
NICENESS="${NICENESS:-10}"
LOG="${LOG:-$HOME/.hermes/logs/tagger.log}"
PIDFILE="$HOME/.hermes/tagger.pid"

mkdir -p "$(dirname "$LOG")"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[serve_diffusion] already running (pid $(cat "$PIDFILE"))" >&2
    exit 0
fi

nohup taskset -c "$CORES" nice -n "$NICENESS" \
    python3 "$HERE/tagger.py" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

# health: tagger logs "ready:" after backend init (model load can take ~30 s)
for _ in $(seq 1 120); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[serve_diffusion] FATAL: tagger died during startup; tail of log:" >&2
        tail -5 "$LOG" >&2
        exit 1
    fi
    if grep -q "ready:" <(tail -20 "$LOG" 2>/dev/null); then
        echo "[serve_diffusion] healthy: pid=$PID cores=$CORES nice=$NICENESS log=$LOG"
        exit 0
    fi
    sleep 1
done
echo "[serve_diffusion] WARNING: no ready line after 120 s (model still loading?); pid=$PID" >&2
