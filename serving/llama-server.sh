#!/usr/bin/env bash
# Serve NexN2 B70 Turbo (Q5_K_M) on the Intel Arc B70 via llama.cpp SYCL.
# OpenAI-compatible endpoint at http://127.0.0.1:8090/v1

# Source oneAPI before strict mode (setvars.sh is not `set -u` safe).
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL="${MODEL:-$HOME/models/Nex-N2-mini-B70-Turbo-Q5_K_M.gguf}"   # the frontier winner
MMPROJ="${MMPROJ:-$HOME/models/mmproj-f16.gguf}"                   # vision (optional)

# Loop-agent profile (default): Flash-Attention on, f16 KV, 131072 context, prompt cache.
# The published GGUFs already carry the MTP/NextN metadata, so no --override-kv is needed.
# Knobs:  CTX=262144 for the full trained window | FA=off | MMPROJ_ENABLE=1 for vision (lower CTX to fit).
CTX="${CTX:-131072}"
FA="${FA:-on}"
ARGS=(-m "$MODEL" --alias nex-n2-mini -ngl 99 -fa "$FA" -ctk f16 -ctv f16 -c "$CTX" -np 1
      --host 127.0.0.1 --port 8090 --jinja)
if [ -n "${MMPROJ_ENABLE:-}" ]; then ARGS+=(--mmproj "$MMPROJ"); fi

# Dry-run: print the resolved command and exit (no GPU launch): DRYRUN=1 ./llama-server.sh
if [ -n "${DRYRUN:-}" ]; then printf '%q ' "$BIN/llama-server" "${ARGS[@]}"; echo; exit 0; fi

exec "$BIN/llama-server" "${ARGS[@]}"

# Measured (Q5_K_M, B70, reorder + FA on): decode 81 t/s @ctx0, 65 @32k, 41 @131k, 27 @262k; prefill ~1139 t/s.
# NexN2 is a reasoning model: it emits a <think> trace, so use generous max_tokens.
