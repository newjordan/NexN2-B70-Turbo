#!/usr/bin/env bash
# Serve the winning NX2 quant (Q5_K_M) on the Intel Arc B70 via llama.cpp SYCL.
# OpenAI-compatible endpoint at http://127.0.0.1:8090/v1  (8080 was taken on this box).

# oneAPI's setvars.sh is NOT compatible with `set -u` (it references unset vars) and may
# return non-zero, so sourcing it under `set -euo pipefail` aborted this script (exit 1,
# empty log) — which is why launches always had to be hand-rolled. Source it FIRST and
# tolerate its exit code, THEN enable strict mode.
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL="${MODEL:-$HOME/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf}"   # the frontier winner
MMPROJ="${MMPROJ:-$HOME/models/nex-n2-mini/mmproj-f16.gguf}"        # vision (optional)

# Metadata is already fixed to block_count=40 / nextn_predict_layers=0, so NO --override-kv needed.
#
# LOOP-AGENT PROFILE (default): FA on + f16 KV + 131072 ctx + prompt cache.
#   - FA verified faithful on the B70 dGPU (2026-06-08): PPL(FA on)=6.7254 vs PPL(FA off)=6.7348 (within
#     noise; NOT the #19276 iGPU corruption). FA nearly DOUBLES decode at depth: +94% @131k, +27% @32k,
#     -1.8% @ctx0. Agents run at depth, so FA is the win. (results/longctx-fa.csv)
#   - f16 KV, NOT q8_0: q8_0 KV is 4.7x SLOWER here (8.7 vs 41 t/s @131k; SYCL FA+quant-KV path unoptimized).
#   - ctx 131072: KV is tiny (hybrid 10/40 attn layers = 20 KiB/tok, ~2.5 GiB @131k). 262144 also fits and
#     decodes 26.8 t/s, but cold-prefilling it takes ~19 min (O(n^2)); prompt cache amortizes per-turn.
# Overrides: CTX=262144 to allow the full trained window; FA=off to disable flash-attn.
# Vision: opt-in via MMPROJ_ENABLE=1 (verified working) BUT mmproj's ~5.5 GB scratch will NOT co-fit at
#   131k on Q5 - drop CTX (e.g. CTX=32768) or use a smaller quant when serving vision + long context.
CTX="${CTX:-131072}"
FA="${FA:-on}"
ARGS=(-m "$MODEL" --alias nex-n2-mini -ngl 99 -fa "$FA" -ctk f16 -ctv f16 -c "$CTX" -np 1
      --host 127.0.0.1 --port 8090 --jinja)
if [ -n "${MMPROJ_ENABLE:-}" ]; then ARGS+=(--mmproj "$MMPROJ"); fi

# Dry-run: print the resolved command and exit (tests env + arg construction, no GPU launch): DRYRUN=1 ...
if [ -n "${DRYRUN:-}" ]; then printf '%q ' "$BIN/llama-server" "${ARGS[@]}"; echo; exit 0; fi

# NOTE (harness): a sandboxed Bash tool call SIGKILLs GPU model-load (exit 144, empty log). Launch this
# script with the Bash tool's dangerouslyDisableSandbox=true, then nohup it; see docs/findings.md.
exec "$BIN/llama-server" "${ARGS[@]}"

# Measured (Q5_K_M, B70, reorder+FA on): decode 81 t/s @ctx0, 65 @32k, 41 @131k, 27 @262k; prefill ~1139 t/s.
# NX2 is a reasoning model: it emits a <think> trace first, so use generous max_tokens on requests.
