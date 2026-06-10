#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

LLAMA="${LLAMA:-/home/frosty40/llama.cpp-sycl-moe-clean}"
BIN="${BIN:-$LLAMA/build-clean/bin}"
MODEL="${MODEL:-/home/frosty40/models/micro-moe/Phi-mini-MoE-instruct-Q4_K_M.gguf}"
OUT_BASE="${OUT:-/home/frosty40/nx2-b70-turbo/results/micro-moe}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/$STAMP"

mkdir -p "$OUT_DIR"

COMMON=(-m "$MODEL" -ngl 99 -fa off -p 0 -n 128 -r 5 -o json)

GGML_SYCL_DISABLE_OPT=1 "$BIN/llama-bench" "${COMMON[@]}" \
  > "$OUT_DIR/phi-mini-moe-q4km-opt-off.json" \
  2> "$OUT_DIR/phi-mini-moe-q4km-opt-off.log"

GGML_SYCL_DISABLE_OPT=0 "$BIN/llama-bench" "${COMMON[@]}" \
  > "$OUT_DIR/phi-mini-moe-q4km-opt-on.json" \
  2> "$OUT_DIR/phi-mini-moe-q4km-opt-on.log"

timeout 180 env GGML_SYCL_DEBUG=1 GGML_SYCL_DISABLE_OPT=0 \
  "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa off -p 0 -n 1 -r 1 \
  > "$OUT_DIR/path-debug.log" 2>&1 || true

grep -E "ggml_sycl_mul_mat_id|ffn_.*_exps.weight|type=q4_K;ne=\\[[0-9]+, [0-9]+, [0-9]+, 1\\]" \
  "$OUT_DIR/path-debug.log" > "$OUT_DIR/path-debug-filtered.log" || true

jq -r '.[0] | "avg_ts=\(.avg_ts) stddev_ts=\(.stddev_ts) model=\(.model_type)"' \
  "$OUT_DIR/phi-mini-moe-q4km-opt-off.json" \
  "$OUT_DIR/phi-mini-moe-q4km-opt-on.json"

echo "wrote $OUT_DIR"
