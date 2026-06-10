#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

BIN="${BIN:-/home/frosty40/llama.cpp/build/bin}"
MODEL="${MODEL:-/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf}"
OUT_BASE="${OUT:-/home/frosty40/nx2-b70-turbo/results/nx2-controls}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/$STAMP"

mkdir -p "$OUT_DIR"

bench() {
  local label="$1"
  local disable_opt="$2"
  local fa="$3"
  local depth="$4"

  GGML_SYCL_DISABLE_OPT="$disable_opt" "$BIN/llama-bench" \
    -m "$MODEL" -ngl 99 -fa "$fa" -p 0 -n 128 -d "$depth" -r 5 -o json \
    > "$OUT_DIR/$label.json" \
    2> "$OUT_DIR/$label.log"
}

bench "stock-ctx0-reorder-off-fa-off" 1 off 0
bench "deployed-ctx0-reorder-on-fa-on" 0 on 0
bench "stock-131k-reorder-off-fa-off" 1 off 131072
bench "deployed-131k-reorder-on-fa-on" 0 on 131072

jq -r '.[0] | "\(.model_filename) avg_ts=\(.avg_ts) n_depth=\(.n_depth) flash_attn=\(.flash_attn)"' \
  "$OUT_DIR"/*.json

echo "wrote $OUT_DIR"
