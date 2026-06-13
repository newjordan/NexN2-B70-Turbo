#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

ROOT="${ROOT:-/home/frosty40/nx2-b70-turbo}"
BIN="${BIN:-/home/frosty40/llama.cpp/build/bin}"
Q5_MODEL="${Q5_MODEL:-/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf}"
OUT_BASE="${OUT:-$ROOT/results/nx2-tensor-split-claim}"
REPS="${REPS:-5}"
TOKENS="${TOKENS:-128}"
DEPTH="${DEPTH:-0}"
NGL="${NGL:-99}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1/1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/$STAMP"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN/llama-bench" ]]; then
  echo "missing llama-bench: $BIN/llama-bench" >&2
  exit 1
fi
if [[ ! -f "$Q5_MODEL" ]]; then
  echo "missing Q5 model: $Q5_MODEL" >&2
  exit 1
fi

cat >"$OUT_DIR/MANIFEST.txt" <<EOF
date_utc=$STAMP
bin=$BIN
q5_model=$Q5_MODEL
reps=$REPS
tokens=$TOKENS
depth=$DEPTH
n_gpu_layers=$NGL
tensor_split=$TENSOR_SPLIT
EOF

bench() {
  local label="$1"
  local mode="$2"
  local gate_up="$3"
  local weighted_sum="$4"
  local tail_add="$5"
  local args=()

  if [[ "$mode" == "tensor" ]]; then
    args=(-sm tensor -ts "$TENSOR_SPLIT")
  else
    args=(-sm layer)
  fi

  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION="$gate_up" \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION="$weighted_sum" \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION="$tail_add" \
  "$BIN/llama-bench" \
    -m "$Q5_MODEL" -ngl "$NGL" -fa on -p 0 -n "$TOKENS" -d "$DEPTH" -r "$REPS" \
    "${args[@]}" -o json \
    >"$OUT_DIR/$label.json" \
    2>"$OUT_DIR/$label.log"
}

bench "baseline-q5-ctx0"           layer  0 0 0
bench "candidate-q5-ctx0"          layer  3 1 1
bench "baseline-q5-ctx0-sm-tensor" tensor 0 0 0
bench "candidate-q5-ctx0-sm-tensor" tensor 3 1 1

"$ROOT/eval/nx2/summarize_kernel_release_gate.py" --write "$OUT_DIR" | tee "$OUT_DIR/summary.txt"
echo "wrote $OUT_DIR"
