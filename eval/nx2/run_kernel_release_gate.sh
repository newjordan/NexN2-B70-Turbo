#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

ROOT="${ROOT:-/home/frosty40/nx2-b70-turbo}"
BIN="${BIN:-/home/frosty40/llama.cpp/build/bin}"
BASELINE_BIN="${BASELINE_BIN:-}"
Q5_MODEL="${Q5_MODEL:-/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf}"
Q4_MODEL="${Q4_MODEL:-/home/frosty40/models/nex-n2-mini/sweep/NX2-Q4_K_M.gguf}"
OUT_BASE="${OUT:-$ROOT/results/nx2-kernel-release-gate}"
REPS="${REPS:-7}"
LONG_REPS="${LONG_REPS:-3}"
RUN_LONGCTX="${RUN_LONGCTX:-1}"
RUN_TENSOR_SPLIT="${RUN_TENSOR_SPLIT:-0}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1/1}"
RUN_PPL="${RUN_PPL:-1}"
RUN_BACKEND_OPS="${RUN_BACKEND_OPS:-1}"
WIKI="${WIKI:-/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw}"
PPL_CHUNKS="${PPL_CHUNKS:-30}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/$STAMP"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN/llama-bench" ]]; then
  echo "missing candidate llama-bench: $BIN/llama-bench" >&2
  exit 1
fi
if [[ -n "$BASELINE_BIN" && ! -x "$BASELINE_BIN/llama-bench" ]]; then
  echo "missing baseline llama-bench: $BASELINE_BIN/llama-bench" >&2
  exit 1
fi
if [[ ! -f "$Q5_MODEL" ]]; then
  echo "missing Q5 model: $Q5_MODEL" >&2
  exit 1
fi
if [[ ! -f "$Q4_MODEL" ]]; then
  echo "missing Q4 model: $Q4_MODEL" >&2
  exit 1
fi
if [[ "$RUN_PPL" == "1" && ! -f "$WIKI" ]]; then
  echo "missing WikiText file: $WIKI" >&2
  exit 1
fi

cat >"$OUT_DIR/MANIFEST.txt" <<EOF
date_utc=$STAMP
candidate_bin=$BIN
baseline_bin=${BASELINE_BIN:-same-binary-feature-off}
q5_model=$Q5_MODEL
q4_model=$Q4_MODEL
reps=$REPS
long_reps=$LONG_REPS
run_longctx=$RUN_LONGCTX
run_tensor_split=$RUN_TENSOR_SPLIT
tensor_split=$TENSOR_SPLIT
run_ppl=$RUN_PPL
ppl_chunks=$PPL_CHUNKS
run_backend_ops=$RUN_BACKEND_OPS
EOF

bench_candidate() {
  local label="$1"
  local model="$2"
  local depth="$3"
  local reps="$4"

  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=3 \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=1 \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=1 \
  "$BIN/llama-bench" \
    -m "$model" -ngl 99 -fa on -p 0 -n 128 -d "$depth" -r "$reps" -o json \
    >"$OUT_DIR/$label.json" \
    2>"$OUT_DIR/$label.log"
}

bench_candidate_tensor_split() {
  local label="$1"
  local model="$2"
  local depth="$3"
  local reps="$4"

  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=3 \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=1 \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=1 \
  "$BIN/llama-bench" \
    -m "$model" -ngl 99 -fa on -p 0 -n 128 -d "$depth" -r "$reps" \
    -sm tensor -ts "$TENSOR_SPLIT" -o json \
    >"$OUT_DIR/$label.json" \
    2>"$OUT_DIR/$label.log"
}

ppl_candidate() {
  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=3 \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=1 \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=1 \
  "$BIN/llama-perplexity" \
    -m "$Q5_MODEL" -ngl 99 -fa on -f "$WIKI" --chunks "$PPL_CHUNKS" \
    >"$OUT_DIR/candidate-q5-ppl.log" \
    2>&1
}

ppl_baseline() {
  local base_bin="${BASELINE_BIN:-$BIN}"

  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=0 \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=0 \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=0 \
  "$base_bin/llama-perplexity" \
    -m "$Q5_MODEL" -ngl 99 -fa on -f "$WIKI" --chunks "$PPL_CHUNKS" \
    >"$OUT_DIR/baseline-q5-ppl.log" \
    2>&1
}

ops_candidate() {
  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=3 \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=1 \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=1 \
  "$BIN/test-backend-ops" test -o MUL_MAT_ID \
    >"$OUT_DIR/candidate-test-backend-ops-mul-mat-id.log" \
    2>&1
}

bench_baseline() {
  local label="$1"
  local model="$2"
  local depth="$3"
  local reps="$4"
  local base_bin="${BASELINE_BIN:-$BIN}"

  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=0 \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=0 \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=0 \
  "$base_bin/llama-bench" \
    -m "$model" -ngl 99 -fa on -p 0 -n 128 -d "$depth" -r "$reps" -o json \
    >"$OUT_DIR/$label.json" \
    2>"$OUT_DIR/$label.log"
}

bench_baseline_tensor_split() {
  local label="$1"
  local model="$2"
  local depth="$3"
  local reps="$4"
  local base_bin="${BASELINE_BIN:-$BIN}"

  GGML_SYCL_ENABLE_MOE_GATE_UP_FUSION=0 \
  GGML_SYCL_ENABLE_MOE_NX2_SPECIALIZE=1 \
  GGML_SYCL_ENABLE_MOE_WEIGHTED_SUM_FUSION=0 \
  GGML_SYCL_ENABLE_MOE_TAIL_ADD_FUSION=0 \
  "$base_bin/llama-bench" \
    -m "$model" -ngl 99 -fa on -p 0 -n 128 -d "$depth" -r "$reps" \
    -sm tensor -ts "$TENSOR_SPLIT" -o json \
    >"$OUT_DIR/$label.json" \
    2>"$OUT_DIR/$label.log"
}

# Reverse the primary order so the candidate does not only benefit from warm state.
bench_candidate "candidate-q5-ctx0" "$Q5_MODEL" 0 "$REPS"
bench_baseline  "baseline-q5-ctx0"  "$Q5_MODEL" 0 "$REPS"

bench_baseline  "baseline-q4-ctx0"  "$Q4_MODEL" 0 "$REPS"
bench_candidate "candidate-q4-ctx0" "$Q4_MODEL" 0 "$REPS"

if [[ "$RUN_LONGCTX" == "1" ]]; then
  bench_baseline  "baseline-q5-131k"  "$Q5_MODEL" 131072 "$LONG_REPS"
  bench_candidate "candidate-q5-131k" "$Q5_MODEL" 131072 "$LONG_REPS"
fi

if [[ "$RUN_TENSOR_SPLIT" == "1" ]]; then
  bench_baseline_tensor_split  "baseline-q5-ctx0-sm-tensor"  "$Q5_MODEL" 0 "$REPS"
  bench_candidate_tensor_split "candidate-q5-ctx0-sm-tensor" "$Q5_MODEL" 0 "$REPS"
fi

if [[ "$RUN_PPL" == "1" ]]; then
  ppl_baseline
  ppl_candidate
fi

if [[ "$RUN_BACKEND_OPS" == "1" ]]; then
  ops_candidate
fi

"$ROOT/eval/nx2/summarize_kernel_release_gate.py" --write "$OUT_DIR" | tee "$OUT_DIR/summary.txt"
echo "wrote $OUT_DIR"
