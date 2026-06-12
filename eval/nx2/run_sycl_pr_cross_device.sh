#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

ROOT="${ROOT:-/home/frosty40/nx2-b70-turbo}"
BIN="${BIN:-/home/frosty40/llama.cpp-sycl-moe-ready/build-pr-preflight/bin}"
BASELINE_BIN="${BASELINE_BIN:-}"
MODEL="${MODEL:-/home/frosty40/models/nex-n2-mini/sweep/NX2-Q4_K_M.gguf}"
WIKI="${WIKI:-/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw}"
OUT_BASE="${OUT:-$ROOT/results/sycl-pr-cross-device}"
REPS="${REPS:-5}"
N_GEN="${N_GEN:-128}"
RUN_PPL="${RUN_PPL:-1}"
PPL_CHUNKS="${PPL_CHUNKS:-30}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/$STAMP"

mkdir -p "$OUT_DIR"

require_bin() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "missing executable: $path" >&2
    exit 1
  fi
}

require_bin "$BIN/llama-bench"
require_bin "$BIN/test-backend-ops"
if [[ "$RUN_PPL" == "1" ]]; then
  require_bin "$BIN/llama-perplexity"
  [[ -f "$WIKI" ]] || { echo "missing WikiText file: $WIKI" >&2; exit 1; }
fi
if [[ -n "$BASELINE_BIN" ]]; then
  require_bin "$BASELINE_BIN/llama-bench"
fi
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 1; }

cat >"$OUT_DIR/MANIFEST.txt" <<EOF
date_utc=$STAMP
candidate_bin=$BIN
baseline_bin=${BASELINE_BIN:-none}
model=$MODEL
reps=$REPS
n_gen=$N_GEN
run_ppl=$RUN_PPL
ppl_chunks=$PPL_CHUNKS
candidate_git=$(git -C "$BIN/../.." rev-parse --short HEAD 2>/dev/null || true)
baseline_git=$([[ -n "$BASELINE_BIN" ]] && git -C "$BASELINE_BIN/../.." rev-parse --short HEAD 2>/dev/null || true)
EOF

if [[ -x "$BIN/llama-ls-sycl-device" ]]; then
  "$BIN/llama-ls-sycl-device" >"$OUT_DIR/device.txt" 2>&1 || true
else
  "$BIN/test-backend-ops" -l >"$OUT_DIR/device.txt" 2>&1 || true
fi

"$BIN/test-backend-ops" test -o MUL_MAT_ID \
  >"$OUT_DIR/candidate-test-backend-ops-mul-mat-id.log" \
  2>&1

bench() {
  local label="$1"
  local bin_dir="$2"
  local disable_opt="$3"

  GGML_SYCL_DISABLE_OPT="$disable_opt" "$bin_dir/llama-bench" \
    -m "$MODEL" -ngl 99 -fa on -p 0 -n "$N_GEN" -r "$REPS" -o json \
    >"$OUT_DIR/$label.json" \
    2>"$OUT_DIR/$label.log"
}

bench "candidate-opt-off" "$BIN" 1
bench "candidate-opt-on"  "$BIN" 0

if [[ -n "$BASELINE_BIN" ]]; then
  bench "baseline-opt-on" "$BASELINE_BIN" 0
fi

if [[ "$RUN_PPL" == "1" ]]; then
  GGML_SYCL_DISABLE_OPT=1 "$BIN/llama-perplexity" \
    -m "$MODEL" -ngl 99 -fa on -f "$WIKI" --chunks "$PPL_CHUNKS" \
    >"$OUT_DIR/candidate-ppl-opt-off.log" \
    2>&1
  GGML_SYCL_DISABLE_OPT=0 "$BIN/llama-perplexity" \
    -m "$MODEL" -ngl 99 -fa on -f "$WIKI" --chunks "$PPL_CHUNKS" \
    >"$OUT_DIR/candidate-ppl-opt-on.log" \
    2>&1
fi

python3 - "$OUT_DIR" <<'PY' | tee "$OUT_DIR/SUMMARY.md"
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])

def bench(name):
    p = out / f"{name}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    row = json.loads(p.read_text())[0]
    return row["avg_ts"], row.get("stddev_ts", 0.0), len(row.get("samples_ts", []))

def ppl(name):
    p = out / name
    if not p.exists():
        return None
    m = re.search(r"Final estimate:\s+PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", p.read_text(errors="replace"))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def ops():
    p = out / "candidate-test-backend-ops-mul-mat-id.log"
    if not p.exists():
        return None
    matches = re.findall(r"(\d+)/(\d+) tests passed", p.read_text(errors="replace"))
    if not matches:
        return None
    return tuple(map(int, matches[-1]))

print(f"# SYCL PR Cross-Device Validation: {out.name}\n")
print("## Correctness\n")
op = ops()
print(f"- `test-backend-ops test -o MUL_MAT_ID`: {op[0]}/{op[1]}" if op else "- backend-op result: missing")
print("\n## Decode\n")
print("| case | t/s | stddev | samples | delta vs opt-off |")
print("|---|---:|---:|---:|---:|")
base = bench("candidate-opt-off")
for name in ("candidate-opt-off", "candidate-opt-on", "baseline-opt-on"):
    b = bench(name)
    if not b:
        continue
    delta = ""
    if base and name != "candidate-opt-off":
        delta = f"{100.0 * (b[0] - base[0]) / base[0]:+.2f}%"
    print(f"| {name} | {b[0]:.4f} | {b[1]:.4f} | {b[2]} | {delta} |")

off = ppl("candidate-ppl-opt-off.log")
on = ppl("candidate-ppl-opt-on.log")
if off and on:
    delta = 100.0 * (on[0] - off[0]) / off[0]
    print("\n## PPL\n")
    print("| case | PPL | stderr | delta vs opt-off |")
    print("|---|---:|---:|---:|")
    print(f"| candidate-opt-off | {off[0]:.4f} | {off[1]:.5f} |  |")
    print(f"| candidate-opt-on | {on[0]:.4f} | {on[1]:.5f} | {delta:+.2f}% |")

print("\n## Files\n")
print("- `MANIFEST.txt`")
print("- `device.txt`")
print("- `candidate-test-backend-ops-mul-mat-id.log`")
PY

echo "wrote $OUT_DIR"
