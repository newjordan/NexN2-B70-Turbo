#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -euo pipefail

BIN="${BIN:-/home/frosty40/llama.cpp-sycl-moe-clean/build-clean/bin}"
APP="${APP:-$BIN/llama-completion}"
MODEL="${MODEL:-/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf}"
OUT_BASE="${OUT:-/home/frosty40/nx2-b70-turbo/results/nx2-first-token}"
REPS="${REPS:-3}"
N_GEN="${N_GEN:-129}"
FA="${FA:-on}"
PROMPT="${PROMPT:-The fundamental theorem of arithmetic states that every integer greater than one}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/$STAMP"
SUMMARY="$OUT_DIR/summary.csv"

mkdir -p "$OUT_DIR"
printf 'label,rep,disable_opt,n_predict,load_ms,prompt_eval_ms,prompt_tokens,eval_ms,eval_runs,total_ms\n' > "$SUMMARY"

run_case() {
  local label="$1"
  local disable_opt="$2"
  local n_predict="$3"
  local rep

  for rep in $(seq 1 "$REPS"); do
    local log="$OUT_DIR/${label}-r${rep}.log"

    GGML_SYCL_DISABLE_OPT="$disable_opt" "$APP" \
      -m "$MODEL" -ngl 99 -fa "$FA" \
      -fit off --no-warmup --perf \
      -no-cnv --single-turn --simple-io --no-display-prompt \
      -p "$PROMPT" -n "$n_predict" \
      > "$log" 2>&1

    awk -v label="$label" -v rep="$rep" -v disable_opt="$disable_opt" -v n_predict="$n_predict" '
      /load time =/ {
        line = $0
        sub(/^.*load time = */, "", line)
        sub(/ ms.*$/, "", line)
        load_ms = line
      }
      /prompt eval time =/ {
        line = $0
        sub(/^.*prompt eval time = */, "", line)
        sub(/ ms.*$/, "", line)
        prompt_eval_ms = line
        line = $0
        sub(/^.* ms \\/ */, "", line)
        sub(/ tokens.*$/, "", line)
        prompt_tokens = line
      }
      /eval time =/ && $0 !~ /prompt eval time/ {
        line = $0
        sub(/^.*eval time = */, "", line)
        sub(/ ms.*$/, "", line)
        eval_ms = line
        line = $0
        sub(/^.* ms \\/ */, "", line)
        sub(/ runs.*$/, "", line)
        eval_runs = line
      }
      /total time =/ {
        line = $0
        sub(/^.*total time = */, "", line)
        sub(/ ms.*$/, "", line)
        total_ms = line
      }
      END {
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
          label, rep, disable_opt, n_predict,
          load_ms + 0, prompt_eval_ms + 0, prompt_tokens + 0,
          eval_ms + 0, eval_runs + 0, total_ms + 0
      }
    ' "$log" >> "$SUMMARY"
  done
}

run_case "stock-first-token-reorder-off" 1 1
run_case "turbo-first-token-reorder-on" 0 1
run_case "stock-steady-reorder-off" 1 "$N_GEN"
run_case "turbo-steady-reorder-on" 0 "$N_GEN"

awk -F, '
  NR == 1 { next }
  {
    count[$1]++
    load[$1] += $5
    prompt[$1] += $6
    eval[$1] += $8
    total[$1] += $10
  }
  END {
    for (label in count) {
      printf "%s load_ms=%.2f prompt_eval_ms=%.2f eval_ms=%.2f total_ms=%.2f\n",
        label, load[label]/count[label], prompt[label]/count[label],
        eval[label]/count[label], total[label]/count[label]
    }
  }
' "$SUMMARY" | sort

echo "wrote $OUT_DIR"
