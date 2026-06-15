#!/usr/bin/env bash
# Elastic-precision Pareto sweep: KLD vs weight footprint between the shipped
# all-IQ3 endpoint and the all-Q4_K endpoint, promoting experts by imatrix
# importance / byte. CPU eval (device-independent quality). Stitch -> eval -> rm,
# one model at a time (disk is tight).
#
# NOTE: llama-perplexity --kl-divergence ignores --chunks (it loops over whatever
# is stored in the base logits file: perplexity.cpp:1725/1794). So we first build a
# small 20-chunk Q6_K reference (the SAVE path *does* honor --chunks) and evaluate
# every point against it -> ~15 min/eval instead of ~75.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${NX2_MODELS:-$HOME/models/nex-n2-mini}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"; export LLAMA_CPP
BIN="${BIN:-$LLAMA_CPP/build-iq3nl/bin}"   # LUT-correct IQ3_A770 build (NF3 kvalues_iq3nl)
PY="${PY:-$(command -v python3)}"
CORPUS="${NX2_CORPUS:-$HOME/nx2-eval/wikitext-2-raw/wiki.test.raw}"
CHUNKS=${CHUNKS:-20}
KLDBASE=$M/NX2-Q6K-${CHUNKS}.kld
BASE=$M/NX2-IQ3_A770-mixed-LUT.gguf
PROM=$M/NX2-IQ3_A770-Q4fill.gguf
WORK=$M/elastic-work; mkdir -p "$WORK"
OUT=$ROOT/results/elastic-precision; mkdir -p "$OUT/kld"
CSV=$OUT/pareto.csv

# --- 0. build the small Q6_K reference if absent ----------------------------
if [ ! -f "$KLDBASE" ]; then
  echo "[$(date +%H:%M:%S)] building ${CHUNKS}-chunk Q6_K reference -> $KLDBASE"
  "$BIN/llama-perplexity" -m "$M/NX2-Q6_K.gguf" -f "$CORPUS" \
     --kl-divergence-base "$KLDBASE" --chunks "$CHUNKS" -t 24 \
     > "$OUT/kld/build_ref.log" 2>&1
  echo "[$(date +%H:%M:%S)] reference built ($(du -h "$KLDBASE" | cut -f1))"
fi

echo "point,budget_gb,n_promoted,weight_gb,mean_kld,top1_pct,ppl_ratio" > "$CSV"

eval_model() {  # label  model_path  n_promoted  budget_gb
  local label="$1" model="$2" npro="$3" bud="${4:-NA}"
  local log="$OUT/kld/$label.log"
  local sz; sz=$(du -L -b "$model" | awk '{printf "%.3f",$1/1e9}')
  echo "[$(date +%H:%M:%S)] eval $label  (${sz} GB, $npro promoted)"
  "$BIN/llama-perplexity" -m "$model" -f "$CORPUS" \
     --kl-divergence-base "$KLDBASE" --kl-divergence -t 24 \
     > "$log" 2>&1
  local kld top1 pplr
  kld=$(grep "Mean    KLD:" "$log" | awk '{print $3}')
  top1=$(grep "Same top p:" "$log" | awk '{print $4}')
  pplr=$(grep "Mean PPL(Q)/PPL(base)" "$log" | awk '{print $4}')
  echo "$label,$bud,$npro,$sz,${kld:-NA},${top1:-NA},${pplr:-NA}" >> "$CSV"
  echo "    -> KLD=${kld:-NA} top1=${top1:-NA} pplr=${pplr:-NA}"
}

build_and_eval() {  # budget_gb
  local G="$1" pj mix npro
  pj="$OUT/promote_${G}gb.json"
  $PY $ROOT/scripts/rank-experts.py --imatrix $M/NX2.imatrix --base "$BASE" --promote "$PROM" \
      --budget-gb "$G" --out-json "$pj" 2>>"$OUT/kld/rank.log"
  npro=$($PY -c "import json;print(len(json.load(open('$pj'))))")
  mix="$WORK/mix_${G}gb.gguf"
  $PY $ROOT/scripts/stitch-mixed.py --base "$BASE" --promote "$PROM" \
      --promote-json "$pj" --out "$mix" 2>>"$OUT/kld/stitch.log"
  eval_model "p_${G}gb" "$mix" "$npro" "$G"
  rm -f "$mix"
}

# decisive points first: both endpoints + the realistic A770-headroom budget
eval_model "p00_base_allIQ3" "$BASE" 0 0
build_and_eval 1.0
eval_model "p99_full_Q4K" "$PROM" 96 4.09
# then fill in the rest of the curve
for G in 0.5 1.5 2.0 3.0; do build_and_eval "$G"; done

echo "[$(date +%H:%M:%S)] SWEEP DONE"
column -t -s, "$CSV"
