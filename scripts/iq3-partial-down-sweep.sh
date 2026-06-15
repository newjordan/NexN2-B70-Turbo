#!/usr/bin/env bash
# Partial-down sweep: land down_q4k's quality (0.0567) INSIDE a comfortable 16 GiB
# fit by bumping only the top-K most-important down-proj layers to Q4_K (rest stay
# codebook-free IQ3_A770). Ranking is imatrix activation energy (rank-down-layers.py):
# importance is heavily late-layer-concentrated (blk.39 alone is 40x any other).
#
# Reference points (gate/up=iq3_a770, nonexp=q6_k):
#   all-down iq3_a770 : 13.90 GiB / 0.0754
#   all-down q4_k     : 15.54 GiB / 0.0567  (too tight for 16 GiB)
#   target            : < ~0.060 at <= ~15.0 GiB (real KV headroom)
set -uo pipefail
M=/home/frosty40/models/nex-n2-mini
BIN=/home/frosty40/llama.cpp/build-cpu-iq3/bin
Q=$BIN/llama-quantize; P=$BIN/llama-perplexity
CORPUS=/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw
KLDBASE=$M/NX2-Q6K.kld; CHUNKS=${CHUNKS:-100}
PY=/home/frosty40/nx2-venv/bin/python
OUT=$M/iq3-partial-down; mkdir -p "$OUT"; CSV=$OUT/partial-down.csv
echo "cell,k_layers,bumped_layers,size_GiB,mean_KLD,top1_pct,ppl_ratio" > "$CSV"

# down-proj layers ranked most->least important (imatrix mean act^2)
RANK_ORDER="39 38 35 33 34 31 37 30 27 36 19 32 23 15 24 26 14 21 11 28 10 12 25 29 18 6 7 5 9 20 13 16 4 22 8 17 3 1 2 0"

cell() {  # K  (number of top down layers to bump to q4_k)
  local K="$1"; local label="topk${K}"; local out="$OUT/NX2.$label.gguf"
  local layers; layers=$(echo $RANK_ORDER | tr ' ' '\n' | head -n "$K" | sort -n | tr '\n' ' ')
  echo "[$(date +%H:%M:%S)] === $label : q4_k down on layers [$layers] ==="
  # build per-layer overrides FIRST (first-match-wins), generic iq3_a770 fallback LAST
  local args=( --imatrix "$M/NX2.imatrix"
               --tensor-type ffn_gate_exps=iq3_a770 --tensor-type ffn_up_exps=iq3_a770 )
  local L
  for L in $layers; do args+=( --tensor-type "blk\.${L}\.ffn_down_exps=q4_k" ); done
  args+=( --tensor-type ffn_down_exps=iq3_a770 )
  if [ ! -f "$out" ]; then
    "$Q" "${args[@]}" "$M/NX2-bf16.gguf" "$out" q6_k 24 > "$OUT/quant.$label.log" 2>&1 \
      || { echo "  quantize FAILED"; grep -iE "error|fail" "$OUT/quant.$label.log" | tail -3; return; }
  fi
  local sz; sz=$(du -L -b "$out" | awk '{printf "%.2f",$1/1073741824}')
  "$P" -m "$out" -f "$CORPUS" --kl-divergence-base "$KLDBASE" --kl-divergence \
       --chunks "$CHUNKS" -t 24 > "$OUT/kld.$label.log" 2>&1
  local kld top1 pplr
  kld=$(grep "Mean    KLD:"  "$OUT/kld.$label.log" | awk '{print $3}')
  top1=$(grep "Same top p:" "$OUT/kld.$label.log" | awk '{print $4}')
  pplr=$(grep "Mean PPL(Q)/PPL(base)" "$OUT/kld.$label.log" | awk '{print $4}')
  echo "$label,$K,\"$layers\",$sz,${kld:-NA},${top1:-NA},${pplr:-NA}" >> "$CSV"
  echo "  size=${sz}GiB KLD=${kld:-NA} top1=${top1:-NA}%"
}

cell 6
cell 12
cell 20
cell 28

echo; echo "=== partial-down sweep (target: ~down_q4k 0.0567 at <=15 GiB comfortable fit) ==="
column -t -s, "$CSV"
