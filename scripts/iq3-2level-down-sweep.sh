#!/usr/bin/env bash
# 2-level down sweep: uniform Q3_K floor on ALL down layers + Q4_K on the top-K
# importance layers. Tests finding #2 (uniform bump is the efficient lever): a
# Q3_K floor under selective Q4_K should beat the pure topk series (IQ3 floor)
# at equal size, chasing down_q4k's 0.0567 into a comfortable 16 GiB fit.
#
# Reference (gate/up=iq3_a770, nonexp=q6_k):
#   down_q3k  (K=0, all q3_k)        14.21 / 0.0697
#   topk20    (iq3 floor, 20 q4_k)   14.72 / 0.0663
#   topk28    (iq3 floor, 28 q4_k)   15.05 / 0.0635   <- current winner
#   down_q4k  (all q4_k)             15.54 / 0.0567   (too tight)
set -uo pipefail
M=/home/frosty40/models/nex-n2-mini
BIN=/home/frosty40/llama.cpp/build-cpu-iq3/bin
Q=$BIN/llama-quantize; P=$BIN/llama-perplexity
CORPUS=/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw
KLDBASE=$M/NX2-Q6K.kld; CHUNKS=${CHUNKS:-100}
OUT=$M/iq3-2level-down; mkdir -p "$OUT"; CSV=$OUT/2level-down.csv
echo "cell,k_q4k_layers,size_GiB,mean_KLD,top1_pct,ppl_ratio" > "$CSV"

# down-proj layers ranked most->least important (imatrix mean act^2)
RANK_ORDER="39 38 35 33 34 31 37 30 27 36 19 32 23 15 24 26 14 21 11 28 10 12 25 29 18 6 7 5 9 20 13 16 4 22 8 17 3 1 2 0"

cell() {  # K  (top-K down layers get q4_k; the rest get the q3_k floor)
  local K="$1"; local label="q3floor_q4top${K}"; local out="$OUT/NX2.$label.gguf"
  local layers; layers=$(echo $RANK_ORDER | tr ' ' '\n' | head -n "$K" | sort -n | tr '\n' ' ')
  echo "[$(date +%H:%M:%S)] === $label : q3_k floor + q4_k on [$layers] ==="
  local args=( --imatrix "$M/NX2.imatrix"
               --tensor-type ffn_gate_exps=iq3_a770 --tensor-type ffn_up_exps=iq3_a770 )
  local L
  for L in $layers; do args+=( --tensor-type "blk\.${L}\.ffn_down_exps=q4_k" ); done
  args+=( --tensor-type ffn_down_exps=q3_k )   # floor (LAST = lowest precedence)
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
  echo "$label,$K,$sz,${kld:-NA},${top1:-NA},${pplr:-NA}" >> "$CSV"
  echo "  size=${sz}GiB KLD=${kld:-NA} top1=${top1:-NA}%"
}

cell 8     # ~14.48 GiB
cell 16    # ~14.74 GiB  (compare vs topk20 14.72/0.0663)
cell 24    # ~15.01 GiB  (compare vs topk28 15.05/0.0635)
cell 32    # ~15.27 GiB

echo; echo "=== 2-level down sweep (q3_k floor + q4_k top-K) vs topk28 0.0635@15.05 ==="
column -t -s, "$CSV"
