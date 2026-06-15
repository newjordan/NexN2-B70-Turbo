#!/usr/bin/env bash
# Mixed-precision down-proj sweep for the single A770 16 GiB budget.
#
# Finding that motivates this: uniform IQ3_A770 is only 13.90 GiB — it leaves
# ~1 GiB of the 16 GiB card unspent, while stock iq3_s spends that GiB to reach
# KLD 0.0653 (vs our 0.0754). The doc names down-proj at 3-bit as THE binding
# quality risk. So: keep gate/up codebook-free (iq3_a770, the dp4a thesis) and
# pour the spare headroom into down only. Goal: beat iq3_s's 0.0653 at <=15 GiB.
#
# Baselines (all-uniform, GiB / mean_KLD): iq3_a770 13.90/0.0754,
# iq3_s 14.84/0.0653, iq3_xxs 13.43/0.0814, Q4_K 19.71/0.0389.
set -uo pipefail
M=/home/frosty40/models/nex-n2-mini
BIN=/home/frosty40/llama.cpp/build-cpu-iq3/bin
Q=$BIN/llama-quantize; P=$BIN/llama-perplexity
CORPUS=/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw
KLDBASE=$M/NX2-Q6K.kld; CHUNKS=${CHUNKS:-50}
OUT=$M/iq3-down-sweep; mkdir -p "$OUT"; CSV=$OUT/down-sweep.csv

# Memory gate: the machine cannot hold a second ~14 GiB model load while the
# IQ3_NL KLD benchmark is resident. Wait for it to finish (frees ~10 GiB) before
# loading anything, then record its converged number for the record.
WAIT_PID=${WAIT_PID:-2041461}
if kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] waiting for IQ3_NL benchmark (pid $WAIT_PID) to free memory..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M:%S)] IQ3_NL benchmark done; proceeding."
fi
if [ -f /tmp/kld-iq3nl.log ]; then
  grep -E "Mean    KLD:|Same top p:|Mean PPL\(Q\)" /tmp/kld-iq3nl.log > "$OUT/iq3nl-final.txt" 2>/dev/null
  echo "=== IQ3_NL (LUT) final ==="; cat "$OUT/iq3nl-final.txt"
fi

echo "cell,gateup,down,nonexp,size_GiB,mean_KLD,top1_pct,ppl_ratio" > "$CSV"

cell() {  # label  gateup_type  down_type  nonexpert_base
  local label="$1" gu="$2" dn="$3" base="$4"; local out="$OUT/NX2.$label.gguf"
  echo "[$(date +%H:%M:%S)] === $label  gate/up=$gu  down=$dn  nonexp=$base ==="
  if [ ! -f "$out" ]; then
    $Q --imatrix "$M/NX2.imatrix" \
       --tensor-type ffn_gate_exps=$gu --tensor-type ffn_up_exps=$gu \
       --tensor-type ffn_down_exps=$dn \
       "$M/NX2-bf16.gguf" "$out" "$base" 24 > "$OUT/quant.$label.log" 2>&1 \
       || { echo "  quantize FAILED"; grep -i error "$OUT/quant.$label.log" | tail -3; return; }
  fi
  local sz; sz=$(du -L -b "$out" | awk '{printf "%.2f",$1/1073741824}')
  $P -m "$out" -f "$CORPUS" --kl-divergence-base "$KLDBASE" --kl-divergence \
     --chunks "$CHUNKS" -t 24 > "$OUT/kld.$label.log" 2>&1
  local kld top1 pplr
  kld=$(grep "Mean    KLD:"  "$OUT/kld.$label.log" | awk '{print $3}')
  top1=$(grep "Same top p:" "$OUT/kld.$label.log" | awk '{print $4}')
  pplr=$(grep "Mean PPL(Q)/PPL(base)" "$OUT/kld.$label.log" | awk '{print $4}')
  echo "$label,$gu,$dn,$base,$sz,${kld:-NA},${top1:-NA},${pplr:-NA}" >> "$CSV"
  echo "  size=${sz}GiB KLD=${kld:-NA} top1=${top1:-NA}% ppl=${pplr:-NA}"
}

# gate/up always codebook-free iq3_a770; pour headroom into down.
cell down_q3k   iq3_a770 q3_k   q6_k   # ~14.2 GiB, codebook-free K-quant down (comfortable)
cell down_iq3s  iq3_a770 iq3_s  q6_k   # ~14.2 GiB, codebook down (max-quality reference)
cell down_q4k   iq3_a770 q4_k   q6_k   # ~15.5 GiB, codebook-free, tight (best codebook-free bet)
cell down_q4k_q5base iq3_a770 q4_k q5_k # ~15.2 GiB, q4_k down + Q5 non-expert trim to claw headroom

echo; echo "=== down-proj sweep (target: beat iq3_s 0.0653 @<=15 GiB; uniform iq3_a770 = 0.0754 @13.90) ==="
column -t -s, "$CSV"
