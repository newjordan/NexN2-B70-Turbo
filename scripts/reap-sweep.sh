#!/usr/bin/env bash
# REAP prune x quant sweep -> (size, KLD) Pareto.
# Cells: full256 x IQ3  +  {r25,r50} x {IQ3_A770, Q4_K, Q5_K}.
# Quantizes each (idempotent) with the matching imatrix, then KLD vs NX2-Q6K.kld.
set -uo pipefail
M=/home/frosty40/models/nex-n2-mini
BIN=/home/frosty40/llama.cpp/build-cpu-iq3/bin
Q=$BIN/llama-quantize; P=$BIN/llama-perplexity
CORPUS=/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw
KLDBASE=$M/NX2-Q6K.kld
CHUNKS=${CHUNKS:-50}
OUT=$M/reap-sweep; mkdir -p "$OUT"
CSV=$OUT/reap-pareto.csv
echo "cell,expert_count,quant,size_GiB,mean_KLD,top1_pct,ppl_ratio" > "$CSV"

IQ3="--tensor-type ffn_gate_exps=iq3_a770 --tensor-type ffn_up_exps=iq3_a770 --tensor-type ffn_down_exps=iq3_a770"

# args: label  src_gguf  imatrix  base_ftype  expert_count  extra_quant_args...
run_cell() {
  local label="$1" src="$2" imat="$3" base="$4" ec="$5"; shift 5
  local out="$OUT/NX2.$label.gguf"
  echo "[$(date +%H:%M:%S)] === $label ==="
  if [ ! -f "$out" ]; then
    if [ ! -f "$src" ]; then echo "  SKIP: missing source $src"; return; fi
    $Q --imatrix "$imat" "$@" "$src" "$out" "$base" 24 > "$OUT/quant.$label.log" 2>&1 \
      || { echo "  quantize FAILED (see quant.$label.log)"; return; }
  fi
  local sz; sz=$(du -b "$out" | awk '{printf "%.2f",$1/1073741824}')
  $P -m "$out" -f "$CORPUS" --kl-divergence-base "$KLDBASE" --kl-divergence --chunks "$CHUNKS" -t 24 \
     > "$OUT/kld.$label.log" 2>&1
  local kld top1 pplr
  kld=$(grep "Mean    KLD:" "$OUT/kld.$label.log" | awk '{print $3}')
  top1=$(grep "Same top p:" "$OUT/kld.$label.log" | awk '{print $4}')
  pplr=$(grep "Mean PPL(Q)/PPL(base)" "$OUT/kld.$label.log" | awk '{print $4}')
  echo "$label,$ec,$base,$sz,${kld:-NA},${top1:-NA},${pplr:-NA}" >> "$CSV"
  echo "  size=${sz}GiB  KLD=${kld:-NA}  top1=${top1:-NA}%"
  # IQ3 cells are quantized off a q6_k base then renamed in the label
}

# full 256-expert IQ3 baseline (reuse the already-built model)
[ -f "$M/NX2-IQ3_A770-C.gguf" ] && ln -sf "$M/NX2-IQ3_A770-C.gguf" "$OUT/NX2.full_IQ3.gguf"
run_cell full_IQ3 "$M/NX2-bf16.gguf" "$M/NX2.imatrix" q6_k 256 $IQ3

# r50 first (answers the headline r50xQ5 vs full IQ3), then r25
run_cell r50_Q5   "$M/NX2-bf16.reap-r50.gguf" "$M/NX2.reap-r50.imatrix" Q5_K_M 128
run_cell r50_IQ3  "$M/NX2-bf16.reap-r50.gguf" "$M/NX2.reap-r50.imatrix" q6_k   128 $IQ3
run_cell r50_Q4   "$M/NX2-bf16.reap-r50.gguf" "$M/NX2.reap-r50.imatrix" Q4_K_M 128
run_cell r25_IQ3  "$M/NX2-bf16.reap-r25.gguf" "$M/NX2.reap-r25.imatrix" q6_k   192 $IQ3
run_cell r25_Q5   "$M/NX2-bf16.reap-r25.gguf" "$M/NX2.reap-r25.imatrix" Q5_K_M 192
run_cell r25_Q4   "$M/NX2-bf16.reap-r25.gguf" "$M/NX2.reap-r25.imatrix" Q4_K_M 192

echo; echo "=== REAP (size, KLD) Pareto (CHUNKS=$CHUNKS) ==="
column -t -s, "$CSV"
echo "baseline frontier: Q5_K=0.020  Q4_K=0.039  Q3_K_M=0.105 (full, unpruned)"
