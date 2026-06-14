#!/usr/bin/env bash
# Quality landscape: does NON-UNIFORM / codebook quant beat uniform IQ3_A770 (L3, 0.075)
# on NX2 experts at similar bpw? Probes existing types to decide whether to build NF3.
set -uo pipefail
M=/home/frosty40/models/nex-n2-mini
BIN=/home/frosty40/llama.cpp/build-cpu-iq3/bin
Q=$BIN/llama-quantize; P=$BIN/llama-perplexity
CORPUS=/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw
KLDBASE=$M/NX2-Q6K.kld; CHUNKS=${CHUNKS:-50}
OUT=$M/quality-landscape; mkdir -p "$OUT"; CSV=$OUT/landscape.csv
echo "cell,expert_quant,size_GiB,mean_KLD,top1_pct,ppl_ratio" > "$CSV"

cell() {  # label  expert_ggml_type
  local label="$1" et="$2"; local out="$OUT/NX2.$label.gguf"
  echo "[$(date +%H:%M:%S)] === $label (experts=$et) ==="
  if [ ! -f "$out" ]; then
    $Q --imatrix "$M/NX2.imatrix" \
       --tensor-type ffn_gate_exps=$et --tensor-type ffn_up_exps=$et --tensor-type ffn_down_exps=$et \
       "$M/NX2-bf16.gguf" "$out" q6_k 24 > "$OUT/quant.$label.log" 2>&1 \
       || { echo "  quantize FAILED"; grep -i error "$OUT/quant.$label.log" | tail -2; return; }
  fi
  local sz; sz=$(du -L -b "$out" | awk '{printf "%.2f",$1/1073741824}')
  $P -m "$out" -f "$CORPUS" --kl-divergence-base "$KLDBASE" --kl-divergence --chunks "$CHUNKS" -t 24 \
     > "$OUT/kld.$label.log" 2>&1
  local kld top1 pplr
  kld=$(grep "Mean    KLD:" "$OUT/kld.$label.log" | awk '{print $3}')
  top1=$(grep "Same top p:" "$OUT/kld.$label.log" | awk '{print $4}')
  pplr=$(grep "Mean PPL(Q)/PPL(base)" "$OUT/kld.$label.log" | awk '{print $4}')
  echo "$label,$et,$sz,${kld:-NA},${top1:-NA},${pplr:-NA}" >> "$CSV"
  echo "  size=${sz}GiB KLD=${kld:-NA} top1=${top1:-NA}%"
}

cell exp_iq3s   iq3_s     # 3.44 bpw codebook (non-uniform) -- the key probe
cell exp_iq3xxs iq3_xxs   # 3.06 bpw codebook (non-uniform), lower bpw
cell exp_iq4xs  iq4_xs    # 4.25 bpw non-linear, reference

echo; echo "=== quality landscape (baseline: uniform IQ3_A770 L3 = 0.075 @ 13.9GiB; Q4_K=0.039 Q5_K=0.020) ==="
column -t -s, "$CSV"
