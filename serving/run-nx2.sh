#!/usr/bin/env bash
# run-nx2.sh — Nex-N2-mini (qwen35moe, 256-expert MoE) dual-mode launcher.
#
# One model, two residency/precision points selected by --sm:
#
#   --sm off  (1 GPU) : NX2-IQ3_A770  — experts 3.19 bpw (codebook-free LUT), single 16 GB
#                       card, the fused reorder dp4a decode path (patches 0007/0008).
#                       ~82 tok/s tg on Arc B70 (A770-proxy); KLD see results/nx2-final-eval.
#
#   --sm on   (2 GPU) : NX2-Q4_K      — experts 4.5 bpw, split layer-wise across both cards
#                       (-sm layer -ts 1,1). More accurate; fewer bytes/card than the 1-card
#                       3-bit base (4.5/2 < 3.19), so the split also relieves decode bandwidth.
#                       (2-card throughput is bandwidth-projected — validate on real 2-GPU HW.)
#
# Why 4.5 bpw for the 2-card fill and not 8: per-card read = (active_experts/2)*bpw. The
# break-even vs the 1-card 3.19 base is ~6.4 bpw; below it the split nets *fewer* bytes/card
# (faster); above it (Q8 = 8.5) the split can't pay for the extra bytes.
#
# Usage:
#   ./run-nx2.sh [--sm on|off|auto] [-- <extra llama args>]
#   NX2_TOOL=llama-cli ./run-nx2.sh --sm off -p "Hello"
# Env overrides: NX2_MODELS, NX2_BIN, NX2_MODEL_1C, NX2_MODEL_2C, NX2_TOOL, NX2_NGL
set -euo pipefail

MODELS="${NX2_MODELS:-/home/frosty40/models/nex-n2-mini}"
BIN="${NX2_BIN:-/home/frosty40/llama.cpp/build/bin}"
MODEL_1C="${NX2_MODEL_1C:-$MODELS/NX2-IQ3_A770-mixed-LUT.gguf}"
MODEL_2C="${NX2_MODEL_2C:-$MODELS/NX2-IQ3_A770-Q4fill.gguf}"
TOOL="${NX2_TOOL:-llama-server}"
NGL="${NX2_NGL:-99}"

SM="auto"
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sm)   SM="$2"; shift 2;;
    --sm=*) SM="${1#*=}"; shift;;
    --)     shift; PASS+=("$@"); break;;
    *)      PASS+=("$1"); shift;;
  esac
done

# auto-detect: 2+ Level-Zero GPUs -> --sm on, else off
if [[ "$SM" == "auto" ]]; then
  N_GPU="$(sycl-ls 2>/dev/null | grep -ic 'level_zero:gpu' || true)"
  [[ "${N_GPU:-0}" -ge 2 ]] && SM="on" || SM="off"
fi

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

if [[ "$SM" == "on" ]]; then
  echo "[run-nx2] --sm on  -> 2-card Q4_K (split): $MODEL_2C" >&2
  exec "$BIN/$TOOL" -m "$MODEL_2C" -ngl "$NGL" -sm layer -ts 1,1 "${PASS[@]}"
else
  echo "[run-nx2] --sm off -> 1-card IQ3_A770 (fused dp4a): $MODEL_1C" >&2
  exec "$BIN/$TOOL" -m "$MODEL_1C" -ngl "$NGL" "${PASS[@]}"
fi
