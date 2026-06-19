#!/usr/bin/env bash
# run-nx2.sh — Nex-N2-mini (qwen35moe, 256-expert MoE) dual-precision launcher.
#
# ONE model file, two residency/precision points selected by --sm. The single
# multi-precision GGUF (Nex-N2-mini-Turbo-Phase-Twin.gguf) carries both expert sets; the loader
# (patch 0010, select_tensor_variant) resolves it to one variant at load time and only
# reads that variant's tensor data — so each mode pays only for the bytes it uses.
#
#   --sm off  (1 GPU) : variant 0 — experts IQ3_A770 3.19 bpw (codebook-free LUT), single
#                       16 GB card, fused reorder dp4a decode (patches 0007/0008).
#                       ~82 tok/s tg on Arc B70 (A770-proxy); KLD see results/nx2-final-eval.
#
#   --sm on   (2 GPU) : variant 1 — experts Q4_K 4.5 bpw, split layer-wise across both
#                       cards (-sm layer -ts 1,1), selected via
#                       --override-kv general.tensor_variant.default=int:1.
#                       More accurate; fewer bytes/card than the 1-card 3-bit base
#                       (4.5/2 < 3.19), so the split also relieves decode bandwidth.
#                       (2-card throughput is bandwidth-projected — validate on real 2-GPU HW.)
#
# Why 4.5 bpw for the 2-card fill and not 8: per-card read = (active_experts/2)*bpw. The
# break-even vs the 1-card 3.19 base is ~6.4 bpw; below it the split nets *fewer* bytes/card
# (faster); above it (Q8 = 8.5) the split can't pay for the extra bytes.
#
# Usage:
#   ./run-nx2.sh [--sm on|off|auto] [-- <extra llama args>]
#   NX2_TOOL=llama-cli ./run-nx2.sh --sm off -p "Hello"
# Env overrides: NX2_MODELS, NX2_BIN, NX2_MODEL, NX2_TOOL, NX2_NGL
#   By default the model is read from next to this script and the binaries from
#   ./llama.cpp/build/bin (what build/build.sh produces); override with NX2_BIN /
#   NX2_MODEL if your layout differs, or just put the llama.cpp bins on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${NX2_MODELS:-$HERE}"
# prefer a locally-built tree, then PATH
if [[ -n "${NX2_BIN:-}" ]]; then BIN="$NX2_BIN"
elif [[ -x "$HERE/llama.cpp/build/bin/llama-server" ]]; then BIN="$HERE/llama.cpp/build/bin"
else BIN=""; fi   # empty -> resolve TOOL from PATH
MODEL="${NX2_MODEL:-$MODELS/Nex-N2-mini-Turbo-Phase-Twin.gguf}"   # one file, both precisions
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

EXE="${BIN:+$BIN/}$TOOL"   # $BIN/tool if built locally, else bare tool from PATH

# Multi-turn safety: this hybrid linear-attention model does not survive prompt-cache /
# context-checkpoint restore (turn 2+ goes incoherent). Disable both so every turn is correct.
MT_SAFE=(--cache-ram 0 --ctx-checkpoints 0)

if [[ "$SM" == "on" ]]; then
  echo "[run-nx2] --sm on  -> variant 1 (Q4_K, 2-card split): $MODEL" >&2
  exec "$EXE" -m "$MODEL" -ngl "$NGL" -sm layer -ts 1,1 \
       --override-kv general.tensor_variant.default=int:1 "${MT_SAFE[@]}" "${PASS[@]}"
else
  echo "[run-nx2] --sm off -> variant 0 (IQ3_A770, 1-card): $MODEL" >&2
  exec "$EXE" -m "$MODEL" -ngl "$NGL" "${MT_SAFE[@]}" "${PASS[@]}"
fi
