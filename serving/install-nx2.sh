#!/usr/bin/env bash
# install-nx2.sh — one-command install for Nex-N2-mini Turbo Phase Twin (elastic).
#
# Detects your VRAM, picks the best precision budget that fits, and AUTOPRUNES the
# 34 GiB dual file down to a single right-sized model (~15–20 GiB) — dropping the
# unused variant bytes. Then writes a launcher pointed at the fitted model.
#
#   ./install-nx2.sh                     # autodetect VRAM, autoprune to fit
#   ./install-nx2.sh --vram-gib 16       # force budget (e.g. A770 16 GB)
#   ./install-nx2.sh --keep-dual         # keep the elastic dual; tune at runtime instead
#   ./install-nx2.sh --dual PATH --out PATH
#
# Quality per budget: see results/elastic-precision/ (the convex KLD frontier).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-$(command -v python3)}"
DUAL="${NX2_DUAL:-}"
OUT=""
VRAM_GIB=""
RESERVE_GIB="${RESERVE_GIB:-0.6}"
KEEP_DUAL=0
HF_REPO="Frosty40/Nex-N2-mini-Turbo-Phase-Twin"
HF_FILE="Nex-N2-mini-Turbo-Phase-Twin.gguf"

while [ $# -gt 0 ]; do
  case "$1" in
    --vram-gib)   VRAM_GIB="$2"; shift 2;;
    --reserve-gib) RESERVE_GIB="$2"; shift 2;;
    --dual)       DUAL="$2"; shift 2;;
    --out)        OUT="$2"; shift 2;;
    --keep-dual)  KEEP_DUAL=1; shift;;
    -h|--help)    sed -n '2,18p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# ---- 1. detect VRAM (largest single device) --------------------------------
detect_vram_gib() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | awk 'BEGIN{m=0}{if($1>m)m=$1}END{if(m>0)printf "%.1f", m/1024}'
    return
  fi
  if command -v xpu-smi >/dev/null 2>&1; then
    xpu-smi discovery 2>/dev/null | grep -iE "Memory Physical Size|Memory Size" \
      | grep -oE "[0-9]+" | awk 'BEGIN{m=0}{if($1>m)m=$1}END{if(m>0)printf "%.1f", m/1024}'
    return
  fi
  # Intel i915/xe sysfs (bytes)
  for f in /sys/class/drm/card*/device/mem_info_vram_total /sys/class/drm/card*/lmem_total_bytes; do
    [ -r "$f" ] || continue
    awk '{printf "%.1f",$1/1073741824}' "$f"; return
  done
  echo ""
}

if [ -z "$VRAM_GIB" ]; then
  VRAM_GIB="$(detect_vram_gib)"
  if [ -z "$VRAM_GIB" ]; then
    echo "Could not auto-detect VRAM. Re-run with --vram-gib N (e.g. --vram-gib 16)." >&2
    exit 1
  fi
  echo "[install] detected VRAM: ${VRAM_GIB} GiB"
else
  echo "[install] VRAM (forced): ${VRAM_GIB} GiB"
fi

# ---- 2. locate (or fetch) the dual model -----------------------------------
if [ -z "$DUAL" ]; then
  for c in "$ROOT/models/"*dual*.gguf "$HERE/$HF_FILE" "./$HF_FILE"; do
    [ -f "$c" ] && { DUAL="$c"; break; }
  done
fi
if [ -z "$DUAL" ] || [ ! -f "$DUAL" ]; then
  echo "[install] dual model not found locally; downloading $HF_REPO/$HF_FILE ..."
  DUAL="$HERE/$HF_FILE"
  if command -v hf >/dev/null 2>&1; then
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$HF_REPO" "$HF_FILE" --local-dir "$HERE" || {
      echo "download failed; fetch $HF_REPO/$HF_FILE manually and pass --dual" >&2; exit 1; }
  else
    echo "no 'hf' CLI; pip install huggingface_hub or pass --dual PATH" >&2; exit 1
  fi
fi
echo "[install] dual model: $DUAL"

# ---- 3. compute the fit & report what we'll get ----------------------------
"$PY" "$ROOT/scripts/prune-dual.py" --dual "$DUAL" --vram-gib "$VRAM_GIB" \
      --reserve-gib "$RESERVE_GIB" --out /dev/null --dry-run || exit 1

# ---- 4. fit: autoprune (default) or keep the elastic dual ------------------
LAUNCH="$HERE/run-nx2-fitted.sh"
if [ "$KEEP_DUAL" -eq 1 ]; then
  BUDGET_MB=$(awk "BEGIN{printf \"%d\", ($VRAM_GIB-$RESERVE_GIB)*1024}")
  cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
# elastic launcher: one dual file, precision auto-fit to ${VRAM_GIB} GiB at load time.
exec "\${NX2_TOOL:-llama-cli}" -m "$DUAL" \\
  --override-kv general.tensor_variant.budget_mb=int:$BUDGET_MB "\$@"
EOF
  chmod +x "$LAUNCH"
  echo "[install] kept elastic dual; launcher (load-time budget ${BUDGET_MB} MiB): $LAUNCH"
else
  [ -z "$OUT" ] && OUT="${DUAL%/*}/NX2-fitted-${VRAM_GIB%.*}g.gguf"
  echo "[install] autopruning -> $OUT"
  "$PY" "$ROOT/scripts/prune-dual.py" --dual "$DUAL" --vram-gib "$VRAM_GIB" \
        --reserve-gib "$RESERVE_GIB" --out "$OUT" || exit 1
  if [ "$DUAL" != "$OUT" ]; then
    echo "[install] removing the excess dual ($(du -h "$DUAL" | cut -f1)): $DUAL"
    rm -f "$DUAL"
  fi
  cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
# fitted launcher: single right-sized model for ${VRAM_GIB} GiB.
exec "\${NX2_TOOL:-llama-cli}" -m "$OUT" "\$@"
EOF
  chmod +x "$LAUNCH"
  echo "[install] fitted model: $OUT ($(du -h "$OUT" | cut -f1))"
  echo "[install] launcher: $LAUNCH"
fi
echo "[install] done. Run:  $LAUNCH -p \"Hello\" -n 32"
