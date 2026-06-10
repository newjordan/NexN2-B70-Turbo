#!/usr/bin/env bash
# Overnight NIAH Pareto sweep: relaunches serve.sh per server config and runs
# niah_sweep.py at each context length. Fully resumable — niah_sweep.py skips
# (length, depth, config, sample) rows already present in the CSV.
#
# Env knobs:
#   GATE_512K=on|off   include >256k configs (set from the T3 smoke verdict; default off)
#   IQ4_512K=on|off    include the IQ4_XS+f16 512k production candidate (default off)
#   PHASE=coarse|dense coarse: depths 10,50,90 x1 sample; dense: full depth set x3 samples
#   CSV=...            output csv (default results/niah-pareto.csv)
#
# Run via Bash dangerouslyDisableSandbox (serve.sh launches GPU loads) in the
# background; check the quiet box first.
set -o pipefail
set -u

cd /home/frosty40/nx2-b70-turbo
PY=/home/frosty40/nx2-venv/bin/python
SERVE=eval/niah/serve.sh
CSV="${CSV:-results/niah-pareto.csv}"
GATE_512K="${GATE_512K:-off}"
IQ4_512K="${IQ4_512K:-off}"
PHASE="${PHASE:-coarse}"
IQ4_MODEL=/home/frosty40/models/nex-n2-mini/sweep/NX2-IQ4_XS.gguf

if [ "$PHASE" = dense ]; then
  DEPTHS="0,10,25,50,75,90,100"; SAMPLES="0 1 2"
else
  DEPTHS="10,50,90"; SAMPLES="0"
fi

# config rows: label|kv|ctx|serve-env...
CONFIGS=(
  "native-131k|f16|131072|CTX=131072 KV=f16"
  "native|f16|262144|CTX=262144 KV=f16"
  "native|q8_0|262144|CTX=262144 KV=q8_0"
)
if [ "$GATE_512K" = on ]; then
  CONFIGS+=(
    "yarn1.5|q8_0|393216|CTX=393216 KV=q8_0 ROPE_SCALING=yarn ROPE_SCALE=1.5 YARN_ORIG_CTX=262144"
    "yarn2|q8_0|524288|CTX=524288 KV=q8_0 ROPE_SCALING=yarn ROPE_SCALE=2 YARN_ORIG_CTX=262144"
    "linear2|q8_0|524288|CTX=524288 KV=q8_0 ROPE_SCALING=linear ROPE_SCALE=2"
  )
fi
if [ "$IQ4_512K" = on ]; then
  CONFIGS+=(
    "yarn2-iq4xs|f16|524288|CTX=524288 KV=f16 ROPE_SCALING=yarn ROPE_SCALE=2 YARN_ORIG_CTX=262144 MODEL=$IQ4_MODEL"
  )
fi

# 260000/520000 (not 262144/524288) so rows from the T3 smoke resume-skip
# instead of re-prefilling the most expensive points
LENGTHS=(32768 65536 131072 196608 260000 327680 393216 458752 520000)

quiet_box() {
  local load
  load=$(awk '{print int($1)}' /proc/loadavg)
  if [ "$load" -ge 4 ] || pgrep -f 'hunt' | grep -v $$ >/dev/null 2>&1; then
    echo "[sweep] WARNING: box not quiet (load $load) — timings may be skewed" >&2
  fi
}

vram_log() {
  local pid
  pid=$(cat /home/frosty40/nx2-niah-run/server.pid 2>/dev/null) || return 0
  grep -h 'drm-total-vram0' /proc/"$pid"/fdinfo/* 2>/dev/null | sort -u \
    | awk -v c="$1" '{printf "[sweep] vram %s: %.1f GiB\n", c, $2/1048576}' >&2
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label kv ctx envs <<<"$cfg"
  echo "[sweep] ===== config $label/$kv ctx=$ctx =====" >&2
  quiet_box
  if ! env $envs bash "$SERVE"; then
    echo "[sweep] FATAL: serve.sh failed for $label/$kv — skipping config" >&2
    continue
  fi
  vram_log "$label/$kv"
  for L in "${LENGTHS[@]}"; do
    [ "$L" -gt "$ctx" ] && continue
    # always leave room for the question + generation under the slot ctx
    target=$(( L <= ctx - 4096 ? L : ctx - 4096 ))
    for s in $SAMPLES; do
      echo "[sweep] --- $label/$kv length $target sample $s ---" >&2
      if ! "$PY" eval/niah/niah_sweep.py --tokens "$target" --depths "$DEPTHS" \
            --sample "$s" --rope-config "$label" --kv-type "$kv" \
            --csv "$CSV" --timeout 10800; then
        echo "[sweep] ERROR: probe failed ($label/$kv $target s$s); continuing" >&2
      fi
    done
  done
done
echo "[sweep] sweep complete -> $CSV" >&2
