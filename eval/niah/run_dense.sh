#!/usr/bin/env bash
# Dense validation pass: the coarse sweep found NO accuracy cliff (138/138),
# so densify at the two claimed operating points instead — full depth set,
# 3 samples (distinct haystacks -> one full prefill each):
#   1. yarn2-iq4xs/f16 @ ~520k  (512k production candidate; server already up)
#   2. native Q5/f16   @ ~258k  (native ceiling claim)
#   3. yarn2/q8_0      @ ~520k  sample 1 (cross-quant confirmation, 1 extra sample)
set -o pipefail
set -u
cd /home/frosty40/nx2-b70-turbo
PY=/home/frosty40/nx2-venv/bin/python
CSV=results/niah-pareto.csv
DEPTHS="0,10,25,50,75,90,100"

probe() { # tokens rope kv sample
  "$PY" eval/niah/niah_sweep.py --tokens "$1" --depths "$DEPTHS" \
    --sample "$4" --rope-config "$2" --kv-type "$3" --csv "$CSV" \
    --timeout 10800 \
    || echo "[dense] ERROR: $2/$3 @$1 s$4 failed; continuing" >&2
}

echo "[dense] phase 1: iq4xs @520k x3 samples (server assumed up)" >&2
for s in 0 1 2; do probe 520000 yarn2-iq4xs f16 "$s"; done

echo "[dense] phase 2: native f16 @258048 x3 samples" >&2
CTX=262144 KV=f16 bash eval/niah/serve.sh || exit 1
for s in 0 1 2; do probe 258048 native f16 "$s"; done

echo "[dense] phase 3: yarn2/q8_0 @520k sample 1" >&2
CTX=524288 KV=q8_0 ROPE_SCALING=yarn ROPE_SCALE=2 YARN_ORIG_CTX=262144 bash eval/niah/serve.sh || exit 1
probe 520000 yarn2 q8_0 1

echo "[dense] dense pass complete -> $CSV" >&2
