#!/usr/bin/env bash
# Full validation matrix for the SYCL reorder-on-MoE patch.
# A/B: base = origin/master (no patch), pr = sycl-moe-reorder (patch only).
# Builds each in its own dir, then runs correctness + perf + PPL.
#
# GPU-exclusive: do not run while anything else is using the B70 or while a
# timing-sensitive eval is running (CPU Delta-Net contention).
# Run via Bash dangerouslyDisableSandbox, in the background (hours).
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u

LLAMA=/home/frosty40/llama.cpp
OUT=/home/frosty40/nx2-b70-turbo/results/upstream-pr
Q4=/home/frosty40/models/nex-n2-mini/sweep/NX2-Q4_K_M.gguf
Q5=/home/frosty40/models/nex-n2-mini/sweep/NX2-Q5_K_M.gguf
WIKI=/home/frosty40/nx2-eval/wikitext-2-raw/wiki.test.raw
CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_F16=ON
             -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
PROMPT="The fundamental theorem of arithmetic states that every integer greater than one"
mkdir -p "$OUT"

step() { echo "[matrix $(date +%H:%M:%S)] $*" >&2; }

build() { # ref builddir
  local ref="$1" dir="$LLAMA/$2"
  step "building $ref -> $2"
  git -C "$LLAMA" worktree add --force "/tmp/wt-$2" "$ref" >/dev/null 2>&1 \
    || { git -C "$LLAMA" worktree remove --force "/tmp/wt-$2" 2>/dev/null;
         git -C "$LLAMA" worktree add --force "/tmp/wt-$2" "$ref" >/dev/null; }
  cmake -S "/tmp/wt-$2" -B "$dir" "${CMAKE_FLAGS[@]}" >"$OUT/cmake-$2.log" 2>&1 \
    || { step "FATAL cmake $2"; return 1; }
  cmake --build "$dir" -j"$(nproc)" \
    --target llama-server llama-completion llama-bench llama-perplexity test-backend-ops \
    >"$OUT/build-$2.log" 2>&1 || { step "FATAL build $2"; return 1; }
}

greedy() { # bin model disable_opt tag
  GGML_SYCL_DISABLE_OPT="$3" "$1/bin/llama-completion" -m "$2" -ngl 99 -fa off \
    --temp 0 -s 0 -n 64 -p "$PROMPT" -no-cnv 2>/dev/null \
    | tail -n +2 > "$OUT/greedy-$4.txt"
}

# ---- builds (sequential; each saturates the CPU) -------------------------
build origin/master build-base || exit 1
build sycl-moe-reorder build-pr || exit 1
BASE="$LLAMA/build-base" PR="$LLAMA/build-pr"

# ---- 1. operator correctness vs CPU reference ----------------------------
step "test-backend-ops MUL_MAT_ID (pr)"
"$PR/bin/test-backend-ops" test -o MUL_MAT_ID > "$OUT/backend-ops-mulmatid-pr.log" 2>&1
echo "MUL_MAT_ID(pr): $(grep -cE 'OK$' "$OUT/backend-ops-mulmatid-pr.log") OK, $(grep -cE 'FAIL' "$OUT/backend-ops-mulmatid-pr.log") FAIL" >&2

step "test-backend-ops FULL (pr) — slow"
"$PR/bin/test-backend-ops" test > "$OUT/backend-ops-full-pr.log" 2>&1
echo "FULL(pr): $(grep -cE 'OK$' "$OUT/backend-ops-full-pr.log") OK, $(grep -cE 'FAIL' "$OUT/backend-ops-full-pr.log") FAIL" >&2

# ---- 2. FA-off greedy differentials ---------------------------------------
for m in Q4 Q5; do
  model_var=${m}; model=${!model_var}
  step "greedy differential $m"
  greedy "$PR"   "$model" 0 "pr-$m-opt"
  greedy "$PR"   "$model" 1 "pr-$m-noopt"
  greedy "$BASE" "$model" 1 "base-$m"
  if cmp -s "$OUT/greedy-pr-$m-opt.txt" "$OUT/greedy-pr-$m-noopt.txt"; then
    echo "greedy $m: pr reorder-on == pr reorder-off (token-identical)" >&2
  else
    echo "greedy $m: MISMATCH pr on/off — INVESTIGATE" >&2
  fi
  if cmp -s "$OUT/greedy-pr-$m-opt.txt" "$OUT/greedy-base-$m.txt"; then
    echo "greedy $m: pr == base (token-identical)" >&2
  else
    echo "greedy $m: MISMATCH pr vs base — INVESTIGATE" >&2
  fi
done

# ---- 3. perf: llama-bench base vs pr --------------------------------------
step "llama-bench (base vs pr, Q4+Q5, pp512/tg128)"
for b in base pr; do
  d=$([ "$b" = base ] && echo "$BASE" || echo "$PR")
  "$d/bin/llama-bench" -m "$Q4" -m "$Q5" -ngl 99 -fa 1 -p 512 -n 128 -o json \
    > "$OUT/bench-$b.json" 2>"$OUT/bench-$b.log"
done

# ---- 4. perplexity spot-check (Q5, ~30 chunks) ----------------------------
step "perplexity base vs pr (Q5_K_M, 30 chunks)"
for b in base pr; do
  d=$([ "$b" = base ] && echo "$BASE" || echo "$PR")
  "$d/bin/llama-perplexity" -m "$Q5" -ngl 99 -fa on -f "$WIKI" --chunks 30 \
    > "$OUT/ppl-$b.log" 2>&1
  grep -E 'Final estimate' "$OUT/ppl-$b.log" | sed "s/^/[$b] /" >&2
done

step "matrix complete -> $OUT"
