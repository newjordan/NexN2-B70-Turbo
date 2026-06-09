#!/usr/bin/env bash
###############################################################################
# nx2-phase1-sweep.sh
#
# Phase-1 accuracy+speed sweep of Nex-N2-mini quantizations on an Intel Arc B70
# (llama.cpp SYCL build).  No shims: every quant is produced from the bf16
# source with the imatrix, then measured for KLD/PPL/top-1 agreement (accuracy)
# and pp512/tg128 throughput (speed).
#
# Idempotent: completed quantize steps (atomic tmp->mv) and completed
# measurements (per-quant .row markers) are skipped on re-run.
# Robust: per-step logs under ~/nx2-sweep-logs/, no `set -e` so one failing
# quant never aborts the campaign.
#
# ---------------------------------------------------------------------------
# FLAGS VERIFIED against the real binaries (`--help`) + llama.cpp source:
#
#   llama-quantize:
#     --imatrix <file>                         importance matrix
#     --output-tensor-type <ggml_type>         type for output.weight
#     --token-embedding-type <ggml_type>       type for token_embd.weight
#     --tensor-type <name>=<ggml_type>         per-tensor override; <name> is a
#                                              std::regex matched with
#                                              regex_search (substring), name is
#                                              lower-cased, FIRST match wins.
#                                              <ggml_type> is matched case-
#                                              INSENSITIVELY against GGML *base*
#                                              type names (q3_K,q4_K,q5_K,q6_K,
#                                              q8_0,iq4_xs,...) -- NOT _M/_S
#                                              mixtures.  Positional may repeat.
#     positional:  <src.gguf> <out.gguf> <TYPE> [nthreads]
#     NOTE: quantize does NOT take --override-kv here (just copies tensors);
#           and llama-quant.cpp force-keeps  ffn_gate_inp.weight (router) and
#           *_norm.weight  as F32 -- a --tensor-type override on those is a
#           no-op (override only fires when default_type is already quantized).
#
#   llama-perplexity:
#     -m <model>   -f <file>   -ngl/--n-gpu-layers <N>
#     --override-kv KEY=TYPE:VALUE       (use exactly ONE; repeating keeps last)
#     --kl-divergence                    compute KLD vs the base logits file
#     --kl-divergence-base <FNAME>       (alias --save-all-logits): WITHOUT
#                                        --kl-divergence it WRITES the base
#                                        logits; WITH it, it READS them.
#     stdout summary lines parsed (exact spellings from perplexity.cpp):
#       "Mean PPL(Q)   : <v> ± ..."     -> PPL
#       "Mean    KLD: <v> ± ..."        -> mean KLD
#       "99.9%   KLD: <v>"              -> 99.9% KLD
#       "Same top p: <v> ± ... %"       -> top-1 token agreement %
#
#   llama-bench:
#     -m <model>  -ngl <N>  -p <n_prompt>  -n <n_gen>  -t <threads>
#     --override-kv KEY=TYPE:VALUE
#     -o json   (parsed with jq; CSV mode quotes every field but does NOT
#                escape embedded commas, so jq/json is the safe parser).
#       rows: {n_prompt:512,n_gen:0}->pp_ts=avg_ts ; {n_prompt:0,n_gen:128}->tg_ts
#       also read .model_size and .model_n_params for bpw.
###############################################################################

set -o pipefail
umask 022

# --------------------------------------------------------------------------- #
# Paths / environment
# --------------------------------------------------------------------------- #
BIN="$HOME/llama.cpp/build/bin"
QUANTIZE="$BIN/llama-quantize"
PERPLEXITY="$BIN/llama-perplexity"
BENCH="$BIN/llama-bench"

MODELDIR="$HOME/models/nex-n2-mini"
SRC="$MODELDIR/NX2-bf16.gguf"          # bf16 quant source
IMAT="$MODELDIR/NX2.imatrix"           # importance matrix
KLDBASE="$MODELDIR/NX2-Q6K.kld"        # KLD base (Q6_K reference logits, 100 chunks)
Q8REF="$MODELDIR/NX2-Q6_K.gguf"        # used to (re)generate KLDBASE if missing
WIKI="$HOME/nx2-eval/wikitext-2-raw/wiki.test.raw"

OUTDIR="$MODELDIR/sweep"               # quantized GGUFs land here
LOGDIR="$HOME/nx2-sweep-logs"          # per-step logs + .row markers
RESULTS="$HOME/nx2-results.csv"

OVERRIDE_KV="qwen35moe.block_count=int:40"   # MTP block-40 load fix (exactly one)
CHUNKS=100                             # must match the KLD base's chunk count
NGL=99
NTHREADS_Q=32                          # quantize worker threads
PP=512; NG=128; BT=1                   # bench: -p 512 -n 128 -t 1

mkdir -p "$OUTDIR" "$LOGDIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGDIR/sweep.log"; }
die() { log "FATAL: $*"; exit 1; }

# --------------------------------------------------------------------------- #
# oneAPI / SYCL runtime
# --------------------------------------------------------------------------- #
log "Sourcing Intel oneAPI environment ..."
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >"$LOGDIR/setvars.log" 2>&1 || \
    die "could not source /opt/intel/oneapi/setvars.sh (see $LOGDIR/setvars.log)"
set -u 2>/dev/null || true
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
for b in "$QUANTIZE" "$PERPLEXITY" "$BENCH"; do
    [[ -x "$b" ]] || die "missing binary: $b"
done
command -v jq  >/dev/null 2>&1 || die "jq is required to parse llama-bench json"
command -v awk >/dev/null 2>&1 || die "awk is required"

[[ -f "$SRC"  ]] || die "quant source not found: $SRC"
[[ -f "$IMAT" ]] || die "imatrix not found: $IMAT  (generate it before running)"
[[ -f "$WIKI" ]] || die "wikitext eval file not found: $WIKI"

# Generate the KLD base from the Q8_0 reference if it does not exist yet.
# (Needs the GPU; this is part of the measurement phase.)
if [[ ! -f "$KLDBASE" ]]; then
    if [[ -f "$Q8REF" ]]; then
        log "KLD base missing -> generating from $Q8REF (this is a full PPL pass)"
        klog="$LOGDIR/kldbase-gen.log"
        if "$PERPLEXITY" \
                -m "$Q8REF" -f "$WIKI" -ngl "$NGL" \
                --override-kv "$OVERRIDE_KV" \
                --kl-divergence-base "$KLDBASE.tmp" >"$klog" 2>&1; then
            mv -f "$KLDBASE.tmp" "$KLDBASE"
            log "KLD base written: $KLDBASE"
        else
            rm -f "$KLDBASE.tmp"
            die "failed to generate KLD base (see $klog)"
        fi
    else
        die "KLD base $KLDBASE missing and reference $Q8REF not present to build it"
    fi
fi

# --------------------------------------------------------------------------- #
# Quant definitions
# --------------------------------------------------------------------------- #
# Plain imatrix quants
QUANTS=(Q5_K_M Q4_K_M IQ4_XS Q3_K_M Q3_K_S)

# Dynamic (Unsloth-Dynamic style) mixes.  Base = low-bit experts; sensitive
# tensors (down-projection experts, shared expert, attention, embeddings,
# output) bumped to higher precision.  Router (ffn_gate_inp) is already F32.
# NOTE: --tensor-type types must be GGML *base* types (q5_K not Q5_K_M).
DYN_Q3_BASE="Q3_K"
DYN_Q3_FLAGS=(
    --token-embedding-type Q6_K
    --output-tensor-type    Q6_K
    --tensor-type attn_q=Q5_K
    --tensor-type attn_k=Q5_K
    --tensor-type attn_v=Q6_K
    --tensor-type attn_output=Q5_K
    --tensor-type ffn_down_exps=Q5_K
    --tensor-type ffn_gate_exps=Q3_K
    --tensor-type ffn_up_exps=Q3_K
    --tensor-type ffn_down_shexp=Q6_K
    --tensor-type ffn_gate_shexp=Q5_K
    --tensor-type ffn_up_shexp=Q5_K
)

DYN_Q4_BASE="Q4_K"
DYN_Q4_FLAGS=(
    --token-embedding-type Q6_K
    --output-tensor-type    Q6_K
    --tensor-type attn_q=Q6_K
    --tensor-type attn_k=Q6_K
    --tensor-type attn_v=Q6_K
    --tensor-type attn_output=Q6_K
    --tensor-type ffn_down_exps=Q6_K
    --tensor-type ffn_gate_exps=Q4_K
    --tensor-type ffn_up_exps=Q4_K
    --tensor-type ffn_down_shexp=Q6_K
    --tensor-type ffn_gate_shexp=Q6_K
    --tensor-type ffn_up_shexp=Q6_K
)

# label -> output gguf
declare -A GGUF
for q in "${QUANTS[@]}"; do GGUF["$q"]="$OUTDIR/NX2-$q.gguf"; done
GGUF["Q3_K_dyn"]="$OUTDIR/NX2-Q3_K_dyn.gguf"
GGUF["Q4_K_dyn"]="$OUTDIR/NX2-Q4_K_dyn.gguf"

# CSV / table ordering
ALL_LABELS=("${QUANTS[@]}" "Q4_K_dyn" "Q3_K_dyn")

# --------------------------------------------------------------------------- #
# Quantize helpers (atomic tmp -> mv; skip if final output exists)
# --------------------------------------------------------------------------- #
quantize_one() {                       # $1=label  $2=type  $3..=extra flags
    local label="$1" type="$2"; shift 2
    local out="${GGUF[$label]}" tmp="${GGUF[$label]}.tmp"
    local qlog="$LOGDIR/quantize-$label.log"

    if [[ -s "$out" ]]; then
        log "quantize[$label]: output exists, skipping ($out)"
        return 0
    fi
    rm -f "$tmp"
    log "quantize[$label]: building ($type) -> $out"
    if "$QUANTIZE" --imatrix "$IMAT" "$@" "$SRC" "$tmp" "$type" "$NTHREADS_Q" \
            >"$qlog" 2>&1; then
        mv -f "$tmp" "$out"
        log "quantize[$label]: done ($(du -h "$out" | cut -f1))"
        return 0
    else
        rm -f "$tmp"
        log "quantize[$label]: FAILED (see $qlog)"
        return 1
    fi
}

# --------------------------------------------------------------------------- #
# Measurement helper
#   writes a tab-separated .row marker on success:
#   label  size_GiB  bpw  PPL  mean_KLD  KLD_999  top1%  pp_ts  tg_ts  vram  gbps
# --------------------------------------------------------------------------- #
num_or_na() { local v="$1"; [[ -n "$v" && "$v" != "null" ]] && echo "$v" || echo "NA"; }

measure_one() {                        # $1=label
    local label="$1" g="${GGUF[$1]}"
    local row="$LOGDIR/$label.row"
    local pplog="$LOGDIR/perplexity-$label.log"
    local bjson="$LOGDIR/bench-$label.json"
    local blog="$LOGDIR/bench-$label.log"

    if [[ -f "$row" ]]; then
        log "measure[$label]: already measured, skipping"
        return 0
    fi
    if [[ ! -s "$g" ]]; then
        log "measure[$label]: gguf missing, skipping ($g)"
        return 1
    fi

    # ---- Accuracy: KLD / PPL / top-1 agreement ----------------------------
    log "measure[$label]: perplexity --kl-divergence"
    if ! "$PERPLEXITY" \
            -m "$g" -f "$WIKI" -ngl "$NGL" --chunks "$CHUNKS" \
            --override-kv "$OVERRIDE_KV" \
            --kl-divergence --kl-divergence-base "$KLDBASE" \
            >"$pplog" 2>&1; then
        log "measure[$label]: perplexity FAILED (see $pplog)"
        return 1
    fi

    local ppl mkld kld999 top1
    ppl=$(grep -m1 -E '^Mean PPL\(Q\) +:'   "$pplog" | sed -E 's/^[^:]*:[[:space:]]*//' | awk '{print $1}')
    mkld=$(grep -m1 -E '^Mean +KLD:'        "$pplog" | sed -E 's/^[^:]*:[[:space:]]*//' | awk '{print $1}')
    kld999=$(grep -m1 -E '^99\.9%[[:space:]]+KLD:' "$pplog" | sed -E 's/^[^:]*:[[:space:]]*//' | awk '{print $1}')
    top1=$(grep -m1 -E '^Same top p:'       "$pplog" | sed -E 's/^[^:]*:[[:space:]]*//' | awk '{print $1}')

    # approximate device VRAM: sum the MiB preceding "MiB" on SYCL buffer lines
    local vram_mib vram
    vram_mib=$(awk 'tolower($0) ~ /sycl.*buffer size/ { for (i=1;i<=NF;i++) if ($i=="MiB") s+=$(i-1) } END { printf "%.1f", s }' "$pplog")
    vram=$(awk -v m="$vram_mib" 'BEGIN{ if (m+0>0) printf "%.2f", m/1024; else print "NA" }')

    # ---- Speed: pp512 / tg128 --------------------------------------------
    log "measure[$label]: llama-bench -p $PP -n $NG"
    if ! "$BENCH" \
            -m "$g" -ngl "$NGL" \
            -p "$PP" -n "$NG" -t "$BT" -o json \
            >"$bjson" 2>"$blog"; then
        log "measure[$label]: llama-bench FAILED (see $blog)"
        return 1
    fi

    local pp_ts tg_ts msize nparams
    pp_ts=$(jq -r 'map(select(.n_prompt=='"$PP"' and .n_gen==0))[0].avg_ts // empty'  "$bjson" 2>/dev/null)
    tg_ts=$(jq -r 'map(select(.n_prompt==0 and .n_gen=='"$NG"'))[0].avg_ts // empty'  "$bjson" 2>/dev/null)
    msize=$(jq -r '.[0].model_size // empty'     "$bjson" 2>/dev/null)
    nparams=$(jq -r '.[0].model_n_params // empty' "$bjson" 2>/dev/null)

    # ---- Derived: size on disk, bpw, effective GB/s -----------------------
    local bytes size_gib bpw gbps
    bytes=$(stat -c%s "$g" 2>/dev/null)
    size_gib=$(awk -v b="${bytes:-0}" 'BEGIN{ printf "%.2f", b/1073741824 }')
    if [[ -n "$msize" && -n "$nparams" && "$nparams" != "0" ]]; then
        bpw=$(awk -v s="$msize" -v n="$nparams" 'BEGIN{ printf "%.3f", (s*8)/n }')
    else
        bpw="NA"
    fi
    if [[ -n "$tg_ts" ]]; then
        gbps=$(awk -v s="$size_gib" -v t="$tg_ts" 'BEGIN{ printf "%.1f", s*t }')
    else
        gbps="NA"
    fi

    ppl=$(num_or_na "$ppl");   mkld=$(num_or_na "$mkld")
    kld999=$(num_or_na "$kld999"); top1=$(num_or_na "$top1")
    pp_ts=$(num_or_na "$pp_ts"); tg_ts=$(num_or_na "$tg_ts")

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$size_gib" "$bpw" "$ppl" "$mkld" "$kld999" "$top1" \
        "$pp_ts" "$tg_ts" "$vram" "$gbps" > "$row"
    log "measure[$label]: PPL=$ppl meanKLD=$mkld KLD99.9=$kld999 top1=$top1% pp=$pp_ts tg=$tg_ts t/s vram~${vram}GiB"
    return 0
}

# --------------------------------------------------------------------------- #
# PHASE A: quantize (CPU/IO heavy; does not need the GPU)
# --------------------------------------------------------------------------- #
log "=== PHASE A: quantize ==="
for q in "${QUANTS[@]}"; do
    quantize_one "$q" "$q"
done
quantize_one "Q4_K_dyn" "$DYN_Q4_BASE" "${DYN_Q4_FLAGS[@]}"
quantize_one "Q3_K_dyn" "$DYN_Q3_BASE" "${DYN_Q3_FLAGS[@]}"

# --------------------------------------------------------------------------- #
# PHASE B: measure (needs the B70 free)
# --------------------------------------------------------------------------- #
log "=== PHASE B: measure (accuracy + speed) ==="
for label in "${ALL_LABELS[@]}"; do
    measure_one "$label"
done

# --------------------------------------------------------------------------- #
# PHASE C: assemble CSV + pretty table
# --------------------------------------------------------------------------- #
log "=== PHASE C: results ==="
echo "quant,size_GiB,bpw,PPL,mean_KLD,KLD_999,top1_agree_pct,pp_ts,tg_ts" > "$RESULTS"
rows_tmp="$(mktemp)"
for label in "${ALL_LABELS[@]}"; do
    row="$LOGDIR/$label.row"
    [[ -f "$row" ]] || { log "results: no row for $label (not measured)"; continue; }
    # CSV: first 9 fields, comma-separated
    cut -f1-9 "$row" | tr '\t' ',' >> "$RESULTS"
    cat "$row" >> "$rows_tmp"
done
log "CSV written: $RESULTS"

# Pretty aligned table, sorted by tg_ts (field 9) descending.
echo
echo "================================ Nex-N2-mini sweep ================================"
{
    printf '%-10s %8s %6s %9s %9s %9s %7s %9s %9s %9s %9s\n' \
        quant size_GiB bpw PPL mean_KLD KLD_999 top1% pp_t/s tg_t/s ~VRAM_GiB "~GB/s"
    sort -t$'\t' -k9,9 -gr "$rows_tmp" | \
    while IFS=$'\t' read -r q sz bpw ppl mk k999 t1 pp tg vr gb; do
        printf '%-10s %8s %6s %9s %9s %9s %7s %9s %9s %9s %9s\n' \
            "$q" "$sz" "$bpw" "$ppl" "$mk" "$k999" "$t1" "$pp" "$tg" "$vr" "$gb"
    done
} | tee -a "$LOGDIR/sweep.log"
echo "=================================================================================="
echo "(~GB/s = size_GiB * tg_ts, APPROXIMATE effective decode bandwidth)"
rm -f "$rows_tmp"
log "Done."
