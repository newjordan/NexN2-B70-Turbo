#!/usr/bin/env bash
# docker-build.sh — TURNKEY build. No host oneAPI needed.
#
# Builds a llama.cpp SYCL image (Intel oneAPI + GPU runtime bundled inside) carrying
# the Nex-N2-mini IQ3_A770 + Phase-Twin loader. The ONLY prerequisites are Docker and
# an Intel Arc GPU with the kernel driver (i915/xe) on the host. Everything else —
# the icpx/icx compiler, MKL, Level-Zero, the compute runtime — lives in the image.
#
#   ./docker-build.sh                 # build image 'nexn2-turbo'
#   IMAGE=myname ./docker-build.sh    # custom tag
#
# Why Docker: the bare-metal ./build.sh needs Intel oneAPI installed and sourced on the
# host. If you don't have oneAPI (or hit oneAPI/Dockerfile version errors), use THIS.
set -euo pipefail

BASE=f0156d1401500512ad85042ccf38970568b12253          # pinned llama.cpp base the patch applies on
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/llama.cpp-turbo-phase-twin.patch"          # includes the .devops/intel.Dockerfile build recipe
IMAGE="${IMAGE:-nexn2-turbo}"
SRC="${SRC:-$HERE/llama.cpp}"                           # where llama.cpp is cloned

command -v docker >/dev/null || { echo "error: docker not found — install Docker first." >&2; exit 1; }
command -v git    >/dev/null || { echo "error: git not found." >&2; exit 1; }
[[ -f "$PATCH" ]] || { echo "error: patch not found at $PATCH" >&2; exit 1; }

# 1) pristine llama.cpp at the pinned base (idempotent: re-runnable)
if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$BASE" 2>/dev/null || git -C "$SRC" fetch origin
git -C "$SRC" checkout -q "$BASE"
git -C "$SRC" reset  -q --hard "$BASE"
git -C "$SRC" clean  -fdq

# 2) apply the single patch (source stack + the lean SYCL Dockerfile — nothing to hand-edit)
echo "applying Turbo Phase Twin patch (source + Dockerfile) ..."
git -C "$SRC" apply --3way "$PATCH" || { echo "error: patch failed to apply on $BASE" >&2; exit 1; }

# 3) build the image (compiles llama.cpp inside the oneAPI container; first run pulls a few GB)
echo "building image '$IMAGE' — compiling llama.cpp (SYCL) inside oneAPI ..."
docker build -f "$SRC/.devops/intel.Dockerfile" -t "$IMAGE" "$SRC"

cat <<EOF

============================================================================
done -> image '$IMAGE'

Run the server (1-card IQ3 default; OpenAI-compatible API on http://localhost:8090):

  docker run --rm -it --device /dev/dri \\
    -v "\$PWD":/models -p 8090:8080 \\
    $IMAGE -m /models/Nex-N2-mini-Turbo-Phase-Twin.gguf -ngl 99 --jinja

(Server listens on the image's default 8080; -p maps it to host 8090 so the built-in
healthcheck stays green. \$PWD is mounted at /models — run from the folder with the .gguf.)

Two Arc cards (Q4 phase, near-lossless) — add these flags to the run:
    -sm layer -ts 1,1 --override-kv general.tensor_variant.default=int:1

If the GPU isn't visible inside the container, add:
    --group-add "\$(getent group render | cut -d: -f3)"

(\$PWD is mounted at /models — run the command from the folder holding the .gguf.)
============================================================================
EOF
