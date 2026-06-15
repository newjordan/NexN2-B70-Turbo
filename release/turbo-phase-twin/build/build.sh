#!/usr/bin/env bash
# build.sh — build llama.cpp with the IQ3_A770 + Phase-Twin loader for Nex-N2-mini.
#
# Requires Intel oneAPI (icpx/icx, e.g. /opt/intel/oneapi) for the SYCL backend.
# Produces ./llama.cpp/build/bin/{llama-server,llama-cli,llama-bench}. Run from anywhere;
# clones llama.cpp into the current directory.
#
#   ./build.sh                 # clone + checkout + patch + build
#   ONEAPI=/opt/intel/oneapi/setvars.sh ./build.sh
set -euo pipefail

BASE=f0156d1401500512ad85042ccf38970568b12253       # pinned llama.cpp base the patch applies on
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/llama.cpp-turbo-phase-twin.patch"

: "${ONEAPI:=/opt/intel/oneapi/setvars.sh}"
if [[ -f "$ONEAPI" ]]; then
  # shellcheck disable=SC1090
  source "$ONEAPI"
else
  echo "warning: oneAPI setvars not found at $ONEAPI — ensure icpx/icx are on PATH" >&2
fi
command -v icpx >/dev/null || { echo "error: icpx not found. Install Intel oneAPI and source setvars.sh." >&2; exit 1; }

if [[ ! -d llama.cpp ]]; then
  git clone https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp
git fetch --depth 1 origin "$BASE" 2>/dev/null || git fetch origin
git checkout -q "$BASE"
echo "applying Turbo Phase Twin patch..."
git apply --3way "$PATCH" || { echo "patch failed to apply on $BASE" >&2; exit 1; }

cmake -B build -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx
cmake --build build -j --target llama-server llama-cli llama-bench

echo
echo "done -> $(pwd)/build/bin"
echo "now run the model from the repo root, e.g.:"
echo "  ./run-nx2.sh --sm off -- -p 'The capital of France is' -n 16"
