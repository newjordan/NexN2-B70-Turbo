# Building llama.cpp for the Turbo Phase Twin

This model uses a **codebook-free 3-bit expert type (`IQ3_A770`)** and a **multi-precision
loader** that are not in upstream llama.cpp. You build llama.cpp once with this repo's single
patch; after that the binaries load the model normally.

## TL;DR — two one-command builds

**Docker\* (recommended; needs only Docker + an Intel Arc GPU — no host oneAPI):**

```bash
./docker-build.sh     # builds image 'nexn2-turbo'; prints the ready docker run command
```

> \*The Docker path is new and **still under validation** (patch build-verified; the in-container
> run is in active testing on Arc B70 + by early users). The bare-metal path below is the proven
> fallback. This note is removed once the container build is confirmed green end-to-end.

**Bare-metal (only if you already have Intel oneAPI installed):**

```bash
./build.sh            # from this build/ directory (or repo root: build/build.sh)
```

Both clone llama.cpp, check out the pinned base, apply
`llama.cpp-turbo-phase-twin.patch`, and build the SYCL binaries
(`--target llama-server llama-cli llama-bench`). The patch now also carries the
`.devops/intel.Dockerfile` build recipe, so the Docker path has nothing to edit and no
oneAPI to install — the compiler and GPU runtime live inside the image.

## Prerequisites

- **Intel oneAPI** (DPC++/icpx + MKL) — the SYCL backend's compiler/runtime. Source its
  environment: `source /opt/intel/oneapi/setvars.sh`. `icpx` and `icx` must be on PATH.
- An Intel GPU with the Level-Zero runtime (Arc A770 / Arc Pro B70 / Battlemage etc.).
- cmake ≥ 3.18, git, a C++17 toolchain.

## Manual steps (what build.sh does)

```bash
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout f0156d1401500512ad85042ccf38970568b12253
git apply --3way ../llama.cpp-turbo-phase-twin.patch
cmake -B build -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx
cmake --build build -j --target llama-server llama-cli llama-bench
```

The patch is a single combined diff (the full B70-Turbo IQ3_A770 stack: SYCL MoE reorder +
fused gate/up, the `IQ3_A770` type, and the multi-precision "twin" loader). It applies
cleanly on the pinned base `f0156d140`; `--3way` lets it also apply on nearby commits when the
context still matches.

## Verify

```bash
./build/bin/llama-cli -m ../Nex-N2-mini-Turbo-Phase-Twin.gguf -ngl 99 \
  -p "The capital of France is" -n 8
```

Expect coherent output and, in the load log, the experts loading as `IQ3_A770`. With
`--override-kv general.tensor_variant.default=int:1` you'll see
`select_tensor_variant: selected tensor variant 1 (.v1)` and the Q4_K phase instead.

## Notes / troubleshooting

- **"unknown tensor type 42" / fails to load on stock llama.cpp** — expected; you must use a
  build from this patch.
- **Other GPUs (CUDA/Metal/CPU):** the `IQ3_A770` *type* has a portable CPU path, so the
  model loads and runs on CPU and on non-SYCL backends, but the fused decode speedups are
  SYCL-only. For non-Intel GPUs, the standard-quant collection
  (https://huggingface.co/Frosty40/Nex-N2-mini-B70-Turbo-GGUF) is the better choice.
- **Docker build fails right after `FROM` ("manifest unknown / not found"):** the base image
  tag must exist. Valid `intel/deep-learning-essentials` tags include
  `2025.3.3-0-devel-ubuntu24.04` (the pinned default) and `2026.0.0-devel-ubuntu24.04` — note
  2026.0.0 has **no** `-0-` segment, the 2025.x line does. Don't pass
  `--build-arg ONEAPI_VERSION=...` unless you copy an exact existing tag; the default works.
- **`CMake Error: Unknown argument --target` during the Docker build:** something joined the
  two `cmake` lines in the Dockerfile's build `RUN` with a bare `\` instead of `&& \`. Don't
  hand-edit the Dockerfile — the shipped patch already has the correct recipe; re-apply it.
- **Per-feature patches & methodology:** https://github.com/newjordan/NexN2-B70-Turbo
  (patches `0005`–`0010` for `IQ3_A770` + the loader; `0001`–`0004` for the SYCL MoE chain).
