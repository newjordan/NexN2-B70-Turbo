#!/usr/bin/env python3
"""Derive an 8-entry non-uniform int8 LUT for IQ3_NL from NX2 expert weights.

Samples expert weights, normalizes each 32-elt sub-block by its abs-max, pools the
normalized values, and runs 1-D Lloyd-Max (k-means) to find 8 levels matched to the
weight distribution. Scales the centers to int8 (max |level| -> LMAX) for a
dp4a-friendly LUT. Prints the LUT to paste into the C type.
"""
import sys, numpy as np
sys.path.insert(0, "/home/frosty40/llama.cpp/gguf-py")
from gguf.gguf_reader import GGUFReader

LMAX = 127         # full int8 range for the 8 levels (fine placement; matches kvalues_iq4nl)
NLEV = 8           # 3-bit
SUB  = 32          # sub-block size (matches L3)
M = "/home/frosty40/models/nex-n2-mini/NX2-bf16.gguf"

def bf16_to_f32(u8):              # raw bf16 bytes -> f32
    u16 = u8.view(np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

r = GGUFReader(M)
# sample a few expert tensors across layers; take a subset of experts from each
want = [f"blk.{il}.ffn_{p}_exps.weight" for il in (0, 13, 26, 39) for p in ("gate", "down")]
vals = []
for t in r.tensors:
    if t.name in want:
        x = bf16_to_f32(t.data)            # numpy axis0 = expert
        x = x[:24].reshape(-1)             # first 24 experts, flatten
        vals.append(x.astype(np.float32))
w = np.concatenate(vals)
w = w[np.isfinite(w)]
print(f"sampled {w.size/1e6:.1f}M weights from {len(want)} tensors")

# normalize per 32-elt sub-block by abs-max
n = (w.size // SUB) * SUB
b = w[:n].reshape(-1, SUB)
amax = np.abs(b).max(axis=1, keepdims=True)
amax[amax == 0] = 1.0
norm = (b / amax).reshape(-1)              # in [-1, 1]
# subsample for speed
rs = np.random.default_rng(0)
s = norm[rs.integers(0, norm.size, size=min(norm.size, 4_000_000))]

# 1-D Lloyd-Max, init at uniform quantiles
c = np.quantile(s, (np.arange(NLEV) + 0.5) / NLEV)
for _ in range(50):
    # assign to nearest center
    idx = np.abs(s[:, None] - c[None, :]).argmin(axis=1)
    newc = np.array([s[idx == k].mean() if np.any(idx == k) else c[k] for k in range(NLEV)])
    if np.allclose(newc, c, atol=1e-6): c = newc; break
    c = newc

# scale centers to int8 LUT (max |center| -> LMAX), round, dedupe-safe
scale = LMAX / np.abs(c).max()
lut = np.clip(np.round(c * scale), -127, 127).astype(np.int64)
# report quantization MSE: non-uniform LUT vs uniform {-4..3}
def mse(levels):
    lv = np.asarray(levels, float); lv = lv / np.abs(lv).max()
    q = np.abs(s[:, None] - lv[None, :]).argmin(axis=1)
    return float(((s - lv[q]) ** 2).mean())
uni = np.arange(NLEV) - 4   # {-4..3}
print("centers (norm):", np.round(c, 4).tolist())
print("LUT int8      :", lut.tolist())
print(f"MSE non-uniform={mse(lut):.5f}  uniform(q-4)={mse(uni):.5f}  "
      f"reduction={100*(1-mse(lut)/mse(uni)):.1f}%")
print("\nC array:\nstatic const int8_t kvalues_iq3nl[8] = { " + ", ".join(str(v) for v in lut.tolist()) + " };")
