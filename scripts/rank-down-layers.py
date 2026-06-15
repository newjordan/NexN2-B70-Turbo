#!/usr/bin/env python3
"""Rank the 40 ffn_down_exps layers by imatrix activation importance.

Score per layer = mean over input dims of (in_sum2 / counts) = mean activation^2
seen during calibration -- the exact importance signal the K-quant scale search
uses. Higher => quantization error on that down-proj propagates more => spend the
Q4_K bits there first. Emits a ranked layer list and ready-to-paste --tensor-type
override flags for the top-K layers.
"""
import sys, numpy as np
sys.path.insert(0, "/home/frosty40/llama.cpp/gguf-py")
from gguf.gguf_reader import GGUFReader

IM = "/home/frosty40/models/nex-n2-mini/NX2.imatrix"
r = GGUFReader(IM)
by = {t.name: t for t in r.tensors}

scores = {}
for il in range(40):
    s = by.get(f"blk.{il}.ffn_down_exps.weight.in_sum2")
    c = by.get(f"blk.{il}.ffn_down_exps.weight.counts")
    if s is None:
        continue
    sv = np.asarray(s.data, dtype=np.float64)
    cv = np.asarray(c.data, dtype=np.float64) if c is not None else np.array([1.0])
    cnt = max(float(cv.flat[0]) if cv.size else 1.0, 1.0)
    scores[il] = float(sv.sum()) / cnt / sv.size   # mean activation^2 per input dim

ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
vals = np.array([scores[l] for l in ranked])
print("# down-proj layers ranked by imatrix importance (mean act^2):")
print("# rank: layer  score   (relative to max)")
for i, l in enumerate(ranked):
    print(f"#  {i:2d}:  blk.{l:<2d}  {scores[l]:.4e}  {scores[l]/vals[0]:.3f}")
print(f"# spread: max/min = {vals[0]/vals[-1]:.1f}x")

# emit ranked layer order (most->least important) for the sweep script
print("RANK_ORDER=\"" + " ".join(str(l) for l in ranked) + "\"")
