#!/usr/bin/env python3
"""Rank every MoE expert tensor by imatrix importance / byte, for elastic precision.

For each of the 120 expert tensors (40 layers x {down,gate,up}) compute the
project-blessed importance signal used by rank-down-layers.py:

    score = mean over input dims of (in_sum2 / counts)   # mean activation^2

A tensor is *promotable* if its type differs between --base (the shipped mostly-IQ3
model) and --promote (the all-Q4_K model). Promoting IQ3/Q3_K -> Q4_K reduces quant
error in proportion to that tensor's importance, at a cost of (q4_bytes - base_bytes).
We rank promotable tensors by benefit-per-byte = score / size_delta, greedily, so any
size budget selects the tensors that buy the most quality per byte spent.

Outputs:
  - a CSV manifest (greedy order, per-tensor + cumulative size) to --out-csv
  - with --budget-gb G, a JSON list of tensor names to promote within G GB to --out-json
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp")), "gguf-py"))
from gguf.gguf_reader import GGUFReader

PROJS = ("down", "gate", "up")


def imatrix_scores(im_path, n_layer=64):
    r = GGUFReader(im_path)
    by = {t.name: t for t in r.tensors}
    scores = {}
    for il in range(n_layer):
        for proj in PROJS:
            base = f"blk.{il}.ffn_{proj}_exps.weight"
            s = by.get(base + ".in_sum2")
            c = by.get(base + ".counts")
            if s is None:
                continue
            sv = np.asarray(s.data, dtype=np.float64)
            cv = np.asarray(c.data, dtype=np.float64) if c is not None else np.array([1.0])
            cnt = max(float(cv.flat[0]) if cv.size else 1.0, 1.0)
            scores[base] = float(sv.sum()) / cnt / sv.size   # mean act^2 per input dim
    return scores


def tensor_bytes(path):
    r = GGUFReader(path)
    return {t.name: (int(t.n_bytes), str(t.tensor_type).split(".")[-1]) for t in r.tensors if "_exps." in t.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--base", required=True, help="shipped mostly-IQ3 model (variant 0)")
    ap.add_argument("--promote", required=True, help="all-Q4_K model (the promotion target)")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--budget-gb", type=float, default=None, help="emit promote-set fitting this many extra GB")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    scores = imatrix_scores(args.imatrix)
    base_b = tensor_bytes(args.base)
    prom_b = tensor_bytes(args.promote)

    rows = []
    for name, sc in scores.items():
        if name not in base_b or name not in prom_b:
            continue
        bb, btype = base_b[name]
        pb, ptype = prom_b[name]
        delta = pb - bb
        il = int(name.split(".")[1]); proj = name.split(".")[2].replace("ffn_", "").replace("_exps", "")
        promotable = delta > 0 and btype != ptype
        bpb = (sc / delta) if delta > 0 else 0.0
        rows.append(dict(name=name, layer=il, proj=proj, score=sc,
                         base_type=btype, prom_type=ptype, base_bytes=bb, prom_bytes=pb,
                         delta_bytes=delta, promotable=promotable, benefit_per_byte=bpb))

    # greedy order: promotable first, by benefit/byte desc; no-ops (already Q4) last
    rows.sort(key=lambda r: (r["promotable"], r["benefit_per_byte"]), reverse=True)
    cum = 0
    for i, r in enumerate(rows):
        if r["promotable"]:
            cum += r["delta_bytes"]
        r["rank"] = i
        r["cum_delta_gb"] = cum / 1e9

    n_prom = sum(1 for r in rows if r["promotable"])
    tot_delta = sum(r["delta_bytes"] for r in rows if r["promotable"]) / 1e9
    print(f"{len(rows)} expert tensors | {n_prom} promotable | full promotion = +{tot_delta:.2f} GB", file=sys.stderr)

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("rank,name,layer,proj,score,base_type,prom_type,delta_bytes,cum_delta_gb,promotable,benefit_per_byte\n")
            for r in rows:
                f.write(f"{r['rank']},{r['name']},{r['layer']},{r['proj']},{r['score']:.6e},"
                        f"{r['base_type']},{r['prom_type']},{r['delta_bytes']},{r['cum_delta_gb']:.4f},"
                        f"{int(r['promotable'])},{r['benefit_per_byte']:.6e}\n")
        print(f"wrote {args.out_csv}", file=sys.stderr)

    if args.budget_gb is not None:
        promote, used = [], 0
        for r in rows:
            if not r["promotable"]:
                continue
            if (used + r["delta_bytes"]) / 1e9 > args.budget_gb:
                continue
            promote.append(r["name"]); used += r["delta_bytes"]
        print(f"budget +{args.budget_gb} GB -> promote {len(promote)} tensors, +{used/1e9:.3f} GB actual", file=sys.stderr)
        if args.out_json:
            json.dump(promote, open(args.out_json, "w"))
            print(f"wrote {args.out_json}", file=sys.stderr)
        else:
            print(json.dumps(promote))


if __name__ == "__main__":
    main()
