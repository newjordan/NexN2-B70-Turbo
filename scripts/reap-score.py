#!/usr/bin/env python3
"""REAP scorer: turn a reap saliency GGUF into per-layer expert keep-maps.

Reads NX2.reap.gguf (reap.s_sum.blk.{il} + reap.count.blk.{il}, F32 [n_expert]),
computes S_mean = s_sum / max(count, 1), and for each prune ratio selects the
per-layer top-(1-r) experts (keeps every layer >= n_expert_used and uniform
shapes). Zero-count experts (never routed in calibration) are pruned first.

Usage:
  reap-score.py NX2.reap.gguf --ratios 0.25 0.50 --out-prefix NX2.reap.keep
Emits NX2.reap.keep.r25.json, NX2.reap.keep.r50.json.
"""
import argparse, json, os, sys

def load_reap(path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "llama.cpp", "gguf-py"))
    # fall back to an installed gguf if the relative path is wrong
    try:
        from gguf.gguf_reader import GGUFReader
    except ImportError:
        sys.path.insert(0, "/home/frosty40/llama.cpp/gguf-py")
        from gguf.gguf_reader import GGUFReader
    import numpy as np
    r = GGUFReader(path)
    def kv(k, d=None):
        f = r.fields.get(k)
        return f.contents() if f else d
    n_layer = int(kv("reap.n_layer"))
    n_expert = int(kv("reap.n_expert"))
    n_used = int(kv("reap.n_expert_used"))
    tens = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
    s_sum, count = [], []
    for il in range(n_layer):
        s_sum.append(tens[f"reap.s_sum.blk.{il}"])
        count.append(tens[f"reap.count.blk.{il}"])
    return n_layer, n_expert, n_used, s_sum, count, (kv("reap.n_tokens"), kv("reap.source_model"))

def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("reap_gguf")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--out-prefix", default="NX2.reap.keep")
    ap.add_argument("--min-count", type=int, default=30,
                    help="warn if a kept expert was routed fewer than this many times")
    args = ap.parse_args()

    n_layer, n_expert, n_used, s_sum, count, meta = load_reap(args.reap_gguf)
    print(f"loaded {n_layer} layers x {n_expert} experts (used={n_used}, "
          f"tokens={meta[0]}, src={meta[1]})")

    for ratio in args.ratios:
        keep_n = round(n_expert * (1.0 - ratio))
        keep_n = max(keep_n, n_used)
        keep = {}
        warn_layers, zero_kept = 0, 0
        for il in range(n_layer):
            s_mean = s_sum[il] / np.maximum(count[il], 1.0)
            s_mean[count[il] == 0] = -np.inf            # never-routed -> prune first
            # rank by (S_mean, then expert index) descending; keep top keep_n
            order = sorted(range(n_expert), key=lambda e: (s_mean[e], -e), reverse=True)
            kept = sorted(order[:keep_n])
            keep[str(il)] = kept
            if any(count[il][e] < args.min_count for e in kept):
                warn_layers += 1
            zero_kept += sum(1 for e in kept if count[il][e] == 0)
        out = {
            "arch": "qwen35moe", "ratio": ratio,
            "n_expert_in": n_expert, "n_expert_out": keep_n,
            "n_expert_used": n_used, "source": os.path.basename(args.reap_gguf),
            "keep": keep,
        }
        tag = f"r{int(round(ratio*100))}"
        fn = f"{args.out_prefix}.{tag}.json"
        with open(fn, "w") as f:
            json.dump(out, f)
        print(f"  ratio {ratio:.2f}: keep {keep_n}/{n_expert} per layer -> {fn}"
              f"  ({warn_layers} layers have low-count kept experts, "
              f"{zero_kept} zero-count kept)")

if __name__ == "__main__":
    main()
