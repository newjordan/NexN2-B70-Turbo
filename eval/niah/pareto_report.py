#!/usr/bin/env python3
"""Render the NIAH Pareto results: markdown table + accuracy/cost plot.

Reads results/niah-pareto.csv, writes results/niah-pareto.md and
results/niah-pareto.png. "Reliable" is defined as 100% retrieval across all
probed depths and samples at a given length (and is reported per config with
the probe count, so the strength of the claim is visible).
"""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/home/frosty40/nx2-b70-turbo/results/niah-pareto.csv"
MD = "/home/frosty40/nx2-b70-turbo/results/niah-pareto.md"
PNG = "/home/frosty40/nx2-b70-turbo/results/niah-pareto.png"

CONTENTION_NOTE = (
    "Timing caveat: external load (load avg 21-29) on 2026-06-10 ~01:50-02:35 "
    "skewed `prefill_s`/`decode_ts` for the `yarn2-iq4xs/f16` rows at 32k-196k "
    "(verdicts unaffected). Its 260k/520k timings are from the earlier quiet-box "
    "smoke and are clean.")


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            r["ctx_tokens"] = int(r["ctx_tokens"])
            r["prefill_s"] = float(r["prefill_s"])
            r["decode_ts"] = float(r["decode_ts"])
            rows.append(r)
    return rows


def main():
    rows = load()
    cfgs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cfgs[f"{r['rope_config']}/{r['kv_type']}"][r["ctx_tokens"]].append(r)

    lines = ["# NIAH Pareto sweep — NexN2 on the B70",
             "",
             f"{len(rows)} probes total, "
             f"{sum(r['verdict'] == 'PASS' for r in rows)} PASS, "
             f"{sum(r['verdict'] == 'FAIL' for r in rows)} FAIL. "
             "Multi-needle RULER-style harness (eval/niah/niah_sweep.py), "
             "temp 0, fixed seed, substring grading, max_tokens 400.",
             "", CONTENTION_NOTE, ""]
    for cfg in sorted(cfgs):
        lines += [f"## {cfg}", "",
                  "| tokens | pass | probes | depths×samples | prefill_s | decode t/s |",
                  "|-------:|-----:|-------:|---------------:|----------:|-----------:|"]
        for L in sorted(cfgs[cfg]):
            cell = cfgs[cfg][L]
            npass = sum(r["verdict"] == "PASS" for r in cell)
            depths = sorted({r["needle_depth_pct"] for r in cell}, key=int)
            samples = sorted({r["sample"] for r in cell})
            prefill = max(r["prefill_s"] for r in cell)
            dec = sum(r["decode_ts"] for r in cell) / len(cell)
            lines.append(
                f"| {L:,} | {npass}/{len(cell)} | {len(cell)} "
                f"| {len(depths)}×{len(samples)} | {prefill:.0f} | {dec:.1f} |")
        lines.append("")
    with open(MD, "w") as f:
        f.write("\n".join(lines))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for cfg in sorted(cfgs):
        Ls = sorted(cfgs[cfg])
        acc = [100 * sum(r["verdict"] == "PASS" for r in cfgs[cfg][L])
               / len(cfgs[cfg][L]) for L in Ls]
        dec = [sum(r["decode_ts"] for r in cfgs[cfg][L]) / len(cfgs[cfg][L])
               for L in Ls]
        ax1.plot([l / 1000 for l in Ls], acc, "o-", label=cfg)
        ax2.plot([l / 1000 for l in Ls], dec, "o-", label=cfg)
    ax1.set_xlabel("context (k tokens)"); ax1.set_ylabel("retrieval %")
    ax1.set_ylim(-5, 105); ax1.set_title("NIAH retrieval vs context")
    ax2.set_xlabel("context (k tokens)"); ax2.set_ylabel("decode t/s")
    ax2.set_title("decode speed vs context (cost axis)")
    ax1.grid(alpha=.3); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PNG, dpi=120)
    print(f"wrote {MD} and {PNG}")


if __name__ == "__main__":
    main()
