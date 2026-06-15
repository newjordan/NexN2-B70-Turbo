#!/usr/bin/env python3
"""Plot the elastic-precision Pareto curve (KLD vs weight footprint) and emit a
markdown summary. Tolerant of partial / NA rows so it can run mid-sweep.

gap_closed = (kld_base - kld) / (kld_base - kld_full)  -- fraction of the
IQ3 -> Q4_K quality gap captured at that footprint.
"""
import csv, sys, os

OUT = os.environ.get("ELASTIC_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "elastic-precision"))
CSV = os.path.join(OUT, "pareto.csv")

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        try:
            r["weight_gb"] = float(r["weight_gb"]); r["mean_kld"] = float(r["mean_kld"])
            r["n_promoted"] = int(r["n_promoted"])
            r["top1_pct"] = float(r["top1_pct"]) if r["top1_pct"] not in ("NA", "") else None
        except (ValueError, KeyError):
            continue
        rows.append(r)
rows.sort(key=lambda r: r["weight_gb"])
if len(rows) < 2:
    print(f"only {len(rows)} valid rows so far; need >=2", file=sys.stderr); sys.exit(0)

base = min(rows, key=lambda r: r["weight_gb"])
full = max(rows, key=lambda r: r["weight_gb"])
kb, kf = base["mean_kld"], full["mean_kld"]
gap = kb - kf

def gap_closed(k):
    return (kb - k) / gap if gap else 0.0

# ---- markdown table --------------------------------------------------------
md = []
md.append("| footprint (GB) | Δ vs IQ3 (GB) | experts promoted | mean KLD ↓ | top-1 % ↑ | gap closed |")
md.append("|---|---|---|---|---|---|")
for r in rows:
    dgb = r["weight_gb"] - base["weight_gb"]
    tag = ""
    if r is base: tag = " (shipped all-IQ3)"
    elif r is full: tag = " (all-Q4_K)"
    t1 = f"{r['top1_pct']:.2f}" if r["top1_pct"] is not None else "—"
    md.append(f"| {r['weight_gb']:.2f}{tag} | +{dgb:.2f} | {r['n_promoted']} | "
              f"{r['mean_kld']:.4f} | {t1} | {gap_closed(r['mean_kld'])*100:.0f}% |")
md_table = "\n".join(md)
print(md_table)
print()
# headline: best point at/under +1.0 GB and +0.5 GB
for budget in (0.5, 1.0):
    cands = [r for r in rows if r["weight_gb"] - base["weight_gb"] <= budget + 1e-6 and r is not base]
    if cands:
        b = max(cands, key=lambda r: r["weight_gb"])
        print(f"@ +{budget} GB: KLD {kb:.4f} -> {b['mean_kld']:.4f} "
              f"({(1-b['mean_kld']/kb)*100:.1f}% lower, {gap_closed(b['mean_kld'])*100:.0f}% of the IQ3->Q4 gap closed)")

with open(os.path.join(OUT, "pareto_table.md"), "w") as f:
    f.write(md_table + "\n")

# ---- plot ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r["weight_gb"] for r in rows]; ys = [r["mean_kld"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # naive linear blend between the two endpoints (the "dumb" baseline)
    ax.plot([base["weight_gb"], full["weight_gb"]], [kb, kf], "--", color="gray",
            lw=1.3, label="naive linear blend (endpoints)")
    ax.plot(xs, ys, "-o", color="#1f77b4", lw=2, ms=6, label="elastic per-tensor (imatrix-ranked)")
    ax.scatter([base["weight_gb"]], [kb], color="#d62728", zorder=5, s=90, marker="*",
               label=f"shipped all-IQ3 ({kb:.4f})")
    ax.scatter([full["weight_gb"]], [kf], color="#2ca02c", zorder=5, s=90, marker="*",
               label=f"all-Q4_K ({kf:.4f})")
    # A770 single-card headroom band over the 15.0 GiB shipped weights
    ax.axvspan(base["weight_gb"], base["weight_gb"] + 0.97, color="#ffe9c7", alpha=0.5,
               label="A770 16 GB single-card headroom (~+0.9 GiB)")
    ax.set_xlabel("weight footprint (GB on disk / VRAM)")
    ax.set_ylabel("mean KL divergence vs Q6_K  (lower = better)")
    ax.set_title("Elastic precision: one file, per-tensor IQ3→Q4_K by VRAM budget")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT, "pareto.png")
    fig.savefig(png, dpi=130)
    print(f"\nwrote {png}", file=sys.stderr)
except Exception as e:
    print(f"plot skipped: {e}", file=sys.stderr)
