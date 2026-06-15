#!/usr/bin/env python3
"""Autoprune a dual/elastic GGUF to a single right-sized model for a given VRAM budget.

Reads the multi-precision dual (canonical variant-0 tensors + "<name>.v1" siblings +
general.tensor_variant.promote_order), runs the SAME greedy importance/byte knapsack the
loader uses, and writes a plain single-variant GGUF holding only the selected per-tensor
winners. The result is exactly what the elastic loader would resolve at that budget — but
materialized, so the file shrinks from ~34 GiB to the budget (~15-20 GiB) and the unused
variant bytes ("the excess") are dropped.

  prune-dual.py --dual NX2-IQ3_A770-dual.gguf --vram-gib 16 --out NX2-fitted.gguf
  prune-dual.py --dual ... --budget-mb 16400 --out ...
"""
import argparse, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp")), "gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter
from tqdm import tqdm

RE_SIB = re.compile(r"^(.+)\.v([1-9][0-9]*)$")
TV_KEYS = ("general.tensor_variant.count", "general.tensor_variant.default",
           "general.tensor_variant.budget_mb", "general.tensor_variant.promote_order")


def select_budget(reader, budget_bytes):
    """Return (promoted_set, baseline_bytes, total_bytes). Mirrors select_tensor_variant()."""
    by = {t.name: t for t in reader.tensors}
    # baseline: every tensor that is NOT a ".v<N>" sibling (canonical v0 + non-varianted)
    baseline = sum(int(t.n_bytes) for t in reader.tensors if not RE_SIB.match(t.name))
    # promotable groups: canonical base present AND a .v1 sibling present
    def bytes_of(name):
        return int(by[name].n_bytes) if name in by else 0
    order_field = reader.fields.get("general.tensor_variant.promote_order")
    order = order_field.contents() if order_field else []
    if order and isinstance(order[0], (bytes, bytearray)):
        order = [x.decode() for x in order]
    promoted, total = set(), baseline
    for base in order:
        if base in promoted or base not in by or (base + ".v1") not in by:
            continue
        v0, v1 = bytes_of(base), bytes_of(base + ".v1")
        if v1 <= v0 or total + (v1 - v0) > budget_bytes:
            continue
        promoted.add(base); total += v1 - v0
    return promoted, baseline, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--budget-mb", type=int)
    g.add_argument("--vram-gib", type=float)
    ap.add_argument("--reserve-gib", type=float, default=0.6,
                    help="VRAM held back for KV cache + compute buffers (with --vram-gib)")
    ap.add_argument("--dry-run", action="store_true", help="print the selection and exit; write nothing")
    args = ap.parse_args()

    reader = GGUFReader(args.dual)
    if args.budget_mb is not None:
        budget_bytes = args.budget_mb * (1 << 20)
        budget_desc = f"{args.budget_mb} MiB"
    else:
        budget_mb = int(math.floor((args.vram_gib - args.reserve_gib) * 1024))
        budget_bytes = budget_mb * (1 << 20)
        budget_desc = f"{args.vram_gib} GiB VRAM - {args.reserve_gib} reserve = {budget_mb} MiB"

    promoted, baseline, total = select_budget(reader, budget_bytes)
    print(f"budget: {budget_desc} ({budget_bytes/(1<<30):.2f} GiB)", file=sys.stderr)
    print(f"baseline (all variant-0): {baseline/(1<<30):.2f} GiB", file=sys.stderr)
    print(f"selected: {total/(1<<30):.2f} GiB, {len(promoted)} tensors promoted to v1", file=sys.stderr)
    if args.dry_run:
        return

    arch_field = reader.fields.get(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else None
    if not arch:
        raise SystemExit("dual has no general.architecture")
    writer = GGUFWriter(args.out, arch=arch, endianess=reader.endianess)
    # copy all KVs except arch, writer-managed, and the now-meaningless tensor_variant.* keys
    for field in reader.fields.values():
        if (field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.")
                or field.name in TV_KEYS):
            continue
        vtype = field.types[0]
        stype = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if value is not None:
            writer.add_key_value(field.name, value, vtype, sub_type=stype)

    by = {t.name: t for t in reader.tensors}
    items = []
    for t in reader.tensors:
        if RE_SIB.match(t.name):
            continue                              # drop every .v<N> sibling
        src = by[t.name + ".v1"] if t.name in promoted else t   # promoted -> take the v1 winner
        items.append((t.name, src))
    print(f"writing {len(items)} tensors -> {args.out}", file=sys.stderr)

    for name, t in items:
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    total_b = sum(t.n_bytes for _, t in items)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    bar = tqdm(total=total_b, unit="byte", unit_scale=True, desc="prune")
    for _name, t in items:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        bar.update(t.n_bytes)
    writer.close(); bar.close()
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
