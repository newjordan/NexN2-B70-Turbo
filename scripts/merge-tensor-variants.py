#!/usr/bin/env python3
# merge-tensor-variants.py
#
# Build a multi-precision GGUF: the --base model's tensors are the canonical (variant 0) set,
# and the matching tensors from --variant are appended as a sibling "<name>.v1" variant. The
# loader (src/llama-model-loader.cpp: select_tensor_variant) collapses the file to one variant at
# load time, chosen by `general.tensor_variant.default` and overridable with
#   --override-kv general.tensor_variant.default=int:1
#
# Only tensors whose name contains --pattern (default the MoE experts) are varianted; everything
# else is shared from --base. Requires the IQ3_A770-aware gguf-py in this tree.
#
#   python scripts/merge-tensor-variants.py \
#       --base   models/.../NX2-IQ3_A770.gguf \
#       --variant models/.../NX2-Q4_K.gguf \
#       --out    models/.../NX2-IQ3_A770-dual.gguf
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp")), "gguf-py"))
import csv
import gguf
from tqdm import tqdm


def load_promote_order(ranking_csv: str) -> list:
    """Ranked base names to promote v0->v1, best quality-per-byte first.
    rank-experts.py already writes rows in that greedy order; keep only promotable ones."""
    order = []
    with open(ranking_csv) as f:
        for row in sorted(csv.DictReader(f), key=lambda r: int(r["rank"])):
            if row.get("promotable") == "1":
                order.append(row["name"])
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge two GGUFs into one multi-precision (variant) GGUF")
    ap.add_argument("--base",    required=True, help="canonical / variant-0 model (e.g. IQ3_A770)")
    ap.add_argument("--variant", required=True, help="variant-1 model (e.g. Q4_K)")
    ap.add_argument("--out",     required=True, help="output dual-precision GGUF")
    ap.add_argument("--pattern", default="_exps.", help="substring identifying the varianted tensors")
    ap.add_argument("--ranking", default=None,
                    help="ranking.csv from rank-experts.py; embeds general.tensor_variant.promote_order "
                         "(ranked promotable base names) so the loader can solve the elastic budget knapsack")
    args = ap.parse_args()

    base = gguf.GGUFReader(args.base, "r")
    var  = gguf.GGUFReader(args.variant, "r")

    arch_field = base.fields.get(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else None
    if not arch:
        raise SystemExit("base model has no general.architecture")

    writer = gguf.GGUFWriter(args.out, arch=arch, endianess=base.endianess)

    # copy every base KV except the architecture + writer-managed virtual fields
    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        stype = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if value is not None:
            writer.add_key_value(field.name, value, vtype, sub_type=stype)

    # declare the variant set
    writer.add_key_value("general.tensor_variant.count",   2, gguf.GGUFValueType.UINT32)
    writer.add_key_value("general.tensor_variant.default", 0, gguf.GGUFValueType.UINT32)
    # elastic budget mode is off by default (0 -> use the global default above); a deployer turns it
    # on per run with --override-kv general.tensor_variant.budget_mb=int:<MiB>
    writer.add_key_value("general.tensor_variant.budget_mb", 0, gguf.GGUFValueType.UINT32)
    if args.ranking:
        order = load_promote_order(args.ranking)
        writer.add_key_value("general.tensor_variant.promote_order", order,
                             gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
        print(f"embedded promote_order manifest: {len(order)} ranked promotable tensors "
              f"(first={order[0] if order else '-'})")

    # canonical tensors (all of base) + variant experts (from var) suffixed ".v1", same order for info & data
    items = [(t.name, t) for t in base.tensors]
    n_variant = 0
    for t in var.tensors:
        if args.pattern in t.name:
            items.append((t.name + ".v1", t))
            n_variant += 1
    if n_variant == 0:
        raise SystemExit(f"no tensors in --variant match pattern {args.pattern!r}")
    print(f"base tensors: {len(base.tensors)}  +  variant '.v1' tensors: {n_variant}  =  {len(items)} total")

    for name, t in items:
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    total = sum(t.n_bytes for _, t in items)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    bar = tqdm(total=total, unit="byte", unit_scale=True, desc="merge")
    for _name, t in items:
        writer.write_tensor_data(t.data, tensor_endianess=base.endianess)
        bar.update(t.n_bytes)
    writer.close()
    bar.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
