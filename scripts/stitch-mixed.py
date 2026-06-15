#!/usr/bin/env python3
"""Stitch a single mixed-precision GGUF from two same-shape models.

Every expert tensor named in --promote-json is taken from --promote (the all-Q4_K
model); every other tensor is taken from --base (the shipped mostly-IQ3 model).
The result is a plain, directly-loadable GGUF (no .v1 variant siblings) whose bytes
are identical to what a per-tensor load-time selector would pick -- so its CPU KLD
is the exact quality of that elastic-precision operating point.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp")), "gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--promote", required=True)
    ap.add_argument("--promote-json", required=True, help="JSON list of tensor names to take from --promote")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    promote_set = set(json.load(open(args.promote_json)))
    base = GGUFReader(args.base)
    prom = GGUFReader(args.promote)
    prom_by = {t.name: t for t in prom.tensors}

    arch_field = base.fields.get(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else None
    if not arch:
        raise SystemExit("base model has no general.architecture")
    writer = GGUFWriter(args.out, arch=arch, endianess=base.endianess)

    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        stype = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if value is not None:
            writer.add_key_value(field.name, value, vtype, sub_type=stype)

    # choose source tensor per name, preserving base order
    items, n_prom = [], 0
    for t in base.tensors:
        if t.name in promote_set:
            src = prom_by[t.name]; n_prom += 1
        else:
            src = t
        items.append((t.name, src))
    missing = promote_set - {t.name for t in base.tensors}
    if missing:
        raise SystemExit(f"{len(missing)} promote names not in base, e.g. {sorted(missing)[:3]}")
    print(f"stitch: {len(items)} tensors, {n_prom} taken from --promote", file=sys.stderr)

    for name, t in items:
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    total = sum(t.n_bytes for _, t in items)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    bar = tqdm(total=total, unit="byte", unit_scale=True, desc="stitch")
    for _name, t in items:
        writer.write_tensor_data(t.data, tensor_endianess=base.endianess)
        bar.update(t.n_bytes)
    writer.close(); bar.close()
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
