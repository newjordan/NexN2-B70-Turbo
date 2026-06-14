#!/usr/bin/env python3
"""Slice an imatrix GGUF to a pruned expert set (index-aligned with reap-prune-gguf.py).

The per-expert imatrix entries blk.{il}.ffn_{gate,up,down}_exps.weight.{in_sum2,counts}
have the expert dimension as numpy axis 0; slice them by the same keep-map used for the
model. Router (ffn_gate_inp.*) and everything else are copied verbatim. This keeps the
imatrix expert rows aligned to the pruned model's 0..N-1 expert indices (the 256-expert
imatrix would otherwise misalign). Fast alternative to regenerating the imatrix.

Usage: reap-prune-imatrix.py NX2.imatrix NX2.reap.keep.r50.json NX2.reap-r50.imatrix
"""
import argparse, json, re, sys
sys.path.insert(0, "/home/frosty40/llama.cpp/gguf-py")
import gguf  # noqa: E402

SLICE_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate_exps|up_exps|down_exps)\.weight\.(in_sum2|counts)$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("keepmap"); ap.add_argument("output")
    args = ap.parse_args()
    km = json.load(open(args.keepmap))
    keep = {int(k): v for k, v in km["keep"].items()}
    n_in = int(km["n_expert_in"])

    reader = gguf.GGUFReader(args.input, "r")
    writer = gguf.GGUFWriter(args.output, arch="imatrix", endianess=reader.endianess)
    al = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if al is not None:
        writer.data_alignment = al.contents()

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        v = field.contents()
        if v is not None:
            writer.add_key_value(field.name, v, vt, sub_type=st)

    def sliced_shape(t):
        m = SLICE_RE.match(t.name)
        if not m or int(m.group(1)) not in keep:
            return t.data.shape, t.data.nbytes, False
        assert t.data.shape[0] == n_in, f"{t.name}: axis0 {t.data.shape[0]} != {n_in}"
        ns = (len(keep[int(m.group(1))]),) + tuple(t.data.shape[1:])
        nb = t.data.dtype.itemsize
        for d in ns: nb *= d
        return ns, nb, True

    n_sliced = 0
    for t in reader.tensors:
        shp, nb, sl = sliced_shape(t)
        writer.add_tensor_info(t.name, shp, t.data.dtype, nb, t.tensor_type)
        n_sliced += 1 if sl else 0
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    for t in reader.tensors:
        m = SLICE_RE.match(t.name)
        data = t.data[keep[int(m.group(1))]] if (m and int(m.group(1)) in keep) else t.data
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"{args.output}: sliced {n_sliced} expert imatrix tensors -> keep {km['n_expert_out']}/{n_in}")

if __name__ == "__main__":
    main()
