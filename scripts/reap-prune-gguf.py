#!/usr/bin/env python3
"""REAP GGUF surgery: prune MoE experts from a (bf16) GGUF using a keep-map.

For each layer, slices ffn_{gate,up,down}_exps and the router ffn_gate_inp to the
kept expert ids (numpy axis 0 = expert dim, untouched by bf16 byte-expansion),
leaves ffn_gate_inp_shexp and everything else verbatim, and sets
<arch>.expert_count = N. Two-pass write (shapes then data) holds one sliced
tensor at a time. Modeled on gguf-py/.../gguf_new_metadata.py.

Usage:
  reap-prune-gguf.py NX2-bf16.gguf NX2.reap.keep.r50.json NX2-bf16.reap-r50.gguf
"""
import argparse, json, os, re, sys
from pathlib import Path

sys.path.insert(0, "/home/frosty40/llama.cpp/gguf-py")
import gguf  # noqa: E402

SLICE_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate_exps|up_exps|down_exps|gate_inp)\.weight$")

def expert_axis_len(shape):
    # numpy shape is reversed ne; expert dim is ne[-1] == numpy axis 0
    return shape[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("keepmap")
    ap.add_argument("output")
    args = ap.parse_args()

    keepmap = json.load(open(args.keepmap))
    keep = {int(k): v for k, v in keepmap["keep"].items()}
    n_out = int(keepmap["n_expert_out"])
    for il, ids in keep.items():
        assert len(ids) == n_out, f"layer {il}: keep len {len(ids)} != n_out {n_out}"
        assert ids == sorted(ids), f"layer {il}: keep ids not ascending"

    reader = gguf.GGUFReader(args.input, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    expert_count_key = f"{arch}.expert_count"
    print(f"arch={arch}  pruning experts -> {n_out}  ({args.input} -> {args.output})")

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    # --- copy metadata verbatim, override expert_count ---
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == expert_count_key:
            print(f"  {expert_count_key}: {value} -> {n_out}")
            value = n_out
        if value is not None:
            writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    # --- tensor info pass (shapes only; slice computed without materializing) ---
    def out_meta(t):
        m = SLICE_RE.match(t.name)
        if not m:
            return t.data.shape, t.data.nbytes, False, None
        il = int(m.group(1))
        if il not in keep:
            return t.data.shape, t.data.nbytes, False, None
        ax0 = expert_axis_len(t.data.shape)
        assert ax0 == keepmap["n_expert_in"], f"{t.name}: axis0 {ax0} != n_expert_in"
        new_shape = (n_out,) + tuple(t.data.shape[1:])
        itemsize = t.data.dtype.itemsize
        nbytes = 1
        for d in new_shape:
            nbytes *= d
        nbytes *= itemsize
        return new_shape, nbytes, True, il

    sliced_n = 0
    for t in reader.tensors:
        shape, nbytes, is_sliced, _ = out_meta(t)
        writer.add_tensor_info(t.name, shape, t.data.dtype, nbytes, t.tensor_type)
        sliced_n += 1 if is_sliced else 0
    print(f"  tensors: {len(reader.tensors)} total, {sliced_n} sliced")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # --- data pass (slice on the fly) ---
    for t in reader.tensors:
        m = SLICE_RE.match(t.name)
        if m and int(m.group(1)) in keep:
            data = t.data[keep[int(m.group(1))]]  # fancy index -> contiguous copy, axis 0
        else:
            data = t.data
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)

    writer.close()
    print(f"  wrote {args.output}")

if __name__ == "__main__":
    main()
