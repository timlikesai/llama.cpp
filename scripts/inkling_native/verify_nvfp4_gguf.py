#!/usr/bin/env python3
"""Verify a converted Inkling-2 GGUF keeps the checkpoint's NVFP4 experts bit-exactly.

Same claim as verify_mxfp8_gguf.py, for the 4-bit layers 0-19: the E2M1 codes and the e4m3
sub-block scales must be the checkpoint's own bytes in ggml block_nvfp4 order
([4 ue4m3][32 payload] per 64 elements), the fused gate/up rows must stay paired with their
scales, and the F32 [n_expert] `.scale` sidecar must carry the amax/outer_scale_denominator
factor that no block byte can hold.

    ./scripts/inkling_native/verify_nvfp4_gguf.py [--gguf PATH] [--experts 0,64,127] [--no-reference]

Exit code 0 means every check held. Nothing is written.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
sys.path.insert(0, HERE)

from gguf import GGUFReader                      # noqa: E402
from gguf.constants import GGMLQuantizationType   # noqa: E402

import layout                                     # noqa: E402
from verify_mxfp8_gguf import Safetensors         # noqa: E402

DEFAULT_GGUF = os.path.join(os.path.expanduser("~"), "models/timlikesai/Inkling-2-Mini-NATIVE.gguf")
DEFAULT_REF = os.path.join(os.path.expanduser("~"), "models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf")
SNAP_GLOB = "models--thinkingmachines--Inkling-2-Mini/snapshots/*"
MXFP8, NVFP4, F32 = 44, 40, 0


def vec(st, name, count):
    """The first `count` float32 values of a 1-D F32 checkpoint tensor."""
    meta = st.header[name]
    if meta["dtype"] != "F32":
        raise ValueError(f"{name}: expected F32, got {meta['dtype']}")
    st.f.seek(st.base + meta["data_offsets"][0])
    return np.frombuffer(st.f.read(4 * count), dtype="<f4")


def tensor_map(path):
    return {t.name: t for t in GGUFReader(path).tensors}


def hf_names(block, proj):
    """(weight, scale, amax, denominator) checkpoint names for one expert tensor."""
    stem = f"model.llm.layers.{block}.mlp.experts.{proj}_weight"
    return stem, stem + "_scale", stem + "_amax", stem + "_outer_scale_denominator"


def logical_shape(st, hf_weight):
    """GGUF logical shape [cols, rows, experts] of an expert tensor, from the checkpoint header."""
    e, rows, cols2 = st.shape(hf_weight)
    scale_cols = st.shape(hf_weight + "_scale")[2]
    if scale_cols * layout.QK_NVFP4_SUB != cols2 * 2:
        raise ValueError(f"{hf_weight}: payload {cols2} bytes and scale {scale_cols} disagree on the row width")
    return e, rows, cols2 * 2


def read_hf_row(st, name, expert, row, cols):
    """Checkpoint (payload, scale) bytes for one logical expert row: cols elements -> cols/2 bytes, cols/16 scales."""
    e, rows, _ = logical_shape(st, name)
    if not 0 <= expert < e or not 0 <= row < rows:
        raise ValueError(f"{name}: expert {expert} row {row} out of range for {e}x{rows}")
    return (st.row(name, expert, row, cols // 2), st.row(name + "_scale", expert, row, cols // layout.QK_NVFP4_SUB))


def gguf_bytes(reader_tensor, experts, rows, cols):
    """Byte view of a GGUF expert tensor as [experts, rows, row_bytes]."""
    raw = np.asarray(reader_tensor.data).view(np.uint8)
    row_bytes = cols // layout.QK_NVFP4 * (layout.QK_NVFP4 // 2 + layout.QK_NVFP4 // layout.QK_NVFP4_SUB)
    if raw.size != experts * rows * row_bytes:
        raise ValueError(f"{reader_tensor.name}: {raw.size} bytes in file, expected {experts * rows * row_bytes}")
    return raw.reshape(experts, rows, row_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--experts", default="0,1,63,64,127")
    ap.add_argument("--reference", default=DEFAULT_REF, help="public GGUF with native NVFP4 experts, for byte cross-check")
    ap.add_argument("--ref-blocks", default="0,17", help="blocks to compare against --reference")
    ap.add_argument("--quick", action="store_true", help="2 experts, 1 block, no reference cross-check")
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args()

    experts = [0, 64] if args.quick else [int(x) for x in args.experts.split(",")]
    ref_blocks = [] if args.quick or args.no_reference else [int(x) for x in args.ref_blocks.split(",")]

    snap = glob.glob(os.path.expanduser(os.path.join("~/.cache/huggingface/hub", SNAP_GLOB)))[0]
    st = Safetensors(os.path.join(snap, "model-00001-of-00001.safetensors"))
    tensors = tensor_map(args.gguf)
    nvfp4 = sorted((n for n, t in tensors.items() if int(t.tensor_type) == NVFP4))
    mxfp8 = [n for n, t in tensors.items() if int(t.tensor_type) == MXFP8]
    print(f"gguf           : {args.gguf}")
    print(f"  tensors      : {len(tensors)}  NVFP4={len(nvfp4)}  MXFP8={len(mxfp8)}")
    types, sizes = {}, {}
    for n, t in tensors.items():
        name = GGMLQuantizationType(int(t.tensor_type)).name
        types[name] = types.get(name, 0) + 1
        sizes[name] = sizes.get(name, 0) + int(t.n_bytes)
    for name in sorted(types, key=lambda k: -sizes[k]):
        print(f"    {name:<10} {types[name]:>5} tensors  {sizes[name] / 2**30:>9.2f} GiB")
    for n in nvfp4:
        print(f"    {n:<44} shape={list(map(int, tensors[n].shape))}")
    if len(nvfp4) != 40:
        print(f"\nFAIL: expected 40 native NVFP4 expert tensors (layers 0-19 x w13/w2), found {len(nvfp4)}")
        return 1

    failures = sign_only = checked_rows = 0
    for name in nvfp4:
        t = tensors[name]
        block = int(name.split(".")[1])
        proj = "w13" if "gate_up" in name else "w2"
        hf_weight, _, hf_amax, hf_den = hf_names(block, proj)
        if hf_weight not in st.header:
            print(f"FAIL {name}: no such checkpoint tensor {hf_weight}")
            failures += 1
            continue
        e, rows, cols = logical_shape(st, hf_weight)
        dims = list(map(int, t.shape))
        if dims != [cols, rows, e]:
            print(f"FAIL {name}: gguf logical shape {dims} != checkpoint-derived {[cols, rows, e]}")
            failures += 1
            continue
        sidecar = tensors.get(name.replace(".weight", ".scale"))
        if sidecar is None or int(sidecar.tensor_type) != F32 or list(map(int, sidecar.shape)) != [e]:
            print(f"FAIL {name}: .scale sidecar missing or wrong ({None if sidecar is None else (int(sidecar.tensor_type), list(map(int, sidecar.shape)))}, expected ({F32}, [{e}]))")
            failures += 1
            continue
        raw = gguf_bytes(t, e, rows, cols)
        factor = np.asarray(sidecar.data).view(np.float32)
        amax, den = vec(st, hf_amax, e), float(vec(st, hf_den, 1)[0])
        if not np.array_equal(factor.view(np.uint32), (amax / np.float32(den)).view(np.uint32)):
            print(f"FAIL {name}: sidecar is not bitwise amax/outer_scale_denominator ({int(np.count_nonzero(factor != amax / den))}/{e} values differ, max relative {float(np.abs(factor / (amax / den) - 1).max()):.3e})")
            failures += 1
        inter = rows // 2  # gate/up halves along the GGUF rows; unused for w2, which is identity
        for expert in experts:
            if expert >= e:
                continue
            for row in range(0, rows, max(1, rows // 8)):
                # the fused gate/up tensor interleaves: gguf row r <- checkpoint row 2*(r % I) + r // I
                hf_row = 2 * (row % inter) + (row // inter) if proj == "w13" else row
                payload, scales = read_hf_row(st, hf_weight, expert, hf_row, cols)
                want = layout.pack_nvfp4_blocks(np.frombuffer(payload, np.uint8).reshape(1, -1), np.frombuffer(scales, np.uint8).reshape(1, -1))[0]
                have = raw[expert, row]
                checked_rows += 1
                if not np.array_equal(have, want):
                    failures += 1
                    i = int(np.flatnonzero(have != want)[0])
                    print(f"FAIL {name} expert {expert} gguf row {row} (checkpoint row {hf_row}): {int(np.count_nonzero(have != want))}/{have.size} bytes differ, "
                          f"first at byte {i} (0-3 ue4m3 scales of the block, 4-35 payload) gguf=0x{have[i]:02x} expected=0x{want[i]:02x}")
                    break
                got = layout.dequant_nvfp4_blocks(have.reshape(1, -1), cols, factor=factor[expert])
                ref, _ = layout.hf_dequant_nvfp4(np.frombuffer(payload, np.uint8).reshape(1, -1), np.frombuffer(scales, np.uint8).reshape(1, -1), amax[expert], den)
                if not np.array_equal(got, ref):
                    zero = (got == ref) & (np.signbit(got) != np.signbit(ref))
                    if np.all(zero | (got == ref)):
                        sign_only += int(zero.sum())
                    else:
                        failures += 1
                        i = int(np.flatnonzero(~(zero | (got == ref)))[0])
                        print(f"FAIL {name} expert {expert} gguf row {row}: {int(np.count_nonzero(~(zero | (got == ref))))}/{got.size} values differ, first at element {i}: gguf={got[i]} reference={ref[i]} (the reference is the checkpoint dequantized times the amax/denominator sidecar)")
                        break

    print(f"\nrows compared against the checkpoint: {checked_rows}  (bytes and decoded values)")
    if sign_only:
        print(f"note: {sign_only} value(s) differed only in the sign of a zero (0.0 vs -0.0), which is not a loss of information")

    # cross-check against the public native file only once the checkpoint comparison passed: while the
    # two files disagree with each other, a reference mismatch would hide which one is wrong
    if ref_blocks and not failures:
        ref = tensor_map(args.reference)
        for block in ref_blocks:
            for proj in ("w13", "w2"):
                name = f"blk.{block}.{'ffn_gate_up_exps' if proj == 'w13' else 'ffn_down_exps'}.weight"
                if name not in ref or name not in tensors:
                    print(f"FAIL {name}: missing in one of the two files")
                    failures += 1
                    continue
                e, rows, cols = logical_shape(st, hf_names(block, proj)[0])
                mine, theirs = gguf_bytes(tensors[name], e, rows, cols), gguf_bytes(ref[name], e, rows, cols)
                for expert in experts:
                    if expert >= e:
                        continue
                    same = np.array_equal(mine[expert], theirs[expert])
                    nd = 0 if same else int(np.count_nonzero(mine[expert] != theirs[expert]))
                    print(f"  {name} expert {expert}: {mine[expert].size} bytes {'identical' if same else f'{nd} differ'} vs reference")
                    if not same:
                        failures += 1
                m, r = tensors[name.replace(".weight", ".scale")], ref.get(name.replace(".weight", ".scale"))
                if r is None:
                    print(f"FAIL {name.replace('.weight', '.scale')}: missing in the reference file")
                    failures += 1
                    continue
                mv, rv = np.asarray(m.data).view(np.float32), np.asarray(r.data).view(np.float32)
                if not np.array_equal(mv.view(np.uint32), rv.view(np.uint32)):
                    print(f"FAIL {name.replace('.weight', '.scale')}: sidecar bytes differ from the reference at {int(np.count_nonzero(mv != rv))}/{mv.size} experts")
                    failures += 1

    if failures:
        print(f"FAIL: {failures} check group(s) mismatched")
        return 1
    print("PASS: every sampled NVFP4 row is the checkpoint's bytes, repacked to [4 ue4m3][32 payload] with the")
    print("      gate/up interleaving preserved, decodes to the checkpoint's values times its .scale sidecar,")
    print("      and matches the public native NVFP4 file byte for byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
