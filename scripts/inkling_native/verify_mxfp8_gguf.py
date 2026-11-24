#!/usr/bin/env python3
"""Verify a converted Inkling-2 GGUF keeps the checkpoint's MXFP8 experts bit-exactly.

The claim under test is not "close" but "the same bytes, only moved": every e4m3 code and every
ue8m0 scale byte in blk.N.ffn_{gate_up,down}_exps.weight must be the checkpoint's own, and the
gate/up row interleaving must not desynchronise a row from its scales.

    ./scripts/inkling_native/verify_mxfp8_gguf.py [--gguf PATH] [--experts 0,64,127] [--rows 0..N]

Exit code 0 means every check held. Nothing is written.
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
from gguf import GGUFReader                     # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

DEFAULT_GGUF = os.path.join(os.path.expanduser("~"), "models/timlikesai/Inkling-2-Mini-NATIVE.gguf")
SNAP_GLOB = "models--thinkingmachines--Inkling-2-Mini/snapshots/*"
QK, BS = 32, 33  # QK_MXFP8 / sizeof(block_mxfp8), must match ggml-common.h

MXFP8, NVFP4, F8_E4M3, F32, BF16, Q8_0 = 44, 40, 43, 0, 30, 8
TYPE_NAMES = {MXFP8: "MXFP8", NVFP4: "NVFP4", F8_E4M3: "F8_E4M3", F32: "F32", BF16: "BF16", Q8_0: "Q8_0"}


class Safetensors:
    """Random access into a safetensors file via its header, never loading the whole thing."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def row(self, name, expert, row, nbytes):
        """One row of a [E, R, C] uint8-viewable tensor, as bytes."""
        meta = self.header[name]
        _, _, cols = meta["shape"]
        stride = meta["shape"][1] * cols
        self.f.seek(self.base + meta["data_offsets"][0] + expert * stride + row * cols)
        return self.f.read(nbytes)

    def shape(self, name):
        return tuple(self.header[name]["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--experts", default="0,1,63,64,127")
    ap.add_argument("--quick", action="store_true", help="only 2 experts and 1 layer (fast smoke)")
    args = ap.parse_args()

    snap = glob.glob(os.path.expanduser(os.path.join("~/.cache/huggingface/hub", SNAP_GLOB)))[0]
    st = Safetensors(os.path.join(snap, "model-00001-of-00001.safetensors"))
    r = GGUFReader(args.gguf)

    experts = [0, 64] if args.quick else [int(x) for x in args.experts.split(",")]

    mxfp8 = [t for t in r.tensors if int(t.tensor_type) == MXFP8]
    nvfp4 = [t for t in r.tensors if int(t.tensor_type) == NVFP4]
    print(f"gguf           : {args.gguf}")
    print(f"  tensors      : {len(r.tensors)}  MXFP8={len(mxfp8)}  NVFP4={len(nvfp4)}")
    for t in sorted(mxfp8, key=lambda x: x.name):
        print(f"    {t.name:<44} shape={list(map(int, t.shape))} type={TYPE_NAMES[int(t.tensor_type)]}")

    if len(mxfp8) != 8:
        print(f"\nFAIL: expected 8 native MXFP8 expert tensors (layers 20-23 x w13/w2), found {len(mxfp8)}")
        return 1

    failures = 0
    checked_rows = 0
    for t in mxfp8:
        # blk.N.ffn_gate_up_exps.weight  <->  model.llm.layers.N.mlp.experts.w13_weight
        blk = int(t.name.split(".")[1])
        proj = "w13" if "gate_up" in t.name else "w2"
        hf_name = f"model.llm.layers.{blk}.mlp.experts.{proj}_weight"
        if hf_name not in st.header:
            print(f"FAIL {t.name}: no such checkpoint tensor {hf_name}")
            failures += 1
            continue

        e_hf, rows_hf, cols_hf = st.shape(hf_name)
        dims = list(map(int, t.shape))
        if dims != [cols_hf, rows_hf, e_hf]:
            print(f"FAIL {t.name}: gguf logical shape {dims} != checkpoint-derived {[cols_hf, rows_hf, e_hf]}")
            failures += 1
            continue

        inter = cols_hf // 2 if proj == "w13" else rows_hf  # intermediate_size for the fused half
        row_bytes = cols_hf // QK * BS
        # GGUFReader hands back a memmap shaped in BYTE dims: (n_experts, rows, row_bytes)
        raw = np.asarray(t.data).view(np.uint8)
        if raw.size != e_hf * rows_hf * row_bytes:
            print(f"FAIL {t.name}: {raw.size} bytes in file, expected {e_hf * rows_hf * row_bytes} for logical shape {dims}")
            failures += 1
            continue
        raw = raw.reshape(e_hf, rows_hf, row_bytes)
        for e in experts:
            if e >= e_hf:
                continue
            for gguf_row_idx in range(0, rows_hf, max(1, rows_hf // 8)):
                # The fused gate/up tensor interleaves: gguf row r <- checkpoint row 2*(r%I) + r//I
                hf_row = 2 * (gguf_row_idx % inter) + (gguf_row_idx // inter) if proj == "w13" else gguf_row_idx
                got = st.row(hf_name, e, hf_row, cols_hf)
                got_s = st.row(hf_name + "_scale", e, hf_row, cols_hf // QK)
                want = bytearray()
                for b in range(cols_hf // QK):
                    want += got_s[b:b + 1] + got[b * QK:(b + 1) * QK]
                have = raw[e, gguf_row_idx, :]
                checked_rows += 1
                want_b = np.frombuffer(bytes(want), dtype=np.uint8)
                if not np.array_equal(have, want_b):
                    failures += 1
                    nd = int(np.count_nonzero(have != want_b))
                    print(f"FAIL {t.name} expert {e} gguf row {gguf_row_idx} (checkpoint row {hf_row}): {nd}/{have.size} bytes differ")
                    if nd > 0 and failures < 4:
                        i = int(np.flatnonzero(have != want_b)[0])
                        print(f"     first diff at byte {i}: gguf=0x{have[i]:02x} expected=0x{want_b[i]:02x}"
                              f" (byte 0 of a block is the ue8m0 scale, 1..32 are e4m3 codes)")
                    break
            # scale-vs-payload alignment proof: a row whose scale byte is wrong would show here first

    print(f"\nrows compared against the checkpoint: {checked_rows}")
    if failures:
        print(f"FAIL: {failures} check group(s) mismatched")
        return 1
    print("PASS: every sampled MXFP8 row is the checkpoint's bytes, repacked - scale byte then 32 e4m3 codes -")
    print("      with the fused gate/up interleaving preserved, so payload rows and their scales stayed paired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
