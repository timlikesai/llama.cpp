#!/usr/bin/env python3
"""Are the DEQ-Q80 expert tensors the same numbers as the healthy public IQ4_NL build?

The two files differ in exactly one way: the MoE expert weights are Q8_0 there and IQ4_NL here.
If the sampled rows agree to within 4-bit rounding, the dequantized file's data is correct and
the non-finite logits must come from the path those numbers take at runtime.

    ./scripts/inkling_native/check_deq_vs_iq4nl.py [--rows 8] [--experts 0,1,64,127]

Exit code 0 = every sampled tensor agrees, 1 = at least one disagrees (layout or values).
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
from gguf import GGUFReader                       # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize                 # noqa: E402

DEQ = os.path.expanduser("~/models/timlikesai/Inkling-DEQ-Q80.gguf")
REF = os.path.expanduser("~/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf")


def blocks(raw, qtype):
    size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType(qtype)]
    assert raw.size % type_size == 0, f"{raw.size} bytes is not a whole number of {type_size}-byte blocks"
    n_blocks = raw.size // type_size
    return dequantize(raw.reshape(n_blocks, type_size), GGMLQuantizationType(qtype)).reshape(-1)[:n_blocks * size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--experts", default="0,1,64,127")
    args = ap.parse_args()

    a, b = GGUFReader(DEQ, "r"), GGUFReader(REF, "r")
    A = {t.name: t for t in a.tensors}
    B = {t.name: t for t in b.tensors}
    experts = [int(x) for x in args.experts.split(",")]
    shared = sorted(n for n in A if n in B and "exps" in n)
    print(f"expert tensors in DEQ={sum('exps' in n for n in A)} in IQ4_NL={sum('exps' in n for n in B)}, shared={len(shared)}")
    worst, fails = 0.0, 0
    for name in shared:
        ta, tb = A[name], B[name]
        sa, sb = [int(x) for x in ta.shape], [int(x) for x in tb.shape]
        if sa != sb:
            print(f"  SHAPE MISMATCH {name}: DEQ={sa} IQ4_NL={sb}")
            fails += 1
            continue
        cols, rows, n_exp = sa
        ra = np.asarray(ta.data).view(np.uint8).reshape(-1)
        rb = np.asarray(tb.data).view(np.uint8).reshape(-1)
        ba, ta_size = GGML_QUANT_SIZES[GGMLQuantizationType(ta.tensor_type)]
        bb, tb_size = GGML_QUANT_SIZES[GGMLQuantizationType(tb.tensor_type)]
        assert cols % ba == 0 and cols % bb == 0, f"{name}: {cols} columns is not a multiple of the block size"
        # 3-D GGUF storage is (expert, row, row_bytes), with row_bytes = cols/block*type_size
        A3 = ra.reshape(n_exp, rows, cols // ba * ta_size)
        B3 = rb.reshape(n_exp, rows, cols // bb * tb_size)
        idx = np.array([e for e in experts if e < n_exp])
        rows_idx = np.linspace(0, rows - 1, args.rows).astype(int)
        x = blocks(np.ascontiguousarray(A3[idx][:, rows_idx, :]), ta.tensor_type).reshape(len(idx), len(rows_idx), cols)
        y = blocks(np.ascontiguousarray(B3[idx][:, rows_idx, :]), tb.tensor_type).reshape(len(idx), len(rows_idx), cols)
        diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
        scale = max(float(np.abs(y).max()), 1e-9)
        rel = float(diff.max()) / scale
        print(f"  {name:<40} rows={len(rows_idx)}x{cols} experts={list(map(int, idx))} max|d|={diff.max():.6g} rel={rel:.4%} rms={np.sqrt((diff**2).mean()):.6g} max|x|={np.abs(y).max():.4g}")
        worst = max(worst, rel)
        if rel > 0.05:
            fails += 1
    print(f"\nworst relative disagreement: {worst:.4%}  ({'FAIL' if fails else 'agree within 4-bit rounding'})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
