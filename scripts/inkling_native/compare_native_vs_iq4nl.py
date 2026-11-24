#!/usr/bin/env python3
"""Cross-file probe: decode the same expert row out of the NATIVE (NVFP4/MXFP8) file and out of
the official lossy IQ4_NL file, with the byte conventions each type uses, and compare.

Both files come from the same converter for the row order (all gate rows, then all up rows) and
the IQ4_NL file is known good on CPU and CUDA, so it is an independent ground truth for what the
NATIVE bytes must decode to. Unlike verify_nvfp4_values.py, which compares the NATIVE file against
the HF checkpoint using a python model of the ggml decoder, this one compares two GGUF files, so a
convention that the converter and the python model agree on but the runtime does not shows up as a
mismatch.

Each row is decoded from the NATIVE bytes under several candidate conventions; the one that agrees
with the IQ4_NL reference identifies the bug.

    ./scripts/inkling_native/compare_native_vs_iq4nl.py [--native PATH] [--lossy PATH]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
sys.path.insert(0, HERE)

from gguf import GGUFReader                                          # noqa: E402
from verify_nvfp4_values import ggml_dequant_nvfp4, ggml_dequant_mxfp8  # noqa: E402

GGML_Q8_0, GGML_IQ4_NL, GGML_NVFP4, GGML_MXFP8 = 8, 20, 40, 44
QK_Q8_0, QK_IQ4NL, QK_NVFP4, QK_MXFP8 = 32, 32, 64, 32
KVALUES_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

DEFAULT_NATIVE = os.path.join(os.path.expanduser("~"), "models/timlikesai/Inkling-2-Mini-NATIVE.gguf")
DEFAULT_LOSSY = os.path.join(os.path.expanduser("~"), "models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf")


def ue4m3(b):
    """Port of ggml_ue4m3_to_fp32 (ggml/src/ggml-impl.h:521), including the *0.5."""
    b = np.asarray(b, dtype=np.uint8)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float32)
    v = np.where(exp == 0, np.ldexp(man, -9), np.ldexp(1.0 + man / 8.0, exp - 7)).astype(np.float32)
    v = np.where((b == 0) | (b == 0x7F), np.float32(0.0), v)
    return v * np.float32(0.5)


def e4m3(b):
    b = np.asarray(b, dtype=np.uint8)
    exp = ((b & 0x7F) >> 3).astype(np.int32)
    man = (b & 7).astype(np.float32)
    v = np.where(exp == 0, np.ldexp(man, -9), np.ldexp(1 + man / 8, exp - 7)).astype(np.float32)
    v = np.where(b & 0x80, -v, v)
    v[(b & 0x7F) == 0x7F] = np.nan
    return v


def iq4nl_decode(raw, cols):
    """dequantize_row_iq4_nl (ggml/src/ggml-quants.c:2801) with this build's QK4_NL = 32:
    18 bytes per 32 values, fp16 d + 16 packed nibbles, element j from the low nibble of qs[j]
    and element j + 16 from the high nibble."""
    nb = cols // QK_IQ4NL
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 18)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = blk[:, 2:]
    k = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
    out = np.empty((nb, QK_IQ4NL), dtype=np.float32)
    out[:, :QK_IQ4NL // 2] = k[qs & 0xF]
    out[:, QK_IQ4NL // 2:] = k[qs >> 4]
    return (out * d[:, None]).reshape(cols)


def q8_0_decode(raw, cols):
    """dequantize_row_q8_0: 34 bytes per 32 values, fp16 d + 32 int8."""
    nb = cols // QK_Q8_0
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    return (blk[:, 2:].astype(np.float32) * d[:, None]).reshape(cols)


def nvfp4_checkpoint_style(raw, cols):
    """Candidate wrong convention A: byte j of the payload holds elements 2i (low) and 2i+1 (high),
    i.e. the packing order the checkpoint itself uses, with the 4 scale bytes still in front."""
    nb = cols // QK_NVFP4
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 36)
    d = blk[:, :4]
    qs = blk[:, 4:]
    v = np.empty((nb, QK_NVFP4), dtype=np.float32)
    v[:, 0::2] = KVALUES_MXFP4[qs & 0xF]
    v[:, 1::2] = KVALUES_MXFP4[qs >> 4]
    s = np.repeat(ue4m3(d), 16, axis=1)
    return (v * s).reshape(cols)


def nvfp4_payload_first(raw, cols):
    """Candidate wrong convention B: the 32 payload bytes come first, the 4 scale bytes last."""
    nb = cols // QK_NVFP4
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 36)
    fixed = np.concatenate([blk[:, 32:36], blk[:, :32]], axis=1)
    return ggml_dequant_nvfp4(fixed[None], cols)[0]


def nvfp4_scale_per_32(raw, cols):
    """Candidate wrong convention C: sub-block scales interpreted as 32-element groups (mxfp4 style)."""
    nb = cols // QK_NVFP4
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 36)
    d = ue4m3(blk[:, :4])
    qs = blk[:, 4:]
    v = np.empty((nb, QK_NVFP4), dtype=np.float32)
    v[:, :QK_NVFP4 // 2] = KVALUES_MXFP4[qs & 0xF]
    v[:, QK_NVFP4 // 2:] = KVALUES_MXFP4[qs >> 4]
    s = np.empty((nb, QK_NVFP4), dtype=np.float32)
    for half in range(2):
        for j in range(QK_NVFP4 // 2):
            s[:, half * 32 + j] = d[:, 2 * (j // 16) + half]
    return (v * s).reshape(cols)


def mxfp8_scale_last(raw, cols):
    """Candidate wrong convention for MXFP8: 32 code bytes then the scale byte."""
    nb = cols // QK_MXFP8
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 33)
    fixed = np.concatenate([blk[:, 32:33], blk[:, :32]], axis=1)
    return ggml_dequant_mxfp8(fixed[None], cols)[0]


def mxfp8_half_scale(raw, cols):
    """Candidate wrong convention for MXFP8: scale decoded as e8m0/2 (the mxfp4 doubling rule)."""
    return ggml_dequant_mxfp8(np.frombuffer(raw, dtype=np.uint8)[None], cols)[0] * np.float32(0.5)


def rel(a, b):
    ref = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b))) / ref


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rows_of(reader):
    return {t.name: t for t in reader.tensors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", default=DEFAULT_NATIVE)
    ap.add_argument("--lossy", default=DEFAULT_LOSSY)
    ap.add_argument("--blocks", default="0,17,20,23")
    ap.add_argument("--experts", default="0,7,127")
    ap.add_argument("--rows", default="0,1,2,1151,1152,2303")
    args = ap.parse_args()

    nat = rows_of(GGUFReader(args.native))
    los = rows_of(GGUFReader(args.lossy))

    print(f"native: {args.native}\nlossy : {args.lossy}\n")
    summary = {}
    for block in (int(b) for b in args.blocks.split(",")):
        for proj in ("gate_up", "down"):
            w_name = f"blk.{block}.ffn_{proj}_exps.weight"
            s_name = f"blk.{block}.ffn_{proj}_exps.scale"
            nt, lt = nat.get(w_name), los.get(w_name)
            if nt is None or lt is None:
                print(f"  {w_name}: missing in one of the files (native={nt is not None} lossy={lt is not None})")
                continue
            n_type, l_type = int(nt.tensor_type), int(lt.tensor_type)
            if n_type not in (GGML_NVFP4, GGML_MXFP8):
                print(f"  {w_name}: native file type {n_type}, not a native expert - skipped")
                continue
            cols, rows, n_expert = (int(x) for x in nt.shape)
            l_cols, l_rows, _ = (int(x) for x in lt.shape)
            assert (cols, rows) == (l_cols, l_rows), (cols, rows, l_cols, l_rows)
            if l_type == GGML_IQ4_NL:
                lossy_decode, l_row_bytes = iq4nl_decode, cols // QK_IQ4NL * 18
            elif l_type == GGML_Q8_0:
                lossy_decode, l_row_bytes = q8_0_decode, cols // QK_Q8_0 * 34
            else:
                print(f"  {w_name}: lossy file type {l_type} is not handled - skipped")
                continue
            row_bytes = cols // QK_NVFP4 * 36 if n_type == GGML_NVFP4 else cols // QK_MXFP8 * 33
            ndata, ldata = nt.data, lt.data
            side = nat.get(s_name)
            factor = np.asarray(side.data, dtype=np.float32) if side is not None else None
            cands = ([("ggml", ggml_dequant_nvfp4), ("ckpt-nibbles", nvfp4_checkpoint_style),
                      ("payload-first", nvfp4_payload_first), ("scale-per-32", nvfp4_scale_per_32)]
                     if n_type == GGML_NVFP4 else
                     [("ggml", ggml_dequant_mxfp8), ("scale-last", mxfp8_scale_last), ("half-scale", mxfp8_half_scale)])
            print(f"\n  {w_name}  type={'NVFP4' if n_type == GGML_NVFP4 else 'MXFP8'} "
                  f"[K={cols}, rows={rows}, experts={n_expert}] sidecar={'yes' if factor is not None else 'no'} "
                  f"lossy={'IQ4_NL' if l_type == GGML_IQ4_NL else 'Q8_0'}")
            for e in (int(x) for x in args.experts.split(",")):
                for r in (int(x) for x in args.rows.split(",") if int(x) < rows):
                    # gguf-py memmaps quantized tensor data as [n_expert, rows, bytes_per_row]
                    raw = np.asarray(ndata[e, r], dtype=np.uint8)
                    lraw = np.asarray(ldata[e, r], dtype=np.uint8)
                    assert raw.size == row_bytes and lraw.size == l_row_bytes, (raw.size, lraw.size)
                    want = lossy_decode(lraw, cols)
                    for label, fn in cands:
                        got = (fn(raw[None], cols) if label == "ggml" else fn(raw, cols))
                        got = np.asarray(got).reshape(-1)
                        if factor is not None:
                            got = got * factor[e]
                        key = (block, proj, label)
                        summary[key] = (max(summary.get(key, (0, 0))[0], rel(got, want)),
                                        min(summary.get(key, (0, 0))[1], corr(got, want)))
                        if (e, r) == (int(args.experts.split(",")[0]), int(args.rows.split(",")[0])):
                            print(f"    e{e:<4} r{r:<5} {label:<14} rel={summary[key][0]:9.3e} corr={summary[key][1]:+.4f} "
                                  f"want[0:3]={np.round(want[:3], 5)} got[0:3]={np.round(got[:3], 5)}")

    print("\n=== worst relative error / weakest correlation over all sampled rows ===")
    for (block, proj, label), (bad, c) in sorted(summary.items()):
        flag = "MATCH" if bad < 0.2 and c > 0.99 else ("close" if c > 0.9 else "no")
        print(f"  blk.{block:<3} {proj:<8} {label:<14} rel={bad:9.3e} corr={c:+.4f}  {flag}")
    print("\nFor reference, IQ4_NL vs the exact weights is a few percent, so 'MATCH' is the")
    print("expected verdict for the true convention and only the true convention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
