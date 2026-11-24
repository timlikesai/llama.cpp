#!/usr/bin/env python3
"""Value-level check: does the NATIVE gguf's expert payload, decoded the way ggml's C code
decodes it and multiplied by the `.scale` sidecar, equal the checkpoint's float weight?

The byte-level verifiers (verify_mxfp8_gguf.py / verify_nvfp4_gguf.py) prove the file holds the
checkpoint's bytes in ggml block order. They cannot prove that ggml's *decode* of those bytes
reproduces the model's weights. This does, by porting the two C decoders
(ggml/src/ggml-quants.c:dequantize_row_nvfp4 / dequantize_row_mxfp8, ggml/src/ggml-impl.h:
ggml_ue4m3_to_fp32) and comparing against conversion/inkling2.py:_dequant_expert_chunk.

    ./scripts/inkling_native/verify_nvfp4_values.py [--gguf PATH] [--blocks 0,17,20,23]

Exit code 0 means every sampled row matches the reference to fp32 rounding.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "conversion"))

from gguf import GGUFReader                                    # noqa: E402
from verify_mxfp8_gguf import Safetensors, MXFP8, NVFP4        # noqa: E402

DEFAULT_GGUF = os.path.join(os.path.expanduser("~"), "models/timlikesai/Inkling-2-Mini-NATIVE.gguf")
SNAP_GLOB = "models--thinkingmachines--Inkling-2-Mini/snapshots/*"

# ggml/src/ggml-common.h: kvalues_fp4 = "e2m1 values (doubled)"
KVALUES_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
# conversion/inkling2.py:_E2M1, the true (not doubled) codebook
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def ue4m3_to_fp32(b, half=True):
    """Port of ggml/src/ggml-impl.h:521 ggml_ue4m3_to_fp32 (note the *0.5)."""
    b = np.asarray(b, dtype=np.uint8)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float32)
    v = np.where(exp == 0, np.ldexp(man, -9), np.ldexp(1.0 + man / 8.0, exp - 7)).astype(np.float32)
    v = np.where((b == 0) | (b == 0x7F), np.float32(0.0), v)
    return v * np.float32(0.5) if half else v


def e4m3_to_fp32(b):
    """Signed e4m3fn, as conversion/inkling2.py:_e4m3_table does."""
    b = np.asarray(b, dtype=np.uint8)
    exp = ((b & 0x7F) >> 3).astype(np.int32)
    man = (b & 7).astype(np.float32)
    v = np.where(exp == 0, np.ldexp(man, -9), np.ldexp(1 + man / 8, exp - 7)).astype(np.float32)
    v = np.where(b & 0x80, -v, v)
    v[(b & 0x7F) == 0x7F] = np.nan
    return v


def ggml_dequant_nvfp4(raw, cols, half_scale=True):
    """Port of dequantize_row_nvfp4: raw = [..., cols/64 * 36] -> [..., cols]."""
    nb = cols // 64
    blk = raw.reshape(-1, nb, 36)
    scales = blk[:, :, :4].astype(np.uint8)          # [rows, nb, 4] one per 16 elements
    qs = blk[:, :, 4:]                               # [rows, nb, 32]
    lo = KVALUES_MXFP4[qs & 0xF]
    hi = KVALUES_MXFP4[qs >> 4]
    # byte j of sub-block s holds element j (low nibble) and j+8 (high nibble)
    out = np.empty(lo.shape[:2] + (64,), dtype=np.float32)
    for s in range(4):
        for j in range(8):
            out[:, :, s*16 + j + 0] = lo[:, :, s*8 + j]
            out[:, :, s*16 + j + 8] = hi[:, :, s*8 + j]
    d = ue4m3_to_fp32(scales, half=half_scale)       # [rows, nb, 4]
    d = np.repeat(d, 16, axis=2)                     # [rows, nb, 64]
    return (out * d).reshape(raw.shape[0], cols)


def ggml_dequant_mxfp8(raw, cols):
    """Port of dequantize_row_mxfp8: raw = [..., cols/32 * 33] -> [..., cols]."""
    nb = cols // 32
    blk = raw.reshape(-1, nb, 33)
    e = blk[:, :, 0].astype(np.int32)
    vals = e4m3_to_fp32(blk[:, :, 1:])
    d = np.ldexp(np.ones_like(e, dtype=np.float32), e - 127)
    return (vals * d[:, :, None]).reshape(raw.shape[0], cols)


def f32_flat(st, name):
    """A small F32 checkpoint tensor (amax / outer_scale_denominator) as a flat array."""
    meta = st.header[name]
    lo, hi = meta["data_offsets"]
    st.f.seek(st.base + lo)
    return np.frombuffer(st.f.read(hi - lo), dtype="<f4")


def hf_reference(st, hf_stem, expert, hf_row, cols, mxfp8):
    """One row of the checkpoint weight as float32, per conversion/inkling2.py:_dequant_expert_chunk."""
    if mxfp8:
        packed = np.frombuffer(st.row(hf_stem, expert, hf_row, cols), dtype=np.uint8)
        scales = np.frombuffer(st.row(hf_stem + "_scale", expert, hf_row, cols // 32), dtype=np.uint8)
        v = e4m3_to_fp32(packed).reshape(-1, 32)
        d = np.ldexp(np.ones(cols // 32, dtype=np.float32), scales.astype(np.int32) - 127)
        return (v * d[:, None]).reshape(-1)
    packed = np.frombuffer(st.row(hf_stem, expert, hf_row, cols // 2), dtype=np.uint8)
    scales = np.frombuffer(st.row(hf_stem + "_scale", expert, hf_row, cols // 16), dtype=np.uint8)
    v = np.empty(cols, dtype=np.float32)
    v[0::2] = E2M1[packed & 15]
    v[1::2] = E2M1[packed >> 4]
    v = (v.reshape(-1, 16) * e4m3_to_fp32(scales)[:, None]).reshape(-1)
    a = f32_flat(st, hf_stem + "_amax")[expert]
    d = f32_flat(st, hf_stem + "_outer_scale_denominator").ravel()[0]
    return v * (np.float32(a) / np.float32(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--blocks", default="0,17,20,23")
    ap.add_argument("--experts", default="0,64,127")
    ap.add_argument("--rows", default="0,1,63,1151,1152,2303")
    args = ap.parse_args()

    snap = glob.glob(os.path.expanduser(os.path.join("~/.cache/huggingface/hub", SNAP_GLOB)))[0]
    st = Safetensors(os.path.join(snap, "model-0001" if False else "model-00001-of-00001.safetensors"))
    r = GGUFReader(args.gguf)
    tensors = {t.name: t for t in r.tensors}

    worst = {}
    for block in [int(b) for b in args.blocks.split(",")]:
        for proj, hf_proj in (("gate_up", "w13"), ("down", "w2")):
            name = f"blk.{block}.ffn_{proj}_exps.weight"
            t = tensors.get(name)
            if t is None:
                print(f"  {name}: ABSENT")
                continue
            tt = int(t.tensor_type)
            mxfp8 = tt == MXFP8
            if tt not in (MXFP8, NVFP4):
                print(f"  {name}: type {tt}, not native -> skipped")
                continue
            ne0, ne1, ne2 = (int(x) for x in t.shape)   # GGUF order: [K=cols, rows, n_expert]
            experts, rows, cols = ne2, ne1, ne0
            row_bytes = cols // 32 * 33 if mxfp8 else cols // 64 * 36
            raw = np.asarray(t.data).view(np.uint8).reshape(experts, rows, row_bytes)
            side = tensors.get(f"blk.{block}.ffn_{proj}_exps.scale")
            factor = np.asarray(side.data, dtype=np.float32) if side is not None else np.ones(experts, np.float32)
            hf_stem = f"model.llm.layers.{block}.mlp.experts.{hf_proj}_weight"
            print(f"\n  {name}  type={'MXFP8' if mxfp8 else 'NVFP4'} experts={experts} rows={rows} cols={cols} "
                  f"sidecar={'yes' if side is not None else 'NO'}")
            for e in [int(x) for x in args.experts.split(",")]:
                for gr in [int(x) for x in args.rows.split(",") if int(x) < rows]:
                    # fused w13: gguf rows are [all gate][all up], checkpoint interleaves them
                    if proj == "gate_up":
                        half, local = gr // (rows // 2), gr % (rows // 2)
                        hf_row = 2 * local + half
                    else:
                        hf_row = gr
                    want = hf_reference(st, hf_stem, e, hf_row, cols, mxfp8)
                    row = raw[e, gr]
                    for label, half_scale in (("ggml (0.5)", True), ("no-0.5", False)) if not mxfp8 else (("ggml", True),):
                        got = (ggml_dequant_mxfp8(row[None], cols) if mxfp8
                               else ggml_dequant_nvfp4(row[None], cols, half_scale))[0] * factor[e]
                        err = np.max(np.abs(got - want))
                        ref = max(np.max(np.abs(want)), 1e-30)
                        key = (block, proj, label)
                        rel = err / ref
                        worst[key] = max(worst.get(key, 0.0), rel)
                        if (e, gr) == (int(args.experts.split(',')[0]), int(args.rows.split(',')[0])):
                            print(f"    e{e:<4} r{gr:<5} {label:<11} max|got-want|={err:.3e} "
                                  f"rel={rel:.3e}  want[0:4]={want[:4]} got[0:4]={got[:4]}")
    print("\n=== worst relative error per (block, proj, decode) ===")
    bad = 0
    for (block, proj, label), rel in sorted(worst.items()):
        control = label != "ggml" and not label.endswith("(0.5)")
        flag = "OK" if rel < 1e-6 else ("CONTROL(2x)" if control else "MISMATCH")
        bad += rel >= 1e-6 and not control
        print(f"  blk.{block:<3} {proj:<8} {label:<11} rel={rel:.3e}  {flag}")
    print("\n'no-0.5' is a deliberate control: it drops the *0.5 in ggml_ue4m3_to_fp32, which must be"
          " cancelled by kvalues_fp4 being 2x E2M1.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
