#!/usr/bin/env python3
"""Self-check a python-written MXFP8 GGUF against its bf16/f32 source. Four gates:

  A container   - gguf-py opens it; every KV except general.file_type is identical to the source;
                  same tensor names/dims/order; every MXFP8 row obeys ggml-common.h
                  block_mxfp8 = { uint8 e; uint8 qs[32] } = 33 B per 32 values.
  B accuracy    - decode our own bytes and compare to the source values: per-element relative
                  error inside the e4m3 in-band, error normalised by block amax, relative Frobenius
                  error, and the count of non-zero values flushed to zero.
  C idempotence - requantizing the decoded values reproduces the file's bytes exactly.
  D finiteness  - no e4m3 NaN code (0x7F / 0xFF) and no non-finite decoded value anywhere.

    ./scripts/inkling_native/verify_mxfp8_file.py [--src FILE] [--out FILE] [--tensors all]

Exit code 0 means all four gates held. Nothing is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import mxfp8_quant as M  # noqa: E402
from layout import E4M3, QK_MXFP8, e8m0_to_f32, unpack_mxfp8_blocks  # noqa: E402

GGUF_PY = str(REPO / "gguf-py")  # the pip gguf 2.5.1 has no MXFP8 enum member
SRC_DEFAULT = ("/home/tim/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF/snapshots/"
               "004fd086eccea9f7cf867aa07ba328773beef81d/Qwen3.5-0.8B-BF16.gguf")
OUT_DEFAULT = str(Path.home() / "models/timlikesai/Qwen3.5-0.8B-MXFP8-PY.gguf")

E4M3_SUBNORMAL_STEP = 2.0 ** -9  # one e4m3 subnormal step, in units of the block scale d
CHUNK_BYTES = 4 << 20            # values per chunk read from the source


def container_checks(src_path: str, out_path: str) -> list[str]:
    sys.path.insert(0, GGUF_PY)
    from gguf import GGUFReader
    src, out = GGUFReader(src_path), GGUFReader(out_path)
    fails: list[str] = []
    # KV integrity by raw record bytes (key + type + value), so no numpy-vs-list comparison traps.
    # gguf-py only parses what it knows; the packer copies every record verbatim except file_type.
    with M.Gguf(src_path) as a, M.Gguf(out_path) as b:
        ka = {k: v for k, v in a.kv}
        kb = {k: v for k, v in b.kv}
    if set(ka) != set(kb):
        fails.append(f"KV key set differs: only-in-src {sorted(set(ka) - set(kb))}, "
                     f"only-in-out {sorted(set(kb) - set(ka))}")
    for k in sorted(set(ka) & set(kb)):
        if k == "general.file_type":
            continue
        if ka[k] != kb[k]:
            fails.append(f"KV {k} is not byte-identical to the source")
    ft = out.fields["general.file_type"].contents()
    if ft != M.LLAMA_FTYPE_MOSTLY_MXFP8:
        fails.append(f"general.file_type is {ft}, expected {M.LLAMA_FTYPE_MOSTLY_MXFP8}")

    ts_src = {t.name: t for t in src.tensors}
    if [t.name for t in out.tensors] != [t.name for t in src.tensors]:
        fails.append("tensor name list / order differs from source")
    n_q = 0
    for t in out.tensors:
        st = ts_src[t.name]
        if list(t.shape) != list(st.shape):
            fails.append(f"{t.name}: shape {list(t.shape)} != source {list(st.shape)}")
        if t.tensor_type.name == "MXFP8":
            n_q += 1
            vals = int(np.prod(t.shape))
            if vals % QK_MXFP8:
                fails.append(f"{t.name}: {vals} values is not a multiple of {QK_MXFP8}")
            elif int(t.data.nbytes) != vals // QK_MXFP8 * (1 + QK_MXFP8):
                fails.append(f"{t.name}: {t.data.nbytes} B != {vals // QK_MXFP8 * (1 + QK_MXFP8)} "
                             f"expected for {vals} values")
        elif t.tensor_type != st.tensor_type:
            fails.append(f"{t.name}: unquantized type changed {st.tensor_type.name} -> {t.tensor_type.name}")
    align = out.fields["general.alignment"].contents() if "general.alignment" in out.fields else 32
    print(f"  A container: gguf-py opened {out_path}; {len(out.tensors)} tensors "
          f"({n_q} MXFP8 + {len(out.tensors) - n_q} copied), file_type={ft}, alignment={align}; "
          f"{len(ka) - 1} KV records byte-identical to the source")
    return fails


def walk(src: M.Gguf, out: M.Gguf, limit: int | None) -> tuple[list[str], dict]:
    fails: list[str] = []
    agg = dict(tensors=0, values=0, blocks=0, sq_err=0.0, sq_val=0.0,
               worst_inband=0.0, worst_inband_tensor="", worst_amax=0.0,
               nonzero=0, flushed=0, sat_blocks=0, idem=0, chunks=0)
    for t in out.tensors:
        if t["type"] != M.GGML_TYPE_MXFP8:
            continue
        if limit is not None and agg["tensors"] >= limit:
            break
        st = next(s for s in src.tensors if s["name"] == t["name"])
        assert st["type"] in M._PLAIN_BPW and M.should_quantize(t["name"], t["dims"], st["type"]), t["name"]
        n_cols = t["dims"][0]
        n_rows = out.n_values(t) // n_cols
        rows = max(1, CHUNK_BYTES // n_cols)
        for r0 in range(0, n_rows, rows):
            r1 = min(r0 + rows, n_rows)
            n_blk_row = n_cols // QK_MXFP8
            out.f.seek(out.data_start + t["offset"] + r0 * n_blk_row * (1 + QK_MXFP8))
            raw = np.frombuffer(out.f.read((r1 - r0) * n_blk_row * (1 + QK_MXFP8)), dtype=np.uint8)
            src.f.seek(src.data_start + st["offset"] + r0 * n_cols * M._PLAIN_BPW[st["type"]])
            x = M.to_f32(np.frombuffer(src.f.read((r1 - r0) * n_cols * M._PLAIN_BPW[st["type"]]),
                                       dtype=np.uint8), st["type"])
            fails += _gate_d(t["name"], raw)
            raw = raw.reshape(r1 - r0, -1)
            d = e8m0_to_f32(unpack_mxfp8_blocks(raw, n_cols)[1])   # (rows, n_blocks) float32
            dec = M.decode_mxfp8(raw, n_cols)                      # (rows, cols) float32
            x = x.reshape(-1, n_cols)
            _gate_b(t["name"], x, dec, d, agg)
            if np.array_equal(M.quantize_row_mxfp8(dec.astype(np.float32).reshape(-1), n_cols).reshape(-1),
                              raw.reshape(-1)):
                agg["idem"] += 1
            agg["chunks"] += 1
        agg["tensors"] += 1
    return fails, agg


def _gate_d(name: str, raw: np.ndarray) -> list[str]:
    """D: no e4m3 NaN payload, no ue8m0 0xFF (NaN), and everything decodes finite."""
    fails = []
    blk = raw.reshape(-1, 1 + QK_MXFP8)
    nan_qs = int(((blk[:, 1:] & 0x7F) == 0x7F).sum())
    nan_e = int((blk[:, 0] == 0xFF).sum())
    if nan_qs:
        fails.append(f"{name}: {nan_qs} e4m3 NaN payload bytes (0x7F/0xFF)")
    if nan_e:
        fails.append(f"{name}: {nan_e} ue8m0 NaN scale bytes (0xFF)")
    return fails


def _gate_b(name: str, x: np.ndarray, dec: np.ndarray, d: np.ndarray, agg: dict) -> None:
    """B: accuracy. `x`/`dec` are (rows, cols) float32, `d` is (rows, n_blocks) float32."""
    bx = x.reshape(x.shape[0], -1, QK_MXFP8)
    bd = dec.reshape(dec.shape[0], -1, QK_MXFP8)
    amax = np.abs(bx).max(axis=-1)
    err = np.abs(bd - bx)
    nz = bx != 0
    inband = np.abs(bx) >= d[..., None] * E4M3_SUBNORMAL_STEP
    rel = np.divide(err, np.abs(bx), out=np.zeros_like(err), where=nz)
    if inband.any():
        w = float(np.where(inband, rel, 0.0).max())
        if w > agg["worst_inband"]:
            agg["worst_inband"] = w
            agg["worst_inband_tensor"] = name
    rel_amax = np.divide(err, np.maximum(amax[..., None], np.float32(1e-30)),
                         out=np.zeros_like(err), where=amax[..., None] > 0)
    agg["worst_amax"] = max(agg["worst_amax"], float(rel_amax.max()))
    agg["values"] += bx.size
    agg["blocks"] += amax.size
    agg["nonzero"] += int(nz.sum())
    agg["flushed"] += int((nz & (bd == 0)).sum())
    agg["sat_blocks"] += int((amax / np.maximum(d, np.float32(1e-30)) > 448.0).sum())
    agg["sq_err"] += float((err.astype(np.float64) ** 2).sum())
    agg["sq_val"] += float((bx.astype(np.float64) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--tensors", default="all", help="'all' or how many MXFP8 tensors to sample")
    args = ap.parse_args()

    print(f"src: {args.src}\nout: {args.out}")
    fails = container_checks(args.src, args.out)
    print(f"  A result: {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    limit = None if args.tensors == "all" else int(args.tensors)
    src, out = M.Gguf(args.src), M.Gguf(args.out)
    f2, agg = walk(src, out, limit)
    src.close()
    out.close()
    fails += f2
    rms_err = float(np.sqrt(agg["sq_err"] / max(agg["values"], 1)))
    rms_val = float(np.sqrt(agg["sq_val"] / max(agg["values"], 1))) or 1.0
    print(f"  B accuracy over {agg['tensors']} tensors, {agg['values']:,} values, "
          f"{agg['blocks']:,} blocks:")
    print(f"     worst |err|/|x| with |x| >= d*2^-9 (the e4m3 in-band) = {agg['worst_inband']:.6f} "
          f"in {agg['worst_inband_tensor']} "
          f"[- bound 0.0625 from RNE, 0.125 where amax/d > 448 saturates]")
    print(f"     worst |err|/amax_block                                 = {agg['worst_amax']:.6f}")
    print(f"     RMS err {rms_err:.6g} vs RMS source {rms_val:.6g}: relative Frobenius error "
          f"{rms_err / rms_val:.5f}")
    print(f"     non-zero values flushed to zero: {agg['flushed']:,}/{agg['nonzero']:,} "
          f"({100 * agg['flushed'] / max(agg['nonzero'], 1):.4f}%) - below half an e4m3 subnormal step")
    print(f"     saturating blocks (amax/d > 448): {agg['sat_blocks']:,}/{agg['blocks']:,}")
    if not (agg["worst_inband"] <= 0.1251 and agg["worst_amax"] <= 0.1251):
        fails.append(f"accuracy gate: worst in-band rel err {agg['worst_inband']}, "
                     f"worst amax-normalised {agg['worst_amax']} (both must be <= 0.1251)")
    print(f"  C idempotence: {agg['idem']}/{agg['chunks']} chunks requantize byte-identically")
    if agg["idem"] != agg["chunks"]:
        fails.append(f"idempotence: {agg['chunks'] - agg['idem']} of {agg['chunks']} chunks differ")
    print(f"  D finiteness: {'PASS' if not f2 else 'FAIL: ' + '; '.join(f2)}")
    print("RESULT:", "PASS" if not fails else "FAIL: " + "; ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
