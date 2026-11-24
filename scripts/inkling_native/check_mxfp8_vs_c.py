#!/usr/bin/env python3
"""Cross-check scripts/inkling_native/mxfp8_quant.py against the C MXFP8 quantizer.

Calls ggml_quantize_chunk(GGML_TYPE_MXFP8, ...) from the already-built
build-cuda/bin/libggml-base.so via ctypes - no build, no source change - which
reaches quantize_mxfp8 -> quantize_row_mxfp8_ref (ggml/src/ggml-quants.c:381).
Any byte difference between C and python on the same float32 input is a real
convention bug in one of the two, so this is the gate the python file must pass
before its numbers mean anything.

    python3 scripts/inkling_native/check_mxfp8_vs_c.py [--model SRC.gguf]
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mxfp8_quant as M  # noqa: E402

REPO = HERE.parent.parent
LIB = REPO / "build-cuda" / "bin" / "libggml-base.so.1.0.0"
MXFP8_BLOCK = 1 + M.QK_MXFP8


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB), mode=ctypes.RTLD_GLOBAL)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_type_name.restype = ctypes.c_char_p
    lib.ggml_type_name.argtypes = [ctypes.c_int]
    lib.ggml_blck_size.restype = ctypes.c_int64
    lib.ggml_blck_size.argtypes = [ctypes.c_int]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.dequantize_row_mxfp8.restype = None
    lib.dequantize_row_mxfp8.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    return lib


def c_dequantize(lib: ctypes.CDLL, raw: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """C dequantize_row_mxfp8 (ggml-quants.c:610), one row per call - the decoder oracle."""
    blk = np.ascontiguousarray(raw.reshape(n_rows, n_cols // M.QK_MXFP8 * MXFP8_BLOCK), dtype=np.uint8)
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    for i in range(n_rows):
        lib.dequantize_row_mxfp8(blk[i].ctypes.data_as(ctypes.c_void_p),
                                 out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_cols)
    return out


def c_quantize(lib: ctypes.CDLL, x: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    assert x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    row_bytes = int(lib.ggml_row_size(M.GGML_TYPE_MXFP8, ctypes.c_int64(n_cols)))
    assert row_bytes == n_cols // M.QK_MXFP8 * MXFP8_BLOCK, (row_bytes, n_cols)
    dst = np.zeros(row_bytes * n_rows, dtype=np.uint8)
    got = lib.ggml_quantize_chunk(
        M.GGML_TYPE_MXFP8,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.c_void_p),
        0, n_rows, n_cols, None,
    )
    if got != dst.nbytes:
        raise RuntimeError(f"ggml_quantize_chunk returned {got}, expected {dst.nbytes}")
    return dst


def compare(name: str, x: np.ndarray, n_cols: int, lib: ctypes.CDLL, dec_rows: int = 256) -> tuple[int, int, int]:
    n_rows = x.size // n_cols
    x = np.ascontiguousarray(x, dtype=np.float32)
    c = c_quantize(lib, x, n_rows, n_cols)
    py = M.quantize_row_mxfp8(x, n_cols).reshape(-1)
    if c.shape != py.shape:
        raise AssertionError(f"{name}: shape {py.shape} vs C {c.shape}")
    same = int((c == py).sum())
    if same != c.size:
        nb = int((c.reshape(-1, MXFP8_BLOCK) != py.reshape(-1, MXFP8_BLOCK)).any(axis=1).sum())
        print(f"  MISMATCH {name}: {c.size - same} of {c.size} bytes differ, in {nb} of {c.size // MXFP8_BLOCK} blocks")
        bad = np.flatnonzero((c.reshape(-1, MXFP8_BLOCK) != py.reshape(-1, MXFP8_BLOCK)).any(axis=1))[:3]
        for b in bad:
            print(f"    block {b}: C e={c.reshape(-1, MXFP8_BLOCK)[b, 0]} qs={c.reshape(-1, MXFP8_BLOCK)[b, 1:6]}... "
                  f"PY e={py.reshape(-1, MXFP8_BLOCK)[b, 0]} qs={py.reshape(-1, MXFP8_BLOCK)[b, 1:6]}...")
            blk = x.reshape(-1, M.QK_MXFP8)[b]
            print(f"      amax={np.abs(blk).max():.9g} log2={np.log2(np.abs(blk).max()):.6f}")
        return same, c.size, 0

    # bytes agree, so decode our own blocks with both decoders and demand bit-identical floats
    rows = min(n_rows, dec_rows)
    nbytes = rows * n_cols // M.QK_MXFP8 * MXFP8_BLOCK
    dec_c = c_dequantize(lib, py[:nbytes], rows, n_cols)
    dec_py = M.decode_mxfp8(py[:nbytes].reshape(rows, -1), n_cols)
    if not np.array_equal(dec_c, dec_py):
        d = np.abs(dec_c - dec_py)
        print(f"  DECODE MISMATCH {name}: {(d > 0).sum()} of {dec_c.size} values differ, max |diff| {d.max():.3g}")
        return same, c.size, 1
    print(f"  ok {name}: {c.size:,} bytes identical to C, {dec_c.size:,} decoded values bit-identical to C")
    return same, c.size, 0


def synthetic(lib) -> tuple[int, int]:
    rng = np.random.default_rng(42)
    cases = {
        "randn(1.0)": rng.standard_normal(4096 * 64).astype(np.float32),
        "randn*1e-3": (rng.standard_normal(4096 * 64) * 1e-3).astype(np.float32),
        "randn*1e-6 (subnormal band)": (rng.standard_normal(4096 * 64) * 1e-6).astype(np.float32),
        "randn*100 (saturating)": (rng.standard_normal(4096 * 64) * 100).astype(np.float32),
        "uniform +-1": rng.uniform(-1, 1, 4096 * 64).astype(np.float32),
        "exact powers of two": np.ldexp(np.ones(1024 * 32, dtype=np.float32), rng.integers(-40, 40, 1024 * 32)).astype(np.float32),
        "amax exactly 448": np.full(1024 * 32, 448.0, dtype=np.float32),
        "amax exactly 512": np.full(1024 * 32, 512.0, dtype=np.float32),
        "block with one big": (np.tile(np.array([[0.01, 7.0] * 16], dtype=np.float32), (1024, 1))).reshape(-1),
        "zeros": np.zeros(4096 * 64, dtype=np.float32),
        "mixed zeros and big": np.where(rng.random(4096 * 64) < 0.9, 0.0, rng.standard_normal(4096 * 64) * 1e4).astype(np.float32),
    }
    tot = bad = dec_bad = 0
    for name, x in cases.items():
        s, n, d = compare(name, x, 64, lib)
        tot += n
        bad += n - s
        dec_bad += d
    return tot, bad, dec_bad


def real_tensors(lib, model: str) -> tuple[int, int, int]:
    src = M.Gguf(model)
    tot = bad = dec_bad = 0
    seen = set()
    for t in src.tensors:
        if not M.should_quantize(t["name"], t["dims"], t["type"]):
            continue
        key = t["name"].split(".", 1)[1] if t["name"].startswith("blk.") else t["name"]
        if key in seen:
            continue
        seen.add(key)
        n_cols = t["dims"][0]
        n_rows = src.n_values(t) // n_cols
        rows = min(n_rows, 2048)
        raw = src.read_tensor(t)[: rows * n_cols * M._PLAIN_BPW[t["type"]]]
        x = M.to_f32(raw, t["type"])
        s, n, d = compare(f"{t['name']} [{rows}x{n_cols}]", x, n_cols, lib, dec_rows=64)
        tot += n
        bad += n - s
        dec_bad += d
    src.close()
    print(f"  ({len(seen)} distinct tensor patterns checked)")
    return tot, bad, dec_bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/home/tim/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF"
                                       "/snapshots/004fd086eccea9f7cf867aa07ba328773beef81d/Qwen3.5-0.8B-BF16.gguf",
                    help="GGUF to sample real tensors from")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()

    lib = load_lib()
    print(f"library: {LIB}")
    print(f"GGML_TYPE_MXFP8 = {M.GGML_TYPE_MXFP8} name={lib.ggml_type_name(M.GGML_TYPE_MXFP8).decode()} "
          f"blck_size={lib.ggml_blck_size(M.GGML_TYPE_MXFP8)}")

    tot, bad, dec_bad = synthetic(lib)
    if not args.no_model:
        t2, b2, d2 = real_tensors(lib, args.model)
        tot += t2
        bad += b2
        dec_bad += d2
    print(f"TOTAL: {tot - bad:,}/{tot:,} bytes byte-identical to the C quantizer, {bad} differing; "
          f"{dec_bad} case(s) where the python decoder disagreed with C dequantize_row_mxfp8")
    return 0 if bad == 0 and dec_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
