#!/usr/bin/env python3
"""Requantize a GGUF's 2-D weights to GGML_TYPE_MXFP8 (type 44) in python.

Exists because llama-quantize cannot emit MXFP8 today: the quantizer itself is
wired (ggml/src/ggml-quants.c:2372 quantize_mxfp8 -> ggml.c:8075 ggml_quantize_chunk),
but ggml_validate_row_data (ggml/src/ggml-quants.c:5564) has no GGML_TYPE_MXFP8
case, so llama-quantize's post-quant validation reaches `default` at :5826 and
aborts with "invalid type 44". Model load is unaffected (check_tensors defaults
to false, src/llama-model.cpp:3611).

The encoder is a bit-exact mirror of ggml quantize_row_mxfp8_ref
(ggml/src/ggml-quants.c:381-417) and ggml_fp32_to_f8_e4m3
(ggml/src/ggml-impl.h:601-630):

  per 32 contiguous elements (QK_MXFP8 = 32, blocks run along ne[0]):
    amax = max(|x|)
    e    = amax > 0 ? clamp(floor(log2(amax)) - 8 + 127, 0, 254) : 0
    d    = 2^(e - 127)                       # GGML_E8M0_TO_FP32, NOT the _HALF variant
    qs[] = f8_e4m3(x / d)                    # saturating RNE, |v| >= 448 -> +/-448
  block_mxfp8 = { uint8 e; uint8 qs[32] }    33 B per 32 values = 8.25 bpw

floor(log2(amax)) - 8 puts the block max at 256..448 of 448, so e4m3 saturates only
inside the (448, 512) * d band - the OCP MX rule.

Tensor policy matches llama-quantize: src/llama-quant.cpp:302 tensor_allows_quantization
(2-D+, name ends in "weight", not *_norm.weight, not *ssm_conv1d*, not *shortconv.conv*,
not *altup*, not *laurel*, not ffn_gate_inp.weight) plus src/llama-quant.cpp:466
tensor_type_fallback's shape rule ne[0] % 32 == 0. Everything else is copied verbatim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout import E4M3, QK_MXFP8, e8m0_to_f32, pack_mxfp8_blocks, unpack_mxfp8_blocks  # noqa: E402

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
GGML_TYPE_MXFP8 = 44
LLAMA_FTYPE_MOSTLY_MXFP8 = 43  # include/llama.h:162

MAGIC = b"GGUF"
T_STRING, T_ARRAY = 8, 9
_SCALAR_FMT = {0: "<u1", 1: "<i1", 2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4", 6: "<f4", 7: "<u1", 10: "<u8", 11: "<i8", 12: "<f8"}

# ggml type -> bytes per value, for the source types this tool accepts un-quantized
_PLAIN_BPW = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_BF16: 2}
# excluded by name in tensor_allows_quantization
_NAME_SKIP = ("_norm.weight", "ssm_conv1d", "shortconv.conv.weight", "altup", "laurel", "ffn_gate_inp.weight")


# ---------------------------------------------------------------- e4m3 encode / MXFP8 block


def _rne(x: np.ndarray) -> np.ndarray:
    """ggml_round_to_nearest_even (ggml-impl.h:595-599) on float32 input."""
    f = np.floor(x)
    frac = x - f
    return (f + ((frac > 0.5) | ((frac == 0.5) & ((f.astype(np.int64) & 1) == 1)))).astype(np.int64)


def f32_to_e4m3(x: np.ndarray) -> np.ndarray:
    """float32 -> e4m3fn code, saturating. Mirror of ggml_fp32_to_f8_e4m3 (ggml-impl.h:601)."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.where(np.signbit(x), 0x80, 0).astype(np.uint8)
    a = np.abs(x)
    out = np.zeros(a.shape, dtype=np.uint8)

    m_nan = np.isnan(a)
    m_zero = a == 0
    m_big = np.isinf(a) | (a >= np.float32(448.0))
    m_sub = ~m_nan & ~m_zero & ~m_big & (a < np.float32(0.015625))  # 2^-6 = smallest e4m3 normal
    m_nrm = ~m_nan & ~m_zero & ~m_big & ~m_sub

    out[m_nan] = 0x7F
    out[m_big] = 0x7E
    if m_sub.any():
        out[m_sub] = _rne(a[m_sub] * np.float32(512.0)).astype(np.uint8)
    if m_nrm.any():
        v = a[m_nrm]
        e2 = np.floor(np.log2(v)).astype(np.int64)  # frexpf exponent = e2 + 1 -> encoded_exp = e2 + 7
        mant = v / np.ldexp(np.ones_like(v), e2.astype(np.int32))  # in [1, 2)
        enc_exp = e2 + 7
        enc_man = _rne((mant - np.float32(1.0)) * np.float32(8.0))
        carry = enc_man == 8
        enc_man = np.where(carry, 0, enc_man)
        enc_exp = enc_exp + carry.astype(np.int64)
        ovf = (enc_exp > 15) | ((enc_exp == 15) & (enc_man > 6))
        out[m_nrm] = np.where(ovf, 0x7E, (enc_exp << 3) | enc_man).astype(np.uint8)
    return out | sign


def quantize_row_mxfp8(x_f32: np.ndarray, n_cols: int) -> np.ndarray:
    """float32 flat array (n_rows * n_cols) -> block_mxfp8 bytes, per quantize_row_mxfp8_ref."""
    if n_cols % QK_MXFP8:
        raise ValueError(f"ne[0] = {n_cols} is not a multiple of {QK_MXFP8}")
    if x_f32.size % n_cols:
        raise ValueError(f"{x_f32.size} values is not a multiple of {n_cols}")
    blk = x_f32.reshape(-1, n_cols // QK_MXFP8, QK_MXFP8)
    if not np.isfinite(blk).all():
        raise ValueError("source weight contains non-finite values")

    amax = np.abs(blk).max(axis=-1)
    e = np.zeros(amax.shape, dtype=np.int64)
    pos = amax > 0
    if pos.any():
        e[pos] = np.floor(np.log2(amax[pos])).astype(np.int64) - 8 + 127
    e = np.clip(e, 0, 254)  # 255 is the ue8m0 NaN byte

    d = np.ldexp(np.ones(amax.shape, dtype=np.float32), (e - 127).astype(np.int32))
    qs = f32_to_e4m3(blk / d[..., None])
    return pack_mxfp8_blocks(qs.reshape(x_f32.size // n_cols, -1), e.astype(np.uint8))


def decode_mxfp8(raw: np.ndarray, n_cols: int) -> np.ndarray:
    """block_mxfp8 rows -> float32. Mirror of dequantize_row_mxfp8 (ggml-quants.c:610).

    `raw` is (..., n_cols / 32 * 33); the ue8m0 byte is per block, so the payload has to be
    regrouped to (..., n_blocks, 32) before the scale broadcasts.
    """
    w, s = unpack_mxfp8_blocks(raw, n_cols)
    g = w.shape[:-1] + (n_cols // QK_MXFP8, QK_MXFP8)
    return (E4M3[w].reshape(g) * e8m0_to_f32(s)[..., None]).reshape(-1, n_cols)


# ---------------------------------------------------------------- GGUF container


def _pad(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def _read_str(f) -> str:
    (n,) = np.frombuffer(f.read(8), dtype="<u8")
    return f.read(int(n)).decode("utf-8")


def _skip_value(f, vtype: int) -> None:
    if vtype == T_STRING:
        (n,) = np.frombuffer(f.read(8), dtype="<u8")
        f.seek(int(n), 1)
    elif vtype == T_ARRAY:
        (etype,) = np.frombuffer(f.read(4), dtype="<u4")
        (n,) = np.frombuffer(f.read(8), dtype="<u8")
        for _ in range(int(n)):
            _skip_value(f, int(etype))
    elif vtype in _SCALAR_FMT:
        f.seek(np.dtype(_SCALAR_FMT[vtype]).itemsize, 1)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


class Gguf:
    """GGUF v2/v3 reader that keeps KV records as raw bytes, so metadata copy cannot mangle them."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        f = self.f
        if f.read(4) != MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        self.version = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        self.n_tensors = int(np.frombuffer(f.read(8), dtype="<u8")[0])
        self.n_kv = int(np.frombuffer(f.read(8), dtype="<u8")[0])

        self.kv: list[tuple[str, bytes]] = []
        self.scalar: dict[str, object] = {}
        for _ in range(self.n_kv):
            start = f.tell()
            key = _read_str(f)
            (vtype,) = np.frombuffer(f.read(4), dtype="<u4")
            val_start = f.tell()
            if int(vtype) in _SCALAR_FMT:
                dt = np.dtype(_SCALAR_FMT[int(vtype)])
                self.scalar[key] = np.frombuffer(f.read(dt.itemsize), dtype=dt)[0].item()
            else:
                f.seek(val_start)
                _skip_value(f, int(vtype))
            end = f.tell()
            f.seek(start)
            self.kv.append((key, f.read(end - start)))
            f.seek(end)

        self.tensors: list[dict] = []
        for _ in range(self.n_tensors):
            name = _read_str(f)
            (n_dims,) = np.frombuffer(f.read(4), dtype="<u4")
            dims = [int(d) for d in np.frombuffer(f.read(8 * int(n_dims)), dtype="<u8")]
            ttype = int(np.frombuffer(f.read(4), dtype="<u4")[0])
            offset = int(np.frombuffer(f.read(8), dtype="<u8")[0])
            self.tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

        self.alignment = int(self.scalar.get("general.alignment", 32) or 32)
        self.data_start = _pad(f.tell(), self.alignment)

    def n_values(self, t) -> int:
        n = 1
        for d in t["dims"]:
            n *= int(d)
        return n

    def n_bytes(self, t) -> int:
        """On-disk size of a tensor, for the types this tool reads or writes."""
        n = self.n_values(t)
        if t["type"] in _PLAIN_BPW:
            return n * _PLAIN_BPW[t["type"]]
        if t["type"] == GGML_TYPE_MXFP8:
            return n // QK_MXFP8 * (1 + QK_MXFP8)
        raise ValueError(f"{t['name']}: unsupported ggml type {t['type']}")

    def read_tensor(self, t) -> np.ndarray:
        self.f.seek(self.data_start + t["offset"])
        return np.frombuffer(self.f.read(self.n_bytes(t)), dtype=np.uint8)

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "Gguf":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def to_f32(raw: np.ndarray, ttype: int) -> np.ndarray:
    if ttype == GGML_TYPE_F32:
        return raw.view(np.float32)
    if ttype == GGML_TYPE_BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)  # exact
    if ttype == GGML_TYPE_F16:
        return raw.view(np.float16).astype(np.float32)
    raise ValueError(f"cannot read ggml type {ttype} as float")


def should_quantize(name: str, dims: list[int], ttype: int) -> bool:
    if ttype not in _PLAIN_BPW:
        return False  # llama-quantize would need --allow-requantize; keep out of scope
    if sum(1 for d in dims if d != 1) < 2 or not name.endswith("weight"):
        return False
    if any(s in name for s in _NAME_SKIP):
        return False
    return dims[0] % QK_MXFP8 == 0


# ---------------------------------------------------------------- conversion


def convert(src_path: str, out_path: str, chunk_rows: int = 65536) -> dict:
    src = Gguf(src_path)
    st = {"quantized": 0, "copied": 0, "bytes_in": 0, "bytes_out": 0, "values": 0,
          "worst_rel": 0.0, "worst_rel_tensor": "", "max_abs": 0.0,
          "requant_value_exact": 0, "requant_byte_identical": 0, "sat_blocks": 0, "n_blocks": 0}

    # pass 1: decide types and offsets (sizes are analytic, no data read)
    plan = []
    off = 0
    for t in src.tensors:
        q = should_quantize(t["name"], t["dims"], t["type"])
        n = src.n_values(t)
        nbytes = n // QK_MXFP8 * (1 + QK_MXFP8) if q else src.n_bytes(t)
        plan.append((t, q, nbytes, off))
        st["bytes_in"] += src.n_bytes(t)
        st["bytes_out"] += nbytes
        off += _pad(nbytes, src.alignment)

    with open(out_path, "wb") as out:
        out.write(MAGIC)
        out.write(np.uint32(src.version).tobytes())
        out.write(np.uint64(src.n_tensors).tobytes())
        out.write(np.uint64(src.n_kv).tobytes())
        for key, raw in src.kv:
            if key == "general.file_type":
                k = key.encode()
                out.write(np.uint64(len(k)).tobytes() + k + np.uint32(4).tobytes() + np.uint32(LLAMA_FTYPE_MOSTLY_MXFP8).tobytes())
            else:
                out.write(raw)
        for t, q, nbytes, o in plan:
            nb = t["name"].encode()
            out.write(np.uint64(len(nb)).tobytes() + nb)
            out.write(np.uint32(len(t["dims"])).tobytes())
            for d in t["dims"]:
                out.write(np.uint64(d).tobytes())
            out.write(np.uint32(GGML_TYPE_MXFP8 if q else t["type"]).tobytes())
            out.write(np.uint64(o).tobytes())
        out.write(bytes(_pad(out.tell(), src.alignment) - out.tell()))
        assert out.tell() == src.data_start, (out.tell(), src.data_start)

        # pass 2: stream the data
        for t, q, nbytes, o in plan:
            assert out.tell() == src.data_start + o, (t["name"], out.tell(), src.data_start + o)
            if q:
                blob = _quantize_tensor(src, t, st, chunk_rows)
                st["quantized"] += 1
            else:
                blob = src.read_tensor(t)
                st["copied"] += 1
            assert blob.nbytes == nbytes, (t["name"], blob.nbytes, nbytes)
            out.write(blob.tobytes() if isinstance(blob, np.ndarray) else blob)
            out.write(bytes(_pad(out.tell(), src.alignment) - out.tell()))
    src.close()
    return st


def _quantize_tensor(src: Gguf, t: dict, st: dict, chunk_rows: int) -> np.ndarray:
    n_cols = t["dims"][0]
    n_rows = src.n_values(t) // n_cols
    out = np.empty(n_rows * (n_cols // QK_MXFP8) * (1 + QK_MXFP8), dtype=np.uint8)
    base = src.data_start + t["offset"]
    bytes_per_row = n_cols * _PLAIN_BPW[t["type"]]
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        src.f.seek(base + r0 * bytes_per_row)
        x = to_f32(np.frombuffer(src.f.read((r1 - r0) * bytes_per_row), dtype=np.uint8), t["type"])
        part = quantize_row_mxfp8(x, n_cols)
        out[(r0 * (n_cols // QK_MXFP8) * 33):(r1 * (n_cols // QK_MXFP8) * 33)] = part.reshape(-1)
        _check_chunk(x, part, n_cols, t["name"], st)
    st["values"] += n_rows * n_cols
    return out


def _check_chunk(x_f32, raw, n_cols, name, st):
    """Decode our own bytes; check error scale, value-level requantize idempotence, saturation."""
    blk = x_f32.reshape(-1, n_cols // QK_MXFP8, QK_MXFP8)
    dec = decode_mxfp8(raw, n_cols).reshape(blk.shape)
    err = np.abs(dec - blk)
    denom = np.maximum(np.abs(blk), np.float32(1e-12))
    worst = float((err / denom).max())
    if worst > st["worst_rel"]:
        st["worst_rel"] = worst
        st["worst_rel_tensor"] = name
    st["max_abs"] = max(st["max_abs"], float(np.abs(dec).max()))
    st["n_blocks"] += blk.shape[0] * blk.shape[1]
    # requantize the decoded values: byte-identity is the strong form of the idempotence gate, and
    # value-identity is checked on the same 2-D shape so a reshape cannot fake a mismatch
    again_raw = quantize_row_mxfp8(dec.astype(np.float32).reshape(-1, n_cols), n_cols)
    if np.array_equal(decode_mxfp8(again_raw, n_cols), dec.reshape(-1, n_cols)):
        st["requant_value_exact"] += 1
    if np.array_equal(again_raw, raw):
        st["requant_byte_identical"] += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Requantize a GGUF to GGML_TYPE_MXFP8 in python")
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", action="store_true", help="print the per-tensor plan and exit")
    args = ap.parse_args()
    if not args.list and not args.out:
        ap.error("out is required unless --list is used")

    if args.list:
        src = Gguf(args.src)
        nq = nc = 0
        for t in src.tensors:
            q = should_quantize(t["name"], t["dims"], t["type"])
            nq += q
            nc += not q
        print(f"{args.src}: {nq} tensors -> MXFP8, {nc} copied verbatim "
              f"(file_type {src.scalar.get('general.file_type')}, arch {src.scalar.get('general.architecture')})")
        src.close()
        return 0

    st = convert(args.src, args.out)
    print(f"{args.out}: quantized={st['quantized']} copied={st['copied']} "
          f"{st['bytes_in']/1e6:.1f} MB -> {st['bytes_out']/1e6:.1f} MB "
          f"({8 * st['bytes_out'] / st['values']:.3f} bpw over {st['values']:,} quantized+copied values)")
    print(f"worst per-element relative error = {st['worst_rel']:.4f} in {st['worst_rel_tensor']} "
          f"(e4m3 3-bit-mantissa bound is 1/16 = 0.0625 at the top of a binade)")
    print(f"max |decoded| = {st['max_abs']}")
    print(f"requantize idempotence: value-exact chunks {st['requant_value_exact']}, "
          f"byte-identical chunks {st['requant_byte_identical']}")
    ok = st["worst_rel"] < 0.0626 and st["requant_value_exact"] > 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
