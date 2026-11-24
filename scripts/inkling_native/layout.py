#!/usr/bin/env python3
"""Byte-level layout model for repacking Inkling-2-Mini expert weights without
requantizing them. See docs/mxfp8/06-lossless-repack-design.md for the contract.

Nothing here reads a file. These are the bit-move rules a converter has to obey;
check_layout.py applies them to the real checkpoint and asserts the invariants.

ggml references (branch mxfp8-weights = upstream master 29b52d398):
  block_mxfp4  ggml/src/ggml-common.h:214-219          17 B / 32 elems
  block_nvfp4  ggml/src/ggml-common.h:221-227          36 B / 64 elems (16-elem sub-blocks)
  block_mxfp8  proposed GGML_TYPE_MXFP8                33 B / 32 elems
  decode       ggml/src/ggml-quants.c:572-615
  e8m0, ue4m3  ggml/src/ggml-impl.h:469-535
  repack code  conversion/base.py:1230-1265 (mxfp4), conversion/base.py:1309-1332 (nvfp4)
  HF dequant   conversion/inkling2.py:33-65
"""

from __future__ import annotations

import numpy as np  # type: ignore[import-not-found]  # repo venv: .venv/bin/python

QK_MXFP4 = 32
QK_MXFP8 = 32
QK_NVFP4 = 64
QK_NVFP4_SUB = 16

E8M0_NAN_BYTE = 255  # ue8m0 exponent 0xFF is NaN, never valid as a scale
E4M3_NAN_MASK = 0x7F  # e4m3fn NaN is 0x7F / 0xFF (all exponent and mantissa bits set)


def _e4m3_table() -> np.ndarray:
    # copy of conversion/inkling2.py:18-25, kept identical on purpose
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3 = _e4m3_table()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
# ggml decodes 4-bit codes through this int8 table (2 x E2M1) and pairs it with a
# scale that is halved, so the product matches the E2M1 x scale of the checkpoint.
KVALUES_MXFP4 = (2 * E2M1).astype(np.int8)


def e8m0_to_f32(byte: np.ndarray) -> np.ndarray:
    """ue8m0 scale byte -> float32 2^(b-127). Mirrors ggml_e8m0_to_fp32 (ggml-impl.h:469-492)."""
    b = np.asarray(byte, dtype=np.uint8)
    if np.any(b == E8M0_NAN_BYTE):
        raise ValueError("ue8m0 scale byte 255 is NaN")
    return np.ldexp(np.ones(b.shape, dtype=np.float32), b.astype(np.int32) - 127)


def e8m0_to_f32_half(byte: np.ndarray) -> np.ndarray:
    """GGML_E8M0_TO_FP32_HALF: 0.5 * 2^(b-127). Only correct for 4-bit payloads."""
    return e8m0_to_f32(byte) * np.float32(0.5)


def ue4m3_to_f32_half(byte: np.ndarray) -> np.ndarray:
    """ggml ue4m3 scale decode (ggml-impl.h:521-535): value * 0.5, NaN and 0 map to 0."""
    x = np.asarray(byte, dtype=np.uint8) & 0x7F
    out = np.zeros(x.shape, dtype=np.float32)
    exp = (x >> 3).astype(np.int32)
    man = (x & 7).astype(np.float32)
    sub = (x != 0) & (exp == 0)
    norm = (exp != 0) & (x != 0x7F)
    out[sub] = np.ldexp(man[sub], -9)
    out[norm] = np.ldexp(1 + man[norm] / 8, exp[norm] - 7)
    return out * np.float32(0.5)


def _u8(a: np.ndarray, name: str = "array") -> np.ndarray:
    if a.dtype != np.uint8:
        raise TypeError(f"{name} must be a uint8 view, got {a.dtype}")
    return a


def nibbles_to_elements(packed: np.ndarray) -> np.ndarray:
    """Byte b at position i holds element 2i in the low nibble and 2i+1 in the high one.

    This is the checkpoint (compressed-tensors / ModelOpt) order, confirmed by
    conversion/inkling2.py:42-43 where _E2M1[packed & 15] feeds column 0::2.
    """
    p = _u8(packed, "packed")
    if p.shape[-1] == 0:
        raise ValueError("packed payload is empty")
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = p >> 4
    return out


def elements_to_ggml_nibbles(elements: np.ndarray, group: int) -> np.ndarray:
    """ggml order within `group` elements: byte j holds element j (low) and element j + group/2 (high).

    group is QK_MXFP4 for block_mxfp4 (conversion/base.py:1262) and QK_NVFP4_SUB for the
    block_nvfp4 sub-blocks (conversion/base.py:1325). Applying it to a whole row instead of
    per group is wrong: the halves then come from different scale groups.
    """
    if elements.shape[-1] % group:
        raise ValueError(f"{elements.shape[-1]} elements is not a multiple of {group}")
    g = elements.reshape((*elements.shape[:-1], -1, group))
    half = group // 2
    packed = (g[..., :half] | (g[..., half:] << 4)).astype(np.uint8)
    return packed.reshape((*elements.shape[:-1], -1))


# ------------------------------------------------------------------ MXFP8 --


def pack_mxfp8_blocks(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Checkpoint MXFP8 (e4m3 payload, ue8m0 scale, 1 x 32 along K) -> block_mxfp8.

    weight uint8 [..., rows, k]  one e4m3 code per element, copied verbatim
    scale  uint8 [..., rows, k/32] one ue8m0 byte per 32-element group, copied verbatim
    return uint8 [..., rows, (k/32) * 33] row-major over K: {e, qs[32]} per group
    """
    w = _u8(weight, "weight")
    s = _u8(scale, "scale")
    if w.ndim < 2 or s.ndim != w.ndim:
        raise ValueError(f"expected [..., rows, k], got {w.shape} and {s.shape}")
    cols = w.shape[-1]
    if cols % QK_MXFP8:
        raise ValueError(f"row has {cols} elements, expected a multiple of {QK_MXFP8}")
    n_blocks = cols // QK_MXFP8
    if tuple(s.shape) != tuple(w.shape[:-1] + (n_blocks,)):
        raise ValueError(f"scale shape {s.shape} does not match {w.shape[:-1] + (n_blocks,)}")
    raw = np.empty(w.shape[:-1] + (n_blocks * (QK_MXFP8 + 1),), dtype=np.uint8)
    blk = raw.reshape(w.shape[:-1] + (n_blocks, QK_MXFP8 + 1))
    blk[..., 0] = s
    blk[..., 1:] = w.reshape(w.shape[:-1] + (n_blocks, QK_MXFP8))
    return raw


def unpack_mxfp8_blocks(raw: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """block_mxfp8 bytes -> (e4m3 payload, ue8m0 scale), the inverse of pack_mxfp8_blocks."""
    r = _u8(raw, "raw")
    n_blocks = cols // QK_MXFP8
    blk = r.reshape(r.shape[:-1] + (n_blocks, QK_MXFP8 + 1))
    return (blk[..., 1:].reshape(r.shape[:-1] + (cols,)), blk[..., 0].reshape(r.shape[:-1] + (n_blocks,)))


def dequant_mxfp8_blocks(raw: np.ndarray, cols: int) -> np.ndarray:
    """Decode block_mxfp8 exactly as a future GGML_TYPE_MXFP8 must: e4m3(qs[j]) * 2^(e-127)."""
    w, s = unpack_mxfp8_blocks(raw, cols)
    values = E4M3[w]
    if np.isnan(values).any():
        raise ValueError("e4m3 payload byte is NaN")
    return values.reshape(values.shape[:-1] + (-1, QK_MXFP8)) * e8m0_to_f32(s).reshape(s.shape[:-1] + (s.shape[-1], 1))


def hf_dequant_mxfp8(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Reference decoder, mirroring conversion/inkling2.py:55-62 (the MXFP8 branch)."""
    w = _u8(weight, "weight")
    s = _u8(scale, "scale")
    if np.any(s == E8M0_NAN_BYTE):
        raise ValueError("UE8M0 byte 255 is NaN")
    values = E4M3[w]
    if np.isnan(values).any():
        raise ValueError("E4M3FN payload is NaN")
    block_scale = np.ldexp(np.ones(s.shape, dtype=np.float32), s.astype(np.int32) - 127)
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.reshape(values.shape[:-1] + (-1, QK_MXFP8)) * block_scale.reshape(block_scale.shape[:-1] + (block_scale.shape[-1], 1))
    if not np.isfinite(values).all():
        raise ValueError("non-finite dequantized weight")
    return values.reshape(w.shape)


# ------------------------------------------------------------------ NVFP4 --


def pack_nvfp4_blocks(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Checkpoint NVFP4 -> ggml block_nvfp4 super-blocks, mirroring conversion/base.py:1309-1332.

    weight uint8 [..., rows, k/2] 4-bit E2M1 codes, element 2i low nibble / 2i+1 high nibble
    scale  uint8 [..., rows, k/16] e4m3 sub-block scale bytes; the sign bit is stripped
    return uint8 [..., rows, (k/64) * 36] per super-block: [4 ue4m3 bytes][32 packed code bytes]
    """
    w = _u8(weight, "weight")
    s = _u8(scale, "scale")
    n_blocks = s.shape[-1]
    if n_blocks % (QK_NVFP4 // QK_NVFP4_SUB):
        raise ValueError(f"{n_blocks} sub-blocks is not a multiple of {QK_NVFP4 // QK_NVFP4_SUB}")
    cols = n_blocks * QK_NVFP4_SUB
    if w.shape[-1] * 2 != cols:
        raise ValueError(f"payload {w.shape} and scale {s.shape} disagree on the row width")
    elements = nibbles_to_elements(w)
    d = s & 0x7F
    qs = elements_to_ggml_nibbles(elements, QK_NVFP4_SUB)
    n_super = n_blocks // (QK_NVFP4 // QK_NVFP4_SUB)
    d_grouped = d.reshape(d.shape[:-1] + (n_super, QK_NVFP4 // QK_NVFP4_SUB))
    qs_grouped = qs.reshape(qs.shape[:-1] + (n_super, QK_NVFP4 // 2))
    return np.concatenate((d_grouped, qs_grouped), axis=-1).reshape(w.shape[:-1] + (n_super * (QK_NVFP4 // 2 + QK_NVFP4 // QK_NVFP4_SUB),))


def unpack_nvfp4_blocks(raw: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """block_nvfp4 bytes -> (packed code bytes in ggml order, ue4m3 scale bytes)."""
    r = _u8(raw, "raw")
    n_super = cols // QK_NVFP4
    n_sub = QK_NVFP4 // QK_NVFP4_SUB
    blk = r.reshape(r.shape[:-1] + (n_super, n_sub + QK_NVFP4 // 2))
    return (blk[..., n_sub:].reshape(r.shape[:-1] + (cols // 2,)), blk[..., :n_sub].reshape(r.shape[:-1] + (cols // QK_NVFP4_SUB,)))


def dequant_nvfp4_blocks(raw: np.ndarray, cols: int, factor: np.float32 | None = None) -> np.ndarray:
    """Decode block_nvfp4 as ggml does (kvalues x half-scale), optionally times the per-expert factor.

    factor is the value stored in the `.scale` sidecar: amax / outer_scale_denominator.
    """
    qs, d = unpack_nvfp4_blocks(raw, cols)
    grp = qs.reshape(qs.shape[:-1] + (-1, QK_NVFP4_SUB // 2))
    elements = np.empty(qs.shape[:-1] + (qs.shape[-1] * 2,), dtype=np.float32)
    flat = elements.reshape(elements.shape[:-1] + (-1, QK_NVFP4_SUB))
    # ggml order inside a 16-element sub-block: byte j holds element j (low) and j + 8 (high)
    flat[..., :QK_NVFP4_SUB // 2] = KVALUES_MXFP4[grp & 0x0F]
    flat[..., QK_NVFP4_SUB // 2:] = KVALUES_MXFP4[grp >> 4]
    scales = ue4m3_to_f32_half(d)
    values = (flat * scales.reshape(scales.shape[:-1] + (scales.shape[-1], 1))).reshape(raw.shape[:-1] + (cols,))
    if factor is not None:
        values = values * factor
    return values


def hf_dequant_nvfp4(weight: np.ndarray, scale: np.ndarray, amax: float, denominator: float) -> tuple[np.ndarray, np.float32]:
    """Reference decoder, mirroring conversion/inkling2.py:38-53 (the NVFP4 branch)."""
    w = _u8(weight, "weight")
    s = _u8(scale, "scale")
    block_scale = E4M3[s]
    if np.isnan(block_scale).any():
        raise ValueError("E4M3FN scale is NaN")
    elements = np.empty(w.shape[:-1] + (w.shape[-1] * 2,), dtype=np.float32)
    elements[..., 0::2] = E2M1[w & 15]
    elements[..., 1::2] = E2M1[w >> 4]
    with np.errstate(over="ignore", invalid="ignore"):
        values = (elements.reshape(elements.shape[:-1] + (-1, QK_NVFP4_SUB)) * block_scale.reshape(block_scale.shape[:-1] + (block_scale.shape[-1], 1))).reshape(w.shape[:-1] + (-1,))
    a = np.float32(amax)
    d = np.float32(denominator)
    if not np.isfinite(a) or a < 0:
        raise ValueError("amax is not finite nonnegative")
    if not np.isfinite(d) or d <= 0:
        raise ValueError("outer_scale_denominator is not finite positive")
    factor = a / d
    with np.errstate(over="ignore", invalid="ignore"):
        values = values * factor
    if not np.isfinite(values).all():
        raise ValueError("non-finite dequantized weight")
    return values, factor


# ------------------------------------------------------------------ MXFP4 --


def pack_mxfp4_blocks(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """numpy twin of ModelBase.repack_mxfp4_blocks (conversion/base.py:1230-1265)."""
    p = _u8(packed, "packed")
    s = _u8(scale, "scale")
    n_blocks = s.shape[-1]
    if p.shape[-1] != n_blocks * (QK_MXFP4 // 2):
        raise ValueError(f"payload {p.shape} and scale {s.shape} disagree")
    elements = nibbles_to_elements(p)
    qs = elements_to_ggml_nibbles(elements, QK_MXFP4)
    raw = np.empty(p.shape[:-1] + (n_blocks * (QK_MXFP4 + 1),), dtype=np.uint8)
    blk = raw.reshape(p.shape[:-1] + (n_blocks, QK_MXFP4 + 1))
    blk[..., 0] = s
    blk[..., 1:] = qs.reshape(p.shape[:-1] + (n_blocks, QK_MXFP4 // 2))
    return raw


# --------------------------------------------------- fused gate/up row order --


def w13_hf_row_of_gguf_row(rows: int) -> np.ndarray:
    """GGUF logical row -> checkpoint row for a fused w13 tensor.

    The checkpoint interleaves gate and up (row 2j = gate j, row 2j+1 = up j); master
    stores them blocked as [gate; up] (conversion/inkling2.py:174 row_slice, and the
    matching reshape/transpose of the non-packed path at conversion/inkling2.py:257).
    """
    if rows % 2:
        raise ValueError(f"row count {rows} is not even")
    half = rows // 2
    r = np.arange(rows, dtype=np.int64)
    return 2 * (r % half) + (r // half)


def w13_gate_or_up_of_gguf_row(rows: int) -> np.ndarray:
    """0 for the first half of the GGUF rows (gate), 1 for the second half (up)."""
    half = rows // 2
    return (np.arange(rows, dtype=np.int64) // half).astype(np.int8)
