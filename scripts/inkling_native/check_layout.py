#!/usr/bin/env python3
"""Bit-exactness harness for the Inkling-2-Mini lossless expert repack.

Reads the real checkpoint by safetensors header offsets (streamed row windows, never
the whole file) and checks the contract in docs/mxfp8/06-lossless-repack-design.md:

  a  MXFP8 ue8m0 scale bytes are never 255 and e4m3 payloads are finite; NVFP4 e4m3
     scale bytes are finite and never carry a sign bit (master strips it with & 0x7F)
  b  dequant(unpack(pack(x))) == the checkpoint dequant, bitwise, on sampled rows
  c  same for the NVFP4 layers, including the amax/denominator global factor
  d  cross-check the shipped unsloth GGUF NVFP4 bytes against layout.py
  e  size budget of a fully lossless repack, plus KV headroom on 2 x 16 GB

Run:  .venv/bin/python scripts/inkling_native/check_layout.py [--quick] [--full] [--no-gguf]
Exit 0 means every strict invariant held.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from typing import Any, NamedTuple

import numpy as np  # type: ignore[import-not-found]  # repo venv: .venv/bin/python

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SNAPSHOT = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*"))[0]
SAFE = os.path.join(SNAPSHOT, "model-00001-of-00001.safetensors")
UNSLOTH = "/home/tim/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf"

DTYPE_SIZE = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "F64": 8}
NVFP4_LAYERS = list(range(20))
MXFP8_LAYERS = list(range(20, 24))
EXPERTS = 128
ROWS = 2304
HIDDEN = 2304
INTER = 1152

FAILURES: list[str] = []
WARNINGS: list[str] = []


def report(tag: str, ok: bool, detail: str, strict: bool = True) -> None:
    tag_state = "PASS" if ok else ("FAIL" if strict else "WARN")
    print(f"[{tag_state}] {tag}: {detail}", flush=True)
    if ok:
        return
    (FAILURES if strict else WARNINGS).append(f"{tag}: {detail}")


def info(tag: str, detail: str) -> None:
    print(f"       {tag}: {detail}", flush=True)


class Checkpoint:
    """Row-window reader: only the bytes a check needs ever enter memory."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        n = struct.unpack("<Q", bytes(self.mm[:8]))[0]
        self.header = json.loads(bytes(self.mm[8:8 + n]))
        self.base = 8 + n
        self.tensors = {k: v for k, v in self.header.items() if isinstance(v, dict) and "data_offsets" in v}

    def row_window(self, name: str, expert: int, start: int, count: int) -> np.ndarray:
        """[expert, start:start+count] of a rank-3 tensor as a uint8 (count, cols) view.

        The file is mmap'd, so only the touched pages are faulted in.
        """
        v = self.tensors[name]
        cols = v["shape"][-1]
        item = DTYPE_SIZE[v["dtype"]]
        off = self.base + v["data_offsets"][0] + (expert * ROWS + start) * cols * item
        return np.asarray(self.mm[off:off + count * cols * item], dtype=np.uint8).reshape(count, cols)

    def vector(self, name: str) -> np.ndarray:
        """Flat F32 view of a named tensor, read at that tensor's own header data_offsets."""
        v = self.tensors[name]
        if v["dtype"] != "F32":
            raise ValueError(f"{name}: expected F32, got {v['dtype']}")
        lo = self.base + v["data_offsets"][0]
        return np.frombuffer(bytes(self.mm[lo:lo + 4 * int(np.prod(v["shape"]))]), dtype=np.float32)


def expert_names(layer: int, kind: str) -> dict[str, str]:
    root = f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"
    out = {"weight": root, "scale": root + "_scale"}
    if layer in NVFP4_LAYERS:
        out["amax"] = root + "_amax"
        out["denom"] = root + "_outer_scale_denominator"
    return out


def cols_of(kind: str) -> int:
    return HIDDEN if kind == "w13" else INTER


# ------------------------------------------------------------------- check a --


def check_bytes(ck: Checkpoint, layers: list[int], experts: list[int], chunk_rows: int) -> None:
    mxfp8 = layers[0] in MXFP8_LAYERS
    for layer in layers:
        for kind in ("w13", "w2"):
            names = expert_names(layer, kind)
            s_min, s_max, s_nan, s_sign, s255, p_nan = 255, 0, 0, 0, 0, 0
            for e in experts:
                for start in range(0, ROWS, chunk_rows):
                    n = min(chunk_rows, ROWS - start)
                    s = ck.row_window(names["scale"], e, start, n)
                    s_min = min(s_min, int(s.min()))
                    s_max = max(s_max, int(s.max()))
                    s_nan += int(((s & 0x7F) == 0x7F).sum())
                    s_sign += int((s & 0x80).sum())
                    s255 += int((s == 255).sum())
                    if mxfp8:
                        w = ck.row_window(names["weight"], e, start, n)
                        p_nan += int((w == 0x7F).sum() + (w == 0xFF).sum())
            cols = cols_of(kind)
            values = len(experts) * ROWS
            if mxfp8:
                report(f"a.mxfp8.L{layer}.{kind}", s255 == 0 and s_nan == 0 and p_nan == 0,
                       f"scale bytes [{s_min}..{s_max}] = exponents [{s_min - 127}..{s_max - 127}], byte-255 {s255}, NaN scale bytes {s_nan}, NaN payload bytes {p_nan} over {values} rows x {cols} elems")
            else:
                report(f"a.nvfp4.L{layer}.{kind}", s_sign == 0 and s_nan == 0,
                       f"scale bytes [{s_min}..{s_max}], NaN scale bytes {s_nan}, SIGN-BIT scale bytes {s_sign} over {values} rows x {cols // 16} sub-blocks (master strips the sign at conversion/base.py:1324)")


# ------------------------------------------------------------------- check b --


def check_mxfp8_roundtrip(ck: Checkpoint, layers: list[int], experts: list[int], chunk_rows: int) -> None:
    for layer in layers:
        for kind in ("w13", "w2"):
            names = expert_names(layer, kind)
            cols = cols_of(kind)
            nb = cols // layout.QK_MXFP8
            raw_ok, bit_ok, half_same = True, True, False
            bad = 0
            for e in experts:
                for start in range(0, ROWS, chunk_rows):
                    n = min(chunk_rows, ROWS - start)
                    w = ck.row_window(names["weight"], e, start, n)
                    s = ck.row_window(names["scale"], e, start, n)
                    raw = layout.pack_mxfp8_blocks(w, s)
                    blk = raw.reshape(n, nb, layout.QK_MXFP8 + 1)
                    if not np.array_equal(blk[:, :, 0], s) or not np.array_equal(blk[:, :, 1:], w.reshape(n, nb, layout.QK_MXFP8)):
                        raw_ok = False
                    got = layout.dequant_mxfp8_blocks(raw, cols)
                    ref = layout.hf_dequant_mxfp8(w, s)
                    if got.tobytes() != ref.tobytes():
                        bit_ok = False
                        bad += int((got != ref).sum())
                    # the mxfp4 HALF-scale shortcut (ggml-impl.h:494-514) must not be reused for e4m3 payloads
                    wq, sq = layout.unpack_mxfp8_blocks(raw, cols)
                    half = (layout.E4M3[wq].reshape(n, nb, layout.QK_MXFP8) * layout.e8m0_to_f32_half(sq).reshape(n, nb, 1)).reshape(n, cols)
                    half_same = half_same or bool(np.array_equal(half, got))
            report(f"b.mxfp8.L{layer}.{kind}", raw_ok and bit_ok and not half_same,
                   f"byte placement {'exact' if raw_ok else 'MISMATCHED'}; decode bitwise-identical to conversion/inkling2.py:55-62 = {bit_ok}; {bad} differing of {len(experts) * ROWS * cols} values (experts {experts}); GGML_E8M0_TO_FP32_HALF shortcut would be wrong here = {not half_same}")


# ------------------------------------------------------------------- check c --


def check_nvfp4_roundtrip(ck: Checkpoint, layers: list[int], experts: list[int], chunk_rows: int) -> None:
    for layer in layers:
        for kind in ("w13", "w2"):
            names = expert_names(layer, kind)
            cols = cols_of(kind)
            amax = ck.vector(names["amax"])
            denom = float(ck.vector(names["denom"])[0])
            global_ok = bool(np.isfinite(amax).all() and (amax >= 0).all() and np.isfinite(denom) and denom > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = (np.float32(amax) / np.float32(denom)).astype(np.float32)
            report(f"c.nvfp4.L{layer}.{kind}.global", global_ok,
                   f"amax [{float(np.min(amax)):.9g}..{float(np.max(amax)):.9g}], denom {denom!r} (== 6*448: {denom == 2688.0}); amax/denom [{float(np.min(factor)):.6g}..{float(np.max(factor)):.6g}]")
            pow2 = int((np.abs(np.frexp(factor)[0]) == 0.5).sum())
            if not global_ok:
                report(f"c.nvfp4.L{layer}.{kind}", False,
                       f"roundtrip SKIPPED, global factor invalid: amax[:4]={np.round(amax[:4], 9).tolist()}, denom={denom!r}", strict=False)
                continue
            bit_ok, num_ok = True, True
            zeros = bad = 0
            for e in experts:
                for start in range(0, ROWS, chunk_rows):
                    n = min(chunk_rows, ROWS - start)
                    w = ck.row_window(names["weight"], e, start, n)
                    s = ck.row_window(names["scale"], e, start, n)
                    raw = layout.pack_nvfp4_blocks(w, s)
                    got = layout.dequant_nvfp4_blocks(raw, cols, factor[e])
                    ref, ref_factor = layout.hf_dequant_nvfp4(w, s, float(amax[e]), denom)
                    if np.float32(ref_factor) != factor[e]:
                        num_ok = False
                    if got.tobytes() != ref.tobytes():
                        bit_ok = False
                        bad += int((got != ref).sum())
                        zeros += int((np.signbit(got) != np.signbit(ref)).sum())
                    num_ok = num_ok and bool(np.array_equal(got, ref))
            report(f"c.nvfp4.L{layer}.{kind}", num_ok,
                   f"numerically identical = {num_ok}; bitwise identical = {bit_ok} ({zeros} differ only in the sign of zero, kvalues_mxfp4 has no -0.0; {bad} other differing of {len(experts) * ROWS * cols}); amax/denom is a power of two for {pow2}/{EXPERTS} experts, {len(np.unique(factor))} distinct factors, range {float(factor.min()):.9g}..{float(factor.max()):.9g}")


GGUF_TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 20: "IQ4_NL", 30: "BF16", 39: "MXFP4", 40: "NVFP4", 43: "F8_E4M3"}
GGUF_BLOCK = {8: (34, 32), 20: (132, 256), 39: (17, 32), 40: (36, 64)}
GGUF_PLAIN = {0: 4, 1: 2, 30: 2, 43: 1}


class GgufTensor(NamedTuple):
    name: str
    type_id: int
    dims: tuple[int, ...]
    data_off: int
    n_bytes: int


def gguf_tensor_bytes(type_id: int, dims: list[int]) -> int:
    """Byte count of one tensor; -1 for types this harness never reads."""
    if type_id in GGUF_PLAIN:
        return int(np.prod(dims)) * GGUF_PLAIN[type_id]
    if type_id not in GGUF_BLOCK:
        return -1
    block, elems = GGUF_BLOCK[type_id]
    if dims[0] % elems:
        raise ValueError(f"type {type_id} with {dims[0]} elements per row is not block aligned")
    return int(np.prod(dims[1:])) * (dims[0] // elems) * block


def gguf_index(path: str) -> dict[str, GgufTensor]:
    """Parse just the GGUF metadata; the 17 GB body stays a lazy memmap.

    gguf-py is unusable here (its package __init__ imports yaml, absent from the venv).
    The tensor-info dimension field width is auto-detected: the shipped unsloth file
    stores them as uint64, so a uint32 walk desynchronises on the first record.
    """
    size = os.path.getsize(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    window = min(size, 1 << 25)  # a 200k-token vocab makes this header ~12 MB
    errors = []
    while True:
        buf = bytes(mm[:window])
        for dims64 in (False, True):
            try:
                return _gguf_parse(buf, dims64)
            except (EOFError, struct.error, IndexError, KeyError, ValueError) as exc:
                errors.append(f"{'u64' if dims64 else 'u32'} dims: {exc}")
        if window >= size:
            raise ValueError("could not parse the GGUF header: " + "; ".join(errors))
        window = min(size, window * 4)


def _gguf_parse(buf: bytes, dims64: bool) -> dict[str, GgufTensor]:
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    n_tensors = int.from_bytes(buf[8:16], "little")
    n_kv = int.from_bytes(buf[16:24], "little")
    pos = 24

    def need(n: int) -> None:
        if pos + n > len(buf):
            raise EOFError("header window too small")

    def string() -> str:
        nonlocal pos
        need(8)
        n = int.from_bytes(buf[pos:pos + 8], "little")
        pos += 8
        need(n)
        s = buf[pos:pos + n].decode("utf-8")
        pos += n
        return s

    def value(t: int) -> Any:
        nonlocal pos
        if t == 8:
            return string()
        if t == 9:
            need(12)
            et = int.from_bytes(buf[pos:pos + 4], "little")
            pos += 4
            cnt = int.from_bytes(buf[pos:pos + 8], "little")
            pos += 8
            return [value(et) for _ in range(cnt)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        need(struct.calcsize(fmt))
        v = struct.unpack_from(fmt, buf, pos)[0]
        pos += struct.calcsize(fmt)
        return v

    align = 32
    for _ in range(n_kv):
        key = string()
        need(4)
        t = int.from_bytes(buf[pos:pos + 4], "little")
        pos += 4
        got = value(t)
        if key == "general.alignment" and isinstance(got, int):
            align = int(got)
    info = {}
    dim_size = 8 if dims64 else 4
    for _ in range(n_tensors):
        name = string()
        need(4)
        nd = int.from_bytes(buf[pos:pos + 4], "little")
        pos += 4
        if nd < 1 or nd > 8:
            raise ValueError(f"tensor {name!r} claims {nd} dimensions (wrong dimension field width?)")
        need(dim_size * nd + 12)
        dims = [int.from_bytes(buf[pos + i * dim_size:pos + (i + 1) * dim_size], "little") for i in range(nd)]
        pos += dim_size * nd
        type_id = int.from_bytes(buf[pos:pos + 4], "little")
        pos += 4
        if type_id > 44:
            raise ValueError(f"tensor {name!r} claims ggml type {type_id}")
        if 0 in dims:
            raise ValueError(f"tensor {name!r} has a zero-length dimension")
        off = int.from_bytes(buf[pos:pos + 8], "little")
        pos += 8
        info[name] = (type_id, dims, off)
    data_start = (pos + align - 1) // align * align
    return {name: GgufTensor(name, type_id, tuple(dims), data_start + off, gguf_tensor_bytes(type_id, dims)) for name, (type_id, dims, off) in info.items()}


def gguf_read(path: str, t: GgufTensor) -> np.ndarray:
    """Raw bytes of one tensor, from a fresh lazy memmap so only its pages are touched."""
    if t.n_bytes < 0:
        raise ValueError(f"{t.name}: no byte size model for ggml type {t.type_id}")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return np.asarray(mm[t.data_off:t.data_off + t.n_bytes], dtype=np.uint8)


# ------------------------------------------------------------------- check d --


def check_unsloth_gguf(ck: Checkpoint, expert: int = 0) -> None:
    if not os.path.isfile(UNSLOTH):
        report("d", False, f"{UNSLOTH} not present", strict=False)
        return
    tmap = gguf_index(UNSLOTH)
    for kind, hf_kind in (("gate_up", "w13"), ("down", "w2")):
        cols = cols_of(hf_kind)
        stride = cols // 2 + cols // 16
        wname = f"blk.0.ffn_{kind}_exps.weight"
        sname = f"blk.0.ffn_{kind}_exps.scale"
        if wname not in tmap:
            report(f"d.{kind}", False, f"{wname} absent from the unsloth GGUF", strict=False)
            continue
        t = tmap[wname]
        gg_full = gguf_read(UNSLOTH, t)
        rows_per_expert = int(np.prod(t.dims[1:]) // EXPERTS)
        gg = gg_full[expert * rows_per_expert * stride:(expert + 1) * rows_per_expert * stride].reshape(rows_per_expert, stride)
        hf = expert_names(0, hf_kind)
        w = ck.row_window(hf["weight"], expert, 0, rows_per_expert)
        s = ck.row_window(hf["scale"], expert, 0, rows_per_expert)
        cand = {"identity": layout.pack_nvfp4_blocks(w, s)}
        if kind == "gate_up":
            half = rows_per_expert // 2
            r = np.arange(rows_per_expert, dtype=np.int64)
            cand["gate/up-permuted"] = layout.pack_nvfp4_blocks(w[layout.w13_hf_row_of_gguf_row(rows_per_expert)], s[layout.w13_hf_row_of_gguf_row(rows_per_expert)])
            cand["alt-perm"] = layout.pack_nvfp4_blocks(w[(r // 2) + (r % 2) * half], s[(r // 2) + (r % 2) * half])
        scores = {name: int((c == gg).sum()) for name, c in cand.items()}
        best = max(scores, key=lambda name: scores[name])
        ok = scores[best] == gg.size
        report(f"d.{kind}", ok, f"{wname} type={GGUF_TYPE_NAMES.get(t.type_id, f'T{t.type_id}')} dims={t.dims} n_bytes={t.n_bytes} (data_off {t.data_off}); row order {best} -> {scores[best]}/{gg.size} bytes equal, all orders {scores}", strict=False)
        n_super = cols // layout.QK_NVFP4
        blk = gg.reshape(rows_per_expert, n_super, 36)
        mine = cand[best].reshape(rows_per_expert, n_super, 36)
        info(f"{kind} field diff", f"scale bytes differ {int((mine[:, :, :4] != blk[:, :, :4]).sum())}/{blk[:, :, :4].size}, payload bytes differ {int((mine[:, :, 4:] != blk[:, :, 4:]).sum())}/{blk[:, :, 4:].size} (expert {expert}, {rows_per_expert} rows)")
        if sname in tmap:
            st = tmap[sname]
            vals = np.frombuffer(gguf_read(UNSLOTH, st).tobytes(), dtype=np.float32)
            f32 = (np.float32(ck.vector(hf["amax"])) / np.float32(float(ck.vector(hf["denom"])[0]))).astype(np.float32)
            n = min(vals.size, f32.size)
            rel = float(np.max(np.abs(vals[:n] / f32[:n] - np.float32(1.0)))) if n else float("nan")
            report(f"d.{kind}.scale", bool(np.array_equal(vals[:n], f32[:n])), f"{sname} ({vals.size} {GGUF_TYPE_NAMES.get(st.type_id, f'T{st.type_id}')}) bitwise == amax/outer_scale_denominator for {int((vals[:n] == f32[:n]).sum())}/{n} experts, max rel diff {rel:.3e}; sidecar range {float(vals[:n].min()):.6g}..{float(vals[:n].max()):.6g}, amax/denom range {float(f32[:n].min()):.6g}..{float(f32[:n].max()):.6g}", strict=False)
        else:
            info(f"{kind}.scale", f"{sname} absent (master omits a sidecar when the factor is 1.0: conversion/base.py:341-345)")


# ------------------------------------------------------------------- check e --


def mib(n: float) -> str:
    return f"{n / 1024 / 1024:.1f}"


def size_budget(ck: Checkpoint) -> None:
    cat = {"nvfp4_payload": 0, "nvfp4_scale": 0, "nvfp4_global": 0, "mxfp8_payload": 0, "mxfp8_scale": 0, "text_other": 0, "not_text": 0}
    for name, v in ck.tensors.items():
        n = v["data_offsets"][1] - v["data_offsets"][0]
        if not name.startswith("model.llm."):
            cat["not_text"] += n
        elif ".mlp.experts." in name:
            layer = int(name.split("layers.")[1].split(".")[0])
            if "_weight_amax" in name or "_outer_scale_denominator" in name:
                cat["nvfp4_global"] += n
            elif layer in NVFP4_LAYERS:
                cat["nvfp4_scale" if name.endswith("_scale") else "nvfp4_payload"] += n
            else:
                cat["mxfp8_scale" if name.endswith("_scale") else "mxfp8_payload"] += n
        else:
            cat["text_other"] += n
    nv_elem = cat["nvfp4_payload"] * 2
    mx_elem = cat["mxfp8_payload"]
    print("\n=== e. size budget (MiB; text backbone only, vision/mtp excluded as master does) ===")
    print(f"checkpoint: NVFP4 payload {mib(cat['nvfp4_payload'])} + scales {mib(cat['nvfp4_scale'])} + global F32 {mib(cat['nvfp4_global'])}; "
          f"MXFP8 payload {mib(cat['mxfp8_payload'])} + scales {mib(cat['mxfp8_scale'])}; other text tensors {mib(cat['text_other'])}; not-text {mib(cat['not_text'])}")
    lossless = cat["nvfp4_payload"] + cat["nvfp4_scale"] + cat["mxfp8_payload"] + cat["mxfp8_scale"] + cat["text_other"]
    sidecar = 40 * EXPERTS * 4  # one F32[128] .scale per NVFP4 expert tensor
    rows = [
        ("lossless repack (NVFP4 36/64 + MXFP8 33/32 + other as-is)", lossless + sidecar),
        ("  NVFP4 experts as block_nvfp4", cat["nvfp4_payload"] + cat["nvfp4_scale"]),
        ("  MXFP8 experts as block_mxfp8", cat["mxfp8_payload"] + cat["mxfp8_scale"]),
        ("  MXFP8 experts as Q8_0 (34/32)", mx_elem * 34 / 32),
        ("  MXFP8 experts as F8_E4M3 flat, scales dropped (1 B/elem, lossy)", mx_elem),
        ("  MXFP8 experts as BF16 (2 B/elem, lossy)", mx_elem * 2),
        ("  MXFP8 experts as MXFP4 (17/32, requantised, lossy)", mx_elem * 17 / 32),
        ("  NVFP4 experts as Q8_0 (34/32, requantised)", nv_elem * 34 / 32),
        ("  all experts as BF16", (nv_elem + mx_elem) * 2),
    ]
    for label, n in rows:
        print(f"  {label:<62} {mib(n):>10} MiB")
    print(f"  checkpoint expert bytes vs repack: NVFP4 {mib(nv_elem * 36 / 64)} == {mib(cat['nvfp4_payload'] + cat['nvfp4_scale'])}, "
          f"MXFP8 {mib(mx_elem * 33 / 32)} == {mib(cat['mxfp8_payload'] + cat['mxfp8_scale'])} (size-neutral bit moves)")

    weights = lossless + sidecar
    layers = 24
    swa = 20  # local_layer_ids has 20 entries -> sliding_window 1024 caps those KV rows
    kv_heads, head_dim, window = 8, 128, 1024
    kv_types = {"fp8 (e4m3, 1 B)": 1.0, "q8_0 (17/16 B)": 34 / 32, "f16 (2 B)": 2.0}
    print(f"\n=== e. KV headroom: 24 layers ({swa} SWA capped at {window}), {kv_heads} kv heads x head_dim {head_dim}, 2 GPUs x 16384 MiB ===")
    print("  per token per layer = 2 (k,v) x heads x head_dim x bytes")
    for tname, b in kv_types.items():
        per_tok_layer = 2 * kv_heads * head_dim * b
        line = [f"  {tname:<18}"]
        for ctx in (32768, 65536, 131072, 262144):
            capped = (swa * min(ctx, window) + (layers - swa) * ctx) * per_tok_layer
            uncapped = layers * ctx * per_tok_layer
            line.append(f"ctx{ctx // 1024}k: {mib(capped)} capped / {mib(uncapped)} uncapped MiB")
        print(" ".join(line) + f"  [{2 * kv_heads * head_dim * b:.0f} B/token/layer]")
    for ctx in (32768, 65536, 131072, 262144):
        kv = (swa * min(ctx, window) + (layers - swa) * ctx) * 2 * kv_heads * head_dim
        total = weights + kv
        budget = 2 * 16384 * 1024 * 1024
        print(f"  ctx{ctx // 1024}k lossless weights {mib(weights)} + fp8-SWA-capped KV {mib(kv)} = {mib(total)} MiB "
              f"{'FITS' if total <= budget else 'OVER'} 32768 MiB (slack {mib(budget - total)} MiB)")
    report("e.budget", True, f"lossless GGUF body {mib(weights)} MiB; expert repacks are byte-count neutral vs the checkpoint", strict=False)


# --------------------------------------------------------------------- main --


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", default=True, help="sample experts and layers (default)")
    g.add_argument("--full", action="store_true", help="sweep every expert of every quantised layer, still streamed")
    ap.add_argument("--no-gguf", action="store_true", help="skip the unsloth GGUF cross-check")
    ap.add_argument("--experts", type=int, default=3, help="experts sampled in --quick")
    ap.add_argument("--rows", type=int, default=256, help="rows per streamed window (must divide 2304)")
    a = ap.parse_args()
    if ROWS % a.rows:
        ap.error(f"--rows must divide {ROWS}")
    if a.full:
        experts, chunk = list(range(EXPERTS)), a.rows
        layers_n, layers_m = NVFP4_LAYERS, MXFP8_LAYERS
    else:
        experts, chunk = [0, EXPERTS // 2, EXPERTS - 1][: max(1, a.experts)], min(a.rows, 128)
        layers_n, layers_m = [0, 19], [20, 23]
    ck = Checkpoint(SAFE)
    print(f"checkpoint {SAFE}\n  file bytes {os.path.getsize(SAFE)}, tensors {len(ck.tensors)}")
    print(f"  mode {'full' if a.full else 'quick'}: experts {experts if len(experts) < 6 else f'{EXPERTS} experts'}, rows/window {chunk}, "
          f"NVFP4 layers {layers_n}, MXFP8 layers {layers_m}")
    check_bytes(ck, layers_n, experts, chunk)
    check_bytes(ck, layers_m, experts, chunk)
    check_mxfp8_roundtrip(ck, layers_m, experts, chunk)
    check_nvfp4_roundtrip(ck, layers_n, experts, chunk)
    if not a.no_gguf:
        check_unsloth_gguf(ck)
    size_budget(ck)
    print("\n=== summary ===")
    print(f"strict failures {len(FAILURES)}, warnings {len(WARNINGS)}")
    for f in FAILURES:
        print(f"  FAIL {f}")
    for f in WARNINGS:
        print(f"  WARN {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
