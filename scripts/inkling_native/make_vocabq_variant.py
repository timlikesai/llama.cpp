#!/usr/bin/env python3
"""Build a decode-speed experiment variant of a GGUF by re-encoding ONLY the named tensors.

Why not llama-quantize: it requantizes the whole file to a target ftype, which would destroy the
native NVFP4/MXFP8 expert bytes this project exists to preserve. This script instead performs a
surgical transform:

  * the KV/metadata section is copied byte-for-byte (tensor count and KV count do not change);
  * every tensor except the named ones keeps its exact data bytes AND its exact offset;
  * the replaced tensors move to the end of the data section (their records' type + offset fields are
    rewritten), because q8_0 is smaller than bf16 and shifting everyone else's offsets would defeat
    the "only these bytes changed" property;
  * the new payloads come from gguf-py's Q8_0, whose docstring states it is bit-exact with the
    reference implementation in ggml-quants.c.

Everything else differs from the original by: 2 type fields, 2 offset fields, the ordering of the
tensor-info records (re-sorted so offsets stay increasing), and the tail. `--verify` proves that
claim by hashing every untouched tensor rather than asserting it.

This is an experiment, not a deployment artifact: it trades two bf16 tensors that the checkpoint
ships as bf16 for q8_0, to measure what that is worth (docs/mxfp8/49, TODO #32).
"""
from __future__ import annotations

import argparse
import hashlib
import mmap
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGUFValueType as GGUF_TYPE, GGMLQuantizationType  # noqa: E402
from gguf.quants import Q8_0  # noqa: E402

GGUF_MAGIC = b"GGUF"
SCALAR_FMT = {
    GGUF_TYPE.UINT8: "<B", GGUF_TYPE.INT8: "<b", GGUF_TYPE.UINT16: "<H", GGUF_TYPE.INT16: "<h",
    GGUF_TYPE.UINT32: "<I", GGUF_TYPE.INT32: "<i", GGUF_TYPE.FLOAT32: "<f",
    GGUF_TYPE.BOOL: "<?", GGUF_TYPE.UINT64: "<Q", GGUF_TYPE.INT64: "<q", GGUF_TYPE.FLOAT64: "<d",
}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def read_scalar(f, t):
    """Skip one GGUF value. This exact form is verified against the shipped file (docs/mxfp8/49);
    the metadata bytes themselves are copied verbatim, so only the walk position has to be right."""
    t = GGUF_TYPE(t)
    if t == GGUF_TYPE.STRING:
        return read_str(f)
    if t == GGUF_TYPE.ARRAY:
        (elem,) = struct.unpack("<I", f.read(4))
        if GGUF_TYPE(elem) == GGUF_TYPE.ARRAY:
            raise AssertionError("nested arrays are not supported by this walker (GGUF emits none)")
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_scalar(f, elem) for _ in range(n)]
    fmt = SCALAR_FMT[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def walk(path):
    """Header layout: version, KV section bounds, alignment, tensor-info table, table end."""
    with open(path, "rb") as f:
        assert f.read(4) == GGUF_MAGIC, f"{path}: not a GGUF"
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        kv_start = f.tell()
        align = 32
        for _ in range(n_kv):
            key = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            val = read_scalar(f, t)
            if key == "general.alignment" and val is not None:
                align = int(val)
        kv_end = f.tell()
        # GGUF layout: header, KV section, tensor-info table (NO padding before it), then padding to
        # `alignment`, then the data section. Padding here instead of after the table desyncs record 0.
        tensors = []
        for _ in range(n_tensors):
            # GGUF v2/v3 tensor-info record field order: name, n_dims, dims[], type, offset.
            # n_dims comes BEFORE type - reading type first desyncs on the very second record.
            name = read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            shape = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
            (dtype,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append({"name": name, "type": dtype, "shape": shape, "offset": off})
        table_end = f.tell()
        pad = (-table_end) % align
        data_start = table_end + pad

    # Cross-check the block arithmetic against gguf-py's own reader rather than trusting the
    # (bytes, elements) tuple order in GGML_QUANT_SIZES.
    rd = GGUFReader(path)
    for t in tensors:
        # GGML_QUANT_SIZES maps a type to (elements per block, BYTES per block) - note the order,
        # e.g. Q8_0 -> (32, 34), MXFP8 -> (32, 33), NVFP4 -> (64, 36).
        be, bs = GGML_QUANT_SIZES[t["type"]]
        nelem = 1
        for s in t["shape"]:
            nelem *= s
        assert nelem % be == 0, f"{t['name']}: {nelem} values is not a whole number of {be}-value blocks"
        t["nbytes"] = (nelem // be) * bs
        r = rd.fields.get(t["name"])
        if r is not None and getattr(r, "n_bytes", None):
            assert int(r.n_bytes) == t["nbytes"], \
                f"{t['name']}: computed {t['nbytes']} B, gguf-py says {int(r.n_bytes)} B"
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv, "kv_start": kv_start,
            "kv_end": kv_end, "pad": pad, "table_end": table_end, "data_start": data_start,
            "align": align, "tensors": tensors}


def bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def copy_kv(w, key, fld) -> None:
    """Re-emit one metadata field with the SOURCE's declared GGUF types.

    Type fidelity matters: a python float would be written FLOAT64 where the loader reads float32,
    and gguf's get_arr checks the element type. Pattern taken from make_dspark_draft.py, which
    produced a file llama.cpp loads.
    """
    vt = fld.types[0]
    val = fld.contents()
    if vt == GGUF_TYPE.ARRAY:
        et = fld.types[1] if len(fld.types) > 1 else GGUF_TYPE.STRING
        items = val if isinstance(val, list) else [val]
        if et == GGUF_TYPE.STRING:
            items = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in items]
        w.add_key_value(key, items, GGUF_TYPE.ARRAY, sub_type=et)
        return
    if vt == GGUF_TYPE.STRING:
        w.add_key_value(key, val.decode("utf-8") if isinstance(val, bytes) else str(val), vt)
        return
    w.add_key_value(key, val, vt)


def build(args, hdr):
    names = args.names
    by_name = {t["name"]: t for t in hdr["tensors"]}
    for n in names:
        if n not in by_name:
            sys.exit(f"tensor {n!r} not in file ({len(by_name)} tensors present)")
        if by_name[n]["type"] != int(GGMLQuantizationType.BF16):
            sys.exit(f"tensor {n!r} has type id {by_name[n]['type']}; this script only replaces BF16 (30)")

    replaced = {n: by_name[n] for n in names}
    for t in replaced.values():
        nelem = _prod(t["shape"])
        t["new_nbytes"] = (nelem // 32) * 34
        assert t["shape"][0] % 32 == 0, \
            f"{t['name']}: ne0={t['shape'][0]} is not a multiple of 32, q8_0 cannot represent it"

    # llama.cpp's reader requires tensor offsets to be strictly contiguous in table order
    # (gguf_init_from_reader: "tensor X has offset N, expected M"), so the data section must be
    # rewritten packed. GGUFWriter owns offsets/padding; every tensor we are NOT re-encoding is fed
    # to it as the original raw bytes, so its data stays bit-identical (verify() proves it).
    import tempfile
    tmpdir = os.path.join(REPO, "logs", "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    tempfile.tempdir = tmpdir          # never /tmp (repo rule, and it needs ~18 GiB of headroom)

    from gguf.gguf_writer import GGUFWriter
    src = GGUFReader(args.src)
    arch = src.fields["general.architecture"].contents()
    arch = arch.decode("utf-8") if isinstance(arch, bytes) else str(arch)
    w = GGUFWriter(args.dst, arch, use_temp_file=True)
    n_kv_written = 0
    for key, fld in src.fields.items():
        # GGUFReader keeps header pseudo-fields in the same dict; re-emitting them corrupts the file
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        copy_kv(w, key, fld)
        n_kv_written += 1
    print("metadata: re-emitted %d KV entries (source had %d incl. header pseudo-keys)"
          % (n_kv_written, hdr["n_kv"]))

    with open(args.src, "rb") as fin:
        mm = mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ)
        for i, t in enumerate(hdr["tensors"]):
            base = hdr["data_start"] + t["offset"]
            raw = mm[base:base + t["nbytes"]]
            assert len(raw) == t["nbytes"], f"{t['name']}: wanted {t['nbytes']} B, read {len(raw)}"
            # gguf_writer emits shape[n_dims-1-j], so raw_shape is the reversal of the on-disk ne.
            # For a QUANTIZED type the shape must be given in BYTES: add_tensor_info runs
            # quant_shape_from_byte_shape on it, so passing element dims (or nothing) yields a
            # flattened tensor that llama.cpp rejects with check_tensor_dims. Byte dims = the
            # contiguous axis converted, everything else unchanged.
            raw_dtype = GGMLQuantizationType(t["type"])
            be, bs = GGML_QUANT_SIZES[t["type"]]
            raw_shape = tuple(reversed(t["shape"]))   # element order, for the unquantized branches
            if t["name"] in replaced:
                vals = np.ascontiguousarray(bf16_to_f32(raw).reshape(-1, t["shape"][0]), dtype=np.float32)
                q = Q8_0.quantize(vals)
                assert q.nbytes == t["new_nbytes"], \
                    f"{t['name']}: q8_0 gave {q.nbytes} B, planned {t['new_nbytes']}"
                qbe, qbs = GGML_QUANT_SIZES[GGMLQuantizationType.Q8_0]
                byte_shape = (t["shape"][0] // qbe * qbs,) + tuple(t["shape"][1:])
                w.add_tensor(t["name"], np.frombuffer(q.tobytes(), dtype=np.uint8),
                             raw_shape=tuple(reversed(byte_shape)),
                             raw_dtype=GGMLQuantizationType.Q8_0)
                print("  %-22s bf16 %11d B -> q8_0 %11d B (%.1f%% of original), byte row %d"
                      % (t["name"], t["nbytes"], t["new_nbytes"],
                         100.0 * t["new_nbytes"] / t["nbytes"], byte_shape[0]))
                del vals, q
            elif raw_dtype == GGMLQuantizationType.BF16:
                w.add_tensor(t["name"], np.frombuffer(raw, dtype=np.uint16),
                             raw_shape=raw_shape, raw_dtype=raw_dtype)
            elif raw_dtype == GGMLQuantizationType.F32:
                w.add_tensor(t["name"], np.frombuffer(raw, dtype=np.float32), raw_shape=raw_shape)
            else:
                byte_shape = (t["shape"][0] // be * bs,) + tuple(t["shape"][1:])
                w.add_tensor(t["name"], np.frombuffer(raw, dtype=np.uint8),
                             raw_shape=tuple(reversed(byte_shape)), raw_dtype=raw_dtype)
            del raw
            if (i + 1) % 100 == 0:
                print("  ... %d/%d tensors" % (i + 1, len(hdr["tensors"])), flush=True)
        mm.close()
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    a, b = os.path.getsize(args.src), os.path.getsize(args.dst)
    print("wrote %s" % args.dst)
    print("  original %.3f GiB -> variant %.3f GiB (%.3f GiB smaller)"
          % (a / 2 ** 30, b / 2 ** 30, (a - b) / 2 ** 30))


def _prod(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def _md5_ranges(path, table_end, spans):
    out = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for name, (off, n) in spans.items():
            h = hashlib.md5()
            pos, left = table_end + off, n
            CHUNK = 1 << 24
            while left > 0:
                k = min(CHUNK, left)
                h.update(mm[pos:pos + k])
                pos += k
                left -= k
            out[name] = h.hexdigest()
        mm.close()
    return out


def verify(args, src_hdr):
    dst_hdr = walk(args.dst)
    if src_hdr["n_kv"] != dst_hdr["n_kv"] or src_hdr["n_tensors"] != dst_hdr["n_tensors"]:
        sys.exit("FAIL: KV or tensor count changed")
    names = set(args.names)
    src_by = {t["name"]: t for t in src_hdr["tensors"]}
    dst_by = {t["name"]: t for t in dst_hdr["tensors"]}
    if set(src_by) != set(dst_by):
        sys.exit("FAIL: tensor name sets differ")
    # shapes must survive the rewrite - a byte/element mix-up flattens tensors and the loader then
    # rejects them with check_tensor_dims; content equality alone would not catch it (it did not).
    shape_bad = sorted(n for n in src_by if list(src_by[n]["shape"]) != list(dst_by[n]["shape"]))
    print("%-40s %s  (%d tensors compared)"
          % ("tensor shapes identical", "PASS" if not shape_bad else "FAIL: " + str(shape_bad[:4]),
             len(src_by)))
    if shape_bad:
        for n in shape_bad[:3]:
            print("    %-24s src %s -> dst %s" % (n, src_by[n]["shape"], dst_by[n]["shape"]))
        sys.exit("FAIL")

    # The metadata must PARSE to the same values. Byte equality of the KV section is not the property
    # worth defending once a writer re-emits it; value equality is what the loader consumes.
    def kv_values(path):
        r = GGUFReader(path)
        out = {}
        for k, fld in r.fields.items():
            if k.startswith("GGUF."):
                continue
            v = fld.contents()
            if isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, list):
                v = [x.decode("utf-8", "replace") if isinstance(x, bytes) else x for x in v]
            out[k] = v
        return out

    ka, kb = kv_values(args.src), kv_values(args.dst)
    meta_ok = True
    if set(ka) != set(kb):
        print("%-40s FAIL key sets differ (+%s / -%s)"
              % ("metadata values identical", sorted(set(kb) - set(ka))[:3], sorted(set(ka) - set(kb))[:3]))
        meta_ok = False
    else:
        diff = [k for k in ka if ka[k] != kb[k]]
        if diff:
            print("%-40s FAIL %d entries differ, e.g. %s" % ("metadata values identical", len(diff), diff[:4]))
            meta_ok = False
        else:
            print("%-40s PASS (%d entries)" % ("metadata values identical", len(ka)))

    span_s = {n: (src_by[n]["offset"], src_by[n]["nbytes"]) for n in src_by if n not in names}
    span_d = {n: (dst_by[n]["offset"], dst_by[n]["nbytes"]) for n in dst_by if n not in names}
    hs = _md5_ranges(args.src, src_hdr["data_start"], span_s)
    hd = _md5_ranges(args.dst, dst_hdr["data_start"], span_d)
    bad = sorted(n for n in hs if hs[n] != hd[n])
    gib = sum(v[1] for v in span_s.values()) / 2 ** 30
    print("%-40s %s  (%d tensors, %.2f GiB hashed)"
          % ("untouched tensors bit-identical", "PASS" if not bad else "FAIL: " + str(bad[:5]), len(hs), gib))
    for n in sorted(names):
        print("  replaced %-24s type %d -> %d, %d B -> %d B"
              % (n, src_by[n]["type"], dst_by[n]["type"], src_by[n]["nbytes"], dst_by[n]["nbytes"]))
    if bad or not meta_ok:
        sys.exit("FAIL")
    print("RESULT: ALL PASS - only the intended tensors changed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst")
    ap.add_argument("--replace", default="token_embd.weight,output.weight",
                    help="comma separated tensor names to re-encode (must be BF16)")
    ap.add_argument("--verify", action="store_true", help="prove only the intended bytes differ")
    args = ap.parse_args()
    args.names = [n.strip() for n in args.replace.split(",") if n.strip()]
    if args.verify:
        if not args.dst:
            sys.exit("--verify needs --dst")
        verify(args, walk(args.src))
        return
    if not args.dst:
        sys.exit("--dst is required to build")
    build(args, walk(args.src))


if __name__ == "__main__":
    main()
