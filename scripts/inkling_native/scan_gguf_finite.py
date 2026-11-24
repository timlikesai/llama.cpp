#!/usr/bin/env python3
"""Find where a GGUF carries values it cannot represent, or already stores as non-finite.

Two failure shapes are separated:
  * file-level  : a block scale stored as float16 (Q8_0, IQ4_NL, ...) that is inf/nan on disk,
                  and its implied source magnitude (127 * d for Q8_0);
  * value-level : dequantized elements that are inf/nan for whatever reason.

Every tensor is walked in bounded chunks straight from the memmap, so a 27 GB file costs no RAM.

    ./scripts/inkling_native/scan_gguf_finite.py [--gguf PATH] [--top 12] [--only SUBSTR]

Exit code 0 = file is finite everywhere, 1 = at least one tensor is bad. Nothing is written.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
from gguf import GGUFReader                       # noqa: E402
from gguf.constants import GGMLQuantizationType    # noqa: E402

DEFAULT_GGUF = os.path.expanduser("~/models/timlikesai/Inkling-DEQ-Q80.gguf")
CHUNK_BYTES = 1 << 26

# type -> (bytes per block, elements per block, scale dtype at block offset 0)
HALF_SCALE = {GGMLQuantizationType.Q8_0: (34, 32, 2), GGMLQuantizationType.IQ4_NL: (18, 32, 2),
              GGMLQuantizationType.Q4_0: (18, 32, 2), GGMLQuantizationType.Q5_0: (22, 32, 2),
              GGMLQuantizationType.Q5_1: (26, 32, 4)}
Q8_0_MAX, F16_MAX = 127.0, 65504.0


def scan_half_scale(raw, block_bytes, scale_off=0):
    """Count non-finite float16 block scales and the largest finite one.

    A finite half scale bounds the block's values at 127 * |d| (Q8_0) or ~8 * |d| (IQ4_NL),
    so finiteness of d is sufficient to prove the stored values are finite.
    """
    blocks = raw.reshape(-1, block_bytes)
    scales = np.ascontiguousarray(blocks[:, scale_off:scale_off + 2]).view(np.float16).astype(np.float32)
    bad = ~np.isfinite(scales)
    finite = scales[~bad]
    return int(bad.sum()), (float(np.abs(finite).max()) if finite.size else 0.0)


def scan_values(arr):
    bad = ~np.isfinite(arr)
    finite = arr[~bad]
    return int(bad.sum()), (float(np.abs(finite).max()) if finite.size else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--only", default="", help="substring filter on tensor name")
    args = ap.parse_args()

    r = GGUFReader(args.gguf, "r")
    print(f"gguf     : {args.gguf}")
    print(f"tensors  : {len(r.tensors)}  alignment={r.alignment}  data_offset={r.data_offset}")
    from collections import Counter
    print("types    :", dict(Counter(GGMLQuantizationType(t.tensor_type).name for t in r.tensors)))

    rows, bad_tensors = [], []
    for t in sorted(r.tensors, key=lambda x: x.name):
        if args.only and args.only not in t.name:
            continue
        qtype = GGMLQuantizationType(t.tensor_type)
        name, nbytes = t.name, int(t.n_bytes)
        raw = np.asarray(t.data).view(np.uint8).reshape(-1)
        assert raw.size == nbytes, f"{name}: reader gave {raw.size} bytes, header says {nbytes}"
        n_bad, mag, note = 0, 0.0, ""
        if qtype in HALF_SCALE:
            block_bytes, _els, _sz = HALF_SCALE[qtype]
            assert nbytes % block_bytes == 0, f"{name}: {nbytes} not a whole number of {block_bytes}-byte blocks"
            step = CHUNK_BYTES - CHUNK_BYTES % block_bytes
            for i in range(0, nbytes, step):
                b, m = scan_half_scale(raw[i:i + step], block_bytes)
                n_bad += b
                mag = max(mag, m)
            note = f"scale=f16, max|d|={mag:.6g} -> max|x|~{mag * Q8_0_MAX:.6g}"
            mag = mag * Q8_0_MAX  # rank tensors by the value bound, like the f32/bf16 legs
        elif qtype == GGMLQuantizationType.F32:
            for i in range(0, nbytes, CHUNK_BYTES):
                b, m = scan_values(raw[i:i + CHUNK_BYTES].view(np.float32))
                n_bad += b
                mag = max(mag, m)
            note = f"f32 max|x|={mag:.6g}"
        elif qtype == GGMLQuantizationType.BF16:
            for i in range(0, nbytes, CHUNK_BYTES):
                # bf16 -> f32 by widening the mantissa, no table needed
                b, m = scan_values((raw[i:i + CHUNK_BYTES].view(np.uint16).astype(np.uint32) << 16).view(np.float32))
                n_bad += b
                mag = max(mag, m)
            note = f"bf16 max|x|={mag:.6g}"
        if n_bad:
            bad_tensors.append((name, qtype.name, list(map(int, t.shape)), n_bad, note))
        rows.append((name, qtype.name, nbytes, n_bad, mag, note))

    print(f"\nscanned {len(rows)} tensors, {sum(r[2] for r in rows) / 1e9:.2f} GB of tensor data")
    if not bad_tensors:
        print("RESULT: no non-finite block scales or values found")
    else:
        print(f"RESULT: {len(bad_tensors)} tensor(s) carry non-finite content")
        for name, qt, shape, n, note in bad_tensors[:args.top]:
            print(f"  BAD {name:<44} {qt:<8} ne={shape} non-finite={n}  {note}")
        if len(bad_tensors) > args.top:
            print(f"  ... {len(bad_tensors) - args.top} more")
        layers = Counter(name.split(".")[1] for name, *_ in bad_tensors) if all(
            n.split(".")[0] == "blk" for n, *_ in bad_tensors) else {}
        if layers:
            print("  by layer:", dict(sorted(layers.items(), key=lambda kv: int(kv[0]))))
    print("\nlargest finite magnitudes seen:")
    for name, qt, nbytes, n, mag, note in sorted(rows, key=lambda x: -x[4])[:args.top]:
        print(f"  {name:<44} {qt:<8} max|x|={mag:.6g} non-finite={n}")
    return 1 if bad_tensors else 0


if __name__ == "__main__":
    sys.exit(main())
