#!/usr/bin/env python3
"""Print the DFlare alpha rows (softmax over taps) for the Inkling DSpark drafter.

Why this matters: fuse_weight is [n_draft_layers + 1, n_taps]. Rows 0..2 mix the target taps that are
injected as each draft stage's context; row 3 is the markov head's own mix (compute_f_rnn). The
per-position acceptance decomposition showed the gap versus the independent implementation is
*uniform across slots*, and the stage-context ablation showed context carries ~5x the signal. If any
alpha row is strongly peaked, then the ORDER of the concatenated taps decides which physical layer
gets that weight - so a mis-ordered tap vector degrades every slot uniformly, which is the observed
signature.

Reads the published draft GGUF (which stores dspark_fuse as F32) and, if available, the checkpoint's
mtp.safetensors, and prints both so a mismatch in either values or row count is visible.
"""
import glob
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402


def softmax_rows(a):
    a = np.asarray(a, dtype=np.float64)
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def from_gguf(path):
    r = GGUFReader(path, modes="r")
    for name, t in r.fields.items():
        pass
    for t in r.tensors.values():
        if "fuse" not in t.name:
            continue
        a = np.array(t.data, dtype=np.float32)
        # gguf-py stores shapes in on-file order (fastest dim last); the tensor is [n_taps, n_rows]
        # in torch convention, so reshape by the declared dims reversed.
        shp = list(t.shapes)[::-1]
        print(f"  GGUF  {os.path.basename(path)}")
        print(f"        {t.name}  file-shape={list(t.shapes)} -> logical {shp}  n={a.size}")
        return a.reshape(shp) if a.size == int(np.prod(shp)) else a
    return None


def from_safetensors(patterns):
    import json
    import struct

    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            try:
                with open(p, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(n))
            except Exception as e:
                print(f"  skip {p}: {e}")
                continue
            for k, v in hdr.items():
                if k == "__metadata__" or "fuse" not in k:
                    continue
                ds, de = v["data_offsets"]
                with open(p, "rb") as f:
                    f.seek(8 + n + ds)
                    raw = f.read(de - ds)
                if v["dtype"] == "BF16":
                    a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
                else:
                    a = np.frombuffer(raw, dtype=v["dtype"].lower())
                a = a.reshape(v["shape"])
                print(f"  CKPT  {p}")
                print(f"        {k}  dtype={v['dtype']}  shape={v['shape']}")
                return a
    return None


def main():
    drafts = sorted(glob.glob("/home/tim/models/**/*spark*.gguf", recursive=True)) + \
             sorted(glob.glob("/home/tim/models/**/*DSpark*.gguf", recursive=True))
    seen = set()
    fuse = None
    for d in drafts:
        if d in seen:
            continue
        seen.add(d)
        try:
            a = from_gguf(d)
        except Exception as e:
            print(f"  {os.path.basename(d)}: {type(e).__name__}: {e}")
            continue
        if a is not None and fuse is None:
            fuse = a

    if fuse is None:
        fuse = from_safetensors([
            "/home/tim/models/**/mtp.safetensors",
            "/home/tim/.cache/huggingface/**/mtp.safetensors",
        ])

    if fuse is None:
        print("no fuse tensor found")
        return 1

    fuse = np.asarray(fuse, dtype=np.float64).reshape(fuse.shape[-2], fuse.shape[-1]) \
        if fuse.ndim == 2 else np.asarray(fuse, dtype=np.float64)
    print(f"\nfuse_weight {fuse.shape}  (rows = n_draft_layers + 1, cols = n_taps)")
    al = softmax_rows(fuse)
    taps = [5, 23, 41, 48]
    for i, row in enumerate(al):
        peak = int(np.argmax(row))
        label = f"draft stage {i}" if i < al.shape[0] - 1 else "markov head (f_rnn)"
        print("  alpha row %d  %-22s %s   peak -> tap %s (layer %s) at %.3f" % (
            i, label, np.round(row, 3), peak, taps[peak] if peak < len(taps) else "?", row[peak]))
    print("\nIf a row is peaked, tap ORDER matters: index 2 must be layer 41 for the head row, and")
    print("the stage rows must index the same physical taps. A permutation that moves the peak onto a")
    print("different layer is a uniform, plausible-but-wrong conditioning error - the failure mode")
    print("that generated this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
