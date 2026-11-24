#!/usr/bin/env python3
"""Rewrite the per-expert NVFP4 `.scale` sidecars of a GGUF in place, without touching any payload.

The sidecars are F32 [n_expert] tensors, so their byte size and offsets are unchanged: the runtime sees
a different global factor for every expert while every packed NVFP4/MXFP8 byte stays where it was.
Used to answer one question - does the runtime read the sidecar at all? - by setting it to a value the
encoder provably did not use.

    cp model.gguf model-scale1.gguf
    ./scripts/inkling_native/patch_expert_scales.py --inplace model-scale1.gguf --value 1.0

Exit code 0 means every sidecar tensor was rewritten and reads back as requested.
"""
import argparse
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))

from gguf import GGUFReader  # noqa: E402

SIDECAR = re.compile(r"ffn_\w+_exps\.scale$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inplace", required=True, help="GGUF to patch (payload bytes are left alone)")
    ap.add_argument("--value", type=float, default=None, help="constant written over every sidecar entry")
    ap.add_argument("--multiply", type=float, default=None, help="multiply each existing entry instead")
    args = ap.parse_args()

    if (args.value is None) == (args.multiply is None):
        print("give exactly one of --value or --multiply")
        return 2

    reader = GGUFReader(args.inplace, mode="r+")
    targets = [t for t in reader.tensors if SIDECAR.search(t.name)]
    if not targets:
        print(f"{args.inplace}: no ffn_*_exps.scale tensors found")
        return 1

    with open(args.inplace, "r+b") as f:
        for t in targets:
            before = np.asarray(t.data, dtype=np.float32).copy()
            after = np.full(before.shape, np.float32(args.value)) if args.multiply is None \
                else (before * np.float32(args.multiply)).astype(np.float32)
            f.seek(t.data_offset)
            f.write(after.tobytes())
            print(f"  {t.name:<40} off={t.data_offset:<14} n={before.size} "
                  f"before[{before[0]:.6e},{before[-1]:.6e}] after[{after[0]:.6e},{after[-1]:.6e}]")
        f.flush()
        os.fsync(f.fileno())

    check = GGUFReader(args.inplace)
    bad = 0
    for t in check.tensors:
        if not SIDECAR.search(t.name):
            continue
        got = np.asarray(t.data, dtype=np.float32)
        want = np.full(got.shape, np.float32(args.value)) if args.multiply is None else None
        if want is not None and not np.array_equal(got, want):
            print(f"  VERIFY FAILED {t.name}: {got[:4]}")
            bad += 1
    print(f"patched {len(targets)} sidecars, verify failures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
