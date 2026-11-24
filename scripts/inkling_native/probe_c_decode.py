#!/usr/bin/env python3
"""Ask ggml's *compiled* NVFP4 decoders what the bytes in our file mean.

verify_nvfp4_values.py compares the file against a python port of dequantize_row_nvfp4. A port
can share an assumption with the packer it is checking, so it cannot settle a convention
disagreement. This calls the real functions out of the shared libraries the runtime loads:

  dequantize_row_nvfp4   (build-cuda/bin/libggml-base.so)   the host decode
  ggml_vec_dot_nvfp4_q8_0(build-cuda/bin/libggml-cpu.so)    the CPU compute path (AVX2 SIMD)

For the dot path a one-hot activation is used: dot(row, e_k) is exactly element k, so the compute
path's own idea of the element order is read off directly, one element at a time.

Reference values come from the checkpoint itself, through conversion/inkling2.py's semantics
(pairs (2i, 2i+1) per byte, e4m3 block scale, per-expert amax/outer_scale_denominator), which is
the only source of meaning for these weights.

    ./scripts/inkling_native/probe_c_decode.py [--gguf PATH] [--blocks 0,17] [--experts 0,64]
                                               [--rows 0,1152] [--dot]

Exit code 0 means ggml's C decode x file sidecar == checkpoint for every sampled row, and (with
--dot) the CPU dot path agrees element by element. Nothing is written.
"""
import argparse
import ctypes
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
sys.path.insert(0, HERE)

from gguf import GGUFReader                                        # noqa: E402
from verify_mxfp8_gguf import Safetensors, NVFP4                   # noqa: E402
from verify_nvfp4_values import hf_reference, ggml_dequant_nvfp4   # noqa: E402

DEFAULT_GGUF = os.path.join(os.path.expanduser("~"), "models/timlikesai/Inkling-2-Mini-NATIVE.gguf")
SNAP_GLOB = "models--thinkingmachines--Inkling-2-Mini/snapshots/*"
LIB_DIR = os.path.join(REPO, "build-cuda", "bin")


class Ggml:
    """The two compiled decode entry points, straight out of the libraries llama uses at runtime."""

    def __init__(self, lib_dir=LIB_DIR):
        self.base = ctypes.CDLL(os.path.join(lib_dir, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
        self.cpu = ctypes.CDLL(os.path.join(lib_dir, "libggml-cpu.so"))
        # the x86/ARM UE4M3 and E8M0 scales come from LUTs filled in ggml_cpu_init (ggml-cpu.c:3884);
        # without it every sub-scale reads 0 and every dot returns 0
        self.cpu.ggml_cpu_init()
        f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.base.dequantize_row_nvfp4.argtypes = [ctypes.c_void_p, f32p, ctypes.c_int64]
        self.base.dequantize_row_nvfp4.restype = None
        # int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc
        self.cpu.ggml_vec_dot_nvfp4_q8_0.argtypes = [
            ctypes.c_int, f32p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cpu.ggml_vec_dot_nvfp4_q8_0.restype = None
        self.cpu.quantize_row_q8_0.argtypes = [f32p, ctypes.c_void_p, ctypes.c_int64]
        self.cpu.quantize_row_q8_0.restype = None
        self.base.dequantize_row_q8_0.argtypes = [ctypes.c_void_p, f32p, ctypes.c_int64]
        self.base.dequantize_row_q8_0.restype = None

    def dequant_nvfp4(self, row_bytes, cols):
        out = np.zeros(cols, dtype=np.float32)
        src = np.ascontiguousarray(row_bytes, dtype=np.uint8)
        self.base.dequantize_row_nvfp4(src.ctypes.data, out, ctypes.c_int64(cols))
        return out

    def dot_nvfp4(self, row_bytes, cols, act):
        """One dot product: NVFP4 weight row against a q8_0-quantized float activation."""
        x = np.ascontiguousarray(row_bytes, dtype=np.uint8)
        y = np.zeros(cols // 32 * 34, dtype=np.uint8)  # sizeof(block_q8_0) == 34
        a = np.ascontiguousarray(act, dtype=np.float32)
        self.cpu.quantize_row_q8_0(a, y.ctypes.data, ctypes.c_int64(cols))
        s = np.zeros(1, dtype=np.float32)
        self.cpu.ggml_vec_dot_nvfp4_q8_0(cols, s, 0, x.ctypes.data, 0, y.ctypes.data, 0, 1)
        return float(s[0])

    def dequant_q8_0(self, buf, cols):
        out = np.zeros(cols, dtype=np.float32)
        self.base.dequantize_row_q8_0(buf.ctypes.data, out, ctypes.c_int64(cols))
        return out


def rel(got, want):
    err = float(np.max(np.abs(got - want)))
    return err, err / max(float(np.max(np.abs(want))), 1e-30)


def first_diff(got, want, tol=1e-6):
    scale = np.maximum(np.abs(want), 1e-6)
    bad = np.nonzero(np.abs(got - want) / scale > tol)[0]
    return int(bad[0]) if bad.size else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--blocks", default="0,17", help="NVFP4 expert blocks to sample")
    ap.add_argument("--experts", default="0,64")
    ap.add_argument("--rows", default="0,1,1151,1152,2303", help="gguf row indices within the expert")
    ap.add_argument("--dot", action="store_true", help="also probe ggml_vec_dot_nvfp4_q8_0 with one-hot activations")
    ap.add_argument("--dot-k", default="0,1,7,8,15,16,63", help="element indices to read out through the dot path")
    args = ap.parse_args()

    snap = glob.glob(os.path.expanduser(os.path.join("~/.cache/huggingface/hub", SNAP_GLOB)))[0]
    st = Safetensors(os.path.join(snap, "model-00001-of-00001.safetensors"))
    r = GGUFReader(args.gguf)
    tensors = {t.name: t for t in r.tensors}
    gg = Ggml()
    print(f"gguf  : {args.gguf}")
    print(f"ckpt  : {snap}")
    print(f"libs  : {LIB_DIR}")

    fails = 0
    for block in [int(b) for b in args.blocks.split(",")]:
        for proj, hf_proj in (("gate_up", "w13"), ("down", "w2")):
            name = f"blk.{block}.ffn_{proj}_exps.weight"
            t = tensors.get(name)
            if t is None or int(t.tensor_type) != NVFP4:
                print(f"\n  {name}: absent or not NVFP4, skipped")
                continue
            ne0, ne1, ne2 = (int(x) for x in t.shape)
            experts, rows, cols = ne2, ne1, ne0
            row_bytes = cols // 64 * 36
            raw = np.asarray(t.data).view(np.uint8).reshape(experts, rows, row_bytes)
            side = tensors[f"blk.{block}.ffn_{proj}_exps.scale"]
            factor = np.asarray(side.data, dtype=np.float32)
            hf_stem = f"model.llm.layers.{block}.mlp.experts.{hf_proj}_weight"
            print(f"\n  {name}  experts={experts} rows={rows} cols={cols}")

            for e in [int(x) for x in args.experts.split(",")]:
                for gr in [int(x) for x in args.rows.split(",") if int(x) < rows]:
                    if proj == "gate_up":
                        half, local = gr // (rows // 2), gr % (rows // 2)
                        hf_row = 2 * local + half
                    else:
                        hf_row = gr
                    want = hf_reference(st, hf_stem, e, hf_row, cols, mxfp8=False)
                    row = raw[e, gr]

                    got_c = gg.dequant_nvfp4(row, cols) * factor[e]
                    got_p = ggml_dequant_nvfp4(row[None], cols)[0] * factor[e]
                    ec, rc = rel(got_c, want)
                    ep, rp = rel(got_p, want)
                    ecp, rcp = rel(got_c, got_p)
                    k = first_diff(got_c, want)
                    tag = "MATCH" if rc < 1e-6 else "MISMATCH"
                    print(f"    e{e:<4} r{gr:<5} C-dequant  rel={rc:.3e} {tag}"
                          f" | python port rel={rp:.3e} | C vs port rel={rcp:.3e}"
                          f" | first diff idx={k}")
                    if k >= 0:
                        s = max(k - 1, 0)
                        print(f"        want[{s}:{s+6}] = {np.array2string(want[s:s+6], precision=6)}")
                        print(f"        C   [{s}:{s+6}] = {np.array2string(got_c[s:s+6], precision=6)}")
                        fails += 1
                    if rc >= 1e-6:
                        fails += 1

                    if args.dot:
                        buf = row.copy()
                        worst = 0.0
                        for kk in [int(x) for x in args.dot_k.split(",") if int(x) < cols]:
                            act = np.zeros(cols, dtype=np.float32)
                            act[kk] = 1.0
                            d = gg.dot_nvfp4(buf, cols, act) * factor[e]
                            w = want[kk]
                            # q8_0 of a one-hot is exact (127 * 1/127), so any error is convention
                            err = abs(d - w) / max(abs(w), 1e-6)
                            worst = max(worst, err)
                            if err > 1e-3:
                                print(f"        dot e_k k={kk:<4} -> {d:+.6f} want {w:+.6f} rel={err:.3e}")
                        tag = "MATCH" if worst < 1e-3 else "MISMATCH"
                        print(f"    e{e:<4} r{gr:<5} CPU dot path  worst rel over one-hots = {worst:.3e} {tag}")
                        if worst >= 1e-3:
                            fails += 1
    print(f"\n=== {'OK' if fails == 0 else f'{fails} MISMATCHES'} ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
