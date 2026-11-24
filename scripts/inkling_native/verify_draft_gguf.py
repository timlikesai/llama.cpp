#!/usr/bin/env python3
"""Verify the embedded Inkling drafter in a GGUF against mtp.safetensors.

Every one of the 54 draft tensors must be present, correctly named, and numerically identical to
the checkpoint: 2-D tensors byte-for-byte (bf16 stays bf16), 1-D tensors value-for-value (the
converter widens them to F32, which is exact for a bf16 input). Also checks the metadata the
runtime needs to place the draft outside the trunk: block_count = trunk + draft,
nextn_predict_layers = draft count, and the per-layer arrays long enough to cover the draft.

usage: verify_draft_gguf.py --gguf <file> [--snapshot <dir>]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

# use this checkout's gguf-py: it knows GGML_TYPE_MXFP8 = 44, the installed copy does not and
# raises "np.uint32(44) is not a valid GGMLQuantizationType" while parsing the file header
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "gguf-py"))

from safetensors import safe_open  # noqa: E402
from gguf import GGUFReader  # noqa: E402

DEFAULT_SNAP = (pathlib.Path.home() / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
                / "snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337")

# the name mapping, kept here as an independent copy of conversion/inkling2.py::_draft_name so a
# wrong mapping in the converter cannot be confirmed by the same wrong assumption here
def draft_name(hf_name: str, trunk_layers: int) -> str:
    n = hf_name[len("mtp."):] if hf_name.startswith("mtp.") else hf_name
    if n.startswith("layers."):
        _, stage, rest = n.split(".", 2)
        return f"blk.{trunk_layers + int(stage)}.nextn.{rest}"
    if n == "fuse_weight":
        return "nextn.fuse.weight"
    if n == "mask_embed":
        return "nextn.mask_embed.weight"
    if n == "norm.weight":
        return "nextn.output_norm.weight"
    if n.startswith("markov_head."):
        return f"nextn.{n}"
    raise ValueError(f"unmapped draft tensor {hf_name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAP))
    args = ap.parse_args()

    cfg = json.loads((pathlib.Path(args.snapshot) / "config.json").read_text())
    trunk = int(cfg["text_config"]["num_hidden_layers"])
    n_draft = int(cfg["mtp_config"]["num_hidden_layers"])
    arch = "inkling2_loop"

    reader = GGUFReader(args.gguf)
    fields = reader.fields
    def get(k):
        f = fields.get(f"{arch}.{k}")
        if f is None:
            return None
        v = f.contents
        # gguf-py has exposed this both as a property returning a callable and as a method; unwrap either
        while callable(v) and not isinstance(v, (int, float, str, bytes, np.ndarray, list)):
            v = v()
        return v

    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    print(f"gguf   : {args.gguf}")
    print(f"file   : {pathlib.Path(args.gguf).stat().st_size:,} B")
    print(f"trunk  : {trunk} layers, drafter: {n_draft} layers\n")

    print("metadata")
    bc = get("block_count")
    nn = get("nextn_predict_layers")
    check(bc is not None and int(bc) == trunk + n_draft, f"block_count = {bc} (want {trunk + n_draft})")
    check(nn is not None and int(nn) == n_draft, f"nextn_predict_layers = {nn} (want {n_draft})")
    for key, want_last in (("feed_forward_length", int(cfg["mtp_config"]["intermediate_size"])),
                           ("attention.head_count", int(cfg["mtp_config"]["num_attention_heads"]))):
        v = get(key)
        arr = np.atleast_1d(np.asarray(v if v is not None else []))
        check(arr.size >= trunk + n_draft and int(arr[-1]) == want_last,
              f"{key}: len {arr.size}, draft entry {int(arr[-1]) if arr.size else None} (want {want_last})")
    pat = np.atleast_1d(np.asarray(get("attention.sliding_window_pattern") or []))
    # inkling2.cpp:38 requires the trunk pattern array to be exactly n_layer() == block_count - nextn
    check(pat.size == trunk, f"trunk sliding_window_pattern entries = {pat.size} (want exactly n_layer() = {trunk})")
    dpat = [bool(x) for x in np.atleast_1d(np.asarray(get("mtp.attention.sliding_window_pattern") or []))]
    check(dpat == [True, False, True],
          f"mtp.attention.sliding_window_pattern = {dpat} (want [True, False, True] per local_layer_ids)")
    for key in ("mtp.block_size", "mtp.markov_rank", "mtp.markov_head_type", "mtp.target_layer_ids"):
        v = get(key)
        check(v is not None, f"{key} present ({v if v is not None else 'MISSING'})")
    check(int(get("mtp.block_size") or 0) == int(cfg["mtp_config"]["block_size"]), "mtp.block_size == mtp_config")
    check(int(get("mtp.markov_rank") or 0) == int(cfg["mtp_config"]["markov_rank"]), "mtp.markov_rank == mtp_config")
    check(str(get("mtp.markov_head_type")) == cfg["mtp_config"]["markov_head_type"], "mtp.markov_head_type == mtp_config")

    print("\ndraft tensors vs checkpoint")
    tmaps = {t.name: t for t in reader.tensors}
    n_exact2d = n_exact1d = 0
    orders: dict[str, int] = {}
    with safe_open(str(pathlib.Path(args.snapshot) / "mtp.safetensors"), framework="pt") as f:
        keys = sorted(f.keys())
        check(len(keys) == 54, f"checkpoint has {len(keys)} draft tensors (want 54)")
        for k in keys:
            want = f.get_tensor(k)                       # bf16, as shipped
            name = draft_name(k, trunk)
            t = tmaps.get(name)
            if t is None:
                fails.append(f"{name}: MISSING from gguf")
                print(f"  FAIL {name}: missing")
                continue
            raw = np.asarray(t.data)
            if raw.dtype == np.uint8 and raw.size % 2 == 0 and want.ndim >= 2:
                # gguf-py hands BF16 tensor data back as a byte view (2 uint8 per element) - reinterpret
                # as the bf16 bit pattern so the comparison is against what the file actually stores
                raw = raw.view(np.uint16)

            if want.ndim >= 2:
                # compare the bf16 bit patterns: no dequantization, no rounding, no dtype map involved
                bits = want.view(torch.uint16).numpy()
                same = raw.shape == bits.shape and np.array_equal(raw, bits)
                trans = raw.shape == bits.T.shape and np.array_equal(raw, bits.T)
                ok = bool(same or trans)
                tag = "row-major" if same else ("transposed (ggml [in,out])" if trans else "MISMATCH")
                orders[tag] = orders.get(tag, 0) + 1
                n_exact2d += ok
                detail = f"bf16 bits {tag}, shape {raw.shape}"
            else:
                ok = raw.shape == tuple(want.shape) and np.array_equal(
                    raw.astype(np.float64), want.float().numpy().astype(np.float64))
                n_exact1d += ok
                detail = f"exact after bf16->f32 widen, shape {raw.shape}"

            if not ok:
                detail += f"  (checkpoint {tuple(want.shape)} vs gguf {raw.dtype}{raw.shape})"
            print(f"  {'ok  ' if ok else 'FAIL'} {name:52s} {detail}")
            if not ok:
                fails.append(f"{name}: value mismatch")

    print(f"\n2-D byte-exact: {n_exact2d}   1-D value-exact: {n_exact1d}   total: {n_exact2d + n_exact1d}/54")
    print(f"2-D layouts seen: {orders}")
    check(n_exact2d + n_exact1d == 54 and n_exact2d == 31 and n_exact1d == 23,
          "all 54 draft tensors match the checkpoint (31 bf16 matrices, 23 vectors)")

    print("\ntrunk experts still native")
    from gguf.constants import GGMLQuantizationType as QT
    nv = sum(1 for t in reader.tensors if t.tensor_type == QT.NVFP4)
    mx = sum(1 for t in reader.tensors if t.tensor_type == QT.MXFP8)
    bf = sum(1 for t in reader.tensors if t.tensor_type == QT.BF16)
    sc = sum(1 for t in reader.tensors if t.name.endswith(".scale") and t.tensor_type == QT.F32)
    print(f"  NVFP4 {nv}  MXFP8 {mx}  BF16 {bf}  F32 .scale {sc}  total tensors {len(reader.tensors)}")
    check(nv == 40, f"NVFP4 tensors = {nv} (want 40)")
    check(mx == 8, f"MXFP8 tensors = {mx} (want 8)")
    check(sc == 40, f"F32 .scale sidecars = {sc} (want 40)")
    check(len(reader.tensors) == 798, f"tensor count = {len(reader.tensors)} (want 744 trunk + 54 draft)")

    print()
    if fails:
        print(f"RESULT: FAIL ({len(fails)} problems)")
        for x in fails:
            print(f"  - {x}")
        return 1
    print("RESULT: PASS - every original drafter weight is present and exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
