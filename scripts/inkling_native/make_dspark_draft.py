#!/usr/bin/env python3
"""Derive a loadable Inkling DSpark draft GGUF from the embedded nextn tensors.

The drafter weights are the *original* `mtp.safetensors` values, already carried
losslessly into the main model file by `conversion/inkling2.py` as
`blk.<24..26>.nextn.*` + `nextn.*`.  This script only re-names and re-packages
them into the flat `dflash`-architecture layout that `src/models/dflash.cpp`
loads, so the published spec-decoding machinery (`--spec-type draft-dspark`) can
drive them.

Why derive from FULL.gguf instead of the safetensors:
  * FULL.gguf's tensors were written through the repo's standard add_tensor path,
    so the PyTorch [out, in] -> ggml [in, out] flip is applied by the same code
    as every other model, and they are byte-verified against the checkpoint
    (verify_draft_gguf.py).
  * it guarantees the embedded copy inside the main file and this separate draft
    file can never silently disagree.

The metadata (and the expected tensor names/orientations) are taken from the
public reference draft, which is the contract llama.cpp's loader enforces.  Every
2-D shape is asserted against it: an orientation mistake is exactly the failure
that "loads without error" hides, because the loader derives the markov rank from
stored metadata (see docs/mxfp8/34).

Usage:
    python3 scripts/inkling_native/make_dspark_draft.py \
        --full  ~/models/timlikesai/Inkling-2-Mini-FULL.gguf \
        --ref   ~/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf \
        --out   ~/models/timlikesai/Inkling-2-Mini-DSpark-BF16.gguf
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.constants import GGUFValueType, GGMLQuantizationType  # noqa: E402

# ref draft tensor (blk.N.<suffix>) -> embedded nextn suffix, per draft stage N
STAGE_MAP = {
    "attn_q":          "self_attn.q_proj",
    "attn_k":          "self_attn.k_proj",
    "attn_v":          "self_attn.v_proj",
    "attn_output":     "self_attn.o_proj",
    "attn_q_norm":     "self_attn.q_norm",
    "attn_k_norm":     "self_attn.k_norm",
    "attn_k_ctx":      "self_attn.k_proj_ctx",
    "attn_v_ctx":      "self_attn.v_proj_ctx",
    "attn_k_ctx_norm": "self_attn.k_norm_ctx",
    "attn_norm":       "input_layernorm",
    "ffn_norm":        "post_attention_layernorm",
    "dflare_norm":     "dflare_norm",
    "ffn_gate":        "mlp.gate_proj",
    "ffn_up":          "mlp.up_proj",
    "ffn_down":        "mlp.down_proj",
}

# ref global name -> embedded global name (the head's dflare norm is what the
# loader calls dspark_norm; see ref/dspark.py compute_f_rnn)
GLOBAL_MAP = {
    "markov_w1":          "nextn.markov_head.markov_w1",
    "joint_proj":         "nextn.markov_head.rnn_proj",
    "dspark_hidden_out":  "nextn.markov_head.hidden_out",
    "dspark_norm":        "nextn.markov_head.dflare_norm",
    "dspark_fuse":        "nextn.fuse",
    "mask_embd":          "nextn.mask_embed",
    "output_norm":        "nextn.output_norm",
}

# general.* keys that describe the file we are writing, not the reference
DROP_KEYS = {"general.file_type", "general.size_label"}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:  # kept for ad-hoc use
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_tensor(reader: GGUFReader, name: str) -> tuple[GGMLQuantizationType, list[int], bytes]:
    """Return (ggml_type, ne, raw_bytes) for one tensor of an open GGUF.

    The buffer is copied verbatim: ggml stores weights column-major with ne[0]
    contiguous, and both files agree on ne, so no transpose is wanted here.
    """
    for t in reader.tensors:
        if t.name != name:
            continue
        ne = [int(x) for x in t.shape]
        ty = t.tensor_type
        buf = np.ascontiguousarray(t.data).tobytes()
        n_el = int(np.prod(ne)) if ne else 1
        # F32 is 4 B/elem, BF16 2 B/elem; gguf-py may hand back either a byte view
        # or a uint16 view for BF16, so validate on bytes rather than on .dtype.
        want = n_el * (4 if ty == GGMLQuantizationType.F32 else 2)
        if len(buf) != want:
            raise SystemExit(f"{name}: {len(buf)} bytes for {ne} {ty.name} (expected {want})")
        return ty, ne, buf
    raise SystemExit(f"tensor {name} not found")


def f32_from(ty: GGMLQuantizationType, buf: bytes) -> np.ndarray:
    if ty == GGMLQuantizationType.F32:
        return np.frombuffer(buf, dtype=np.float32)
    u16 = np.frombuffer(buf, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="main model file carrying the embedded nextn tensors")
    ap.add_argument("--ref", required=True, help="public draft file: source of the name/metadata contract")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft-window", type=int, default=None,
                    help="override dflash.attention.sliding_window (the reference converter writes "
                         "sliding_window_size + 1, because a llama.cpp SWA query sees n_swa - 1 past "
                         "positions while the trained block sees the full window before the anchor)")
    args = ap.parse_args()

    full = GGUFReader(args.full)
    ref = GGUFReader(args.ref)

    n_layer = int(ref.fields["dflash.block_count"].contents())
    embd = int(ref.fields["dflash.embedding_length"].contents())

    # --- build the name mapping, asserting the orientation contract as we go ---
    plan: list[tuple[str, str, GGMLQuantizationType, list[int], bytes]] = []
    for t in sorted(ref.tensors, key=lambda x: x.name):
        rname = t.name
        base, _, suffix = rname.rpartition(".")          # suffix: weight|bias
        stem = base or rname
        if stem.startswith("blk."):
            stage = int(stem[4:].split(".", 1)[0])
            leaf = stem[4:].split(".", 1)[1]             # e.g. "attn_k_ctx" or "ffn.gate_proj"
            if "." in leaf:
                raise SystemExit(f"unexpected nested ref tensor name {rname}")
            mapped = STAGE_MAP.get(leaf)
            if mapped is None:
                raise SystemExit(f"no mapping for ref tensor {rname}")
            src = f"blk.{stage + n_layer_inkling_offset(full, n_layer)}.nextn.{mapped}.{suffix}"
        else:
            mapped = GLOBAL_MAP.get(stem)
            if mapped is None:
                raise SystemExit(f"no mapping for ref tensor {rname}")
            src = f"{mapped}.{suffix}" if suffix else mapped
            # nextn.* names in the main file keep the .weight/.bias suffix already
        if not any(x.name == src for x in full.tensors):
            raise SystemExit(f"mapped source tensor missing in main file: {src} (for {rname})")
        sty, sne, sbuf = read_tensor(full, src)
        rne = [int(x) for x in t.shape]
        if sorted(sne) != sorted(rne) or int(np.prod(sne)) != int(np.prod(rne)):
            raise SystemExit(
                f"{rname}: orientation/size disagreement\n  ref  {rne}\n  main {sne}\n"
                f"  (same element count required; a transpose is fine, a reshape is not)")
        plan.append((rname, src, sty, sne, sbuf))

    print(f"mapped {len(plan)} tensors "
          f"({sum(1 for p in plan if p[2] == GGMLQuantizationType.BF16)} BF16, "
          f"{sum(1 for p in plan if p[2] == GGMLQuantizationType.F32)} F32)")

    # --- write ---
    w = GGUFWriter(args.out, arch="dflash")
    for key, fld in sorted(ref.fields.items()):
        # GGUFReader keeps the *header* fields (GGUF.magic/.version/.tensor_count/.kv_count)
        # in the same dict as the KV section; re-emitting those corrupts the file.
        if key.startswith("GGUF.") or key == "general.architecture" or key in DROP_KEYS:
            continue
        copy_kv(w, key, fld)
    # MOSTLY_BF16 = 32 (include/llama.h): this file keeps the original dtypes
    if args.draft_window is not None and "%s.attention.sliding_window" % "dflash" in ref.fields:
        over = "dflash.attention.sliding_window"
        w.add_key_value(over, int(args.draft_window), GGUFValueType.UINT32)
        print("override %s = %d" % (over, args.draft_window))

    w.add_uint32("general.file_type", 32)
    w.add_string("general.size_label", "318M")

    # the tap mixing weights are the one tensor the runtime refuses to read as bf16:
    # build_dflare_alpha asserts F32 because the reference derives the weights with
    # softmax(fuse.float()), and bf16 alpha would round the mixing coefficients. The widening
    # is exact (every bf16 value is representable in f32), so the tensor stays lossless.
    WIDEN_TO_F32 = {"dspark_fuse.weight"}

    total = 0
    widened: list[str] = []
    for rname, src, sty, sne, sbuf in plan:
        ggty = GGMLQuantizationType.BF16 if sty == GGMLQuantizationType.BF16 else GGMLQuantizationType.F32
        # write_tensor_info emits shape[n_dims-1-j] (gguf_writer.py:272), so raw_shape is
        # the *torch*-order tuple, i.e. the reversal of the on-disk ne we read.
        raw_shape = tuple(reversed(sne))
        if ggty == GGMLQuantizationType.BF16 and rname not in WIDEN_TO_F32:
            # carry the original bf16 bit patterns; no f32 round trip, no requantization
            arr = np.frombuffer(sbuf, dtype=np.uint16)
            w.add_tensor(rname, arr, raw_shape=raw_shape, raw_dtype=ggty)
            total += len(sbuf)
        elif ggty == GGMLQuantizationType.BF16:
            vals = f32_from(sty, sbuf)
            back = (vals.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            if not np.array_equal(back, np.frombuffer(sbuf, dtype=np.uint16)):
                raise SystemExit(f"{rname}: bf16 -> f32 -> bf16 is not exact, refusing to widen")
            w.add_tensor(rname, np.ascontiguousarray(vals, dtype=np.float32), raw_shape=raw_shape)
            total += vals.nbytes
            widened.append(rname)
        else:
            w.add_tensor(rname, np.frombuffer(sbuf, dtype=np.float32), raw_shape=raw_shape)
            total += len(sbuf)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {args.out}: {os.path.getsize(args.out)} B, {len(plan)} tensors, "
          f"{total / 1e6:.1f} MB of tensor data")
    if widened:
        print("widened to f32 (lossless, required by build_dflare_alpha): %s" % ", ".join(widened))
    return 0


def n_layer_inkling_offset(full: GGUFReader, n_draft: int) -> int:
    """The main file stores the draft stages at blk.<n_layer + i>; derive n_layer."""
    bc = int(full.fields["inkling2_loop.block_count"].contents())
    nxt = int(full.fields["inkling2_loop.nextn_predict_layers"].contents())
    if nxt != n_draft:
        raise SystemExit(f"draft block_count {n_draft} != main nextn_predict_layers {nxt}")
    return bc - nxt


def copy_kv(w: GGUFWriter, key: str, fld) -> None:
    """Re-emit one metadata field with the reference file's *declared* GGUF types.

    Type fidelity matters: a python float would be written FLOAT64 where the loader
    reads float32 (and gguf's get_arr checks the element type), so the element type
    is taken from the source field rather than inferred from the value.
    """
    vt = fld.types[0]
    val = fld.contents()
    if vt == GGUFValueType.ARRAY:
        et = fld.types[1] if len(fld.types) > 1 else GGUFValueType.STRING
        items = val if isinstance(val, list) else [val]
        if et == GGUFValueType.STRING:
            items = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in items]
        w.add_key_value(key, items, GGUFValueType.ARRAY, sub_type=et)
        return
    if vt == GGUFValueType.STRING:
        w.add_key_value(key, val.decode("utf-8") if isinstance(val, bytes) else str(val), vt)
        return
    w.add_key_value(key, val, vt)


if __name__ == "__main__":
    raise SystemExit(main())
