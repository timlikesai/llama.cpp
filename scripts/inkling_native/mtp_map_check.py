#!/usr/bin/env python3
"""Verify the Inkling-2 MTP head (mtp.safetensors, 54 BF16) against the public
dspark-Inkling-2-Mini-Q8_0.gguf (54 tensors), tensor by tensor.

Proves the name mapping and the dtype rule. Run: python3 scripts/inkling_native/mtp_map_check.py
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    "/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/mtp.safetensors")
REF = os.path.expanduser("~/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf")

# HF name -> GGUF name. The per-layer part is a plain rename of the HF block fields.
LAYER_MAP = {
    "input_layernorm.weight":             "attn_norm.weight",
    "post_attention_layernorm.weight":    "ffn_norm.weight",
    "dflare_norm.weight":                 "dflare_norm.weight",
    "self_attn.q_proj.weight":            "attn_q.weight",
    "self_attn.k_proj.weight":            "attn_k.weight",
    "self_attn.v_proj.weight":            "attn_v.weight",
    "self_attn.o_proj.weight":            "attn_output.weight",
    "self_attn.k_proj_ctx.weight":        "attn_k_ctx.weight",
    "self_attn.v_proj_ctx.weight":        "attn_v_ctx.weight",
    "self_attn.q_norm.weight":            "attn_q_norm.weight",
    "self_attn.k_norm.weight":            "attn_k_norm.weight",
    "self_attn.k_norm_ctx.weight":        "attn_k_ctx_norm.weight",
    "mlp.gate_proj.weight":               "ffn_gate.weight",
    "mlp.up_proj.weight":                 "ffn_up.weight",
    "mlp.down_proj.weight":               "ffn_down.weight",
}
ROOT_MAP = {
    "fuse_weight":                     "dspark_fuse.weight",
    "norm.weight":                     "output_norm.weight",
    "mask_embed":                      "mask_embd.weight",
    "markov_head.dflare_norm.weight":  "dspark_norm.weight",
    "markov_head.markov_w1.weight":    "markov_w1.weight",
    "markov_head.hidden_out.weight":   "dspark_hidden_out.weight",
    "markov_head.hidden_out.bias":     "dspark_hidden_out.bias",
    "markov_head.rnn_proj.weight":     "joint_proj.weight",
    "markov_head.rnn_proj.bias":       "joint_proj.bias",
}

TYPE_NAME = {"0": "F32", "8": "Q8_0", "30": "BF16"}


def load_st():
    """All safetensors tensors as float32, in HF (out, in) order."""
    import json, struct
    fsize = os.path.getsize(SNAP)
    with open(SNAP, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        whole = np.memmap(SNAP, dtype=np.uint8, mode="r", offset=base, shape=(fsize - base,))
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            assert v["dtype"] == "BF16", (k, v["dtype"])
            a, b = v["data_offsets"]
            raw = np.ascontiguousarray(whole[a:b])
            out[k] = raw.view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32).reshape(v["shape"])
    return out


def gg_hf_order(t):
    """GGUF tensor -> float32 in HF (out, in) order.

    GGML lays a logical (C, R) tensor out column-major, so a C-order view of the buffer
    with the dims reversed is the HF (R, C) array.
    """
    flat = np.ascontiguousarray(dequantize(t.data, t.tensor_type)).ravel()
    return flat.reshape([int(x) for x in t.shape][::-1])


def derive(hf_names):
    m = dict(ROOT_MAP)
    for n in hf_names:
        if n.startswith("layers."):
            _, i, rest = n.split(".", 2)
            m[n] = "blk.%s.%s" % (i, LAYER_MAP[rest])
    return m


def main():
    hf = load_st()
    rd = GGUFReader(REF)
    ref = {t.name: t for t in rd.tensors}
    mapping = derive(list(hf))

    unmapped = sorted(set(hf) - set(mapping))
    orphan = sorted(set(ref) - set(mapping.values()))
    print("HF tensors: %d   reference tensors: %d   mapped: %d" % (len(hf), len(ref), len(mapping)))
    print("unmapped HF :", unmapped or "none")
    print("orphan  REF :", orphan or "none")

    print("\n%-42s %-27s %-6s %-16s %-10s %s" % ("HF", "GGUF", "type", "hf shape", "exact", "max rel err"))
    n_exact = n_q8 = 0
    for hk in sorted(mapping):
        gk = mapping[hk]
        a = hf[hk]
        b = gg_hf_order(ref[gk])
        ty = TYPE_NAME.get(str(ref[gk].tensor_type), str(ref[gk].tensor_type))
        if a.shape != b.shape:
            print("%-42s %-27s %-6s SHAPE %s vs %s" % (hk, gk, ty, a.shape, b.shape))
            continue
        exact = bool(np.array_equal(a, b))
        denom = np.maximum(np.abs(a), 1e-6)
        rel = float((np.abs(a - b) / denom).max())
        n_exact += exact
        n_q8 += (not exact) and ty == "Q8_0"
        print("%-42s %-27s %-6s %-16s %-10s %.3e" % (hk, gk, ty, str(list(a.shape)), exact, rel))

    print("\nexact: %d/%d ; lossy (all Q8_0): %d" % (n_exact, len(mapping), n_q8))

    print("\n=== dtype rule by tensor class ===")
    for gk in sorted(ref):
        ty = TYPE_NAME.get(str(ref[gk].tensor_type), str(ref[gk].tensor_type))
        if ty != "Q8_0":
            print("  %-28s %s" % (gk, ty))
    q8 = [gk for gk in ref if TYPE_NAME.get(str(ref[gk].tensor_type)) == "Q8_0"]
    print("  Q8_0 (%d): all 2-D per-layer weights: %s" % (len(q8), "blk.* only"
          if all(n.startswith("blk.") for n in q8) else sorted(q8)))

    print("\n=== fuse_weight (HF) ===")
    fw = hf["fuse_weight"]
    print(np.array2string(fw, precision=6))
    print("row sums:", np.round(fw.sum(axis=1), 6).tolist())
    print("col sums:", np.round(fw.sum(axis=0), 6).tolist())
    print("abs > 1 count:", int((np.abs(fw) > 1).sum()), "of", fw.size)


if __name__ == "__main__":
    main()
