#!/usr/bin/env python3
"""Compare candidate quantizations of Inkling's two vocabulary tensors.

token_embd.weight is consumed by GET_ROWS, output.weight by a 1-column MUL_MAT at decode.
Both are BF16 in the checkpoint, so every option here is a deviation from "unadulterated" -
the question is which deviation costs least. This measures error and its effect on logits;
it does not touch the GPU.

Usage: vocab_quant_compare.py [--gguf PATH] [--samples 64]
"""
import argparse
import numpy as np
import torch
from gguf import GGUFReader  # repo gguf-py: GGML_QUANT_SIZES is (elements, bytes)

E4M3_MAX = 448.0            # max finite normal of float8_e4m3fn
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def q_q8_0(w, bs=32):
    """int8 + one f16 absolute-max scale per block (8.5 bpw)."""
    v = w.view(-1, bs).float()
    amax = v.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = amax / 127.0
    q = torch.round(v / scale).clamp(-127, 127)
    return (q * scale).view(w.shape).to(torch.bfloat16), 2 + bs          # bytes per block


def q_mxfp8(w, bs=32):
    """e4m3 payload + ue8m0 (power-of-two) scale per block (8.25 bpw)."""
    v = w.view(-1, bs).float()
    amax = v.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / E4M3_MAX))                     # exponent only, no mantissa
    scale = torch.pow(2.0, e)
    q = (v / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).view(w.shape).to(torch.bfloat16), 1 + bs          # 1 shared exponent byte


def q_nvfp4(w, bs=16):
    """e2m1 payload + e4m3 micro-scale per block (4.5 bpw)."""
    v = w.view(-1, bs).float()
    amax = v.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = amax / 6.0
    sq = scale.to(torch.float8_e4m3fn).float()                      # the scale itself is fp8
    x = (v / sq).abs().unsqueeze(-1)
    idx = (x - E2M1_LEVELS).abs().argmin(dim=-1)
    q = E2M1_LEVELS[idx] * v.sign()
    return (q * sq).view(w.shape).to(torch.bfloat16), 1 + bs // 2        # 1 scale byte per 2 elements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/tim/models/timlikesai/Inkling-2-Mini-FULL.gguf")
    ap.add_argument("--samples", type=int, default=64)
    a = ap.parse_args()

    r = GGUFReader(a.gguf)
    # this gguf-py exposes .tensors as a list, not a name-keyed dict
    by_name = {t.name: t for t in r.tensors}
    names = ["token_embd.weight", "output.weight"]
    src = {}
    for n in names:
        t = by_name[n]
        src[n] = torch.from_numpy(t.data.copy()).view(torch.bfloat16)
        print("%-18s %s bf16 = %.0f MiB" % (n, tuple(t.shape), src[n].numel() * 2 / 2**20))

    # Plausible decode-time inputs for the head: embedding rows live in the same space.
    emb = src["token_embd.weight"].float()
    idx = torch.randint(0, emb.shape[1], (a.samples,))
    hs = emb[:, idx].T.contiguous()                                  # [samples, n_embd]

    print("\n%-18s %-8s %9s %9s %9s %9s %9s" % (
        "tensor", "type", "GiB", "relRMSE", "maxAbs", "logitRMS", "top1flip"))
    for n in names:
        w = src[n]
        ref_logits = (hs.to(torch.bfloat16).float() @ w.float()).double()
        top2 = torch.topk(ref_logits, 2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1])
        for label, fn in (("q8_0", q_q8_0), ("mxfp8", q_mxfp8), ("nvfp4", q_nvfp4)):
            dq, bpw = fn(w)
            bytes_per = bpw / (32 if label != "nvfp4" else 16)
            size_gib = w.numel() * bytes_per / 2**30
            err = (dq.float() - w.float())
            rel = float(err.norm() / w.float().norm())
            new_logits = (hs.float() @ dq.float()).double()
            lerr = float((new_logits - ref_logits).pow(2).mean().sqrt())
            flips = int((new_logits.argmax(1) != ref_logits.argmax(1)).sum())
            # how close is the error to the tightest decision boundary we sampled?
            print("%-18s %-8s %9.3f %9.4f %9.4f %9.4f %5d/%d  minmargin=%.3f" % (
                n, label, size_gib, rel, float(err.abs().max()), lerr, flips, a.samples,
                float(margin.min())))
            del dq, err, new_logits
    print("\nNote: bf16 baseline for both tensors is %.3f GiB total." % (
        sum(v.numel() for v in src.values()) * 2 / 2**30))


if __name__ == "__main__":
    main()
