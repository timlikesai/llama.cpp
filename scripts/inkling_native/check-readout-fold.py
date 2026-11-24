#!/usr/bin/env python3
"""Does contracting hidden_out -> lm_head in two f32 steps differ meaningfully from the
reference's single fp64 fold (fold_weight = lm_head @ hidden_out.T)?

The graph never folds: latent(f32) -> hidden_out(bf16 w, f32 acc) -> f32 -> lm_head(bf16 w,
f32 acc) -> f32. The reference folds once in fp64 and applies that to the latent. Both use the
same bf16 weights, so the delta is purely the order/precision of the contraction - 256 -> 2304
-> vocab versus 256 -> vocab. Measured here on the real checkpoint tensors, with the decision
metric being whether it can move an argmax, since a drafter's error shows up as a rejected token.

Usage: check-readout-fold.py [--seed N]
"""
import argparse, pathlib
import numpy as np
import torch
from safetensors import safe_open

SNAP = pathlib.Path.home() / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337"
SOFTCAP = 30.0  # final_logit_softcapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()

    with safe_open(str(SNAP / "mtp.safetensors"), framework="pt") as f:
        H  = f.get_tensor("markov_head.hidden_out.weight")   # [n_embd, rank] bf16
        Hb = f.get_tensor("markov_head.hidden_out.bias")     # [n_embd] bf16
    with safe_open(str(SNAP / "model-00001-of-00001.safetensors"), framework="pt") as f:
        U = f.get_tensor("model.llm.unembed.weight")  # [n_vocab, n_embd] bf16

    rank, n_embd, n_vocab = H.shape[1], H.shape[0], U.shape[0]
    print(f"hidden_out {tuple(H.shape)} {H.dtype}   unembed {tuple(U.shape)} {U.dtype}   rank={rank}")
    assert n_embd == 2304 and rank == 256, (n_embd, rank)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    U, H, Hb = U.to(dev), H.to(dev), Hb.to(dev)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    # the reference takes the latent through tanh, so it is bounded in (-1, 1)
    lat = torch.tanh(torch.randn(a.n, rank, generator=g)).to(dev)

    # exact: one fp64 fold of the same bf16 weights, applied in fp64
    #   h = l @ H.T + b ; bias = h @ U.T  =  l @ (U @ H).T + b @ U.T   -> fold_weight = U @ H  [n_vocab, rank]
    W_fold64 = (U.double() @ H.double())                   # [n_vocab, rank]
    bias64 = lat.double() @ W_fold64.T + Hb.double() @ U.double().T
    # as the graph does it: two f32 contractions with bf16 weights, f32 activations throughout
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    h32 = lat.float() @ H.float().T
    if dev == "cuda":
        h32 = h32.to(torch.bfloat16).float()                # hidden_out weight is bf16 in the file
    h32 = h32 + Hb.float()
    bias32 = h32 @ U.float().T

    d = (bias32 - bias64).abs()
    ref = bias64.abs()
    print(f"\nbias magnitude : mean {ref.mean():.4f}  std {bias64.std():.4f}  max {ref.max():.4f}")
    print(f"abs error      : mean {d.mean():.3e}  max {d.max():.3e}  (logit units, before softcap)")
    print(f"rel error      : mean {(d / (ref + 1e-9)).mean():.3e}  p99 {np.quantile((d / (ref + 1e-9)).cpu().numpy(), 0.99):.3e}")
    print(f"error / std    : {d.max().item() / bias64.std().item():.3e}  (how big the error is vs the signal spread)")

    # the metric that matters for a drafter: can it move the argmax of the capped logits?
    base64 = (torch.randn(a.n, n_vocab, generator=g).to(dev) * 2.14).double()  # f32 RNG (fp64 randn is CPU-only)
    def top1(x):
        return torch.softmax((SOFTCAP * torch.tanh(x.float() / SOFTCAP)).float(), dim=-1).argmax(dim=-1)
    t_ref, t_g = top1(base64 + bias64), top1(base64 + bias32)
    flips = (t_ref != t_g).sum().item()
    print(f"\nargmax flips over {a.n} rows: {flips} ({100.0 * flips / a.n:.2f}%)")

    # the alternative we declined: fold at load and store the table like the other weights (bf16).
    # If that is worse than contracting in two steps, not folding is the more accurate choice, not just the simpler one.
    W_foldbf16 = W_fold64.to(torch.bfloat16).double()
    bias_foldbf16 = lat.double() @ W_foldbf16.T + Hb.double() @ U.double().T
    df = (bias_foldbf16 - bias64).abs()
    print(f"\nfor comparison, fold-then-round-to-bf16: mean {df.mean():.3e}  max {df.max():.3e}"
          f"  ({df.mean().item() / d.mean().item():.1f}x the two-step error)")
    tf = top1(base64 + bias_foldbf16)
    print(f"argmax flips for the bf16 fold        : {(tf != t_ref).sum().item()} ({100.0 * (tf != t_ref).sum().item() / a.n:.2f}%)")

    # margin of the runner-up, on a subsample (a full fp64 sort of 201024 columns OOMs)
    sub = min(64, a.n)
    s = torch.sort((base64 + bias64)[:sub].float(), dim=-1, descending=True).values
    margin = (s[:, 0] - s[:, 1])
    print(f"top-2 margin (64 rows) : mean {margin.mean():.4f}  min {margin.min():.4f}"
          f"  frac below two-step max err: {(margin < d.max()).float().mean():.4f}")
    print("\nVERDICT:", "two-step readout is indistinguishable at argmax" if flips == 0 else
          f"{flips} flips - needs a closer look")


if __name__ == "__main__":
    main()
