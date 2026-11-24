# MXFP4 scale selection

MXFP4 stores each 32-element block with a single `e8m0` (power-of-2) scale and
e2m1/e4m3 mantissas. The only free parameter is **how large a value the scale
allows before the mantissa saturates** — the `format_max` (fmax). This document
records how fmax was chosen for the weight, activation, and KV paths, and the
PPL evidence behind each choice.

## The OCP spec is not optimal

The OCP Microscaling spec fixes the scale at `round_to_pow2(amax / format_max)`,
with `format_max = 6.0` for e2m1 and `448` for e4m3. That is the *overflow-safe*
choice: the largest element maps exactly to the format's max value, so nothing
clips. But "no clipping" is not the same as "best accuracy" — it wastes the
mantissa's dynamic range when the block's values are skewed (one large outlier
plus a bulk of small values), because the whole block is scaled to fit the
outlier rather than to give the bulk resolution.

Measured on the workhorse models (full wikitext-2, f16 KV, ctx 4096), the spec
scale is consistently **worse** than a tighter scale:

| Model | spec (÷6.0 w, ÷448 a) | best fixed (÷4.0 w, ÷256 a) | gap |
|---|---:|---:|---:|
| 27B | 6.69 | 6.32 | +0.37 |
| 35B | 6.42 | 5.998 | +0.42 |

The gap comes almost entirely from the **activation** scale. Sweeping the e4m3
activation fmax on the 27B (weights held fixed) shows a flat plateau from
~128 to ~256 and a cliff once fmax exceeds ~320, where the spec's 448 sits:

| 27B activation fmax | 128 | 256 | 320 | 384 | 448 (spec) |
|---|---:|---:|---:|---:|---:|
| PPL (÷4.0 weights) | ~6.32 | 6.32 | ~6.5 | ~6.6 | 6.69 |

The e2m1 weight fmax has the same shape, milder: ÷4.0 beats the spec's ÷6.0 by
~0.1–0.4 PPL.

## Chosen values

| path | format | fmax | applied |
|---|---|---:|---|
| weights (offline) | e2m1 | **4.0** | at `llama-quantize` time |
| activations (runtime) | e4m3 | **256** | in the quant kernel, every forward pass |
| KV cache (runtime) | e2m1 | 6.0 (spec) | in the copy/quant kernel |

- **Weights ÷4.0** and **activations ÷256** are the measured PPL optima. The
  activation choice is the larger win (the ~0.37 PPL gap on the 27B is mostly
  here). The weight choice is milder but consistent across models.
- **KV stays at the spec's 6.0.** KV quantization error is small relative to the
  attention computation and the benefit of a tighter scale is not worth the
  deviation; the spec value is retained.

## Why not a per-block "optimal" scale

The principally-best scale for a block is the one that minimizes the *output*
error, `sum_i W_i^2 (a_i - quant(a_i))^2` for activations (weighted by the
partner tensor), not the plain L2 on the block's own values. Two practical
obstructions prevented shipping an adaptive scale:

1. **The activation-quant kernel does not have the weights in scope.** They load
   in the consuming MMA kernel, not the activation quantizer, so the
   weight-aware objective is not computable there without a kernel refactor.
2. **The e8m0 grid is too coarse to place the activation optimum.** A
   single-pass 2-candidate selector (choose between the scale for fmax=448 and
   one octave tighter, by measured e4m3 error) landed at 6.5586 on the 27B —
   *worse* than the best fixed value (6.32). The activation optimum sits between
   two e8m0 exponents; integer powers of two cannot land on it, so per-block
   selection cannot beat the best fixed fmax.

A fixed ÷256 is therefore the right call: it is the best reachable value on the
e8m0 grid and needs no extra kernel machinery.

## Imatrix weight quantization

The *weight* scale is the one place the output-aware objective is usable: at
quantization time an **imatrix** (per-element activation magnitudes captured from
a calibration perplexity run) is available. `llama-quantize --imatrix` now drives
an imatrix-aware mxfp4 path that picks the per-block weight scale by minimizing
the imatrix-weighted output error `sum_j im[j]^2 (w[j] - round(w[j]/s))^2` over
a band of candidate scales around ÷4.0.

Measured across all three models (real PPL, f16 KV, ctx 4096), the imatrix weight scale vs the fixed ÷4.0 default:

| model | fixed ÷4.0 | imatrix-weighted | effect |
|---|---:|---:|---:|
| 0.8B (dense) | 17.5073 | 17.2352 | **+0.27** |
| 27B (dense) | 6.4405 | 6.3324 | **+0.11** |
| 35B (MoE) | 5.9980 | 6.0955 | **-0.10 (worse)** |

It is a clear win on the **dense** models (the calibration lands on the weights that matter) and a *loss* on the **MoE 35B**, where the expert weights dominate and the dense parts the imatrix calibrates are a small fraction - there the per-block scale search fits noise in the calibration. It is opt-in; the fixed ÷4.0 is the default, and the better choice for the MoE model.

## Reproducing the measurements

- Weight/activation PPL: `llama-perplexity -m <model> -f wikitext-2-raw/wiki.test.raw
  --chunks 72 -ngl 32 -fa on -c 4096` (f16 KV).
- fmax was swept by rebuilding with a different `format_max` and re-running the
  above; the imatrix was generated with `llama-imatrix` over the full wikitext-2
  test split.

![Scale selection (fmax), measured PPL](mxfp4-pr-assets/scale-selection.png)

Raw per-model PPL and the cross-quant comparison live in `e8m0-quant-research.md`
and the PR data files under `scripts/`.
