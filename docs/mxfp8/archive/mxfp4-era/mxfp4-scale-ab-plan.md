# MXFP4 scale-selection A/B test plan (UOS vs old floor(log2)-2)

## Goal

Show, with measured **PPL** data, that the **new UOS** scale-selection formula (this PR) beats the
**old `floor(log2(amax)) - 2`** formula, on the two dimensions that matter:
1. **Weight** quant (the `.gguf` `LLAMA_FTYPE_MOSTLY_MXFP4`, E2M1) with **f16 KV**.
2. **KV cache** (`-ctk/-ctv mxfp4`, E2M1).
In **both** comparisons the W4A8 / mxfp8 **activation** quant uses the **old** scale (pinned in
both arms), so the only thing that varies is the dimension under test. KLD for the new W4A8
path is already done and is out of scope here.

The expectation: **UOS PPL <= old-scale PPL** on both weights and KV. Perf is ~identical by
construction (the scale changes only the stored e8m0 byte), so it is not the point of this test.

---

## The two formulas

Both take the block `amax` and return an e8m0 exponent `e` (scale = 2^(e-127), stored as a
single byte per 32 values).

**New (UOS, this PR)** — distribution-free, qmax-aware. `ggml_mxfp_scale(amax, qmax)`:
```
int A, Q;  r_a = frexpf(amax, &A);  m = frexpf(qmax, &Q);
e = 127 + A - Q + (r_a > m) - (r_a <= 0.5*m);   // clamp [0,254]
```
`qmax` = 7.25 (E2M1) / 464 (E4M3). (ref arXiv 2607.24377, q* = 29/4 for E2M1)

**Old (pre-PR, the `floor(log2)-2` rule)** — qmax-agnostic power-of-2 (as in the existing
SYCL path, `ggml-sycl/cpy.hpp:323`):
```
e = floor(log2(amax)) - 2 + 127;   // clamp [0,254], 0 if amax==0
```
The old rule just lands the max value near the top of the grid via a raw power of two; it
does not account for the format's actual `qmax`, so it wastes headroom (the max maps to a
scale that can overshoot the grid), which is where the UOS quality gain comes from.

> Note: for the **activation** (E4M3 / mxfp8) path the pre-PR format used a *ue4m3* (4-bit
> exponent) scale, not e8m0. The "old" e8m0 comparison here is the same `floor(log2)-2` rule
> applied to e8m0 — i.e. "what if we had shipped mxfp8 with the naive scale instead of UOS".
> That is the fair A/B: same format, same e8m0 byte, different scale rule.

---

## Code change (the toggle)

All three sites funnel through **two** functions, so a single toggle controls them. We need the
toggle to act **independently per site** so the activation can be pinned to the old scale in both
arms while weight / KV are varied:

| Site | Function | File:line | qmax |
|------|----------|-----------|------|
| weight (CPU ref)      | `ggml_mxfp_scale`         | `ggml/src/ggml-impl.h:517`  | E2M1 |
| weight (CUDA cpy)     | `ggml_cuda_mxfp_scale`    | `ggml/src/ggml-cuda/common.cuh:454` | E2M1 |
| KV cache (CUDA cpy)   | `ggml_cuda_mxfp_scale` (via `quantize_f32_mxfp4_block`) | `ggml/src/ggml-cuda/cpy-utils.cuh:194` | E2M1 |
| activation (W4A8)     | `ggml_cuda_mxfp_scale` (via `quantize_mmq_mxfp8`) | `ggml/src/ggml-cuda/quantize.cu:382` | E4M3 |

**Mechanism:** a runtime env var read once at startup, e.g. `GGML_MXFP_SCALE=<weight>|<kv>|<act>
where each field is `uos` or `old` (default `uos`). The host `ggml_mxfp_scale` and the device
`ggml_cuda_mxfp_scale` each branch on the relevant field:
```c
const bool old = (scale_mode_for_this_site() == OLD);   // weight / kv / act
if (old)  e = amax>0 ? clamp(0,254, floorf(log2f(amax)) - 2 + 127) : 0;   // old floor(log2)-2
else     e = <existing UOS body>;                                          // qmax-aware
```
- **Weight A/B:** `GGML_MXFP_SCALE=weight:old,act:old`  (W-OLD)  vs  `weight:uos,act:old` (W-UOS).
  (quantize the .gguf with the matching mode; activation pinned old in both.)
- **KV A/B:**   same .gguf; run with `GGML_MXFP_SCALE=kv:old,act:old` (KV-OLD) vs
  `kv:uos,act:old` (KV-UOS). KV scale is chosen at runtime on the KV write path.
- Activation is `act:old` in **every** run (both arms of both A/B's).

> Default (no env) = all `uos` = the shipped build. The dequant / MMQ / FA paths are
> **unchanged** - only the scale byte written at quantize / KV-write time changes, so the
> candidates run on the exact same kernels and perf is ~identical by construction.

### Candidate artifacts

- `W-UOS.gguf` / `W-OLD.gguf`  - mxfp4 weights, UOS vs old scale (for the weight PPL A/B).
- The same mxfp4 `.gguf` reused for the KV A/B (KV scale is a runtime choice, not a file).
- No KLD, no activation A/B, no RMSE in this round - PPL only.
---

## What to measure

**Models:** Qwen3.5-0.8B (dense), Qwen3.8-27B (dense), Qwen3.6-35B-A3B (MoE) — same as the
rest of the PR. Hardware: 2x RTX 5070 Ti (local) and 2x RTX 5060 Ti (150W limit) (Beast). KLD/PPL are
hardware-independent, so quality can be taken from either box.

### The two PPL comparisons (W4A8 / mxfp8 activations on the OLD scale in both arms)

KLD is already done for the new W4A8 path - skip it here. The activation (W4A8/mxfp8) scale
is set to the **old** formula in *both* arms, so the only thing that varies between an arm's two
candidates is the dimension under test (weight scale, or KV scale).

**Common to every run:** `-sm tensor -fa on` (W4A8 activations, old scale), full wikitext-2,
ctx 4096. Models: Qwen3.5-0.8B, Qwen3.8-27B, Qwen3.6-35B-A3B. PPL is hardware-independent.

### 1. Weight A/B  -  mxfp4 weights, f16 KV
Two candidates, both f16 KV + old-scale mxfp8 activations:
- `W-UOS`: mxfp4 weights quantized with the **UOS** scale (default build).
- `W-OLD`: mxfp4 weights quantized with the **old** scale (`GGML_MXFP_SCALE=weight` build).
Compare **PPL** (full wikitext-2): expect W-UOS <= W-OLD.

### 2. KV cache A/B  -  mxfp4 weights, mxfp4 KV
Same mxfp4 `.gguf`; KV scale chosen at runtime on the KV write path. Two runs,
both old-scale mxfp8 activations:
- `KV-UOS`: `-ctk/-ctv mxfp4` under the **UOS** build.
- `KV-OLD`: `-ctk/-ctv mxfp4` under the **old-KV** build (`GGML_MXFP_SCALE=kv`).
Compare **PPL** (full wikitext-2, ctx 4096): expect KV-UOS <= KV-OLD.

> Perf (pp/tg) is optional here - the point is the PPL delta; the scale changes only the
> stored e8m0 byte, so perf is ~identical by construction.

---

## Data tables to produce (for the PR)

Two small tables, one per dimension. Columns: `model | PPL (UOS) | PPL (old scale) | delta`.
Activation (W4A8/mxfp8) is the old scale in every cell. Headline: **UOS PPL <= old-scale PPL**
on both weights and KV.

```
Weight (mxfp4 weights, f16 KV, old-scale mxfp8 act):
| model | PPL UOS | PPL old |
| 0.8B  | ... | ... |
KV cache (mxfp4 weights, mxfp4 KV, old-scale mxfp8 act):
| model | PPL UOS-KV | PPL old-KV |
```

Optional chart: 3 models x (UOS/old) grouped bars of PPL for weights and KV, "act: old" noted.
Keep it small - the tables carry the evidence.

---

## Execution notes

- **PPL only** this round: `llama-perplexity -f wikitext-2 -ngl 999 -sm tensor -fa on -c 4096`
  (full split), with `GGML_MXFP_SCALE=...` set per the table above. Hardware-independent, so one
  box suffices (Beast is fine; the full-KLD run there is separate and independent).
- **Weight A/B:** quantize BF16 -> `W-UOS.gguf` (default) and `W-OLD.gguf` (weight:old build);
  PPL each with f16 KV + old-scale activations.
- **KV A/B:** same mxfp4 `.gguf`; PPL with `-ctk/-ctv mxfp4` under `kv:uos` vs `kv:old` (both
  old-scale activations). No base file needed (PPL, not KLD).
- Keep the UOS default build as the shipped artifact; the old-scale configs are test-only.

## Out of scope

- No new ftypes, no layout/kernel changes - purely a scale-rule A/B on PPL.
- No KLD (already done for the new W4A8 path), no activation A/B (pinned old in both arms),
  no RMSE this round. Do not touch the W4A4 (nvfp4) path.
