# MXFP4 imatrix weight-quantization: accuracy audit + MoE regression hypotheses

Context: we re-introduced imatrix-based weight quantization for MXFP4. Dense models
looked good, but the **MoE model got *worse* with imatrix** vs without. This doc
records what the code actually does, the concrete defects found, ranked hypotheses
for the MoE regression, and a focused test plan to run on beast.

The goal is the *downstream* accuracy: after the weight is quantized to mxfp4 (E2M1),
real mxfp8 (e4m3) activations multiply it through attention. We want the attention
output as close to the f16 reference as possible.

---

## 1. What the imatrix actually is

The imatrix tool (`tools/imatrix/imatrix.cpp`) accumulates, per channel, `acc[j] += x[j]*x[j]`
(line 333 dense, line 333 MoE) over calibration activations, and the loader divides by
per-channel counts (`tools/quantize/quantize.cpp:214`). So the loaded per-channel value is

    im[j] = E[a^2]   (mean of squared activations, a SECOND moment, NOT an RMS)

- Dense tensor: one count, `e[j] = sums[j]/count`.
- MoE tensor (`MUL_MAT_ID`): stored as `src1->ne[0] * n_as` = `n_per_row * n_experts`,
  per-expert slice at `e.values.data() + ex * ne0` (line 325). An expert never routed to
  during calibration gets `count==0` -> the slice is filled with **1** (placeholder,
  `quantize.cpp:216`).

## 2. How the quantizer consumes it (the defect)

`ggml/src/ggml-quants.c`:
- `mxfp4_block_err2(x, im, qk, e)` (line 384): `w = im[jj]` (line 392), then
  `s += w * w * df * df` (line 394). Since `im[jj]` is **already E[a^2]**, this weights by
  **E[a^2]^2 = a fourth moment**. The downstream attention-output MSE wants weight **E[a^2]**
  (a second moment), i.e. `s += w * df * df`.
- `quantize_row_mxfp4_ref_imatrix` (line 397): per 32-value block, computes `e_base` from the
  (RNE) log2 rule, then searches `e_base - 2 .. e_base + 3` minimizing that imatrix-weighted
  `mxfp4_block_err2`. `quantize_mxfp4` (line 2349) dispatches to it when `quant_weights` is
  non-null, else the plain path.

So the **weight-reconstruction objective is squared one power too high** for the actual
imatrix units. This is a real accuracy bug in the objective, on *every* tensor, dense and MoE.

---

## 3. Ranked hypotheses for the MoE regression

The dense model survived because dense imatrix is dense (every channel exercised). MoE
amplifies the defects below. Ranked by likelihood:

### H1 (most likely): the `count==0` -> 1 placeholder, weighted by E[a^2]^2
For MoE, experts not visited by the calibration set get their whole imatrix slice set to
1. With the **correct** second-moment objective, an all-1 slice is a *neutral* weight
(equal weighting, equivalent to no imatrix) — harmless. With the **fourth-moment** bug,
1 is a *small* weight (real channels have E[a^2] that can be >> 1, so their square is even
bigger). The optimizer then *down-weights* the real channels relative to the placeholder,
i.e. it optimizes the *wrong* channels. More unvisited experts => more of the block's
weight is misplaced => a systematic, direction-agnostic quality loss. This is the cleanest
explanation for "imatrix made MoE *worse*": the placeholder interacts with the squared
objective to actively mis-weight.

Fix: (a) use the second moment (`s += w * df * df`), and (b) treat `count==0` experts as
*no-imatrix* (skip the imatrix path for those slices / pass `im=NULL` for them) rather
than the constant 1. The `count` array is available in the loader; the current code discards
it (`load_imatrix` only keeps the normalized `sums/count` values).

### H2: the placeholder=1 is wrong even before the squared bug
Independent of H1, an all-1 slice is *not* a good prior. If the true E[a^2] for an unvisited
expert is, say, 0.01 or 100, weighting its 32-value blocks by 1 (instead of ~the true second
moment) biases the scale search toward a generic, not expert-typical, scale. For MoE the
unvisited fraction is large (top-k routing over many experts on a finite calibration set).
Same fix as H1(b).

### H3: the search window is too narrow / biased
`e_base - 2 .. e_base + 3`. The E2M1 max is 6.0; the "optimal" scale usually lands *just
below* `lrintf(log2(amax) - 2)` so amax maps near (not past) 6.0. The window explores 2
below and 3 above — asymmetric the *wrong* way for accuracy, and only 6 candidates. Widening
(e.g. `e_base - 5 .. e_base + 2`, or a symmetric +/-4) and letting the *correct* objective
pick is cheap and likely to help. This is a second-order effect vs H1/H2 but trivial to test.

### H4 (rule out): the per-expert imatrix slice is misaligned
Checked: the imatrix file layout is `[n_experts][n_per_row]` (imatrix tool line 325) and the
quantizer slices `imatrix + (row_global / nrows_per_expert) * n_per_row`
(`llama-quant.cpp:760`), with `nrows_per_expert = ne[1]`, `n_per_row = ne[0]`,
`nrows_total = ne[1]*ne[2]` (line 1282). The layouts **match** (expert order, per-expert
slice of `n_per_row`). So the slicing is *consistent* — not a misalignment bug. Listed here
only to record that it was checked and is OK. (One caveat: if a MoE tensor's `ne[2]` expert
count differs between the imatrix run and the quantize run, `imatrix_for_row` would read
out of bounds — worth a bounds assert, see test plan.)

### H5 (rule out for scale choice): e4m3 activation quantization error
The real pipeline quantizes activations to e4m3. That activation error is *independent of the
weight scale e*, so it is a constant floor added to every e candidate — it does **not** change
which e wins. It does not bias the scale search. (The cross term E[a * act_err] is higher
order and also e-independent.) So it is not the cause, but it is a real accuracy floor the
imatrix can't remove.

---

## 4. The fix (proposed)

In `ggml-quants.c`:

1. **Second moment, not fourth** — in `mxfp4_block_err2`, change `s += w*w*df*df` to
   `s += w*df*df` (im is already E[a^2]).
2. **Unvisited experts = no imatrix** — thread the per-channel `count` through
   (`llama-quant.cpp` / `quantize.cpp` currently drops it) and pass `im=NULL` (or a flag)
   for a block whose expert has `count==0`, so the plain RNE rule is used instead of the
   (wrong) 1-weighted search. Minimal version: in the quantize tool, if a MoE expert's
   count is 0, zero its imatrix slice and have the ref treat a zero-weight block as "no
   imatrix" (fall back to `e_base` with no search).
3. **Widen the search window** to `e_base - 5 .. e_base + 2` (or symmetric +/-4).

Items 1 and 2 are the real fixes; 3 is a cheap bonus. All three are small and local to the
ref + the quantize tool's imatrix load (keep the count array).

---

## 5. Test plan (run on beast)

Run this **on beast** (this doc lives at `/home/tim/code/services/repos/llama.cpp/`,
checked out on the box you're running it on). Work from that repo; the build is in
`build/bin/`. Use `llama-quantize` + `llama-perplexity` (full wikitext-2 test, f16 KV).
## Where to put files (read this first - we just lost ~134G of scattered quants to cleanup)

All test artifacts go in ONE place, under the repo, so they are versioned alongside this doc and never get swept into a `~/.lmstudio` or home-dir cleanup:

    /home/tim/code/services/repos/llama.cpp/logs/mxfp4-imatrix/

Layout (create once, reuse for every run):
    logs/mxfp4-imatrix/
      imatrix/     # *.imatrix files (per model: 27b, 35b) - KEEP, these are cheap-ish to regen but don't waste it
      quants/      # the .gguf outputs for each arm (plain/bug/fix/... ) - safe to delete & remake
      results/     # PPL/KLD .log + .csv outputs - NEVER DELETE (the actual data)
      base/        # the BF16 base .kld logits (cheap to keep, needed for KLD) - KEEP

Rules:
- **Logs/results are permanent.** Never `rm` anything under `results/`. If a run is bad, move it to
  `results/stale/` - do not delete. (Same rule as the rest of the mxfp4 work.)
- **Quants are disposable** - put them in `quants/`, name them `<model>-<arm>.gguf`
  (e.g. `35b-imx-fix.gguf`, `27b-imx-bug.gguf`). Delete freely when remaking; that is the
  whole point of the `quants/` subdir. Do NOT scatter quants into `~/.lmstudio/models`,
  `quant-data/`, `mxfp4-test/`, or the repo root - those are exactly where we lost 134G.
- **One arm = one quant file + one result log**, both named by the arm, so a run is
  reproducible from the names alone.
- When quantizing, always redirect the tool's stdout to `results/<model>-<arm>.log`
  (e.g. `llama-quantize ... > logs/mxfp4-imatrix/results/35b-imx-fix.log 2>&1`).
Models: ggml-org **Qwen3.6-35B-A3B** (MoE) and **Qwen3.8-27B** (dense) BF16.

Imatrix: generate with the *current* `llama-imatrix` over a representative calibration set
(the same set used before), for both models.

For each model, measure PPL (wikitext-2, full) of:

| arm | weights | notes |
|-----|---------|-------|
| base | BF16 | reference |
| MXFP4-plain | mxfp4, no imatrix | current `quantize_mxfp4` plain path |
| MXFP4-imx-bug | mxfp4, imatrix as-is (4th moment + 1-placeholder) | reproduces the regression |
| MXFP4-imx-fix | mxfp4, imatrix with fix (2nd moment, unvisited->no-imatrix, wider window) | the candidate |

Expected: for MoE, `MXFP4-imx-fix` PPL <= `MXFP4-plain` and clearly better than
`MXFP4-imx-bug`; for dense, `MXFP4-imx-fix` ~= or better than plain.

Sub-checks (isolate the hypotheses):
- **Isolate H1/H2 (placeholder):** MoE, but with the fix's "unvisited -> no imatrix" only
  (keep the 4th-moment objective). If this alone recovers the regression, the placeholder is
  the culprit. If not, the 4th-moment weighting is.
- **Isolate H3 (window):** MoE with the 2nd-moment fix but the *narrow* window vs the
  *wide* window. Delta = the window's contribution.
- **Bounds assert:** add `GGML_ASSERT(imatrix && imatrix_for_row < imatrix_end)` in
  `llama_tensor_quantize_impl` to catch any expert-count mismatch between imatrix and
  quantize runs (H4).

Also verify the *inference* side still agrees: the KV/activation mxfp8 path is unchanged by
this (weights only), so inference perf is expected identical; only weight quality moves.
