# 35 — Drafter step 2: DFlare per-stage tap mixing (committed `55bf894f8`)

Implements §4 row 3 of `33-drafter-reference-semantics.md` for `VARIANT_INKLING`. Step 1 (`6e89360b6`) made the published draft file load; this makes its **conditioning path** correct. The `rnn_hidden` markov head (#39) is still unimplemented, so the variant still throws at graph build — narrowed to name only that gap (`src/models/dflash.cpp:619-622`).

Written by the parent after the implementing lane was interrupted; the lane had already committed, so this doc records what is actually in the tree, verified by re-running the tests.

## Design

DFlash fuses the tapped target features **once** for the whole draft: `fc {n_taps*n_embd, n_embd}` then a shared `enc.output_norm`, and every stage sees the same vector. DFlare's point (arXiv 2606.02091) is that this is a *narrow conditioning bottleneck*: each draft stage gets its own weighted mixture of the taps. Inkling ships `dspark_fuse {4,4}` plus a per-stage `blk.N.dflare_norm` and **no `fc` at all**, so the shared path cannot express it.

Three small builders on `llama_model_dflash` (`src/models/models.h:1688-1694`):

| builder | what it does |
|---|---|
| `build_dflare_alpha(ctx, fuse)` | `ggml_soft_max(fuse)` — **one softmax for the whole matrix**. `soft_max` reduces `ne[0]`, which is the tap axis of the stored `{n_taps, n_mix}` layout, so this equals the reference's `softmax(fuse_weight.float(), dim=-1)` exactly. Asserts F32, because the reference derives the weights in fp32. |
| `build_dflare_taps(ctx, feats, n_embd)` | splits the concatenated injection features into `{n_taps, n_embd, n_tokens}` via `reshape` + `cont(permute)`, **once per injection batch** — every mix reads the same copy. |
| `build_dflare_mix(ctx, taps, alpha, i_mix, norm_w, eps)` | `sum_taps(alpha[i_mix][t] * taps[t])` then `rms_norm` with that stage's own `dflare_norm` weight. |

Two non-obvious details worth keeping:

- **The taps are materialized contiguous** because the CPU binary-op kernel and `ggml_sum_rows` both require a contiguous `ne[0]`. Avoiding the copy looked attractive and is not available to this graph shape.
- `build_inject_inp` lost its fusion half to `build_inp_features`; `build_inject()` does the per-stage mixing **only when `dspark_fuse` is present**, so drafts with an `fc` get the previous nodes in the previous order. That is what makes the z-lab DFlash2 regression meaningful rather than hopeful.

## Numeric oracle and its mutation proof

A wrong `alpha` is **invisible in generated text** — the target verifies every token, so a mis-mixed drafter still emits correct output and merely drafts badly. So correctness here is carried by numbers, not by behavior:

`tests/test-dflare-mix.cpp` runs all three builders against a numpy-`float64` oracle transcribed from the reference (`tests/test_dflare_mix.py`), and a **second ctest re-derives the pasted constants** so they cannot rot silently:

```
tiny     taps= 4 embd=   6 tokens= 2: max err 1.19e-07 OK
aligned  taps= 4 embd=  64 tokens= 3: max err 1.19e-07 OK
inkling  taps= 4 embd=2304 tokens= 5: max err 4.77e-07 OK
one hot                                           : max err 1.19e-07 OK
```

`scripts/inkling_native/check-dflare-oracle.sh` then proves the oracle is load-bearing by breaking the code twice and restoring from `docs/mxfp8/drafter-dflare-mix.patch`:

| mutation | outcome |
|---|---|
| **M1** swap the tap and hidden axes in the split | `GGML_ASSERT(alpha->ne[0] == n_taps …)` at `dflash.cpp:285`, exit 134 — caught structurally |
| **M2** read the wrong `dspark_fuse` column per stage (changes **no shape**, purely numeric) | **all cases FAIL**, max err 0.565 / 0.432 / 0.97 / 2.74 — caught numerically |
| restored | all four pass, `src` verified byte-identical to the patch |

M2 is the whole point: the failure mode that no amount of end-to-end text testing would reveal is caught by the test.

## Regression and gates (parent re-ran the tests; lane measured the rest)

- **z-lab DFlash2 + Qwen3.8-27B, same command before and after**: same **207 drafts**, same **97 accepted**, same greedy text (`0.46860`, mean length `4.23`), **70.9 t/s** decode vs **23.1 t/s** plain. Acceptance — not just "it runs" — is unchanged.
- `ctest -R dflare`: 2/2 pass; `test-dflare-mix` re-run by parent on the committed tree: 4/4 OK, exit 0.
- `test-llama-archs` 591/591; `test-backend-ops -p mxfp8` 306/306 CPU, 237/237 CUDA0.
- Inkling lossless greedy probe: unchanged **on CPU**, and unchanged **with `-fa off` on CUDA**. Note from the lane, worth remembering: **flash attention on changes that string — identically with and without this commit.** So FA numerics move greedy output on this model; it is a property of the backend, not drift from this work. Recorded in `00-status.md`'s baseline table context.

## What remains before the drafter runs

1. **#39 `rnn_hidden` head** — the last unimplemented piece, fully specified in §3 of doc 33: `joint_proj` split into `[state ‖ prev]` and `[hidden ‖ f_rnn]` halves, the GRU-style gate with its explicit bf16 roundings, `hidden_out` folded into the target LM head (or applied directly), the `dflare_norm` row used for `f_rnn`, and per-step state carried across the block with a reset per block. Then delete the throw.
2. **#40** — acceptance-rate measurement on Inkling. Expectation set by the 27B: ~3x decode, but **measured, not assumed**.
3. Converter note: `scripts/inkling_native/convert_mtp_draft.py` emits `VARIANT_DFLASH`-shaped output and still fails on `fc` (#25). Once #39 lands it should be re-pointed at the Inkling contract so the lossless path can produce its own draft from `mtp.safetensors` rather than depending on a third-party file.
