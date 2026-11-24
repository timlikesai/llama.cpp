# 43 — Branch audit: what we own, what is essential, what is residue

**Branch:** `mxfp8-weights` · **32 commits** on top of `upstream/master` · 85 files, +2,979/−341
**Lens (per Tim, 2026-11-23):** *does it help us run this model fast and correctly* — not
PR-readiness. Upstream PRs are optional side-benefits, so "unused but tested" is not a reason to churn.
Data: `logs/branch-commit-stats.txt` (per-commit shortstat, generated this pass).

## Verdict in one paragraph

Everything essential is live. Three commits are superseded-in-place (their code was replaced by the
`#32808` merge) and one adds an API nothing calls anymore — none of them cost us anything at runtime,
so they stay. There is **no debug scaffolding left in tracked files**: the temporary allocator
instrumentation and the ablation env-gates were all reverted, and the one `getenv` in the diff
(`LLAMA_DFLASH_DEV`) is upstream's (`2cbca389f`), not mine. The real residue is at the repo root and
is untracked: stale markdown/figures from the old mxfp4 sessions plus a build scratch dir.

## 1. Essential to running the model at all (14 commits)

| commit | what it does | why it is load-bearing |
|---|---|---|
| `4f3cbd64e` `1a4fb60c3` `03165022a` `f5eb057d9` | MXFP8 type, CPU op routing, file type, round-trip tests | nothing loads type 44 without these (290 insertions) |
| `c9cf0d32b` `d25064c66` | converter keeps NVFP4 **and** MXFP8 experts as native bytes | this *is* the losslessness claim: 40+8 tensors byte-identical to the checkpoint |
| `8193bf1b0` | MXFP8 through the CUDA vector path | correctness fallback for arches without the block-scaled MMA |
| `15649f8e8` `faac4b7d2` `20cb517b0` | MMQ tile path (`mma…mxf8f6f4…e4m3.e4m3`) + fp8-activation tolerances + corrected `QI_MXFP8` | +4.6% prefill; and `QI_MXFP8=2→8` removes a silent ¾-row-drop landmine |
| `1ab82fa7f` | place aliased layers on their physical layer's device | **+78% prefill / +82% decode** — the single largest win on the branch, 24 lines |
| `3a6bafb02` `f0066094f` `d92cf048d` | embed the drafter in the main file (nextn) + per-stage windowing | makes `Inkling-2-Mini-FULL.gguf` the complete model (798 tensors); windowing returns 1 GiB |
| `84d9bc0dc` | adopt upstream `#32808` indexed-bias flash attention | **+28.3% prefill** on the lossless file; retires my fold chain |

## 2. Correctness bug fixes we own (4 commits, all general, none Inkling-only)

| commit | bug | blast radius it fixes |
|---|---|---|
| `53cbb9a1e` | two unguarded interpolations in the device fitter → **SIGFPE** when a probe's memory span is zero | every model; latent upstream, my memory win made it reachable |
| `78c09b17c` | draft logits spanned the **padded** vocabulary → a draft could propose a nonexistent token (`failed to add token 200426`) | every dflash-family draft against a padded target |
| `90c9b6397`* | `can_reuse` compared state *width* but not *presence* → assert/crash when one graph serves both pp and tg ubatches | all ten models using `llama_graph_input_embd_state` |
| `0d1fc1e41` | the Inkling target never published the layers the drafter reads (`layer input tensor is null`) | required for any tap-consuming draft on this arch |

\* landed inside `996836c34`. These four are the best small-PR candidates precisely because they are
not about Inkling — each is a latent upstream defect we hit by accident. Note `78c09b17c` and
`90c9b6397` were both found *because* we adopted someone else's feature and it did not work on this
arch — adopting upstream code is also a bug-finding strategy.

## 3. Performance work that is live and measured (4 commits)

`4f4fd2957` f16 depth mix for batched eval (**+8.7% prefill**, PPL 5.4354 vs 5.4373, decode untouched,
`GGML_PREC_F32` pinned after I caught my own CPU/CUDA divergence); `cb7f67131` carry the sink logit
inside the bias projection (**+4.7%**, PPL bit-identical); `1ab82fa7f` and `84d9bc0dc` above. Also
`20cb517b0` (hygiene with no runtime effect today, since nothing reads those constants yet).

## 4. The drafter (7 commits, ~600 lines — the whole speculative-decoding path)

`5842bbac2` load · `12c23d61d` DFlare per-stage mix + oracle test + mutation proof · `15f9b9593`
`hidden_out` readout through the shared LM head · `996836c34` tap-as-batch-state channel ·
`a93f2339a` run the `rnn_hidden` head when drafting · `9f979fab6` **index taps by layer visit, not cache
slot** (2.8× acceptance — the insight from `#32218`, not mine) · `690fb1d5f` softcap after the markov
bias.

Status: **working and net-positive** — server-verified 47.4 t/s and acceptance 0.536 with the ngram
chain, 0.673 on code at CLI. Two of these (`9f979fab6`, `690fb1d5f`) implement upstream's findings, so
they are *adoption*, not novelty — worth knowing before anyone claims them in a PR.

## 5. Superseded in place (3 commits) — keep, learn from, do not remove

| commit | replaced by | evidence it is dead |
|---|---|---|
| `28c8acdd5` fold the causal mask into the bias table | `84d9bc0dc` | `fold_rel_mask` / `fold_future` / `fold_mask` / `LLAMA_REL_POS_FOLD` → **0 hits** in `src/`, `ggml/` |
| `8def03281` fold the window mask likewise | same | same |
| `e1f2296fa` gather the bias in the mask type | same (the gather no longer exists) | `ggml_get_rows_same_type` has **0 production callers** |

These were not wasted: the fold got the ctx slope from 0.1698 → 0.0728 MiB/token and *proved* the
mechanism, which is what made the upstream indexed-bias merge legible enough to adopt the same night.
Doc `40-indexed-bias-supersedes-fold.md` records the retirement. Removing them now would only re-touch
`inkling2.cpp` and `llama-graph.cpp` for zero speed — explicitly **not** work worth doing.

`dddc5d2d0` (`ggml_get_rows_same_type`, 97 insertions across 9 files incl. 5 backend `supports_op`
guards and 5 new op-coverage cases) is the one genuinely unused API. Kept: it is tested in both
directions, the guards are one-directional (they can only decline, which falls back to CPU), and we
only build CUDA + CPU so the compile-unverified guards are inert here. **The only reason to delete it
would be PR tidiness, which is not the goal.** If a future change needs a narrow gather again, it
already exists.

## 6. Real residue to clean (untracked, cheap)

**Done 2026-11-23:** the stale files below were moved to `docs/mxfp8/archive/mxfp4-era/` — renamed,
nothing deleted, since some contain measurements from branches that no longer build.

Repo root had stale files from the earlier mxfp4/mxfp6 sessions that mislead a fresh reader (and
`grep` hits): `e8m0.md`, `e8m0-quant-research.md`, `mxfp4-imatrix-accuracy.md`,
`mxfp4-pr-diet-plan.md`, `mxfp4-scale-ab-plan.md`, `mxfp4-scale.md`, `mxfp8-scale.md`,
`ppl-vs-size.png`, `scale-selection.png`, `logs-scout-remotebranches.txt`.

`ggml/gen-scratch/` **is not residue** — I checked before moving it: it is 564 KB of generated CUDA
(`fattn-mma-f16-instance-ncols*.cu`), i.e. build output, and it belongs exactly where it is.
Tracked-file residue: none found.

## 7. What is *not* on the branch (deliberate, and easy to re-add by mistake)

- The rejected `mmvq` MXFP8 decode route (measured 21.90–21.98 vs 22.24 → reverted; lesson in
  `docs/mxfp8/`, do not re-try blind).
- Temporary allocator instrumentation in `ggml-alloc.c` and the ablation env-gates (`INK_MIX_F16`,
  `INK_NO_F32`-style switches) — reverted after use; the analysis scripts stayed as *untracked* tooling
  under `scripts/inkling_native/` (intentional: tooling that is not part of the build).
- Full-block (`n_block = 8`) drafting and `sliding_window = 1025` — both tested, both measured no-op
  or worse, neither shipped.
- The `dense kq_mask` allocation removal (#33, zero measured benefit).

## 8. If we had to rebuild this box from scratch

The order that produced the wins, with the payoff each time: **alias-aware placement** (`1ab82fa7f`,
+78/+82%) → **indexed-bias FA** (`84d9bc0dc`, +28% prefill, kills the ctx-proportional term) →
**f16 depth mix** (`4f4fd2957`, +8.7%) → **sink-in-contraction** (`cb7f67131`, +4.7%) → **MXFP8 MMQ**
(`15649f8e8`, +4.6%) → **fitter guards** (`53cbb9a1e`, stops the SIGFPE) → **drafter visit-indexing**
(`9f979fab6`, 2.8× acceptance) → **target q4_1 cache** (256K becomes reachable: `41-…` §5).
Net session arc on the complete lossless model: **507/22 → 1,027/52 t/s** (2.0× prefill, 2.4× decode)
at 256K with drafting.

That sequence is the strongest candidate to become a skill (`#17`): the knowledge is *ordering* plus
about a dozen exact flags, and every one of those flags was learned the expensive way.
