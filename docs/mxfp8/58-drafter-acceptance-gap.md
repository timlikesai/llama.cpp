# 58. Where the drafter's acceptance gap actually is

Their independent implementation (`repos/llama.cpp-pr32773`) reaches higher acceptance than mine on the
same target file and same published draft file — 0.770 vs 0.647 at `n_max 2` — while mine is 24-28%
faster. Acceptance is close to linear in decode throughput, so this is the biggest remaining lever. The
question this page answers is *where* the gap lives, because the answer decides which hypothesis to
chase.

## Method: derive marginal positions from data already on disk

The tree has no per-position instrumentation (the `pos = (...)` lines in `logs/abl*.log` came from a
temporary debug print I removed). But two block widths give two equations. With `sample_from_anchor`,
`mean len` counts the bonus token, so expected accepted-draft tokens per step is `E[k] = mean_len - 1`,
and:

- at `n_max = 2`: `E[k] = p1 + p2`
- at `n_max = 4`: `E[k] = p1 + p2 + p3 + p4`

so `p3 + p4 = E[k](4) - E[k](2)` with no new runs. Source: `logs/ab-MINE-{2,4}.log`,
`logs/ab-THEIRS-{2,4}.log` (2026-11-22, IQ4_NL target, same prompt and flags, both binaries loading the
published `dspark-Inkling-2-Mini-Q8_0.gguf`).

| build | `n_max` | acceptance | mean len | E[k] |
|---|---:|---:|---:|---:|
| mine | 2 | 0.647 | 2.29 | 1.290 |
| mine | 4 | 0.506 | 3.02 | 2.020 |
| theirs | 2 | 0.770 | 2.54 | 1.540 |
| theirs | 4 | 0.640 | 3.54 | 2.540 |

| build | positions 1-2 | positions 3-4 | late/early ratio |
|---|---:|---:|---:|
| mine | 1.290 | 0.730 | 0.57 |
| theirs | 1.540 | 1.000 | 0.65 |

**Gap decomposition: 0.250 tokens/step in positions 1-2, 0.270 in positions 3-4.**

## What that rules out

I went in expecting the slot-row-index lead (my earlier analysis flagged their `i+1` vs my indexing) to
show up as a *progressive* decay in later slots. It does not: the gap is 0.25 early and 0.27 late, an 8%
difference that is well inside the noise of two aggregate measurements. Same for an anchor/embedding
problem, which would concentrate in position 1.

**A roughly uniform multiplicative gap means the draft's logits are systematically a little worse at
every slot** — a quality difference in the head or its numerics, not a wiring mistake in one place.

That also corrects a conclusion I would otherwise have written: the shape of the evidence fits "their
head is slightly better everywhere" and does **not** fit "I have an off-by-one in the block".

## Hypotheses that survive, ranked

1. **bf16 rounding points.** The reference (`vllm/models/inkling2/nvidia/dspark.py`, transcribed in doc
   33) rounds at specific steps: `g = bf16(σ(bf16(raw₀)))`, `cand = bf16(tanh(bf16(raw₁)))`,
   `state' = bf16(bf16(g·state) + bf16(bf16(1−g)·cand))`, `latent = bf16(tanh(bf16(raw₂)))`. My graph runs
   that recurrence in f32 — *more accurate*, but the drafter was **trained** with the bf16 chain, so
   matching the trainer's arithmetic can beat being numerically better. This is the classic
   "different, not wrong" failure and it would be uniform across slots, exactly as observed.
   Test: insert the rounding points, measure acceptance. Cheap, self-contained, reversible.
2. **`f_rnn` source.** They feed it from `h_nextn`; I deliver the anchor's target taps through the batch
   state channel and mix them in-graph with the real `fuse` row 3. Both are defensible readings of the
   reference, and a systematically weaker `f_rnn` input would depress every slot. Note my own ablation
   measured `f_rnn` as near-neutral (0.0922 → 0.0922) — but that was at acceptance ≈ 0.09, in the broken
   regime, so it is weak evidence about the working regime.
3. **Softcap placement.** I cap after the markov bias (`softcap(base + bias)`, matching both the PR and
   the reference). Provably invisible at temperature 0 (monotone), so it cannot explain a greedy-mode
   acceptance gap — kept here only so nobody re-tests it.

## Also worth noting

Their `E[k]` at `n_max 4` is 2.54 vs my 2.02 — they get 26% more accepted tokens per step and still lose
on throughput, which quantifies the cost of their unrolled in-graph draw chain with device-pinned nodes.
If I close the quality gap without adopting that chain, the combination is strictly better than either
build. That is the prize.

## Both surviving hypotheses, tested the same evening — both refuted

### 1. bf16 rounding points in the markov chain — NULL (gate-verified)

Implemented behind `LLAMA_DSPARK_BF16_CHAIN`, reproducing the reference's exact form and rounding:
`g = bf16(σ(bf16(raw₀)))`, `cand = bf16(tanh(bf16(raw₁)))`,
`state' = bf16(bf16(g·state) + bf16(bf16(1−g)·cand))`, `latent = bf16(tanh(bf16(raw₂)))` — including the
`(1−g)·cand` algebraic form rather than my `cand + g·(state − cand)`, which f32 makes equivalent but bf16
does not. Patch kept at `logs/bf16-chain-experiment.patch`.

First attempt produced acceptance identical to five digits — and the first instinct ("no effect") was
wrong twice over: the build had **failed** (`exit=2`, a `\n` eaten by a python string), so both runs used
the old binary. Adding a one-shot `LLAMA_LOG_INFO("markov chain: %s (is_rnn=%d)")` probe and rebuilding
gave a real result:

| prompt | f32 acc / t/s | bf16 acc / t/s | Δacc |
|---|---|---|---:|
| code | 0.63018 / 48.17 | 0.63018 / 47.88 | **+0.00%** |
| prose | 0.69906 / 51.33 | 0.69906 / 51.09 | **+0.00%** |
| repetitive | 0.52000 / 43.61 | 0.52000 / 43.34 | **+0.00%** |

Probe confirmed firing (`markov chain: bf16 rounding (is_rnn=1)` in 3 runs, `f32` in 3). Acceptance is
bit-identical; decode is 0.5% lower, consistent with the extra CPY nodes. Reason it cannot matter: the
markov bias is ~5.6 logit units and bf16 rounds the *state* (256 floats), shifting logits far below the
top-2 margin, so the argmax never moves. Reverted.

**Process note worth keeping:** `VAR=""` makes `getenv(VAR) != nullptr` **true** — an empty environment
variable silently enables a gate. Use `unset`, not `VAR=`, to mean off.

### 2. Is `f_rnn` load-bearing in the working regime? — it is marginal, and not the gap

The earlier ablation said "≈0", but that was measured at acceptance 0.09 (the broken regime), so it said
nothing about the working one. Re-run with `LLAMA_ZERO_FRNN` (patch: `logs/frnn-ablation.patch`):

| prompt | f_rnn live | f_rnn zeroed | Δ acceptance |
|---|---|---|---:|
| code | 0.63018 / 47.97 | **0.64371 / 48.50** | **+2.15%** (removing it HELPS) |
| prose | 0.69906 / 51.05 | 0.69906 / 50.92 | +0.00% |
| repetitive | 0.52000 / 43.43 | 0.50394 / 42.80 | −3.09% (it helps here) |

So `f_rnn` is domain-dependent and small: it helps on repetitive text, does nothing on prose, and
*costs* acceptance on code. Consistent with the weight measurement (`rnn_proj`'s `f_rnn` column block is
rms 0.0044 vs `hidden`'s 0.0118 — the weakest of four inputs). Nothing in it explains a 19% gap.

## The lead that came out of reading their head instead

`repos/llama.cpp-pr32773/src/models/dspark.cpp`:

```c
257:    auto * base = g.build_lora_mm(output, hidden, scale);      // base logits recomputed per slot, inside the head
498:    auto * row      = ggml_view_1d(ctx0, hidden, n_embd, (i + 1)*hidden->nb[1]);
499:    auto * features = ggml_concat(ctx0, row, inp->context, 0);  // second half = the injected target CONTEXT
500:    auto * logits   = inkling2_head_step(*this, model, state, prev, features, row);
```

vs mine (`src/models/dspark.cpp`):

```c
base = res->t_logits;                                   // the draft graph's own logits, computed once
joint = concat(concat(state, w1_prev), hidden_i);       // second half = the draft's hidden
if (f_rnn) joint = concat(joint, anchor);               // third  half = the ANCHOR's tap mix, same column every slot
```

Both operands are 5120 wide, so shapes agree and nothing would ever complain — but the **content of the
second half differs**: they feed slot-specific injected target context (`inp->context`), I feed the
anchor's DFlare tap mix repeated at every position. A systematically weaker or wrong conditioning input
at every slot is exactly the uniform multiplicative gap the position decomposition showed.

Next experiment (filed): feed slot-specific context instead of the repeated anchor column and measure
acceptance on the same three-prompt suite. Cheap, self-contained, and it is the only surviving
explanation for a gap that is uniform across positions and immune to numerics.

## Third and fourth hypotheses, both refuted — and a reframe

### Tap order (refuted)

`fuse_weight` is `[n_draft_layers + 1, n_taps]`, and its softmax rows are strongly peaked:

| row | alpha | peak |
|---|---|---|
| 0 (stage 0) | [0.093, 0.113, **0.688**, 0.106] | tap 2 = layer 41, 68.8% |
| 1 (stage 1) | [0.329, 0.222, **0.348**, 0.100] | tap 2 = layer 41, 34.8% (nearly flat) |
| 2 (stage 2) | [0.181, 0.060, 0.288, **0.472**] | tap 3 = layer 48, 47.2% |
| 3 (markov head) | [0.202, 0.015, **0.748**, 0.035] | tap 2 = layer 41, 74.8% |

(Reproduced by `scripts/inkling_native/fuse_alpha_rows.py` from the checkpoint's `mtp.safetensors`.)
Three of four rows peaked means a mis-ordered tap vector would put most of a stage's weight on the
wrong layer — a *uniform* per-slot error, which is exactly the observed signature. So this was the
best remaining lead.

It is correct. My fill is `common/speculative.cpp:1310-1321`, tap `k` at offset `k*n_embd_tgt` where
`k` indexes `target_layer_ids[k]`, and both sources declare the same order:

- draft GGUF: `dflash.target_layers = [5, 23, 41, 48]`
- checkpoint: `mtp_config.target_layer_ids = [5, 23, 41, 48]`

And the visit-vs-slot indexing that #32218's comment exposed was already fixed in `4bad7e855` + the
visit re-indexing commit, so `llama_get_embeddings_layer_inp(ctx, 41)` reads visit 41.

### Head structure (refuted earlier this session)

Their `build_dspark_markov_head` is structurally equivalent to mine.

## The reframe: the "gap" may not be a defect at all

Everything structural now matches, and I have a measured confound that is the right size: **`-ub`
alone moved acceptance 0.495 → 0.619** (§cache-type study), and a prompt-cache hit moves it
0.658 → 0.47 (doc 60). The head-to-head that produced "theirs 0.770 vs mine 0.647" ran the two
binaries with **`--fit` enabled on both**, because pinning `-ngl 99` OOM'd — so the two builds
planned *different context sizes and therefore different batch shapes* for the same request.

Acceptance is a property of the (model, prompt, shape) triple, not of the drafter alone. Before
spending more effort on an implementation gap, the cheap and possibly larger win is the inverse
question: **which shape maximizes acceptance at the deployed context?** If `-ub` is worth 25 points of
acceptance, choosing it deliberately is worth more than any remaining head fix.

That promotes TODO #33 (why `-ub` changes acceptance) from curiosity to the top of the drafter queue,
and closes #46 as "no defect found; confound identified".
