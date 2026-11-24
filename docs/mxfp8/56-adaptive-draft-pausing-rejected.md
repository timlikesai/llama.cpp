# 56. Adaptive draft pausing: ported, measured, rejected

Prior art from `repos/llama.cpp-pr32773` commit `61031e7d2` (patch kept at
`logs/upstream-adaptive-draft-pausing.patch`): track per-sequence draft acceptance as an EMA, and after
>=8 observations with `rate < 0.33`, **stop drafting for 64 steps**. Two states only, because every
distinct verification batch size forces a graph rebuild. Their comment names the mechanism I had
measured independently: *"~25 ms on a 12k node graph"*.

Ported into `common_speculative_impl_draft_dspark` (my Inkling path — theirs is a dedicated
`impl_inkling2_dspark` class that does not exist here), gated on `LLAMA_SPEC_ADAPT` so it A/Bs with no
rebuild. Harness mode: `measure_variant.sh adapt` (env `ADAPT_N` for generation length).

**Verdict: rejected on measurement.** It raises acceptance on every single configuration and lowers
throughput on five of six.

## First, a latent bug the port exposed

With pausing implemented as "draft zero tokens", the run aborted:

```
adapt_count: seq 0 draft acceptance 0.327 < 0.330 - pausing drafting for 64 steps
ggml/src/ggml-backend.cpp:534: GGML_ASSERT(buf != NULL && "tensor buffer not set") failed
#6 ggml_backend_tensor_set
#7 llm_graph_result::set_inputs(llama_ubatch const*)
#8 llama_context::process_ubatch
#9 llama_context::decode(llama_batch_ext const&)
#10 common_speculative_impl_draft_dspark::draft(...)
```

Cause: with `sample_from_anchor = false`, a "zero draft" still adds the anchor, producing a **width-1
draft batch**, and that trips the assert in `set_inputs`. Skipping the whole step (not just the block
width) avoids it. The underlying fragility is independent of this experiment — a draft context asked to
process a single anchor token appears to be an unhandled shape. Noted as a separate item; I did not
isolate whether it is my state-input plumbing or the shared path, so it is filed as "needs a clean
repro", not as a confirmed upstream bug.

## Short generations (192 tokens, ctx 32,768, `-ub 256`, `q4_1`, chain off, 0 allocator retries)

| prompt | n_max | adapt off | adapt on | delta |
|---|---:|---:|---:|---:|
| code | 2 | 44.61 | 44.43 | −0.4% |
| prose | 2 | 46.07 | 46.15 | +0.2% |
| repetitive | 2 | 41.44 | 36.71 | **−11.4%** |
| code | 4 | 51.21 | 36.54 | **−28.6%** |
| prose | 4 | 43.51 | 40.68 | −6.5% |
| repetitive | 4 | 35.16 | 34.47 | −2.0% |

A 64-step pause on a 192-token generation is a third of the run, so this test is biased against the
mechanism. The long run below is the fair one.

## Long generations (1,500 tokens) — the fair test, still negative

| prompt | n_max | adapt off | adapt on | delta | acceptance off → on | pauses |
|---|---:|---:|---:|---:|---|---:|
| code | 2 | 50.55 | 48.97 | −3.1% | 0.681 → 0.722 | 3 |
| prose | 2 | 44.76 | 42.22 | −5.7% | 0.540 → 0.588 | 8 |
| repetitive | 2 | 42.59 | 40.01 | −6.1% | 0.491 → 0.525 | 3 |
| code | 4 | 51.98 | 51.35 | −1.2% | 0.557 → 0.692 | 6 |
| prose | 4 | 40.31 | 41.00 | **+1.7%** | 0.363 → 0.466 | 8 |
| repetitive | 4 | 39.19 | 38.41 | −2.0% | 0.348 → 0.408 | 10 |

**Acceptance improved in all six cases; throughput got worse in five.** That is the trap in the metric:
pausing deletes the worst drafts, so the ratio of accepted-to-drafted rises while the user gets fewer
tokens per second. Measuring a drafter by acceptance alone would have called this a win.

## Why there is no operating range here

Pausing only pays when drafting is **net negative** — when verification overhead exceeds the tokens the
drafter saves. On this box that boundary is measurable:

- no-draft baseline: **40.7 – 41.9 t/s**
- worst drafted case in the table (repetitive, `n_max 4`): **39.19 t/s** at acceptance 0.348

So drafting is roughly break-even at acceptance ≈ 0.30 and clearly profitable above it. The port's
threshold is 0.33 — i.e. it pauses exactly in the region where drafting still pays, and every pause
costs 64 steps of the slower baseline plus two graph rebuilds on the transitions. To be useful the
threshold would have to sit near 0.15–0.20, which this model never reaches.

Two corollaries worth keeping:

1. **Shorter pauses do not rescue it.** The cost is dominated by transitions (each pause and each resume
   changes the verification batch width → ugraph rebuild), and more, shorter pauses means more
   transitions.
2. **A binary state is the wrong shape for this model.** A per-step decision — draft only when the
   confidence head says the next token is likely — avoids the batch-width churn entirely. That is the
   `p_min` path, which this tree already implements via the draft's confidence output
   (`llama_get_embeddings_nextn`, `speculative.cpp` dspark `draft()`), and it needs no state machine.

## Reusable numbers

The 3-domain × 2-width grid above is a decent regression baseline for this model: acceptance ranks
code > prose > repetitive, and `n_max 4` beats `n_max 2` on code (51.98 vs 50.55) but loses on prose
(40.31 vs 44.76) — so `n_max 2` remains the right shipped default.

## Decision

Reverted; nothing shipped. `LLAMA_SPEC_ADAPT` does not exist in the tree. The mechanism may still be
right for a box where drafting is genuinely net-negative (more experts on host, slower interconnect, a
weaker drafter) — if anyone revisits it, the number to compute first is the break-even acceptance, not
the threshold.
