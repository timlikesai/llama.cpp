# Cache state, acceptance, and greedy non-reproducibility on the deployed server

Measured 2026-11-24 against the **running container** (`draft-dspark,ngram-mod,ngram-map-k`,
ctx 262,144, `q4_1` caches, vision on so `--cache-reuse` is inert but `--ctx-checkpoints
256 --checkpoint-min-step 32768 --cache-idle-slots` are live).

## The experiment

Four byte-identical `temperature: 0` requests, 256 `max_tokens`, to the same slot
(`measure_variant.sh srvdet`).

| req | `cache_n` | `prompt_n` | prefill ms | `draft_n` | accepted | rate | mean len | decode t/s |
|---|---|---|---|---|---|---|---|---|
| 1 (cold) | 0 | 38 | 2,558.6 | 237 | 156 | **0.658** | 2.59 | 57.0 |
| 2 (hit)  | 34 | 4 | 47.1 | 316 | 157 | 0.497 | 2.64 | 54.4 |
| 3 (hit)  | 34 | 4 | 47.7 | 355 | 167 | 0.470 | 2.96 | 55.8 |
| 4 (hit)  | 34 | 4 | 46.6 | 417 | 195 | 0.468 | **4.61** | **72.6** |

**All four responses differ.** md5s distinct; `reasoning_content` 831 / 848 / 892 / 828 chars.
(`content` is empty in all four — the thinking-model trap, 256 tokens spent inside the block.)

## What this is NOT

Three hypotheses, each killed by evidence:

1. **RNG.** The only generator in `common/speculative.cpp` is `dflash2_rng` (line 1589), seeded at
   `:1680` and read at `:1718`, and it is gated by `sample = params.probabilistic && dp.result_q`.
   Greedy never reaches it. The **ngram impls contain no RNG at all** (verified by scanning the
   class bodies). So no random source is on the greedy path.
2. **#8 (missing target taps at the prompt boundary).** The `no target taps recorded` warning fires
   **once per process**, not once per cache hit — and it also appears in the CLI runs that are fully
   deterministic. So it is real but bounded (1 of ~83 draft rounds) and is *not* this effect.
3. **A cache bug.** The cold request is the *correct* baseline; nothing is corrupted.

## What it is

A prompt-cache hit changes the **effective batch shape**: 38 prompt tokens become 4. That changes
floating-point reduction order in the target's forwards, which changes the draft's conditioning
inputs, which flips near-tie greedy tokens — and a flipped token changes every subsequent draft and
verification. This is the same mechanism already documented for speculative decoding generally
(*"greedy output with the drafter is not guaranteed to be bit identical … the target's batched
verification can flip near-tie tokens on CUDA"*), now observed through the cache rather than through
`--spec-type`.

The acceptance-rate decline is **not a degradation**. `draft_n` rises 237 → 417 while accepted stays
roughly flat in absolute terms (156 → 195), so the *ratio* falls while **mean accepted length rises
2.59 → 4.61 and throughput rises 57.0 → 72.6 t/s**. That is the ngram speculators finding matches in
repeated context — `ngram_map_begin` appears in the container log — which is exactly what they are
for. The server reports a single aggregate acceptance figure, so per-speculator attribution would
need instrumentation; the behavioural signature (more proposals, longer accepted runs, faster) is
unambiguous.

## Consequences

**For measurement (this is the important one).** Acceptance is only comparable within a cache state.
Every number in this session that came from `profile_shipped.sh` (llama-server, caching flags intact)
is on a different footing from every number that came from `llama-cli` (which cannot even accept
`--cache-reuse` / `--checkpoint-min-step` / `--cache-idle-slots` — verified against `llama-cli
--help`; only `--ctx-checkpoints` is shared). That explains the "0.609 / 0.429 / 0.618 on three
identical requests" I had attributed to an unknown cause, and it means:

- The **head-to-head acceptance gap vs the #32218/#32773 builds (0.770 vs 0.647)** was measured
  CLI-to-CLI, both cold — so that comparison stands.
- Any server-derived acceptance figure must state its cache state, or be discarded.
- New rule: to compare acceptance, either restart the server per configuration or send each prompt
  exactly once per process.

**For operation.** Warm requests are *faster*, not slower. Greedy reproducibility across cache
states does not hold; if a caller needs identical output for identical input, it must disable prompt
caching (`--no-context-shifting`? no — the knob is to not reuse: restart, or drop
`--ctx-checkpoints`/`--cache-idle-slots`).

## Control measurement

`measure_variant.sh cachedet` runs the same prompt three times through `llama-cli` with the `SHIPPED`
vector (no caching flags available to it): acceptance **0.63018 (213/338) on all three reps**, decode
48.18 / 48.15 / 48.24. The CLI path is deterministic, which is what isolates the cache as the source
of the server-side variation.

## Open

- Whether the cold-vs-warm acceptance difference can be separated per speculator (needs a stats
  split in the server, or a chain of one).
- #8 remains open but is now known to be **small**: one degraded draft round per process, not per
  request.
