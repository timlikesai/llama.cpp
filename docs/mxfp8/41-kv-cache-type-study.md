# 41. KV cache type study at 200K, target vs draft, q4_1 vs f8_e4m3

Question: does `q4_1` beat `f8_e4m3` (fp8) for the target cache and for the draft cache, at the
config we actually run — 200K context, lossless 19.6 GB file, drafter engaged.

Box: spare-parts-2, 2x RTX 5060 Ti 16 GB, `-sm layer -fa on -b 2048 -ub 256`, embedded/DSpark draft
via `--spec-type draft-dspark --spec-draft-n-max 2 --spec-draft-n-ubatch 64`, 256 generated tokens,
`--temp 0 -s 42`, prompt `logs/prompt-code.txt`. Logs: `logs/k41-*.log`, parsed with
`scripts/inkling_native/parse_spec_run.py`.

## 1. Target cache, quality only (PPL, 2048 ctx, 16,705 WikiText-2 tokens, no draft)

| target KV | PPL | vs f16 |
|---|---|---|
| f16 | 5.5656 ± 0.14398 | — |
| f8_e4m3 | 5.5714 | +0.10% |
| q5_1 | 5.5740 | +0.15% |
| **q4_1** | **5.5783** | **+0.23%** |
| q4_0 | 5.5837 | +0.33% |

Every one of these sits inside the ±0.144 confidence interval, so on perplexity none of them is
distinguishable from f16. That is the easy case though: 2048-token chunks stress the cache far less
than a 200K retrieval would (that probe was skipped on request — see #27 for what remains).

## 2. End-to-end at 200K with the drafter engaged

| target / draft cache | prefill | decode | acceptance | acc/gen | mean len |
|---|---|---|---|---|---|
| **q4_0 / q4_0** | 1,031.2 | **52.0** | **0.733** | 151/206 | 2.47 |
| f8_e4m3 / q4_0 | 1,013.0 | 50.5 | 0.655 | 144/220 | 2.31 |
| f8_e4m3 / q4_1 | 1,019.4 | 50.1 | 0.634 | 142/224 | 2.27 |
| q4_1 / q4_0 | 1,027.8 | 46.6 | 0.592 | 138/233 | 2.18 |
| q4_1 / q4_1 | 1,031.8 | 45.1 | 0.565 | 135/239 | 2.12 |
| f8_e4m3 / f8_e4m3 | — | — | — | refused, see §3 |
| q4_1 / f8_e4m3 | — | — | — | refused, see §3 |
| f16 / q4_0 (what ran before this study) | 762.9 | 49.8 | 0.609 | 140/230 | 2.22 |
| no draft, f16 cache | — | ~40.7 | — | — | — |

Two things stand out. **Quantizing the target cache was the real lever, not the draft cache**: the
target had been running f16 all along, and moving it to q4_0/f8_e4m3 lifted prefill 763 -> ~1,030
(+35%) because those ~4 GiB of cache were competing with compute buffers for the same cards. And
**q4_1 loses to q4_0 on every axis that matters** — it stores the same 4.5 bits per weight (one extra
`min` per block), so it dequantizes slower for no capacity gain: 45.1-46.6 t/s vs 52.0, and lower
acceptance. Its only edge is 0.1% of PPL, which is noise here.

> **[CORRECTED 2026-11-23 late — `50-cache-type-and-ubatch.md`] The causal story here is wrong.**
> The 45.1-vs-52.0 gap did not come from cache type at all: at equal `-ub`, `q4_1` measures 48.92
> vs `q4_0`'s 49.02 at 8K (0.2%) and 48.24 vs 48.73 at 262,144 (1%), and acceptance 0.655 vs 0.667.
> Two confounds explain the original table: (1) **`-ub` moves acceptance by 20% relative**
> (0.495 at `-ub 128` vs 0.619 at `-ub 256`, identical across contexts) and (2) those runs were the
> *server* with the ngram chain + vision, which this comparison excludes. I also tested and rejected
> the plausible mechanism: `q4_1-q4_1` is missing from `GGML_CUDA_FA_QUANTS` and does log
> "converting K and V to f16 instead (slow)", but compiling it changed nothing (48.92 -> 48.41).

`q4_0` also measured better than `f8_e4m3` on both throughput and acceptance, which is the wrong
direction if the differences were real — see the caveat below.

## 3. Why the two f8-draft rows are missing: it is refused, not slow

```
E llama_init_from_model: FP8 cache is not supported for architecture 'dflash'
E common_speculative_init_result: failed to create MTP context
```

`src/llama-context.cpp:4496` keeps an explicit **denylist** for FP8 KV — `DFLASH`, `DEEPSEEK41`,
`GLM_DSA`, `GLM5_NEXT`, `DOTS3NOTE`, `MISTRAL_LARGE4`, `HY_V4`, `MINIMAX_M3`, `QWEN4EXP` — and the
Inkling DSpark draft loads as a `dflash` arch, so it is covered. Inkling's *target* arch
(`inkling2_loop`) is **not** on that list, which is why `-ctk f8_e4m3` works on the target and fails
at model-load time for the draft. So for the draft the choice is only among
f16 / q8_0 / q4_0 / q4_1 / q5_1 / iq4_nl, and `q4_0` measured best.

Worth reading as a signal: this denylist is upstream saying FP8 KV is known-bad for the newest
MoE/sparse-attention architectures. Anything we do with fp8 caches on models in that family should
be checked, not assumed.

## 4. Recommendation

- **Throughput-first (measured best): `-ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0`, `-ub 256`,
  `--spec-draft-n-max 2`** → 1,031 t/s prefill / 52.0 t/s decode at 200K with the drafter, ~28%
  faster decode than no-draft and ~35% faster prefill than the f16 cache.
- **Quality-conservative**: same but `-ctk f8_e4m3 -ctv f8_e4m3` → 1,013 / 50.5, and f8 is the
  option with the smallest measured PPL delta (+0.10%). Costs ~1.5 t/s.
- **Do not use q4_1 anywhere.** Same bytes as q4_0, slower, no measurable quality benefit.
  **[SOFTENED 2026-11-23]** under controlled `-ub` the q4_1/q4_0 difference is 0.2-1%, not 13% - use
  q4_0 (marginally better and it has a compiled FA kernel) but the earlier "13%" figure was the
  `-ub`/ngram-chain confound, see the correction above and `50-cache-type-and-ubatch.md`.
- **[NEW, and it reverses another conclusion here] `f8_e4m3` on the TARGET cache is the best choice at
  any context it fits**: +13% to +17% decode and +19% to +24% acceptance vs `q4_0` at equal `-ub`,
  *and* the most accurate (PPL +0.10% vs f16). It costs ~2x the KV bytes of q4_0, so it fits
  196,608 at `-ub 256` and 262,144 only at `-ub 64` (where prefill drops to 586 t/s). The claim above
  that q4_0 beat f8 on throughput does not survive a controlled comparison.
- `-ub 512` at 200K only works with q4_0 caches (and even then the fitter logged an allocation
  failure it recovered from); with f8 target cache it OOMs outright. So `-ub 256` is the honest
  ceiling at 200K.

## 5. Caveat I am not smoothing over

Acceptance moved a lot with the *target* cache type (0.733 for q4_0 vs 0.592-0.634 for q4_1/f8),
which is suspicious: the target cache should only perturb logits slightly. It is not a bug in the
draft — a slightly different target logit stream changes which token wins near ties, so the model
generates a *different continuation*, and acceptance is strongly domain-dependent (measured elsewhere
on this head: code 0.67, English prose ~0.2, reasoning 0.3-0.4). Each cell here is one 256-token
greedy run on one prompt, so treat differences of ±0.1 in acceptance as unreliable and the
throughput numbers as the trustworthy part of this table. The three-prompt matrix in #50/#56 is what
should decide any future acceptance-driven change.

## 6. And that closed 256K with the drafter

The ~288 MiB shortfall that blocked 256K + drafting was never compute buffers or placement — it was
this: **the target model's KV cache had been f16 for the whole session**, because `-ctkd/-ctvd` only
set the *draft's* cache. Quantizing the target's cache freed ~2 GiB and the same configuration that
failed at 262144 now runs clean, with **zero** allocation failures and all 17.8 GiB of weights
resident (model 8,355 + 9,451 MiB):

```
-c 262144 -sm layer -fa on -b 2048 -ub 256 \
  -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 \
  --spec-type draft-dspark --spec-draft-model <dspark draft> \
  --spec-draft-n-max 2 --spec-draft-n-ubatch 64
```

Four repeats: prefill 1,026-1,029 t/s, decode 51.95-52.30 t/s, acceptance 0.73301 (151/206) identical
every run. No-draft at the same context is ~39.5-40.8 t/s, so drafting is **+29%**. Other clean
micro-batches: `-ub 128` = 780/49.2, `-ub 192` = 941/46.1.

Session arc on the complete lossless model (all original weights + drafter): **507/22 t/s at the
start -> 1,027/52 t/s now** — 2.0x prefill, 2.4x decode. The two changes that did it were alias-aware
device placement (doc 27) and removing the materialized relative bias (§1-2 above, upstream #32808);
the cache type was the last 30%.

Reproducibility note worth keeping: every config whose log contained `failed to allocate` (the fitter
backing off and retrying) produced 5-18% run-to-run spread, and configs without it were stable to
under 1%. So when a measurement looks noisy here, check for fitter retries in that log before
believing a delta — several of my earlier "variances" were that, not code behavior.

## 7. Verified in the docker server, not just the CLI (2026-11-23)

The compose `llama-cpp` service now runs this model: `/models/timlikesai/Inkling-2-Mini-FULL.gguf`
with `--spec-draft-model /models/timlikesai/Inkling-2-Mini-DSpark-BF16.gguf`, `-c 262144`,
`--cache-type-k/v q4_1`, `--spec-draft-type-k/v q4_1`, `-ub 256 -b 2048`, `--split-mode layer`,
`--flash-attn on`, `--parallel 1`, `--fit on` (with `-ngl` unset). `q4_1` on both sides is the
operator's choice; §1-2 above is why `q4_0` would be ~7 t/s faster.

Measured through the HTTP API (not a CLI harness), 400-token code generation, temp 0:

| |
|---|
| decode **47.42 t/s** (`predicted_per_second`), 400 tokens in 8.41 s |
| **draft acceptance 0.53646** (206 accepted / 384 generated), mean len 2.07 |
| CUDA graphs reused 416, reallocs 2 (same size) — no rebuild storms |
| VRAM 15,281 + 15,161 MiB across the two cards, **no host spill** |
| `/v1/chat/completions` on `17 times 4?` → `"68"` ✓ |

Two operational notes that fell out:

1. **The prompt-throughput number is not meaningful from the server** (`prompt_per_second 81` on
   that request was 4 uncached tokens — the rest came from the prompt cache, `cached_tokens 40`).
   Prefill rate has to be measured with a cold cache or read from a CLI run.
2. **Inkling answers inside a thinking block.** With `max_tokens: 400` the whole budget went into
   `reasoning_content` and `content` came back empty. Callers must either allow a large budget or
   the server should set `--reasoning-budget`. This is a usage trap, not a defect.

Reboot safety, checked because the boot unit's description says it "fetches llama.cpp updates":
`docker-compose-stack.service` has its `git fetch` / `git pull` / `compose build` lines **commented
out**, and `llama-cpp.Dockerfile` does `COPY repos/llama.cpp/. .` (it never invokes git for the
sources). So the image is whatever branch is checked out in `repos/llama.cpp` — currently
`mxfp8-weights` — and boot reuses Tim's rebuild rather than silently reverting to master. Verified
by symbol, not by assumption: the running image's `libggml-base.so` exports
`ggml_flash_attn_ext_add_indexed_bias`, which exists only after the #32808 merge. If anyone rebuilds
this image from `master`, Inkling stops loading (type 44 unknown).
