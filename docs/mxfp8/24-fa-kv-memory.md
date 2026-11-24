# 24 — Flash Attention vs f16 KV memory (measured), and a bigger anomaly it exposed

Status: **matrix measured by the parent after the assigned lane timed out with zero artifacts on disk.** **CORRECTED 14:55 — the "anomaly" section below was a measurement artifact of my own making; see §Correction at the top of the file for the true numbers.** Method: `llama-completion -v ... -n 1 -p hi` and reading the **final** `sched_reserve` / `llama_kv_cache` reservation pass.

## CORRECTION — my "KV is flat with context" claim was wrong (a two-line cache I summed as one)

A cache with `iswa` (this model has both dense and sliding-window layers) prints **two** `llama_kv_cache: size =` lines — one for the dense cache, one for the SWA cache. My extraction took only the last, so I reported the SWA cache alone as if it were all of KV and then invented an anomaly around it. A sibling lane found this from my own logs. The true numbers (`logs/fav-fa-on-8k.log`, `logs/fav-fa-on-32k.log`, `logs/fav-fa-on-e4m3*.log`):

| config | ctx | dense cache | SWA cache | **KV total** | compute total |
|---|---|---|---|---|---|
| FA on, f16 | 8192 | 256 MiB (8192 cells, **8 layers**) | 240 MiB (1536 cells, **40 layers**) | **496 MiB** | 5,479 MiB |
| FA on, f16 | 32768 | 1024 MiB (32768 cells, 8 layers) | 240 MiB (1536 cells, 40 layers) | **1,264 MiB** | 9,652 MiB |
| FA on, f8_e4m3 | 8192 | 128 MiB | 120 MiB | **248 MiB** | 5,485 MiB |
| FA on, f8_e4m3 | 32768 | 512 MiB | 120 MiB | **632 MiB** | 9,652 MiB |
| FA off, f16 | 8192 | 256 MiB | 240 MiB | **496 MiB** | 6,466 MiB |

**What is actually true:**
- **KV scales with context, exactly as it should** — the dense cache grows linearly (256 -> 1024 MiB for 8k -> 32k); only the SWA cache is flat (1536 cells = the 1024 window + padding). Nothing anomalous.
- **fp8 KV halves it precisely** (1,264 -> 632 MiB at 32k; 496 -> 248 at 8k), and it halves both caches.
- **FA does not change KV at all** (496 vs 496, 1,264 vs 1,264) and makes compute **987 MiB cheaper** at 8k. That part of the original finding stands.
- **The looped layers do double the dense KV slots — and correctly so.** 4 unique full-attention layers (pattern indices 5, 11, 17, 23) appear as **8** cache layers, and 20 unique SWA layers as **40** slots: 48 total, matching `n_layer_all`. That is required, not waste — `inkling2-loop.cpp` shares *weights* but each application keeps its own KV and conv state, and the slots inherit the right type because `is_swa_impl`/`is_recr_impl` are propagated (`:38-39`). But it does mean the loop costs **2x dense-KV bytes**, which is the real, quantified version of the expansion intuition.
- **`relative_attention_extent` is not a context cap** (settled by task #26): it is the width of the learned relative-position bias table, `attn_rel_bias {n_rel, n_extent}` (`inkling2.cpp:110,124`), and distances beyond it are **clamped into one extra learned bucket** (`src/llama-kv-cache.cpp:2018` gathering against `ggml_pad(layer.attn_rel_bias, 0, 1, 0, 0)`) rather than dropped. Past 8192 a dense layer stops distinguishing *how far* away a key is; it still sees the key. Allocation uses `n_ctx_seq` (`src/llama-model.cpp:3418` -> `llama-kv-cache-iswa.cpp:69` -> `llama-kv-cache.cpp:233`) and the dense cache is built with `LLAMA_SWA_TYPE_NONE`, so it is never window-masked.

**Still open and now the live question (#29):** the **compute** buffers really do grow 5,479 -> 9,652 MiB from 8k to 32k (+4,173 MiB ~= 0.174 MiB/token) while FA is enabled, which is not expected to be context-proportional at all. That term is the actual obstacle to 256K/512K, and unlike KV it is unaffected by cache type.

**Extrapolation for Tim's 256K/512K question, using dense KV = 4 KiB/token/slot f16 (2 KiB fp8) x 8 slots:** 256K needs ~8.0 GiB KV in f16 or **~4.0 GiB in f8_e4m3**; 512K needs ~16.1 GiB / **~8.1 GiB**. Plus the ~3.2 GiB of SWA/flat terms and whatever compute ends up costing. So 256K at fp8 KV is plausible on 2x16 GB and 512K is marginal — the deciding unknown is the compute-buffer coefficient, not KV.


Two method traps that produced wrong numbers on the first attempt, recorded so nobody repeats them: (1) the tool prints `KV buffer size` twice per device (reserve + final) and, at large contexts, **many** `sched_reserve` passes (54 KV lines and 46 compute lines at 32k, one per ubatch size) — summing all matches gave a nonsense 129,871 MiB; take the **last** pass only. (2) The cache-type string is **`f8_e4m3`** (`ggml_type_name(GGML_TYPE_F8_E4M3)`, `common/arg.cpp:285-300`), *not* `fp8` — `--cache-type-k fp8` fails with `Unsupported cache type` and, because the failure was in argument parsing, the earlier run produced no log lines at all and looked like a silent hang.

## Measured matrix — Inkling-2-Mini-NATIVE, 2x RTX 5060 Ti, `-b 2048`

| config | ctx | dense cache | SWA cache | **KV total** | compute total | notes |
|---|---|---|---|---|---|---|
| FA off, f16 KV | 8192 | 256 MiB | 240 MiB | **496 MiB** | 6,466 MiB | |
| FA **on**, f16 KV | 8192 | 256 MiB | 240 MiB | **496 MiB** | **5,479 MiB** | FA on is **987 MiB cheaper** |
| FA off, f8_e4m3 KV | 8192 | 128 MiB | 120 MiB | **248 MiB** | 6,466 MiB | fp8 halves KV exactly |
| FA on, f8_e4m3 KV | 8192 | 128 MiB | 120 MiB | **248 MiB** | 5,485 MiB | |
| FA on, f16 KV | 32768 | 1024 MiB | 240 MiB | **1264 MiB** | 9,652 MiB | dense cache **grew 4x** from 8k |
| FA on, f8_e4m3 KV | 32768 | 512 MiB | 120 MiB | **632 MiB** | 9,652 MiB | |

The first version of this table had a single `KV total` column reading 240 / 240 / 120 / 120 / 240 / 120 MiB. Every one of those numbers is the **SWA cache alone** - see "The real anomaly" below, which is where the split columns come from. Compute totals are unaffected and stand as first measured.

## Answer to the reported symptom

**On this model, enabling Flash Attention does not increase KV memory at all** - dense + SWA totals are identical with and without FA at 8k, for both cache types (496 MiB f16, 248 MiB f8_e4m3), and it *reduces* compute-buffer demand by ~15%. The combination people describe (FA costing more than a plain f16 cache) does not reproduce here; if it is real it is model- or config-specific, and the first thing to check on another model is whether `kv_unified` differs (it is **false** here) and whether the per-layer `is_swa` array is fully populated for every applied layer.

**Two loop-model suspicions I checked and cleared, with the code, rather than assuming:**
- *Do replayed slots misclassify as dense and grab full-context buffers?* **No.** `src/models/inkling2-loop.cpp:38-39` propagates both flags when expanding slots: `hparams.is_swa_impl[il] = hparams.is_swa_impl[p]` and `is_recr_impl[il] = is_recr_impl[p]`. The GGUF carries `sliding_window_pattern` as a **24**-entry array (`[true x5, false, ...]`, full-attention at indices **5, 11, 17, 23**) while `n_layer_all = 48`, so that copy is load-bearing — and it is present and correct. I want this on the record because I had written the expansion up as the likely cause before reading it.
- *Is the 18.8 GiB allocation failure KV growth?* **No** — it is a **compute** buffer (the same class as the `sched_reserve: failed to allocate CUDA0 buffer` we hit when a pinned `-ngl` disables fitting). KV and compute must be logged separately or this confuses every reader, as it did us.

## The anomaly I thought this exposed (retracted — see CORRECTION above)

Kept for the record because the reasoning was wrong in an instructive way: I claimed KV was identical at 8192 and 32768 (240/120 MiB) and that four dense layers at 32k would need 512 MiB, therefore dense layers must be capped. The measurement was of one of two cache lines. **Both branches of the dilemma were fictitious** — dense layers are allocated at full context width and are never window-masked. - SETTLED: it was my own measurement error

**There is no anomaly. Neither of the two candidates I had posted - "capped at `relative_extent` by design" and "silently truncated to `n_swa`" - was right, and the shared premise of both was false.** I summed only one of the two KV cache lines that `llama_kv_cache_iswa` prints. An iswa cache is *two* `llama_kv_cache` objects, and each emits its own `I llama_kv_cache: size = ... MiB` line. I captured the second one (the SWA cache) and dropped the first (the dense cache), then reasoned from the partial total.

The allocator prints its own cell counts, at default verbosity, in the same pass - I did not look for them. From the logs already on disk (`logs/fav-fa-on-8k.log:1345,1396,1399,1450` and `logs/fav-fa-on-32k.log:1345,1396,1399,1450`):

```
8k   non-SWA KV cache, size =  8192 cells | size =  256.00 MiB ( 8192 cells,  8 layers)  <- missed
     SWA     KV cache, size =  1536 cells | size =  240.00 MiB ( 1536 cells, 40 layers)  <- the "240 MiB"
32k  non-SWA KV cache, size = 32768 cells | size = 1024.00 MiB (32768 cells,  8 layers)  <- missed
     SWA     KV cache, size =  1536 cells | size =  240.00 MiB ( 1536 cells, 40 layers)  <- the "240 MiB"
```

**The dense cache grows exactly as it should: 256 -> 1024 MiB (4x) going 8k -> 32k.** True KV totals are 496 MiB at 8k and 1264 MiB at 32k, not 240 at both. My 240 MiB is ctx-independent *by construction*, which is what made the artifact look like a finding: `size_swa = GGML_PAD(min(n_ctx, n_swa + n_ubatch), 256)` = 1536 cells at both 8192 and 32768 (`src/llama-kv-cache-iswa.cpp:73`). I had summed the one cache that cannot grow with context and concluded global attention did not grow with context.

### Answer: full-attention layers are bounded by neither `relative_extent` nor the SWA window

Checked at all three levels that would matter, each independently:

1. **Allocation.** `attn_kv_size = cparams.n_ctx_seq` (`src/llama-model.cpp:3411-3419`) -> `size_base = kv_size` (`src/llama-kv-cache-iswa.cpp:69`) -> `ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)` (`src/llama-kv-cache.cpp:233`). Dense slots get `n_ctx` cells, full stop. **Nothing in the allocation path reads `n_relative_extent`** - the only three readers in `src/` are the hparam load, the saver, and `inkling2.cpp` (weights and graph input). Verified above in the logs: 8192 cells at 8k, 32768 at 32k.
2. **Masking.** The dense cache is built with `n_swa = 0, LLAMA_SWA_TYPE_NONE` (`src/llama-kv-cache-iswa.cpp:94-98`), and `is_masked_swa` returns false unconditionally for `SWA_TYPE_NONE` (`src/llama-hparams.h:563-569`). So no window mask can be applied to a dense layer at any distance. The 1024 window only ever reaches the 40 SWA slots.
3. **What `relative_extent` actually is.** Not an attention span - the width of the learned relative-position **bias table**. `inkling2.cpp:110` exists solely to shape `attn_rel_bias = {n_rel, n_extent}` (`:124`). Distances at or past the extent are clamped into one extra learned bucket, not dropped: `distance >= 0 && distance < extent ? distance : extent` (`src/llama-kv-cache.cpp:2018`), gathered against `ggml_pad(layer.attn_rel_bias, 0, 1, 0, 0)` (`src/models/inkling2.cpp:357`) - the pad is the +1 column that makes index `extent` legal. Past 8192 tokens a dense layer stops distinguishing *how far* away a key is; it still sees the key.

**So the planning consequence in candidate 1 is void: contexts beyond 8192 do buy additional attention reach on Inkling.** Chasing 262k for retrieval is not pointless, and `n_ctx_train = 1048576` is not misleading for memory planning. Candidate 2 (silent 1024-token truncation, upstream bug report) is likewise refuted - there is nothing to report upstream.

The doc's own arithmetic was correct and should have overruled the observation: it computed that 32k needs 512 MiB for the dense layers, "more than the entire observed 240 MiB". The right inference from that gap was *my total is missing a cache*, not *the allocator is capping global attention*.

**Disposition of the experiments this section asked for:** the code read (item 1) is what settled it, and the allocator does print the per-cache cell counts at default verbosity (item 4 - I had simply not looked for those two lines). The 32k needle probe and the IQ4_NL cross-check are no longer needed to answer this question, since there is no truncation to detect; run them only if someone wants independent behavioural confirmation of long-context quality for its own sake.

## What is safe to conclude right now

- fp8 KV works on this model and saves exactly 2x (496 -> 248 MiB total; each of the two caches halves on its own); with FA on it costs nothing extra.
- Flash Attention is currently **enabled and beneficial** post-`e046294a7`; before that commit the alias-straddling split force-disabled it, which is why these numbers were never obtainable earlier tonight.
- Do not tell anyone "FA over-allocates KV on Inkling" - measured, it does not.
- Do not tell anyone "Inkling global attention caps at 8192" or "truncates at 1024" - both came from the same half-read of the allocator log, and both are false. Dense layers hold `n_ctx` cells and are never SWA-masked.
- Method rule for any iswa/hybrid model, from the above: an iswa cache prints **two** `llama_kv_cache: size =` lines (non-SWA first, then SWA). Sum both, and cross-check the total against `n_embd_k_gqa x cells x 4 B x slots` before writing a conclusion. This is the same trap as the double `KV buffer size` print noted at the top of this doc, in a new disguise.

## Artifacts

`logs/fav-{fa-off-8k,fa-on-8k,fa-off-fp8-8k,fa-on-fp8-8k,fa-on-32k,fa-on-e4m3,fa-off-e4m3,fa-on-e4m3-32k,probe}.log`. Note `logs/fav-fa-on-fp8-8k.log` and `fav-fa-off-fp8-8k.log` are the failed `fp8`-named runs kept as evidence of the argument-parsing trap. Nothing deleted.
