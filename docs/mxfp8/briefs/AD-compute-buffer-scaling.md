# Brief AD — Where does the ~0.174 MiB/token **compute-buffer** growth come from? (Suspect: per-applied-layer allocation in the loop model)

## Read this first: the previous question in this area was answered wrong by me

I reported that "KV is flat from 8k to 32k" and built an anomaly around it. **That was a measurement error**: an `iswa` cache (dense + sliding-window layers) prints **two** `llama_kv_cache: size =` lines and I summed one. The corrected numbers are in `docs/mxfp8/24-fa-kv-memory.md` (see its CORRECTION section) and task #26 is settled: dense layers are allocated at full context width, never window-masked, and `relative_attention_extent` is only the width of the learned relative-bias table (beyond it distances clamp into one extra learned bucket — the key is still seen).

**Read `docs/mxfp8/24-fa-kv-memory.md` before anything else**, then verify the numbers yourself from the logs rather than trusting either of us.

## The term that IS real and unexplained

Measured from `logs/fav-fa-on-8k.log` and `logs/fav-fa-on-32k.log` (`sched_reserve`, final pass, summed over CUDA0+CUDA1+CUDA_Host), FA **on**, `-b 2048`, `n_seq 1`:

| ctx | KV (f16) | compute total | CUDA0 compute | CUDA1 compute | Host compute |
|---|---|---|---|---|---|
| 8192 | 496 MiB | **5,479 MiB** | 1,434.77 | 3,907.77 | 136.59 |
| 32768 | 1,264 MiB | **9,652 MiB** | 4,170.77 | 5,056.75 | 424.59 |

**+4,173 MiB over +24,576 tokens = ~0.174 MiB/token (≈182 KB/token).** With Flash Attention enabled there should be no context-proportional per-layer scratch — avoiding exactly that is FA's purpose. So this term is the anomaly, and unlike KV it is **not** affected by cache type, which makes it the actual obstacle to 128K/256K/512K.

Note also CUDA1's compute (3,908 MiB) is already 2.7x CUDA0's at 8k, and the host term tripled (137 -> 425 MiB). Those asymmetries are clues, not noise.

## Tim's hypothesis, which I share

It's the **looped memory**: something sized per **applied** layer (48) instead of per **unique** layer (24). This is the same failure class we just fixed for device placement (`e046294a7`, where code iterated `n_layer_all` without consulting `layer_phys`) — and there is independent evidence the loop inflates per-slot things by exactly 2x: the dense KV cache reports **8 layers for 4 unique dense layers** and the SWA cache **40 slots for 20 unique**. That KV doubling is *correct* (each application keeps its own state). A matching compute-buffer doubling may not be.

## Do the arithmetic before the code

Fit the coefficient analytically against candidate buffer formulas, so you can name the winner instead of guessing:

| candidate | per-slot bytes/token | x48 slots | vs measured 182 KB/tok |
|---|---|---|---|
| f16 vector of `n_embd` 2304 | 4,608 B | 221 KB | **~1.2x — closest match so far** |
| f32 vector of `n_embd` | 9,216 B | 442 KB | 2.4x too big |
| f16 scores row, `n_head` 24 | 48 B | 2.3 KB | 80x too small |
| f16 K/V for one dense slot (8 kv heads x 128) | 4,096 B | — (already in KV) | not compute |

So the shape of the winner looks like **roughly one f16 `n_embd`-wide buffer per applied layer, context-proportional**. Where could a context-proportional `n_embd`-wide per-layer buffer live when FA is on? Candidates to check in order:
1. `attn_res` / controller path (Inkling's looped layers run down/gain/forget controllers over `n_embd`; `n_shortconv_l_cache`, `loop_k_ctrl_*`) — check `src/models/inkling2*.cpp` graph build for any `ggml_new_tensor` or `dup` whose size depends on `n_ctx`.
2. `llama-graph`'s `cparams`-derived buffers: `src/llama-memory*.cpp`, `llama-kv-cache-iswa.cpp` (the dense/SWA split, `n_stream`), and the **relative-bias gather**: `ggml_pad(layer.attn_rel_bias, 0, 1, 0, 0)` gathered per token — a `{n_rel, n_extent}` table gather for 8192-wide extent is context-independent, but a per-token materialization of gathered bias would be O(tokens), not O(ctx). Distinguish those two: they have different signatures (batch-proportional vs context-proportional). **A key experiment: vary `-b` at fixed ctx, and ctx at fixed `-b`, and see which axis the term follows.** If it follows ctx at constant batch, it is an allocation, not a scratch buffer.
3. `ggml` cgraph node memory that can't be reused across the 3 passes of a replayed layer because the aliasing makes buffers look live simultaneously (this would also explain the CUDA0/CUDA1 asymmetry via `resolve_fused_ops`).
4. Graph capture buffers: `GGML_CUDA_GRAPH_CHUNK_SIZE`, `USE_GRAPHS=1`, and the `graphs reused / reallocs` counters.

## Method

1. **Write `docs/mxfp8/29-compute-buffer-scaling.md` in your first 4 tool calls** with this table and your candidate ranking; update as you measure.
2. **Sweep** ctx `{2k,4k,8k,16k,32k,64k}` x `-b {512,2048}` with `n_seq 1`, FA on, and record per-pass `sched_reserve` totals per device + both `llama_kv_cache: size =` lines + `llama_memory_recurrent: size`. From that, fit whether the term is linear in ctx, in batch, or in both — **this single experiment discriminates allocation vs scratch and is the highest-information thing you can do.** Parse only the final `sched_reserve` pass and **both** KV lines (the two mistakes I already made).
3. **Control on a non-looping model** of similar dims so you know what "normal" is: `~/models/timlikesai/Qwen3.5-0.8B-Q8_0.gguf` and, if usable, a non-loop Inkling build is not available — so compare per-layer-per-token coefficients instead of absolutes, and say how you normalized.
4. **Name the allocation.** Use the `-v` reserve logs; if they can't attribute it, add a *temporary* instrumented print behind an env var in the reserve path, capture the evidence, then **remove it**. `nsys`/`ncu` are installed if you need allocation timelines (`logs/w-nsys-prefill.nsys-rep` shows the earlier methodology).
5. **Fix or report.** If it is a 48-vs-24 sizing miss, a minimal alias-aware patch (analogous to `layer_phys`) with before/after numbers is the goal. If it is inherent (each replayed slot genuinely needs its buffer), then document the true cost model for long context and say plainly that 256K/512K needs a different lever — that's a valuable negative and I'd rather have it than a fabricated fix.

## Gates / rules

- Any change must keep: `Q: What is 17 times 4? A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on CUDA **and** `-ngl 0` CPU; `test-backend-ops -p mxfp8` 237/237 CUDA0+CUDA1 and 306/306 CPU; prefill/decode not regressed below **901 / 40 t/s** (harness: `-c 8192 -f logs/g0-prompt-long.txt -n 64 --temp 0 -no-cnv -s 42`).
- **Builder access:** one other lane is running but it is forbidden from building, so `cmake --build build-cuda -j 22` is yours. **Never reconfigure.** Tee to `logs/ad-build-<n>.log` and read the complete output.
- Never pin a nonzero `-ngl` (disables fitting, OOMs on this 2-card box). `llama-bench` is unusable for this model. Cache type string is `f8_e4m3`, never `fp8`. `llama-perplexity` needs >= 2x ctx tokens.
- You own: `src/**`, `ggml/src/**` (propose, don't spray — one focused change), `scripts/inkling_native/`, `docs/mxfp8/29-*`, `logs/`. **Stay out of `conversion/**` and `docs/mxfp8/23-mtp-draft.md`** — the other lane owns MTP packaging there.
- No cherry-picks from stale branches, no rebase, never touch `master`. Check `git status` before editing; never revert work that isn't yours. **Never delete a file or log**; nothing in `~/models/timlikesai/` is deletable. No docker/systemctl.
- 50-minute budget; finalize the doc at 42. Naming the allocation with evidence is a complete success even with no patch.
