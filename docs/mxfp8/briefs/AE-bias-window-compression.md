# Brief AE — Make the relative-bias mask O(extent) instead of O(n_ctx): the lossless compression that unlocks 128K+

## Why this is the highest-value remaining item

`docs/mxfp8/29-compute-buffer-scaling.md` (read it fully; it is the basis for everything here) established, with a coefficient fit accurate to 3.5%:

- The context-proportional compute term is **one dense layer's materialized relative-position bias chain**, `kq_b`, living in ~3.5 copies: `ggml_get_rows` (f32) → `ggml_cont(ggml_permute(...))` (f32) → `ggml_add(kq_b, kq_mask)` (f32) → `ggml_cast(F16)` for FA. Unit `n_kv × n_ubatch × n_head × 4 B`; predicted 0.164 MiB/token vs **measured 0.1698**.
- Alternatives excluded by the same number: all 8 dense slots live = 1.31 MiB/token (7.8×), the loop's 2× = 0.328 (2.0×). **So this is not a 48-vs-24 loop bug** — my loop hypothesis was tested and disproven with arithmetic.
- Cost model: `compute ≈ n_ubatch × (8.0 MiB + 3.5·n_head·4B·n_ctx)`, giving **25.0 GiB at ctx 131072** and 46 GiB at 262144 (`-ub 512`) on a box with ~13 GiB free after 18.4 GiB of weights. Directly observed: `-c 32768 -ub 2048` requests a 20,860 MiB CUDA0 compute buffer and fails in `cudaMalloc` (`logs/ad-cb-ink-ctx32768-ub2048.log:675-678`).

## The insight to exploit (mine, derived from the lane's own code quote — verify it before building on it)

`src/llama-kv-cache.cpp:2018` computes the gather index as `distance >= 0 && distance < extent ? distance : extent`, and task #26 confirmed the semantic: beyond `n_relative_extent` the dense layer **stops distinguishing how far a key is — it still sees the key** (extent is **1024** here, not 8192 — see the correction below). Table shape is `{n_rel=16, n_extent+1=1025}` per head, projected by `Rcur` (`inkling2.cpp:357`), so:

**for every key at distance ≥ 1024, the bias is exactly the same value per (head, token) — the bucket-`extent` column.** Therefore the O(n_ctx) gathered tensor is redundant beyond the window, and an O(extent + n_ubatch) representation plus a per-head-token scalar tail is **numerically identical, not approximate**. That is what makes this a clean optimization instead of a quality tradeoff — and the burden of proof is to demonstrate that identity.

Note the interaction with the additive `kq_mask`: for dense layers the causal padding is `-INF`-style, so the *combination* must stay right too — the tail constant must still be added to the real mask, not replace it. Read `build_attn(llm_graph_input_mem_hybrid_iswa_rel *, ...)` (`src/llama-graph.cpp:4482-4512`) before deciding where the compression goes.

## Correction after dispatch (parent)

I wrote `n_relative_extent = 8192` in the brief above. **The artifact says 1024**: `inkling2_loop.attention.relative_extent = 1024`, `relative_length = 16` (that is `n_rel`, the rank of the pre-gather table), `sliding_window = 1024`, `relative_clip = 8.0`. I had conflated it with a context figure. This makes the win much larger — the constant tail begins at 1024, so ~7/8 of the gathered columns are already redundant at ctx 8192 and ~99.4% at 256K, and the per-token table itself is only `{24 heads x 1025} f32 ≈ 98 KB/token`. Predicted post-fix slope should be **near-flat in ctx**, not merely reduced. Steered to the lane on receipt.

## Work order

1. **Fill `docs/mxfp8/30-bias-window-compression.md` in your first 4 tool calls**: the claim, your verification plan, and a results table. Update as you go — nineteen lanes have timed out this session with nothing on disk.
2. **Prove the constant-tail property empirically before writing the optimization.** Small python probe or a temporary instrumented build: for a dense layer at ctx > extent, dump/derive `rel_pos` and confirm all columns beyond the window equal the bucket-`extent` value, per head and per token. If that is true, the compression is exact. If it is *not* true (e.g. the clamp only applies to SWA layers, or the sign/offset differs), **stop and report that** — the whole idea dies and that is a good outcome to know early.
3. **Implement, cheapest-first, and measure each step separately** (do not stack them):
   - **(a) f16 the chain** — the `ggml_add` result is cast to f16 one line later (`llama-graph.cpp:4510`), so `get_rows`/`cont`/`add` can be f16: 3.5 → 1.5 units (**−57%**). Caveat to quantify, not hand-wave: this changes rounding (f32 gather then cast vs f16 gather). Measure the numeric delta on the mask and confirm the `17 times 4` probe and PPL/KL are unchanged.
   - **(b) drop the `ggml_cont(ggml_permute(...))`** by letting the broadcast `ggml_add` consume the strided view (3.5 → 2.5 units, no numeric change) if `ggml` can stride that 4D broadcast — check `ggml_can_mul_mat`/the CPU+CUDA add paths for the needed stride order, and confirm no perf regression from an uncoalesced read.
   - **(c) the window compression** — restrict the gathered bias to `min(n_kv, extent + n_ubatch)` and add the clamped bucket as a per-head broadcast for the remainder. This is the one that actually kills the ctx term. With `extent = 1024`, expect the slope to fall from ~0.17 MiB/token to essentially flat: the variable part is sized by `1024 + n_ubatch`, independent of `n_ctx`.
   Land (a) and/or (b) even if (c) turns out too invasive; a −57% term is worth shipping, and (c) can be reported upstream with the numbers we generated.
4. **Re-run the lane's own sweep** with `scripts/inkling_native/ad_cbuf_sweep.py` (ctx {2k,8k,32k} × `-ub` {256,512,2048} × FA {on,off}, plus the Qwen3.5-0.8B control) and report the **fitted slope before/after** — the slope is the deliverable, not just a t/s number. Then recompute the §4 cost model and state the new maximum feasible context, and try it for real: does `-c 131072` (and if plausible `-c 262144`) now reserve and run, at `-ub 512`, with f8_e4m3 KV? Report actual pass/fail and the `sched_reserve` numbers.

## Gates (every one, with logs kept)

1. Correctness invariant: `llama-completion -m ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf -c 4096 -p "Q: What is 17 times 4? A:" -n 24 --temp 0 -no-cnv -s 42` → `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on CUDA; and on `-ngl 0` CPU (note: **CPU batched eval of Inkling nans for every quantization — task #21 — so use a short prompt, not the perplexity corpus, for the CPU check**).
2. `test-backend-ops -p mxfp8` stays **237/237 CUDA0, 237/237 CUDA1, 306/306 CPU**; `test-llama-archs` and `test-quantize-fns` still pass; and add/extend an op test that covers the **relative-bias path** if one exists (`grep -rn "rel_pos\|REL_BIAS\|kq_b" tests/`) — if the mask path has no coverage, say so, since that's how a numeric change could hide.
3. **Numerical identity or measured delta for (c):** compare post-change `kq_mask` values (or final logits via `--kl-divergence-base` on a fixed corpus) against a pre-change baseline build. For (a) report the rounding delta explicitly rather than asserting "should be fine".
4. Perf must not regress: baseline **prefill 901.28 / decode 40.36 t/s** (`-c 8192 -f logs/g0-prompt-long.txt -n 64`); the 0.8B dense control must also not regress (this touches shared graph code, so it is the real guard against Inkling-specific reasoning).
5. Commit green work separately in llama.cpp style with the fitted slope before/after in the message; `docs/mxfp8/**` and `scripts/**` stay untracked.

## Rules

- **You are the only builder.** `cmake --build build-cuda -j 22`, **never reconfigure**, tee to `logs/ae-build-<n>.log`, read the complete output.
- Fence: `src/models/inkling2.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache*.cpp/.h`, `ggml/src/ggml-cuda/*` only if needed for the strided add, `scripts/inkling_native/`, `docs/mxfp8/30-*`, `logs/`. **Do not touch `conversion/**`, `tools/mtmd/**`, `ggml/src/ggml-cuda/mmq*`** (committed and green), and leave `docs/mxfp8/23-*`/`24-*`/`29-*` alone except to read them.
- Check `git status` before editing; never revert work that isn't yours. No cherry-picks from stale branches, no rebase, never touch `master` — take *facts* from branches, structure from master (see `docs/mxfp8/18-mma-mxfp8.md` STEP 0).
- Never pin a nonzero `-ngl`; `llama-bench` is unusable for this model; KV cache type strings are `f8_e4m3`/`q8_0`/`q4_0`/`q4_1`/`q5_1`/`iq4_nl` (**`fp8` is invalid and fails at argument parsing**); read **both** iswa `llama_kv_cache: size =` lines and only the **final** `sched_reserve` pass — I made both mistakes earlier and they manufactured a phantom bug (see `docs/mxfp8/24-fa-kv-memory.md` CORRECTION).
- **Never delete a file or log.** Nothing in `~/models/timlikesai/` is deletable. No docker/systemctl.
- Honest gaps welcome: the flat ~4.0 GiB batch-proportional term is unattributed (MoE activations across 48 slots) and does *not* grow with context; if (c) lands, that flat term becomes the next wall, so name it as the follow-up rather than pretending 512K is now free.
- 60-minute budget; finalize the doc at 52. A negative result on step 2 is a complete, valuable answer — report it and stop rather than building a compression on a false premise.
