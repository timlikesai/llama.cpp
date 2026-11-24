# Brief Z — Attribute the sync storm, then the FA/KV memory question

Two jobs, both already scoped by measurement. You are the only active builder (a python-only KLD lane is running and does not compile), so `cmake --build build-cuda -j 22` is yours — never reconfigure, always tee to `logs/`, read the full output.

## Job 1 (primary): find *which op* causes ~7,000 `cudaStreamSynchronize` calls per run

Established by the parent from the nsys capture a previous lane left unanalyzed (`docs/mxfp8/22-prefill-perf.md` has the full table — read it first):

- `cudaStreamSynchronize` = **64.8% of CUDA API time, 4.457 s across 6,991 calls** (avg 638 µs, max 97 ms).
- `cudaMemcpyAsync` = 1,493 calls / 1.201 s, median 3 µs but **max 61 ms**.
- `cudaGraphInstantiate` = **429 calls** while the run reports `graphs reused = 94 / reallocs = 0`.
- Consequence: prefill ~506 t/s ≈ 3.2 TFLOP/s and ~4.4 GB/s against tens of TFLOP/s and ~900 GB/s on two 5060 Tis. **The device is idle waiting, not busy.**

Evidence already on disk — reuse it, don't recapture blindly: `logs/w-nsys-prefill.nsys-rep` (4.3 MB), `logs/w-nsys-prefill.sqlite`, `logs/w-nsys-kern-sum.txt`, `logs/w-nsys-api-sum.txt`.

Start with:
```
nsys stats --report cuda_api_trace --report cuda_gpu_trace logs/w-nsys-prefill.nsys-rep
```
and correlate: what kernel/API pattern repeats per layer, and what the syncs sit between.

Already ruled out (do not redo): **MoE cache is not it** — `--moe-cache-mib` defaults to 0/disabled and prefill is identical with it explicit (506.26 vs 509.17 t/s; `logs/x-sync-{default,nocache}.log`).

Ranked hypotheses to test, cheapest first:
1. **`ggml_cuda_mul_mat_id_needs_sync` returning true for our types** (`ggml/src/ggml-cuda/ggml-cuda.cu`, `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`). The function is named for exactly this symptom; if it says "needs sync" for `MUL_MAT_ID` with block-scaled weights or with `ids`, MoE falls out of graph capture and syncs per application. Read it, then instrument (temporary counter behind an env var, removed before you finish) to get a per-op sync count.
2. **Does the count scale with layer *applications* (48) or layers (24)?** This model shares physical layers (`src/models/inkling2-loop.cpp` `post_arch_tensors`: `layers[il] = layers[layer_phys(il)]`). 6,991 syncs / 2 batches of prefill is ~3500/batch — compare against 48 MoE applications to see whether the loop replay multiplies the cost. Same arithmetic for decode.
3. **The per-expert `.scale` path** (`src/llama-graph.cpp:1904-1911`: `reshape_3d → repeat_4d → get_rows(ids) → ggml_mul` per MoE layer) — check whether `get_rows` on a host-resident or CPU-pinned tensor, or the `repeat_4d`, forces a sync or an upload (the 1,493 `cudaMemcpyAsync` calls are unexplained; 1,493 ≈ 31 × 48, worth checking against layer count).
4. **`cudaMemGetInfo` 250 calls** — who polls device memory that often? If it's inside a per-layer path that is a stall of its own.
5. Graph instantiate ×429 — figure out whether graphs are being rebuilt per ubatch rather than reused.

Deliver a **fix or a precise upstream report**. If the fix is in `src/llama-graph.cpp` / `src/models/inkling2-loop.cpp` (outside your fence), write the exact patch into the doc for the parent to land.

**Gates for any change**: prefill and decode t/s before/after on the fixed harness (`llama-completion -m ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf -c 8192 -f logs/g0-prompt-long.txt -n 64 --temp 0 -no-cnv -s 42`, baseline 506 / ~22), re-run the same nsys report and show the sync count/time **dropping** — that's the metric, not vibes. Correctness invariant: `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on CUDA, and `test-backend-ops -p mxfp8` 237/237 CUDA0+CUDA1, 306/306 CPU. **Never pin a nonzero `-ngl`** (disables fitting, OOMs); `llama-bench` is unusable here (no fitter, `--tensor-split` defaults to 0).

## Job 2: Flash Attention memory (task #22 — Tim's report, likely the looped layers)

Users report enabling FA uses **more** memory than an f16 KV cache. Measure the matrix **{FA off, FA on} × {cache f16, cache fp8} × {ctx 2k, 8k, 32k}**, reading per-device KV and compute-buffer bytes from the load log, against theoretical KV size (Inkling: `full_attention_interval 4` → 20 SWA layers @ `sliding_window 1024` + 4 full-attention; `head_dim_ctl 128`, `head_count_ctl [16,8]`; 48 layer *applications*). Check:
1. KV allocated per applied layer (48) instead of per unique layer (24) when FA is on — the expansion Tim suspects.
2. Whether `sliding_window` is honoured in the FA path or degrades to full-length rows.
3. FA's own workspace/scratch allocations vs the non-FA path.
4. Whether shared loop slots still get separate KV state in FA's allocator.
Context: the 2-card split currently disables FA outright (`layer 24 assigned to CUDA1 but FA assigned to CUDA0`), so to measure FA at all you may need a single-card config with `--override-tensor` or a smaller ctx — document what you had to do.

## Rules

- Fence: you own `ggml/src/ggml-cuda/**`, `scripts/inkling_native/`, `docs/mxfp8/22-*` (append), `docs/mxfp8/26-sync-attribution.md`, `logs/`. `src/**`, `conversion/**`, `tools/**`, `tests/**`, `common/**` are read-only — propose patches in the doc.
- **Never delete a file or log.** Do not touch anything in `~/models/timlikesai/`. No docker/systemctl. A python KLD lane may be using the GPU — if numbers wobble, say so and re-run.
- No cherry-picks from stale branches, no rebase, never touch `master`. Prior art in `docs/mxfp8/02-prior-art-branches.md`, `docs/mxfp8/ref/`, and branches `mxfp-fa-cuda` / `mxfp4-flash-attention-v3` / `mxfp4-fused-moe` (`d65231c7e` "fused GLU is 7% faster than unfused") are relevant to both jobs — take **facts**, follow **master's conventions** (see `docs/mxfp8/18-mma-mxfp8.md` STEP 0, where comparing beat porting).
- Write `docs/mxfp8/26-sync-attribution.md` within your first 4 tool calls with this data in it, and update after every measurement. Thirteen lanes have timed out this session with nothing on disk; that is the failure mode to avoid above all. Dead hypothesis + number = success.
- 50-minute budget; finalize at 42.
