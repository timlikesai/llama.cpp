# Brief T — Phase 4c: MXFP8 decode route (`mmvq`) so single-token MoE stops paying the vector-dequant cost

## Why

Prefill is now on tensor cores (`c39ab0a00`, +4.6%: 484.50 → 506.95 t/s), but **decode did not move: 22.06 → 21.95 t/s.** Single-token `MUL_MAT_ID` over 128 experts — which is what Inkling actually is (top-8 of 128, 24 layers) — goes through `mmvf` with on-the-fly dequantization, because MXFP8 was never added to `mmvq`. Decode t/s is the number that matters for interactive use, so this is the highest-value remaining perf item.

**The hard constraint you must design around:** `mmvq`'s dot is built on `ggml_cuda_dp4a` over **int8 pairs**. An e4m3 payload has no integer form, so you cannot reuse the dp4a trick that makes MXFP4 cheap there. Look at how `mmvq.cu`/`vecdotq.cuh` handle types that are not dp4a-able (e.g. `vec_dot_nvfp4_q8_1` at `vecdotq.cuh:334`, and the f16/half2 dots) and follow that pattern: decode e4m3 → half2 with `__nv_cvt_fp8x2_to_halfraw2` and accumulate in f16x2/f32.

## Establish the baseline before optimizing anything

Measure first, so a "win" is real:
1. Current numbers: `llama-completion -m ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf -c 8192 -f logs/g0-prompt-long.txt -n 32 --temp 0 -no-cnv --seed 42` → prefill **506.95**, decode **21.95** (`logs/r-perf-auto.log`).
2. Which kernel serves decode today? Read `ggml-cuda.cu`'s MUL_MAT_ID dispatch and confirm it's `mmvf`. Note: **`GGML_CUDA_FORCE_MMQ` is not read by the CUDA backend on current master** — do not trust env-var A/Bs that assume it is; prove routes behaviourally (the mmq route, for instance, shifts `type_b=f32` NMSE from <5e-4 to ~7e-4 because activations are quantized to e4m3).
3. Sanity target: the lossy IQ4_NL reference decodes at **23.62 t/s** — that's the parity line for a bit-exact file.

## Reference material (already extracted — do not re-derive)

- `docs/mxfp8/02-prior-art-branches.md` — the block-scaled MMA facts from the stale branches (`kind::mxf8f6f4 … e4m3.e4m3`, `sa >> 8*frag` scale packing, `tidx_A = g + (t%2)*8`, arch guard + cuBLAS fallback policy).
- `docs/mxfp8/ref/` — 8 code excerpts from `mxfp68-quant` / `mxfp4-soa-model-weights`.
- `docs/mxfp8/18-mma-mxfp8.md` — **STEP 0's lesson: master reuses tile geometry; COMPARE before writing new loaders.** The same discipline applies here — check whether `mmvq` already has a non-dp4a template slot you can instantiate rather than a new kernel.
- Type facts: `GGML_TYPE_MXFP8 = 44`, `block_mxfp8 { uint8_t e; uint8_t qs[32]; }` = 33 B / 32 values, `QK_MXFP8 32`, `QR_MXFP8 4`, `QI_MXFP8 2`.

## Critical correctness trap

Scale must be `ggml_cuda_e8m0_to_fp32(e)` with **no `* 0.5`** — the half in the MXFP4 dots exists only because `kvalues_mxfp4` are doubled; e4m3 payloads are not. A stray half is a silent 2× that the model's text output will reveal as garbage, not as an error.

## Fences

- You own: `ggml/src/ggml-cuda/mmvq.cu`, `mmvq.cuh`, `vecdotq.cuh`, `mmv.cuh` if needed, `dequantize.cuh` **read-only for reference**, `ggml-cuda.cu` **only** the `MUL_MAT_ID`/`mmvq` dispatch decision, and `docs/mxfp8/19-decode-mxfp8.md`.
- Do NOT touch: `conversion/**`, `scripts/**`, `src/**`, `tests/test-backend-ops.cpp` (its MXFP8 tolerances are already set), `mmq*` (just landed and green — if you believe a `mmq` change is needed, write it up instead of editing).
- A parallel python-only lane may be measuring perplexity; it won't build. **You are the only builder**: `cmake --build build-cuda -j 22` (never reconfigure), tee to `logs/t-build-<n>.log`, read the complete output.

## Gates

1. `test-backend-ops -b CUDA0 -p mxfp8` and `-b CUDA1 -p mxfp8` still **237/237**; `-b CPU -p mxfp8` still **306/306**.
2. Correctness unchanged on the real file: `Q: What is 17 times 4? A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on CUDA **and** CPU (`-ngl 0`), plus `The capital of France is → Paris.` **Never pass an exact `-ngl` other than 0** — any pinned nonzero value disables memory fitting and OOMs this 2-card setup.
3. **Decode must actually improve**, measured with the command in "Why" step 1 at the same context/prompt: report prefill and decode t/s before/after. If the new route is not faster, **say so and leave it unselected** — a documented negative result is a success; an unjustified dispatch change that regresses is not.
4. State which kernel executed and how you know (behavioural signature, a temporary instrumented log you then remove, or a route-specific numeric difference).
5. Commit only if green: `ggml-cuda : add an MMQ/MMVQ route for MXFP8 decode` style, with the measured numbers in the message. `docs/mxfp8/**` and `scripts/**` stay untracked.

## Process rules (these have bitten us all session)

- **Write `docs/mxfp8/19-decode-mxfp8.md` within your first 4 tool calls** (baseline numbers, the dispatch finding, your hypothesis) and update it as you measure. Seven delegated tasks this session timed out having written nothing; that is the failure mode to avoid above all others.
- **No cherry-picks from the stale branches, no rebase, never touch `master`.** Adapt by hand to current conventions and cite provenance in a comment.
- If a task turns out to be a dead end (e.g. no non-dp4a slot exists and a new kernel won't beat `mmvf`), record exactly why in the doc — that's a durable answer and stops the next agent re-trying it.
- No docker/systemctl. Both GPUs free (the `llama-cpp` container was removed on Tim's order). **Never delete a file or log.** `~/models/timlikesai/` holds the artifacts — do not delete anything there.
- 45-minute budget; finalize the doc at 38; leave the build green.
