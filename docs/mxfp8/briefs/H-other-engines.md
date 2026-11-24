# Brief H — Is there already an MXFP8/NVFP4 implementation on this box (Strata, vLLM, others)?

**Deliverable:** write `/home/tim/code/services/repos/llama.cpp/docs/mxfp8/09-other-engines-mxfp8.md`.
Read-only except that one file. **Do not build anything, do not start servers, do not touch docker.** Print a 12-line summary to stdout. Cite `file:line` for every claim.

## Why

Before we write MXFP8 support in llama.cpp we must know whether proven implementations already exist on this machine that we can read as reference (or, in the worst case, use as a stopgap to run the model at all). Fleet notes say the boxes moved to an engine called **Strata** around 2026-11-13, and that a `vllm` checkout also lives here.

## Where to look

- `/home/tim/code/services/repos/Strata` — a llama.cpp-derived engine? Check: `git log --oneline -15`, `git remote -v`, README/docs, whether it is a fork of llama.cpp (and how far from upstream), and crucially whether it has **MXFP8 / NVFP4 / FP8 block-scaled** support: grep the tree for `MXFP8`, `mxf8f6f4`, `ue8m0`, `e8m0`, `NVFP4`, `block_scale`, `F8_E4M3`, `inkling` (case-insensitive). Report what each hit means (type def? kernel? converter?) and whether any of it is *weight* support (not KV cache).
- `/home/tim/code/services/repos/vllm` — same greps, plus: which MXFP8/NVFP4 GEMM kernels ship for **consumer Blackwell (sm_120)** vs datacenter (sm_100) — vLLM has CUTLASS/TRTLLM/FlashInfer backends; find the MXFP8 (`e4m3` + `e8m0`, block 32) path: kernel entry points, how it lays out the scale tensor (`e8m0` swizzle/granularity 1x32), whether it uses `mma...kind::mxf8f6f4` or a CUTLASS blockscaled collective, and whether it needs 128x4 scale swizzling (that detail matters a lot for us: MXFP8 in some stacks requires a *swizzled* scale layout, whereas llama.cpp GGUF wants linear 1×32 per row — if vLLM's kernel needs swizzle, note it and how it converts).
- The Inkling checkpoint's producer side: `pip show nvidia-tensorrt-model-optimizer modelopt 2>/dev/null`, `python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`, and search site-packages for a `modelopt`/`inkling2` quant config consumer, so we know the **canonical** byte layout semantics for `weight_block_shape [32, 32]` and `amax_denominator` (what does the reference code do with them? if the reference dequant is available locally, cite its file:line — that is our ground-truth for Phase 2's verifier).
- Any other local engine with fp8 weight support: `/home/tim/code/services/repos/` listing, `llama-cpp-entrypoint.sh`, `/home/tim/code/services/docs/*`, `~/.cache/huggingface/hub/models--*` (is there a *different* Inkling packaging, e.g. an `Inkling-2-Mini-NVFP4`/`-MXFP8` repo cached anywhere, or only the base + GGUFs?).
- Check the **remote LM Studio box** is irrelevant (192.168.68.79) — do not ssh; just note in one line that any stopgap there is out of scope.

## Also answer

1. Could **Strata** run this checkpoint today (does it have an `inkling2` model + a type for these experts)? If yes, that is a fast path to a *reference implementation of the architecture* — but state clearly it does not satisfy "llama.cpp master".
2. Which engine has the most complete, readable **MXFP8 weight + FP8-activation GEMM** on sm_120, and what exactly should we port from it (list the 2–4 functions/kernels with file:line and a one-line "why this is the reference").
3. Any sign of **upstream llama.cpp** MXFP8 work we should track instead of duplicating (search this box's cloned llama.cpp for stale patches/branches mentioning MXFP8 *weights*, and note if a `GGML_TYPE_MXFP8` id was ever used locally so we don't collide).

## Output shape

`## Bottom line` (3 bullets), `## Strata`, `## vLLM`, `## ModelOpt/reference dequant`, `## Other`, `## What to port`, `## What NOT to reuse` (with reasons). Dense; no filler.
