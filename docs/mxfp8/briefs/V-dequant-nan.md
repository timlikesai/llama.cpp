# Brief V — Diagnose (and if cheap, fix) why `Inkling-DEQ-Q80.gguf` emits non-finite logits

## Why this matters more than a broken side artifact

This file is the **positive control** I used to localize tonight's bug: I concluded "our converter's values are correct because the dequantized build answers `Q: What is 17 times 4? A: 68`" and used that to exonerate the converter and blame the native byte path. It generates that string fine — and it produces **nan perplexity on all 8 WikiText-2 chunks**, on two independent code paths (`perplexity` and `perplexity_v2`/`--kl-divergence-base`), and nan as a *candidate* against a known-good base (lane U's evidence: `docs/mxfp8/20-ppl-kl-validation.md` §2.1, §3.3). So either the control was partly bogus, or there's a real defect. Either way the project's stated numeric reference is currently unusable, and the direct `NATIVE == dequantized` KL has therefore never been measured.

## What the parent already ruled out (do not redo)

- **Not a dtype-policy problem.** Tensor-type inventory comparison against the healthy public file: `DEQ-Q80` = 478 F32 + 226 Q8_0, `IQ4_NL` = 478 F32 + 186 Q8_0 + 40 IQ4_NL. **F32-in-IQ4_NL-but-not-in-DEQ-Q80 = 0 tensors.** The two files differ in exactly one way: the 40 expert tensors are `Q8_0` there vs `IQ4_NL` here (226 − 186 = 40).
- **Not the native repack.** `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` is healthy: PPL **5.5757 ± 0.1442**, self-KLD 0 (residual 7e-5 = the u16 logit downcast), same-top-p 100%, and it answers the 4-prompt suite correctly on CPU and CUDA.
- **Not quantization quality** — a wrong scale convention reads as PPL 5.26e6 (the `scale1` control), not nan.

So the search space is: **(a) the dequantized expert values are inf/nan for some blocks, or (b) the `Q8_0`-as-3D-`MUL_MAT_ID`-weight path is broken** (a combination IQ4_NL doesn't exercise), or **(c) something about this file's size/CPU spill** (27.8 GB vs 18 GB; auto-fit gives layer 24 to CUDA1, which also disables Flash Attention per §5 of doc 20).

## Bisection plan (cheapest discriminator first)

1. **Offline value scan of the experts — do this first, it settles (a) in one shot.** Read the 40 `ffn_*_exps.weight` tensors from `~/models/timlikesai/Inkling-DEQ-Q80.gguf` with `gguf-py`, dequantize (`gguf.quants.dequantize` — note `ReaderTensor.data` is a *byte* view for quantized types and for a 3-D tensor the array is `(E, R, byte_cols)`; a wrong reshape gave me 128× element counts twice this session, so assert the shape rather than trusting it), and count non-finite values per tensor. Also scan the F32/BF16 tensors for inf/nan. Report which tensors/layers are affected — in particular whether it's the **ex-MXFP8 layers (blk.20-23)** specifically: their dequant is `e4m3 × ue8m0`, and `e = 254/255` is non-finite by convention, so if the checkpoint contains such a scale byte (or the dequant helper mishandles it) that group is the culprit while the NVFP4 layers stay fine.
2. **Cross-check the source of truth.** For any tensor that shows non-finite values, recompute the same block from the HF checkpoint via `conversion/inkling2.py:_dequant_expert_chunk` and compare, to say whether the bug is in the dequant formula or in what got written.
3. **Test (b) by requantizing:** `llama-quantize` DEQ-Q80 (or a slice) to a type that keeps experts 2-D-friendly and re-run the tiny PPL probe. If experts-as-`IQ4_NL` is finite while experts-as-`Q8_0` is nan *with finite source values*, that's a runtime/backend bug in Q8_3D `MUL_MAT_ID` — write the reproducer up, don't hack around it in the converter.
4. **Test (c):** run the same PPL probe with all experts forced to CPU (`--override-tensor`), and with `-sm`/tensor-split variations, and note whether the nan follows the device.

## Measurement rules (learned the hard way this session)

- **`llama-perplexity` footgun: `-b` must equal `-c`.** `-c 512` silently becomes a 4-slot 2048-token context (`perplexity.cpp:2012` computes `n_parallel = n_batch / n_ctx`, `:2014` multiplies back). Use the existing driver `scripts/inkling_native/ppl_kl_u2.sh` (working, NATIVE-base chain) rather than hand-rolling flags.
- Corpus is already downloaded under `logs/corpora/` — reuse it, don't refetch. A short probe (a few hundred tokens) is enough to distinguish finite from nan.
- **Never pass a pinned nonzero `-ngl`** — it disables memory fitting and OOMs this 2-card setup. Use auto-fit or `-ngl 0`.
- Interpret nan carefully: nan that appears only in *some* chunks/tensors points at data; nan everywhere points at the path.

## Fences and hard rules

- **NO BUILDS.** Another agent is compiling `build-cuda` and editing `mmvq.cu`/`vecdotq.cuh`/`mmq*`. Use existing binaries + python only. If a C++ fix is indicated, write the exact patch into your doc and stop.
- You may edit: `conversion/inkling2.py` (python only — a converter fix needs no build) and create files under `scripts/inkling_native/`, `docs/mxfp8/21-*`, `logs/`. Do **not** touch `src/**`, `ggml/**`, `tools/**`, `tests/**`, `common/**`.
- If you change `conversion/inkling2.py`, you must **not** disturb the native path: after any edit, regenerate with `--native-experts` into a scratch file and confirm its inventory is still 40 NVFP4 + 8 MXFP8 + 178 BF16 + 518 F32 = 744 and that `scripts/inkling_native/verify_mxfp8_gguf.py --gguf <scratch>` exits 0. Keep `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` untouched.
- **Never delete a file or log** — including the 3.27 GB `logs/corpora/u-kld-*.bin` base files and the 27.8 GB DEQ-Q80 file. Disk is ~250 GB free; if you need space, ask the parent rather than deleting.
- Write `docs/mxfp8/21-dequant-nan.md` within your first 4 tool calls (this context + your step-1 result) and update as you measure. No invisible success: nine delegated tasks this session timed out having written nothing.
- 35-minute budget; finalize at 30.

## Definition of done

Root cause named with evidence (which tensors/blocks, data vs path), the direct `NATIVE vs dequantized` KL either measured or declared unmeasurable with a reason, and — if a converter fix was made — the regenerated reference validated and `docs/mxfp8/20-ppl-kl-validation.md` §7's caveat updated. If it turns out the control's conclusion was wrong, say that plainly; tonight's diagnosis rests on `690b87d88` + the PPL/KL table, so being told the real answer matters more than being told the convenient one.
