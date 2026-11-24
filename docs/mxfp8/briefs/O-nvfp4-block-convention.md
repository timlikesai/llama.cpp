# Brief O — Find the exact NVFP4 block-layout disagreement between ggml's C decoder and our repack

**Scope is one question:** for the *same bytes*, does `dequantize_row_nvfp4` (C, `ggml/src/ggml-quants.c:646`) return what our converter/python model believes those bytes mean? Everything else is already settled — do not re-litigate it.

## Settled (measured by parent; do not redo)

- Converter **values** are right: `--outtype q8_0` (dequantized, no native bytes) generates `Q: What is 17 times 4? A: 68 B: 64 C: 72 D: 70 …` correctly on CUDA (`logs/deq-q80-run.log`, converter `logs/deq-q80-convert.log`).
- Converter **row order** is right: our NATIVE `blk.0.ffn_gate_up_exps.weight` expert 0 vs the working IQ4_NL file's same tensor gives `corr = +0.9974` at **identity** row mapping, and ~0.001 for every permutation alternative (un-interleave, de-interleave, swap halves).
- MXFP8 is exonerated (device matrix: MXFP8-on-CPU output is byte-identical to all-GPU).
- Native NVFP4 works in general (`esatapedico` Qwen3.8 NVFP4 model is coherent on CUDA) — but its experts are **unfused**, so it does not exercise our fused+`.scale` combination.
- `.scale` is consumed by both backends (patching all 40 to 1.0 corrupts both).

⇒ The fault lives in the **NVFP4 block byte layout that our repack writes vs what the C decoder reads** (nibble pairing inside the 32 payload bytes, and/or which 16-element group each of `d[0..3]` covers, and/or the ue4m3 sign-stripped-vs-signed reading).

## The experiment that answers it (do this first, ~15 minutes)

Get ggml's **own C decode** of our actual file bytes, and compare it elementwise to the python model.

Cheapest route: **requantize the native file to a float type with the built tool** — `./build-cuda/bin/llama-quantize ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf <out> q8_0 --tensor-type ...` or simply `f32`/`bf16` — and see whether the tool accepts an NVFP4 source. Whatever lands, decode the *same* expert row in python and compare per-element; report the first index where they differ and what the two interpretations are (e.g. C reads byte j as (elem j, elem j+8) vs python's (2i, 2i+1)).
If the tool refuses to requantize NVFP4, fall back to: build a 1-tensor GGUF holding **one known expert row** (write the header by hand with gguf-py, or use `llama-gguf`), plus a ~20-line C program linked against `build-cuda/bin/libggml*.so` that calls `ggml_row_*`/`dequantize_row_nvfp4` and prints the 64 floats. Compile with `cc -I ggml/include -L build-cuda/bin -lggml -lm`. A tiny C probe is completely acceptable here and is the most direct evidence.

**Second, confirm against the true source of meaning:** the checkpoint's own semantics in `conversion/inkling2.py:_dequant_expert_chunk` (`values[:, 0::2] = _E2M1[packed & 15]; values[:, 1::2] = _E2M1[packed >> 4]`, then `reshape(rows, -1, 16) *= block_scale`). Whatever the C decoder expects must be *derived*, not assumed — read `dequantize_row_nvfp4` and `ggml_cuda_mmq_load_tiles_nvfp4`/`vec_dot_nvfp4_q8_1` and state the convention each one implements, in words, side by side.

## Then fix

- If our pack is wrong: correct `conversion/inkling2.py` (`_pack_nvfp4_chunk`), regenerate the NATIVE file (`--native-experts`, ~50 s) and re-run the probes. Note `scripts/inkling_native/verify_nvfp4_gguf.py` and `layout.py` encode the SAME (possibly wrong) assumption and cross-validated against unsloth's file — so "verifier passes" is not evidence of correctness here; the verifier must be updated to assert agreement with the **C** decoder instead.
- If our pack matches C but the MoE/fused path is wrong: write the minimal `src/llama-graph.cpp` or `ggml-cuda` patch into the doc, and say precisely which assumption is violated; the parent lands it.

## Gates (all with logs kept, nothing deleted)

1. `A: 68`-style correct output from `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` on **CUDA auto-fit** (never pass an exact `-ngl`; it disables memory fitting and OOMs), and on CPU too, for the 4 prompts in `scripts/inkling_native/probe_device_matrix.sh`.
2. CPU-vs-CUDA agreement.
3. `test-backend-ops -b CUDA0 -p mxfp8` 237/237 and `-b CPU -p mxfp8` 306/306 still green.
4. Prefill and decode t/s for the working lossless file vs G0 (589.72 / 21.94 long / 32.00 short).
5. Update `docs/mxfp8/15-inkling-native-expert-fix.md` §2-§5 (currently scaffolded with `_TBD_`) with the real measurements.

## Rules

- Write into `docs/mxfp8/15-inkling-native-expert-fix.md` in your first 4 tool calls and keep updating; three siblings died at the timeout having written nothing.
- Build with `cmake --build build-cuda -j 22` only (never reconfigure); tee to `logs/`; read full output.
- Files you own: `conversion/inkling2.py`, `scripts/inkling_native/`, `docs/mxfp8/15-*`, `logs/`, and — if your diagnosis proves the fault is there — `src/llama-graph.cpp` / `src/llama-model.cpp` / `ggml/src/ggml-cuda/**` **except** `mmq*` and `template-instances/**` (another agent may be adding the MXFP8 mmq fast path; check `git status` before editing shared files and never revert someone else's work).
- No docker/systemctl. Both GPUs are free. Never delete files or logs. Delete no model file you did not create. 35-minute budget; leave the tree building.
