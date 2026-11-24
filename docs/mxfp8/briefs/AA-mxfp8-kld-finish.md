# Brief AA — Finish the isolated MXFP8 KLD study: produce ONE valid MXFP8 GGUF, then measure

**Scope is deliberately small.** A previous lane timed out after doing most of the setup; do not redo what already exists.

## Already done — reuse, verify, don't regenerate

| artifact | state |
|---|---|
| `~/models/timlikesai/Qwen3.5-0.8B-Q8_0.gguf` | **valid** — 335 tensors, 822.6 MB data, F32 x140 + Q8_0 x195 |
| `~/models/timlikesai/Qwen3.5-0.8B-MXFP4.gguf` | present (423 MB) — verify it loads before using |
| `~/models/timlikesai/Qwen3.5-0.8B-MXFP8-C.gguf` | **BROKEN — `GGUFReader` raises `ValueError: GGUF magic invalid`.** A truncated/aborted write. Do not try to repair it; write a fresh file under a new name and leave this one (never delete) |
| bf16 source | `/home/tim/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF/snapshots/004fd086eccea9f7cf867aa07ba328773beef81d/Qwen3.5-0.8B-BF16.gguf` (1,557,662,496 B) |
| harness doc | `docs/mxfp8/25-mxfp8-isolated-kld.md` — skeleton with the claim, the table, and §1's checklist. **Fill that doc in place; it is the deliverable** |
| drivers | `scripts/inkling_native/ppl_kl_u2.sh` (working KL/PPL flag pattern), `scripts/inkling_native/layout.py:114 pack_mxfp8_blocks`, `tests/test-mxfp8-repack.py` (decoder oracle) |
| corpus | `logs/corpora/` has `wiki.test.slice64k.raw` etc. Reuse; do not refetch |

## Why it isn't a C++ job today

`llama-quantize` **cannot** emit MXFP8 — `LLAMA_FTYPE_MOSTLY_MXFP8 = 43` exists but `src/llama-quant.cpp` and `tools/quantize/quantize.cpp` have no MXFP8 path (filed as task #23). **You must not build** — another lane owns `build-cuda`. So: python packer, and if you conclude the C++ wiring is the better route, write the exact patch into the doc instead of applying it.

## Job 1 — write ONE valid MXFP8 GGUF, and prove it before measuring

1. Read the bf16 source with `gguf-py`, take 2-D weight tensors, quantize per 32-value block to `{uint8 e /*ue8m0*/; uint8 qs[32] /*e4m3*/}` (33 B/block, `v = e4m3 * 2^(e-127)`); keep 1-D tensors and norms F32; choose and **state** the scale rule (`ceil(log2(amax/448))+127` is the OCP convention — record what you use and assert every payload lands in e4m3 range).
2. Write with `GGUFWriter`: copy source metadata, set `general.file_type = 43`, tensor type **44**. Mind the traps that have produced wrong results three times this session: `ReaderTensor.data` is a **byte view** for quantized types, `GGUFReader.tensor.shape` is `(C, R, E)` — reversed from numpy order — and a quantized row length must go through `quant_shape_from_byte_shape`. **Verify size sanity: ~450-500 MB for this model. If you write something an order of magnitude off, stop and fix the reshape.**
3. **Self-check gate (do not skip, do not measure before it passes):** (a) `GGUFReader` opens the file; (b) decode your own blocks and compare to the bf16 originals — report max/mean per-element relative error (e4m3 expectation: ~6% worst case) and RMS; (c) requantizing the decoded values reproduces your bytes (idempotence); (d) **cross-check the python decoder against the C decoder's semantics** using `tests/test-mxfp8-repack.py` as oracle. If python and C disagree on any block, that is the headline finding — write it up immediately and stop the measurement work.
4. Confirm the runtime accepts it: `llama-perplexity -m <file> -c 1024 -b 1024 -ub 512 -f <small slice>` must produce **finite** PPL on **both** `-ngl 0` (CPU) and default (CUDA). A dense model has none of Inkling's arch defects, so any nan here is a genuine MXFP8 bug — report it loudly.

## Job 2 — the measurement

base = the **bf16 source** so every candidate is compared to identical ground truth. Candidates: bf16 (self, noise floor), q8_0 (8.5 bpw), **mxfp8 (8.25 bpw)**, mxfp4 (4.25 bpw). Report mean PPL ± err, mean KLD ± err, same-top-pct, plus file size for each.

Flag gotchas already discovered: **`-b` must equal `-c`** (`perplexity.cpp:2012-2014` derives `n_parallel = n_batch/n_ctx` and multiplies back, so `-c 512` silently becomes a 4-slot 2048-token context), and the corpus needs **≥2×ctx tokens** or the tool errors out. Use a real slice (~64k tokens if time allows; state exactly what you used).

**Then the CPU-vs-CUDA axis on the same mxfp8 file** — a backend disagreement there would be a genuine compute-path bug, and it is currently undetectable on Inkling.

**Claim under test:** at ~8.25 bpw, MXFP8 should sit near q8_0, nowhere near mxfp4. If it's materially worse than q8_0 at similar size, say so plainly with the numbers.

## Rules

- You own: `scripts/inkling_native/` (new packer + driver), `docs/mxfp8/25-*`, `logs/`, and new GGUFs under `~/models/timlikesai/` (~0.5 GB; disk fine).
- **No builds, no source edits** (`ggml/**`, `src/**`, `tools/**`, `tests/**`, `conversion/**` read-only). **Never delete a file or log**, including the broken `-C` stub and everything Inkling-related.
- No docker/systemctl. Another lane is profiling on the GPU — keep runs short; if a number looks off, note the contention and re-run rather than reporting it.
- **Fill `docs/mxfp8/25-mxfp8-isolated-kld.md` as you go, not at the end** — §1's checklist first, within your first 4 tool calls. Fourteen lanes have timed out this session with nothing on disk.
- 35-minute budget; finalize the doc at 30, including a clear §6 verdict on whether MXFP8 is fit for a PR.
