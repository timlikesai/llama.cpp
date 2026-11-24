# Brief Y — Isolate and validate the MXFP8 code path on a dense 0.8B model (no Inkling confounds)

## Why this exists

Inkling-2-Mini has architectural defects that mask type-level results — CPU batched eval returns nan for **every** quantization including the public lossy build (task #21), and the 2-card split silently disables Flash Attention. So KLD/PPL numbers on Inkling cannot cleanly attribute error to MXFP8. `ggml-org/Qwen3.5-0.8B-BF16.gguf` (1,557,662,496 B, at `/home/tim/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF/snapshots/004fd086eccea9f7cf867aa07ba328773beef81d/`) is small, dense, bf16, and finite on both backends — so it can decide the question directly: **how much error does MXFP8 actually add, and do CPU and CUDA agree on it?**

## Known blocker — do not try to work around it with a build

`LLAMA_FTYPE_MOSTLY_MXFP8 = 43` exists (`include/llama.h:162`) and ggml has `to_mxfp8`/`from_mxfp8`, **but `src/llama-quant.cpp` and `tools/quantize/quantize.cpp` contain no MXFP8 references at all** — `llama-quantize 2>&1 | grep mxfp8` is empty. So the tool cannot emit MXFP8 today; that is a real gap (filed separately) and it is a *2-line-ish C++ wiring change the parent will land after the profiling lane releases `build-cuda`*.

**You must not run any build.** Another agent owns `build-cuda`. Therefore:

1. **Quantize in python** to produce the MXFP8 file. Reference implementations you should reuse rather than reinvent — `scripts/inkling_native/layout.py:114 pack_mxfp8_blocks(weight, scale)`, the packer in `conversion/inkling2.py` (`_pack_mxfp8_chunk`, byte-identical to `layout.py` on real bytes, verified), and the decoder oracle `tests/test-mxfp8-repack.py`. Block format: `block_mxfp8 { uint8_t e /*ue8m0*/; uint8_t qs[32] /*e4m3*/ }`, 33 B per 32 values, `v = e4m3 * 2^(e-127)`.
   - Scale selection: derive the per-32 `ue8m0` as `ceil(log2(amax/448)) + 127` **and verify the resulting values stay in e4m3 range**; record whatever convention you choose and justify it — a lazy `amax/448` power-of-two rule is the OCP microscaling convention, but state it, don't assume it.
   - Write a valid GGUF: copy the source metadata, set `general.file_type` to 43, take 2-D weight tensors → MXFP8, keep 1-D tensors and norms F32, keep `output.weight`/embeddings at a sane choice, and set the tensor type field to **44** (`GGML_TYPE_MXFP8`). Mind the traps that have already cost this project three wrong results: `GGUFWriter` wants logical dims, `ReaderTensor.data` is a **byte** view for quantized types, and `GGUFReader` `.shape` is `(C, R, E)` — reversed from numpy order.
   - **Self-check before any measurement**: decode your own file with the reference decoder and assert per-block `|decoded - original| / |original|` is within e4m3 expectations (~6% worst per element), and that requantizing the decoded values reproduces your bytes. If your python packer and the C `from_mxfp8` disagree on any block, stop and report that — it would be a type-level bug, which is the whole point of this exercise.
2. **Cross-check against the C decoder, not just python.** `test-backend-ops -b CPU -p mxfp8` and `-b CUDA0 -p mxfp8` are already green (306/306, 237/237) — re-run them as a regression gate, and additionally confirm the *real model* runs: `llama-perplexity -m <your-mxfp8-file>` on CPU **and** GPU must be finite and sane for a dense model. If the tiny model nans anywhere, that IS an MXFP8 bug (unlike Inkling, where nan is a known arch defect) and it's the headline.

## The measurement

Baseline is the **same model** in other types, all from that BF16 source, using `llama-perplexity --kl-divergence-base` (KL is native to this build; a sibling lane proved it works and documented the failure mode):

| candidate | how to get it |
|---|---|
| `q8_0` | `./build-cuda/bin/llama-quantize <bf16> <out> q8_0` — supported today |
| `bf16` (identity) | the source file itself |
| `f32` | `llama-quantize … f32` if offered, else skip |
| `mxfp4` | `llama-quantize … mxfp4` (in the tool's list) — the closest existing block-scaled sibling |
| **`mxfp8`** | your python-produced file |

Use the **base = bf16 source** so every candidate is compared to the same ground truth, and report per candidate: mean PPL ± error, KLD ± error, same-top-pct. Reuse `scripts/inkling_native/ppl_kl_u2.sh` for the flag pattern. **Critical flag gotchas already discovered:** `-b` must equal `-c` (`perplexity.cpp:2012-2014` derives `n_parallel = n_batch/n_ctx` and multiplies back, so `-c 512` silently becomes a 2048-token 4-slot context), and the tool needs **≥2×ctx tokens** in the corpus. Corpus is available at `logs/corpora/wiki.test.slice64k.raw` / `wt2` sources — for a 0.8B model you can afford a real slice (say 64k tokens); say exactly what you used.

**The claim you are testing:** MXFP8 (8.25 bpw, block-scaled) should land *between* `q8_0` (8.5 bpw) and `mxfp4` (4.25 bpw) on KLD, and be far closer to q8_0 than to mxfp4 given the similar bits-per-weight. If MXFP8 is materially worse than q8_0 at similar size, something in our type or its runtime use is wrong — report it loudly with the numbers. Also compare **CPU vs CUDA KLD for the same mxfp8 file**: a backend disagreement here would be a genuine compute-path bug, and it is currently impossible to detect on Inkling because of the arch defect.

## Deliverables and rules

- `docs/mxfp8/25-mxfp8-isolated-kld.md`, **created within your first 4 tool calls** and updated as numbers land. Do not be the thirteenth task this session that timed out with nothing on disk. Include: the exact packer convention, the self-check results, the KLD/PPL table, the CPU-vs-CUDA comparison, and a verdict on whether MXFP8 is fit for a PR.
- You own: `scripts/inkling_native/` (new python quantizer + driver), `docs/mxfp8/25-*`, `logs/`, and output GGUFs under `~/models/timlikesai/` (small — a 0.8B model is ~0.5-1 GB per variant; ~250 GB free).
- **Never delete a file or log.** Do not touch anything Inkling-related: `Inkling-2-Mini-NATIVE.gguf`, `Inkling-DEQ-Q80.gguf`, `Inkling-2-Mini-vision-mmproj.gguf`, `Inkling-2-Mini-NATIVE-scale1.gguf`, `conversion/**`, `src/**`, `ggml/**`, `tests/**`.
- No docker/systemctl, no builds. A prefill-profiling lane and an MTP lane are running: keep GPU runs short, and if a timing or PPL looks odd, note the contention and re-run rather than reporting it.
- Report negatives plainly. "MXFP8 is worse than q8_0 by X" or "my packer disagreed with the C decoder on Y blocks" are the results I most need to hear.
- 40-minute budget; finalize the doc at 34.
