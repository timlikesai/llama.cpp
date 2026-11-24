# Brief L — CUDA correctness path (Tier A) for `GGML_TYPE_MXFP8` weights

**Deliverables (own ONLY these; `conversion/` is off-limits — another agent is there):**
`ggml/src/ggml-cuda/dequantize.cuh`, `mmvf.cu`, `convert.cu`, `convert.cuh`, `common.cuh`, `getrows.cuh`/`getrows.cu`, `ggml-cuda.cu`, plus new `docs/mxfp8/13-cuda-tierA.md`.
Do **not** modify: `conversion/*`, `scripts/inkling_native/*`, `ggml/src/ggml.c`, `ggml-common.h`, `docs/mxfp8/0[0-9]*` (parent owns 00/01).

**Goal / gate:** `./build-cuda/bin/test-backend-ops -b CUDA0 -p mxfp8` and `-b CUDA1 -p mxfp8` pass for `MUL_MAT`, `MUL_MAT_ID`, `MUL_MAT_FUSION(use_id=1)`, `GET_ROWS`, `CPY`, `DUP`; and CPU-vs-CUDA outputs agree. Currently every MXFP8 case reports `not supported` on CUDA (verified — that is the clean state we start from). Full sweep on CPU is already 306/306 green.

**Why this route (already analysed — do not re-derive, do not "simplify" into mmvq):**
- `ggml_cuda_should_use_mmf` returns false for **all** quantized types (`mmf.cu:138`), so there is no dequant-f16 mmf path to reuse.
- `mmvq`'s per-type dot uses **`ggml_cuda_dp4a`** on int8×int8 (`vecdotq.cuh:310-359`). FP8 payloads have no integer form, so an `mmvq` MXFP8 dot would be a new float-dot kernel — do that later, in Phase 4, together with the real tensor-core path.
- **`mmvf` is the right first door.** Master already routes `GGML_TYPE_F8_E4M3` (an 8-bit float weight type) through `ggml_cuda_should_use_mmvf` — including inside `ggml_cuda_mul_mat_id`'s decision ladder (`ggml-cuda.cu` ~:2615, ~:2664, and `ggml_cuda_mul_mat_id_needs_sync`). `mmvf` dequantizes src0 on the fly via `dequantize.cuh` and dots against f32 activations, so MXFP8 needs a *decoder*, not a new MMA. This gives correct-and-reasonably-fast MoE decode, which is the dominant path for this model (top-8 of 128 experts).

**Type facts (already committed on this branch — `git log --oneline -6`):**
`GGML_TYPE_MXFP8 = 44`; `block_mxfp8 { uint8_t e; uint8_t qs[32]; }` = 33 B / 32 elements; `QK_MXFP8 = 32`, `QR_MXFP8 = 4`, `QI_MXFP8 = 2` (`ggml-common.h:115-117, 233-238`). Dequant is `value = e4m3(qs[j]) * 2^(e - 127)`.
- **Use `ggml_cuda_e8m0_to_fp32(b)`, NOT the `* 0.5f` / `_HALF` variant** that `vec_dot_mxfp4_q8_1` applies at `vecdotq.cuh:330` — MXFP4 halves because `kvalues_mxfp4` are doubled. Halving here is a silent 2× error. Same rule already recorded for the CPU path (`ggml-quants.c` `dequantize_row_mxfp8`) and in the C unit test.
- Scale byte `255` is NaN per OCP. `ggml_e8m0_to_fp32` does not special-case it and yields +inf. Do not paper over it; this checkpoint has no 255 bytes (verified across all 8 MXFP8 tensors), and the Python side already rejects them.
- Payload decode helper exists device-side: check `dequantize.cuh`/`common.cuh` for the `ggml_fp8_e4m3` → float conversion already used by `F8_E4M3` (`__nv_cvt_fp8x2_to_halfraw2` / `ggml_cuda_f8_e4m3_to_fp32`-style) and reuse it — do not hand-roll a 256-entry LUT unless it is clearly faster, and if you do, say why in the doc.

**Work list, in the order that keeps the build green at every step:**
1. `common.cuh`: add `template<> struct ggml_cuda_type_traits<GGML_TYPE_MXFP8> { qk=QK_MXFP8; qr=QR_MXFP8; qi=QI_MXFP8; bs=sizeof(block_mxfp8); }` (mirror the MXFP4 block at `:1161-1167`).
2. `dequantize.cuh`: add the MXFP8 decoder used by `mmvf`/`getrows` (note there is a `block_mxfp4` decoder at `:441` to model it on).
3. `convert.cu` (+ any `.cuh` decl): add `dequantize_row_mxfp8_cuda` and register it in **all three** `ggml_get_to_fp16_cuda` / `to_bf16` / `to_fp32` switches (`convert.cu:540, 602, 661` — they each list MXFP4/NVFP4). Missing one of these is how types end up half-working.
4. `mmvf.cu`: accept `GGML_TYPE_MXFP8` in `ggml_cuda_should_use_mmvf` and in the kernel type dispatch; make sure the `ne00 % (2*ts)`/alignment assumptions hold for a 33-byte block (this is the most likely place to discover the block size is awkward — if so, document it rather than hacking an unsafe path).
5. `ggml-cuda.cu`: the op-support matrix (see the `GGML_TYPE_MXFP4`/`NVFP4` cases around `:6447` and `:6503`) plus any `ggml_cuda_cvt_tensor_data` / row-size / `is_cpu_only` logic. Add MXFP8 wherever MXFP4 appears; list any place you deliberately did NOT add it, and why.
6. `getrows.cu`: MXFP8 → f32 rows (needed by the MoE sorted-expert fallback and by embedding lookups).

**Constraints & gotchas for this box:**
- Build: `cmake --build build-cuda -j 22` (incremental ~1-3 min; do NOT reconfigure — `build-cuda` is already configured for CUDA 13.2 / sm_120a). Tee every build to `logs/phase3-build-<n>.log`. Read the FULL output; the project's rules ban `| tail`/`| grep` on builds to hide errors — filter only *after* you have seen the whole thing.
- **Never delete any file or log.** No `rm`.
- Both GPUs are free right now: the `llama-cpp` container was removed by Tim's order (`docker compose down llama-cpp`), so the old ~60 s VRAM-grab crash loop is gone. Do **not** run any `docker`/`systemctl` command.
- Useful measured baseline for context (do not chase it): the 17.63 GB IQ4_NL Inkling file runs 590 t/s prefill @2817 tokens and 22 t/s decode on 2 cards; short-context decode 32 t/s. `-ngl <exact number>` **disables** auto-fitting and OOMs this model — omit `-ngl` (or use `llama-completion`'s default) so `common_fit_params` runs.
- The Inkling model file is 17.6 GB and takes ~1.5 s to load; for iterating on numerics prefer `test-backend-ops` (synthetic tensors) over loading real models.
- Another agent is running a large model conversion; expect disk/CPU contention, and do not start a second full-model conversion yourself.

**Acceptance:**
1. `test-backend-ops -b CUDA0 -p mxfp8` and `-b CUDA1 -p mxfp8` fully green (paste the pass counts into the doc + keep the logs).
2. `test-backend-ops -b CPU -p mxfp8` still 306/306 (you must not regress the CPU path).
3. Numeric agreement: for `MUL_MAT` and `MUL_MAT_ID` with MXFP8 src0, CUDA vs CPU reference within the harness tolerance (that is what the tests assert) — and state in the doc what the *device vec_dot type* ended up being (`Q8_1` via mmvf-f32 activations, or whatever you implemented) and what the known perf ceiling is.
4. `docs/mxfp8/13-cuda-tierA.md`: what you changed per file (with `file:line`), what you deliberately deferred to Phase 4 (the `mxf8f6f4` W8A8 tensor-core path), and any place where master's structure fought the 33-byte block.
5. Commit your work on branch `mxfp8-weights` with 1-3 focused commits in llama.cpp commit style (`ggml-cuda : ...`). Do not commit `docs/mxfp8/*` or `scripts/` — those stay untracked like the repo's other working notes.

**Timebox: 30 minutes.** Write `docs/mxfp8/13-cuda-tierA.md` within your first 5 tool calls and keep updating it as you go — a stream of sibling tasks died at the timeout with nothing on disk, and a partial file is a good outcome. If you run out of time, leave the build green and the doc accurate about what is done vs pending; do not leave the tree broken.
