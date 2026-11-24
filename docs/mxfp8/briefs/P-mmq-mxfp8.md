# Brief P — Phase 4a: `mmq` tile path for MXFP8 so CUDA prefill stops riding the dequant route

**Current state (do not re-verify):** MXFP8 is committed and *correct* end to end. `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` (18,974,614,688 B, 40 NVFP4 + 8 MXFP8 expert tensors, bit-exact) generates `A: 68 B: 64 C: 72 …` identically on CPU and CUDA, matching the reference file. `test-backend-ops` is 306/306 CPU and 237/237 CUDA0 for `-p mxfp8`. Perf vs the lossy IQ4_NL reference: prefill 484.50 vs 492.32 t/s (−1.6%), decode 22.06 vs 23.62 t/s (−6.6%) @2817 tokens.

**Why this task exists:** MXFP8 currently has **no `mmq` tile path** — `ggml_cuda_should_use_mmq` does not list `GGML_TYPE_MXFP8`, so large-batch/prefill matmuls fall to `mmvf`/dequant-to-float. That is correct but leaves the tensor cores unused. Add the tile path. Native `mma…kind::mxf8f6f4` block-scaled W8A8 is a *separate later* step (brief R) — do **not** attempt it here; land the A16-style tile path first and measure.

**Files you own:** `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh`, `mmq-config-*.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `mmq-few.cu` (if needed), `ggml/src/ggml-cuda/template-instances/**` (incl. `generate_cu_files.py`), and `docs/mxfp8/16-mmq-mxfp8.md`. Also allowed: adding `GGML_TYPE_MXFP8` to the `should_use_mmq` switch.
**Do NOT touch:** `src/**`, `conversion/**`, `scripts/inkling_native/**`, `mmvf.cu`, `mmvq.cu`, `convert.cu`, `dequantize.cuh`, `ggml-cuda.cu` (a read-only agent may be inspecting those; and the Tier A route already works). If a change genuinely must land outside your fence, write it into your doc and stop.

**Type facts:** `GGML_TYPE_MXFP8 = 44`, `block_mxfp8 { uint8_t e; uint8_t qs[32]; }` = 33 B / 32 elements, `QK_MXFP8 32`, `QR_MXFP8 4`, `QI_MXFP8 2` (`ggml-common.h:115-117, 233-238`). Device vec-dot type is `Q8_0` on CPU; on CUDA the existing types use `Q8_1`.
**CRITICAL:** use `ggml_cuda_e8m0_to_fp32(e)` **without** the `* 0.5` that `vec_dot_mxfp4_q8_1` applies (`vecdotq.cuh:330`) — MXFP4 halves because `kvalues_mxfp4` are doubled; e4m3 payloads are not. A stray half is a silent 2× error that `test-backend-ops` will catch only if the reference is computed independently, so also run the real model (below).

**Config selection:** in `mmq.cuh` both `ggml_cuda_mmq_get_config` paths special-case NVFP4/MXFP4 (`if (prec_src1 != GGML_PREC_Q4 && (type == NVFP4 || type == MXFP4)) → ampere config`). Give MXFP8 the same treatment initially (ampere-style f16 tile config) so you are not blocked on Blackwell-native MMA.

**Method**
1. Create `docs/mxfp8/16-mmq-mxfp8.md` within 4 tool calls: goals, the touchpoints you identified, and a live progress table. Keep updating it.
2. Read the three files that define a tile type end-to-end for an existing analogue: `template-instances/mmq-instance-mxfp4.cu`, its `load_tile_*` in `mmq-load-tiles.cuh`, and its `ggml_cuda_mmq_vec_dot` specialization in `mmq-vec-dot.cuh`, plus the relevant `mmq-config-blackwell.cuh` / `mmq-config-ampere.cuh` entry. Mirror them for MXFP8.
3. `python3 ggml/src/ggml-cuda/template-instances/generate_cu_files.py` (read `CMakeLists.txt` first to see whether generation is required or the files are checked in) and add `mul_mat_q_case<GGML_TYPE_MXFP8>` instantiation in `mmq.cu`.
4. Build `cmake --build build-cuda -j 22` (NEVER reconfigure), tee to `logs/mmq-build-<n>.log`, read the **whole** output.
5. Prove the new path is actually taken — otherwise you may be measuring `mmvf` again. Use `GGML_CUDA_FORCE_MMQ=1` and/or `GGML_CUDA_GRAPH_GRAPH=0`/debug env, or add a temporary print behind an env var; state in the doc how you proved dispatch, and remove any temporary instrumentation before finishing.

**Gates (all with logs kept; never delete anything)**
1. `test-backend-ops -b CUDA0 -p mxfp8` **and** `-b CUDA1 -p mxfp8` fully green, and `-b CPU -p mxfp8` still 306/306.
2. `MUL_MAT_ID` / `MUL_MAT_FUSION(use_id=1)` at the real Inkling expert shapes (`n_ff 1152/2304`, 128 experts, top-8) — the MoE case is what matters here.
3. Real model, unchanged expectations: `llama-completion -m ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf -c 8192 -f logs/g0-prompt-long.txt -n 32 --temp 0 -no-cnv --seed 42` must still print `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on CUDA **and** with `GGML_CUDA_FORCE_MMQ=1`. **Never pass an exact `-ngl`** — it disables memory fitting and OOMs this model on 2 cards; let auto-fit run, and don't use `llama-bench` for this model.
4. Measured delta: prefill and decode t/s at 2817 tokens for (a) current mmvf route = 484.50 / 22.06, (b) with mmq enabled, (c) `GGML_CUDA_FORCE_MMQ=1`. State the numbers in the doc with the exact commands. If mmq is *slower*, say so plainly and leave it not-selected by default — a negative result is a real result, do not fudge the dispatch heuristic.
5. Commit only if green, in llama.cpp style (`ggml-cuda : add MXFP8 to MMQ`), with the measurement summary in the message. `docs/mxfp8/**` and `scripts/**` stay untracked.

**Rules:** no docker/systemctl; both GPUs are free (container removed on Tim's order); a second agent may be reading the tree — check `git status` before you edit and never revert anything that isn't yours; 35-minute budget; leave the build green even if unfinished.
