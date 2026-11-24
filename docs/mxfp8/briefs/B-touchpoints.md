# Brief B — Exhaustive touchpoint checklist for a new `GGML_TYPE_MXFP8` weight type

**Deliverable:** write `/home/tim/code/services/repos/llama.cpp/docs/mxfp8/03-touchpoint-checklist.md`.
**Do not edit source.** Read-only trace of current master + one markdown output.
Cite `file:line` for every insertion point. Print a 15-line summary to stdout.

Repo: `/home/tim/code/services/repos/llama.cpp`, branch `mxfp8-weights` = master `29b52d398`.

**Method:** trace how `GGML_TYPE_MXFP4` (39) and `GGML_TYPE_NVFP4` (40) are wired through every layer of the stack, and mirror it. `git grep -n GGML_TYPE_MXFP4` and `git grep -n GGML_TYPE_NVFP4` are your spine; list **every** hit with a verdict for MXFP8.

**Target type semantics (already decided):** 32 elements/block = `{uint8_t e /* ue8m0, bias 127 */; uint8_t qs[32] /* e4m3 payload */}` = 33 bytes/block (8.25 bpw). Dequant: `value = e4m3(qs[i]) * 2^(e - 127)`. Byte 255 in the scale is NaN by spec — must not occur in this checkpoint. It is a *weight* type (also usable as a KV-cache type later, but that is out of scope here).

## Required sections (each an actionable checkbox list, with exact current line numbers)

**A. ggml core** — `ggml/include/ggml.h` enum + `GGML_TYPE_COUNT`; `ggml/src/ggml-common.h` block struct; `ggml/src/ggml.c` type_traits row (cf. `[GGML_TYPE_NVFP4]` at :769), `to_float`, `from_float_ref`, `quantize_row` dispatch (:8063 area), `ggml_type_size`/`block_size` tables, `ggml_..._allows_prompt`/fusion-capability switches, `ggml/src/ggml-impl.h`; every array indexed by type or by `GGML_TYPE_COUNT` anywhere in the tree (this is a silent-corruption risk — grep hard, include `.py`).

**B. ggml-cpu** — `ggml-cpu.c` (vec_dot type selection, `is_dot_dotx`, `from_float` conversions), `ops.cpp`, `repack.cpp` (AVX2/NEON), `arch-fallback.h`, `quants.h`. State clearly: reference-only path is acceptable for the first cut; say where a `GGML_ABORT("not implemented")` must go instead of silently producing garbage.

**C. ggml-cuda** — `common.cuh` (`type_size`, `to_typed`, row-size helpers), `convert.cu`, `mmq.cu` (`mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>` at :75), `mmq.cuh` (`native_fp4_available` at :1040, `ne_block` at :963, :257/:291/:749/:1047), `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `mma.cuh` (:1288 `kind::mxf4`, :1295 `kind::mxf4nvf4`), `mmq-config-blackwell.cuh`, `template-instances/generate_cu_files.py` + `mmq-instance-*.cu`, `mmvq.cu`, `quantize.cu`, `getrows.cu`, `ggml-cuda.cu` (op support matrix, `ggml_backend_buft_is_supported`, row-size/alloc math). Explicitly separate: (i) **minimum viable correct-but-slow** CUDA path (which existing kernel would serve — dequant to F16/BF16? `mmvq` with a new vec-dot? `dmmv`?) and where the fallback decision lives (`ggml_cuda_should_use_mmx`, `wsize` computations); (ii) the **fast block-scaled mmq** path.

**D. llama** — `src/llama-arch.cpp/.h` (type name table; how "mxfp4"/"nvfp4" strings map), `src/llama-model.cpp` (`.scale` / `.input_scale` sidecar tensors: `LLM_TENSOR_*` declarations ~:1900-2010, incl. the `ffn_up_exps.scale` vs `ffn_gate_up_exps.weight` conflict check at :1917 — describe how the per-expert f32 scale is applied in the graph and whether that works for MUL_MAT_ID experts), `src/llama-model-loader.cpp` (per-type fixups, `GGML_TYPE_F8_E4M3` handling), `src/llama-quant.cpp` (ftype tables), `src/llama-graph.cpp` (MUL_MAT_ID construction + any weight-type allow-list for experts), `src/llama-model-saver.cpp`, `tools/quantize/` ftype CLI table, `common/` arg parsing.

**E. python** — `gguf-py/gguf/constants.py`: `GGMLQuantizationType` (add `MXFP8 = 44`), `LlamaFileType` (`MOSTLY_F8_E4M3 = 42` is last → `MOSTLY_MXFP8 = 43`), `QUANT_SIZE_MAP` (`MXFP4: (32, 1+16)` at :7585 → `MXFP8: (32, 1+32)`); `conversion/base.py`: `repack_mxfp4_blocks` (~:1240), `_mxfp4_expert_tensor`, `_nvfp4_pack` (~:1310), `_repack_nvfp4`, `_generate_nvfp4_tensors` (:1345), `_flush_nvfp4_experts`, `_prepare_fp8_e4m3_tensors` (:692, note it only handles *per-tensor* fp8 scales → `prod(scale.shape) != 1` → `continue`), `_write_scale_tensor`/`_write_scales_tensor` (:335-345), `_tag_prec_a4` (:939), and the ftype selection at :1688-1690. Enough detail that a later task can add an MXFP8 export without re-reading all of base.py.

**F. tests + other backends** — `tests/test-backend-ops.cpp` (existing MXFP4/NVFP4 cases: names, shapes, tolerances, how prec is set), `tests/test-quantize-fns.cpp`, `test-quantize-perf.cpp`, `test-ggml.cpp`; and for `ggml-metal`, `ggml-vulkan`, `ggml-hexagon`, `ggml-sycl`, `ggml-openvino`: state whether MXFP8 must be added, is auto-unsupported, or would *assert*/misbehave (find how NVFP4 is gated in `ggml-metal-ops.cpp` / `ggml-vulkan.cpp` :6794 and whether unsupported types fall back cleanly).

## End the doc with

1. **Ordered build sequence** — 6–10 commits, each one compiling and independently testable, from "type exists and dequantizes in C" → "converts + loads on CPU" → "runs correct-but-slow on CUDA" → "mmq fast path".
2. **Risk list** — spots where a missing case *silently corrupts* rather than failing loudly (type-indexed arrays, python block-size tables, ftype maps, `to_float` fallbacks, KV-cache type validation).
