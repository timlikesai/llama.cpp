# Phase 4a (CUDA Tier B) - `mmq` tile path for `GGML_TYPE_MXFP8`

Brief: `docs/mxfp8/briefs/P-mmq-mxfp8.md`. Scope is the MMQ tile route only. Native
`mma...kind::mxf8f6f4` block-scaled MMA is brief R, deliberately out of scope here.

## Goal

`ggml_cuda_should_use_mmq` does not list `GGML_TYPE_MXFP8`, so prefill / large-batch
mul_mat falls through to `mmvf` (dequant-free but SIMT) or dequant-to-float + `mmf`.
Add the A16-style tile path (fp16 accumulator tiles, e4m3 payload dequantized in the
vec_dot) so tensor cores are used, then measure prefill and decode against the `mmvf`
route. A negative measurement is an acceptable outcome; the dispatch stays off if so.

## Touchpoints identified so far

| File | What it needs |
|---|---|
| `ggml/src/ggml-cuda/mmq.cu` | `mul_mat_q_case<GGML_TYPE_MXFP8>` instantiation (or a new template instance file) |
| `ggml/src/ggml-cuda/mmq.cuh` | `ggml_cuda_get_device_quant_size`/`type_can_use_int_mma`-style classification, `load_scale_k`/dequant path selection, `ggml_cuda_should_use_mmq` case, config selection for MXFP8 |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `load_tile_x_q<K, NS>` specialization for `block_mxfp8` |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot<block_mxfp8, ...>` specialization |
| `ggml/src/ggml-cuda/mmq-config-ampere.cuh` / `mmq-config-blackwell.cuh` | MXFP8 entries in the config tables |
| `template-instances/**` + `generate_cu_files.py` | instance file if needed (check whether GLOB needs a reconfigure) |

## Notes / decisions

- Mirror `mmq-instance-mxfp4.cu` + its `load_tile_x_q` / `ggml_cuda_mmq_vec_dot` entries.
- Scale decode **must** be `ggml_cuda_e8m0_to_fp32(e)` with **no** `* 0.5`. The 0.5 in
  `vec_dot_mxfp4_q8_1` exists only because `kvalues_mxfp4` are doubled. e4m3 payloads are
  not, so a stray half is a silent 2x error.
- MXFP8 is not an int-MMA type: payload is fp8 e4m3, so `mmq_vec_dot` must take the A16
  (fp16) path with the payload converted like `dequantize_mxfp8`, not the INT8/DP4A path.
- Config: give MXFP8 the same treatment as NVFP4/MXFP4 in `ggml_cuda_mmq_get_config`
  (ampere-style f16 tile config) to stay off the Blackwell-native MMA requirement.

## Progress

| Step | State |
|---|---|
| Read analogue files (mxfp4 instance, load-tiles, vec-dot, configs) | in progress |
| Implement load_tile + vec_dot + config + dispatch | not started |
| Build clean (`logs/mmq-build-<n>.log`) | not started |
| Prove the mmq path is actually taken | not started |
| Gate 1: `test-backend-ops` CUDA0 + CUDA1 + CPU | not started |
| Gate 2: MoE shapes (MUL_MAT_ID / fusion use_id=1) | not started |
| Gate 3: real Inkling model, `A: 68 B: 64 C: 72 ...` with and without FORCE_MMQ | not started |
| Gate 4: prefill/measure vs mmvf (484.50 / 22.06 baseline) | not started |
| Gate 5: commit if green | not started |
