# Phase 3 (CUDA Tier A) — `GGML_TYPE_MXFP8` on the mmvf/dequantize/convert paths

Working doc, updated as the implementation lands. Scope: briefs/L-cuda-tierA.md.

## Status

- [ ] dequantize.cuh: `dequantize_mxfp8` pair decoder
- [ ] convert.cu: `MXFP8` in to_fp16 / to_bf16 / to_fp32 (contiguous + non-contiguous)
- [ ] getrows.cu: MXFP8 rows
- [ ] mmvf.cu: MXFP8 in the fused vector kernel + `ggml_cuda_should_use_mmvf`
- [ ] ggml-cuda.cu: op-support matrix + MUL_MAT_ID dispatch (MMVQ must not claim MXFP8)
- [ ] builds clean on sm_120a
- [ ] `test-backend-ops -b CUDA0 -p mxfp8` green
- [ ] `test-backend-ops -b CUDA1 -p mxfp8` green
- [ ] `test-backend-ops -b CPU -p mxfp8` still 306/306

## Notes

(filled in below as work lands)
