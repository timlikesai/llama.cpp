// one mxfp family (mxfp4/mxfp6/mxfp8): 512-value tiles, MXFP8 SRAM layout, fp4-capable
#define MXFP_CASES(type_, sram_layout_) \
    CASE(type_, 256, 1, 128,   8, sram_layout_, MMQ_ITER_K_FP4, true, true);  \
    CASE(type_, 256, 1, 128,  16, sram_layout_, MMQ_ITER_K_FP4, true, true);  \
    CASE(type_, 256, 1, 128,  32, sram_layout_, MMQ_ITER_K_FP4, true, true);  \
    CASE(type_, 256, 1, 128,  64, sram_layout_, MMQ_ITER_K_FP4, true, true);  \
    CASE(type_, 256, 1, 128, 128, sram_layout_, MMQ_ITER_K_FP4, true, true);  \
    CASE(type_, 256, 1, 128,   8, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  16, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  24, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  32, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  40, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  48, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  64, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  80, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128,  96, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128, 112, sram_layout_, MMQ_ITER_K_FP4, true, false); \
    CASE(type_, 256, 1, 128, 128, sram_layout_, MMQ_ITER_K_FP4, true, false)

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_blackwell(ggml_type type, int J, bool fallback) {
    MXFP_CASES(GGML_TYPE_MXFP4, GGML_CUDA_MMQ_SRAM_LAYOUT_MXFP8);
    MXFP_CASES(GGML_TYPE_MXFP6, GGML_CUDA_MMQ_SRAM_LAYOUT_MXFP8);
    MXFP_CASES(GGML_TYPE_MXFP8, GGML_CUDA_MMQ_SRAM_LAYOUT_MXFP8);

    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);

    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
}

#undef MXFP_CASES
