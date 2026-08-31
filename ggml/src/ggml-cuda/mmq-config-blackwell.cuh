// mxfp4, mxfp6 and mxfp8 share the same tile geometry
#define CASE_MXFP(J_, nthreads_, occupancy_, I_, sram_layout_, K_vram_, stream_k_, fallback_)                                           \
    if ((type == GGML_TYPE_MXFP4 || type == GGML_TYPE_MXFP6 || type == GGML_TYPE_MXFP8) && J == (J_) && fallback == (fallback_)) {                                                              \
        return ggml_cuda_mmq_config(type, (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
    }

#define CASE_NVFP4(J_, nthreads_, occupancy_, I_, sram_layout_, K_vram_, stream_k_, fallback_)                                          \
    if (type == GGML_TYPE_NVFP4 && J == (J_) && fallback == (fallback_)) {                                                              \
        return ggml_cuda_mmq_config(GGML_TYPE_NVFP4, (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
    }

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_blackwell(ggml_type type, int J, bool fallback) {
    CASE_MXFP(8,   256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, true);
    CASE_MXFP(16,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, true);
    CASE_MXFP(32,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, true);
    CASE_MXFP(64,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, true);
    CASE_MXFP(128, 256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, true);
    CASE_MXFP(8,   256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(16,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(24,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(32,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(40,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(48,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(64,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(80,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(96,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(112, 256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);
    CASE_MXFP(128, 256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP8, MMQ_ITER_K_FP4, true, false);

    CASE_NVFP4(8,   256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE_NVFP4(16,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE_NVFP4(32,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE_NVFP4(64,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE_NVFP4(128, 256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE_NVFP4(8,   256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(16,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(24,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(32,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(40,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(48,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(64,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(80,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(96,  256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(112, 256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE_NVFP4(128, 256, 1, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);

    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
}

#undef CASE_MXFP
#undef CASE_NVFP4
