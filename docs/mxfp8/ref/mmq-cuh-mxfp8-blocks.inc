};

static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq)  == sizeof(block_q8_1_mmq),    "Unexpected block_fp4_mmq size");

// activation block for the mxfp block-scaled mma (Blackwell):
// 128 e4m3 values, d4[f] byte 0 = ue8m0 scale of values 32f..32f+31
struct block_fp8_mmq {
    uint32_t d4[4];
    int8_t   qs[QK_FP8_MMQ];
};
static_assert(sizeof(block_fp8_mmq) == sizeof(block_q8_1_mmq), "Unexpected block_fp8_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_MXFP6:
        case GGML_TYPE_MXFP8:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_NVFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            default:
                return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
        }
    }

// ---------------------------------------------------------------------------------------------

#ifdef BLACKWELL_MMA_AVAILABLE
    switch (type) {
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_MXFP6:
        case GGML_TYPE_MXFP8:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_mxfp<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_mxfp_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_NVFP4:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        default:
