#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "mxfp-common.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

// ============================================================================
// MXFP SoA set_rows kernel — Struct-of-Arrays layout: [qs0..qsN | e0..eN]
//
// One warp (32 threads) per 32-element block. Follows the CPU scalar
// quantize_row_mxfp*_soa, using shared constructs from
// ggml-common.h for all per-element math.
//
// CANNOT reuse k_set_rows_quant — that template is for AoS block types.
// ============================================================================

template <ggml_type mxfp_type, bool apply_hadamard, typename idx_t>
static __global__ void k_set_rows_mxfp_soa(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        char * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3,
        const int64_t ne11,
        const int64_t ne12) {

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;

    // Global warp index → which 32-element block to process
    const int64_t global_warp = (int64_t)blockIdx.x * warps_per_block + warp;

    const int nblocks_per_row = (int)(ne00 / 32);
    const int64_t total_blocks = (int64_t)nblocks_per_row * ne01 * ne02 * ne03;
    if (global_warp >= total_blocks) return;

    // Decompose into (block_in_row, i01, i02, i03)
    int64_t tmp = global_warp;
    const int block_in_row = (int)(tmp % nblocks_per_row); tmp /= nblocks_per_row;
    const int i01 = (int)(tmp % ne01); tmp /= ne01;
    const int i02 = (int)(tmp % ne02); tmp /= ne02;
    const int i03 = (int)tmp;

    // Index mapping
    const int i12 = i03 % (int)ne12;
    const int i11 = i02 % (int)ne11;
    const int i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    // Source float data for this block
    const float * src_block = src0 + i01*s01 + i02*s02 + i03*s03 + block_in_row * 32;

    // Destination: SoA byte region for this row
    char * dst_row_ptr = dst + dst_row * nb1 + i02 * nb2 + i03 * nb3;

    // SoA layout constants — from shared type traits
    constexpr int qs_per_blk = mxfp_type_traits<mxfp_type>::qs_per_blk;

    // Step 1: Each lane loads one element (matches CPU's x[i*32 + j])
    float val = src_block[lane];

    // Step 2: Optional Hadamard rotation (warp-cooperative)
    if (apply_hadamard) {
        val = mxfp_hadamard_warp(val);
    }

    // Step 3: Warp-wide amax (matches CPU's loop over 32 elements)
    const float amax = mxfp_warp_amax(val);

    // Step 4: E8M0 computation — calls shared mxfp_compute_e8m0 template
    uint8_t e8m0 = 0;
    if (lane == 0) {
        e8m0 = mxfp_compute_e8m0<mxfp_type>(amax);
        // Write E8M0 to SoA scale region
        ((uint8_t *)dst_row_ptr)[MXFP_SOA_E8M0_OFFSET(nblocks_per_row, qs_per_blk) + block_in_row] = e8m0;
    }
    e8m0 = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)e8m0, 0);

    // Step 5: Per-type quantize + write to SoA qs region
    uint8_t * qs = (uint8_t *)dst_row_ptr + MXFP_SOA_QS_OFFSET(block_in_row, qs_per_blk);

    if constexpr (mxfp_type == GGML_TYPE_MXFP4) {
#if CUDART_VERSION >= 12080
        // Intrinsic path: full E8M0 scale, intrinsic quantize uses true E2M1 values.
        const float d     = ggml_cuda_e8m0_to_fp32(e8m0);
        const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
        const uint8_t idx = mxfp4_quantize_intrinsic(val * inv_d);
#else
        // Portable path: full scale + shared quantize from ggml-common.h.
        const float d     = ggml_cuda_e8m0_to_fp32(e8m0);
        const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
        const uint8_t idx = ggml_mxfp_float_to_fp4_e2m1(val * inv_d);
#endif
        // Nibble packing: lanes 0-15 = low nibble, lanes 16-31 = high nibble
        // Matches CPU: qs[j] = x0 | (x1 << 4) where x0=elem[j], x1=elem[j+16]
        const uint8_t hi = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)idx, lane + 16);
        if (lane < 16) {
            qs[lane] = idx | (hi << 4);
        }
    } else if constexpr (mxfp_type == GGML_TYPE_MXFP8) {
        const float d = ggml_cuda_e8m0_to_fp32(e8m0);
        const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
#if CUDART_VERSION >= 12080
        qs[lane] = mxfp8_quantize_intrinsic(val * inv_d);
#else
        qs[lane] = ggml_mxfp_float_to_fp8_e4m3(val * inv_d);
#endif
    } else if constexpr (mxfp_type == GGML_TYPE_MXFP6) {
        const float d = ggml_cuda_e8m0_to_fp32(e8m0);
        const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
#if CUDART_VERSION >= 12080
        const uint8_t elem = mxfp6_quantize_intrinsic(val * inv_d);
#else
        const uint8_t elem = ggml_mxfp_float_to_fp6_e2m3(val * inv_d);
#endif
        // Pack 4 elements per 3 bytes — gather group of 4 via shuffles
        const int group = lane / 4;
        const uint8_t v0 = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)elem, group * 4 + 0);
        const uint8_t v1 = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)elem, group * 4 + 1);
        const uint8_t v2 = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)elem, group * 4 + 2);
        const uint8_t v3 = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)elem, group * 4 + 3);
        if (lane % 4 == 0) {
            const uint8_t vals[4] = { v0, v1, v2, v3 };
            ggml_mxfp_pack_fp6x4(vals, qs + group * 3);
        }
    }
}

// MXFP SoA set_rows dispatch helper
template <ggml_type mxfp_type, typename idx_t>
static void set_rows_mxfp_soa_cuda(
        const float * src0_d, const idx_t * src1_d, char * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const int64_t ne11, const int64_t ne12,
        bool hadamard, cudaStream_t stream) {

    GGML_ASSERT(ne00 % 32 == 0);
    const int nblocks_per_row = (int)(ne00 / 32);
    const int64_t total_warps = (int64_t)nblocks_per_row * ne01 * ne02 * ne03;
    if (total_warps == 0) return;

    const int warps_per_cuda_block = 4;
    const int threads_per_cuda_block = warps_per_cuda_block * 32;
    const int num_cuda_blocks = (int)((total_warps + warps_per_cuda_block - 1) / warps_per_cuda_block);

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    if (hadamard) {
        k_set_rows_mxfp_soa<mxfp_type, true, idx_t><<<num_cuda_blocks, threads_per_cuda_block, 0, stream>>>(
            src0_d, src1_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3, ne11, ne12);
    } else {
        k_set_rows_mxfp_soa<mxfp_type, false, idx_t><<<num_cuda_blocks, threads_per_cuda_block, 0, stream>>>(
            src0_d, src1_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3, ne11, ne12);
    }

}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_MXFP4 || dst->type == GGML_TYPE_MXFP8 || dst->type == GGML_TYPE_MXFP6) {
        // MXFP uses Struct-of-Arrays layout: [qs0..qsN | e0..eN] per row.
        // Cannot reuse k_set_rows_quant — that's for AoS block types.
        const bool hadamard = ((const int32_t *)dst->op_params)[0] != 0;
        if (dst->type == GGML_TYPE_MXFP4) {
            set_rows_mxfp_soa_cuda<GGML_TYPE_MXFP4, idx_t>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                nb01, nb02, nb03, nb10, nb11, nb12,
                nb1, nb2, nb3, ne11, ne12, hadamard, stream);
        } else if (dst->type == GGML_TYPE_MXFP8) {
            set_rows_mxfp_soa_cuda<GGML_TYPE_MXFP8, idx_t>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                nb01, nb02, nb03, nb10, nb11, nb12,
                nb1, nb2, nb3, ne11, ne12, hadamard, stream);
        } else {
            set_rows_mxfp_soa_cuda<GGML_TYPE_MXFP6, idx_t>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                nb01, nb02, nb03, nb10, nb11, nb12,
                nb1, nb2, nb3, ne11, ne12, hadamard, stream);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
