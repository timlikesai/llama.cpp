#pragma once

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "hadamard.cuh"

using namespace ggml_cuda_mma;

// Fast exp(x) via cubic polynomial approximation of 2^x (Flash Attention 4 technique).
// Splits into 2^floor(x*log2e) (bit manipulation) * poly(frac) (3 FMAs).
// Reduces SFU contention by using CUDA cores instead of the Special Function Unit.
// Max relative error ~0.3% over the softmax-relevant range (x <= 0).
static __device__ __forceinline__ float fast_expf(const float x) {
    const float x_log2e = x * 1.4426950408889634f; // x * log2(e)
    const float xi = floorf(x_log2e);

    // Clamp: exponent below -126 would underflow IEEE 754 normal range.
    if (xi < -126.0f) {
        return 0.0f;
    }

    const float r  = x_log2e - xi;

    // Cubic polynomial: 2^r ≈ ((0.07711909*r + 0.22756439)*r + 0.69514614)*r + 1.0
    const float poly = fmaf(fmaf(fmaf(0.07711909f, r, 0.22756439f), r, 0.69514614f), r, 1.0f);

    // 2^floor via IEEE 754 exponent field manipulation.
    const int exp_bits = ((int)xi + 127) << 23;
    float pow2_floor;
    memcpy(&pow2_floor, &exp_bits, sizeof(float));

    return pow2_floor * poly;
}

// MXFP4 MMA config: Blackwell-only, cols_per_warp = 8 (FP4 B-tile width).
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_mxfp4_get_config(
        const int DKQ, const int DV, const int ncols) {
    // nbatch_V2 = half2 count for dequantized V tile. nstages_target unused (no cp.async).
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2,  64,  32,  32,  32, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 3,  64,  64,  64,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_mxfp4_get_config(
        const int DKQ, const int DV, const int ncols, const int /*cc*/) {
    return ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols);
}

// ------------------------------------------------------------------------------------------------------------------
// Loading helpers
// ------------------------------------------------------------------------------------------------------------------

// Load K from global to shared memory (SoA layout).
template<int DKQ, int nwarps, int nbatch_fa, int stride_k_qs, int stride_k_sc, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_load_K(
        const char * const __restrict__ K_row_base,
        const int K_qs_head_off,
        const int K_e_head_off,
        const int K_pos_stride,
        int      * const __restrict__ tile_K_qs,
        uint32_t * const __restrict__ tile_K_sc,
        const int k_VKQ_0,
        const int k_VKQ_sup) {
    constexpr int ints_per_row = DKQ / 8;
    constexpr int blocks_per_head = DKQ / 32;

#pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps) {
        const int i = i0 + threadIdx.y;

        if (i0 + nwarps > nbatch_fa && i >= nbatch_fa) {
            break;
        }

        const char * row_i = K_row_base + int64_t(k_VKQ_0 + i) * K_pos_stride;

#pragma unroll
        for (int k0 = 0; k0 < ints_per_row; k0 += WARP_SIZE) {
            const int k = k0 + threadIdx.x;
            if (k0 + WARP_SIZE > ints_per_row && k >= ints_per_row) {
                break;
            }

            if (oob_check && i >= k_VKQ_sup) {
                tile_K_qs[i * stride_k_qs + k] = 0;
            } else {
                tile_K_qs[i * stride_k_qs + k] =
                    *reinterpret_cast<const int *>(row_i + K_qs_head_off + k * 4);
            }
        }

        // Pack 2 E8M0 scales per uint32_t for scale_vec::2X MMA.
#pragma unroll
        for (int s0 = 0; s0 < blocks_per_head / 2; s0 += WARP_SIZE) {
            const int s = s0 + threadIdx.x;
            if (s0 + WARP_SIZE > blocks_per_head / 2 && s >= blocks_per_head / 2) {
                break;
            }

            if (oob_check && i >= k_VKQ_sup) {
                tile_K_sc[i * stride_k_sc + s] = 0;
            } else {
                const uint8_t e0 = *(row_i + K_e_head_off + 2 * s);
                const uint8_t e1 = *(row_i + K_e_head_off + 2 * s + 1);
                tile_K_sc[i * stride_k_sc + s] = (uint32_t)e0 | ((uint32_t)e1 << 8);
            }
        }
    }
}

// Load compact 1-bit sign residual K: expand sign bits to +-1.0 FP4 nibbles in smem.
template<int DKQ, int nwarps, int nbatch_fa, int stride_k_qs, int stride_k_sc, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_load_K_res_compact(
        const char * const __restrict__ K_row_base,
        const int K_sign_off,
        const int K_res_e_off,
        const int K_pos_stride,
        int      * const __restrict__ tile_K_qs,
        uint32_t * const __restrict__ tile_K_sc,
        const int k_VKQ_0,
        const int k_VKQ_sup) {
    constexpr int ints_per_row = DKQ / 8;
    constexpr int blocks_per_head = DKQ / 32;

#pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps) {
        const int i = i0 + threadIdx.y;

        if (i0 + nwarps > nbatch_fa && i >= nbatch_fa) {
            break;
        }

        const char * row_i = K_row_base + int64_t(k_VKQ_0 + i) * K_pos_stride;

#pragma unroll
        for (int k0 = 0; k0 < ints_per_row; k0 += WARP_SIZE) {
            const int k = k0 + threadIdx.x;
            if (k0 + WARP_SIZE > ints_per_row && k >= ints_per_row) {
                break;
            }

            if (oob_check && i >= k_VKQ_sup) {
                tile_K_qs[i * stride_k_qs + k] = 0;
            } else {
                const int b  = k / 4;
                const int j0 = 4 * (k % 4);

                const uint32_t signs = *reinterpret_cast<const uint32_t *>(
                    row_i + K_sign_off + b * 4);

                // Expand sign bits to +-1.0 FP4 nibbles: 0x2 = +1.0, 0xA = -1.0.
                uint32_t result = 0;
#pragma unroll
                for (int j_off = 0; j_off < 4; ++j_off) {
                    const uint32_t lo_sign = (signs >> (j0 + j_off))      & 1u;
                    const uint32_t hi_sign = (signs >> (j0 + j_off + 16)) & 1u;
                    result |= (0x22u | (lo_sign << 3) | (hi_sign << 7)) << (j_off * 8);
                }
                tile_K_qs[i * stride_k_qs + k] = (int)result;
            }
        }

        // Load residual E8M0 scales (2 per uint32_t, same as primary).
#pragma unroll
        for (int s0 = 0; s0 < blocks_per_head / 2; s0 += WARP_SIZE) {
            const int s = s0 + threadIdx.x;
            if (s0 + WARP_SIZE > blocks_per_head / 2 && s >= blocks_per_head / 2) {
                break;
            }

            if (oob_check && i >= k_VKQ_sup) {
                tile_K_sc[i * stride_k_sc + s] = 0;
            } else {
                const uint8_t e0 = *(row_i + K_res_e_off + 2 * s);
                const uint8_t e1 = *(row_i + K_res_e_off + 2 * s + 1);
                tile_K_sc[i * stride_k_sc + s] = (uint32_t)e0 | ((uint32_t)e1 << 8);
            }
        }
    }
}

// Dequantize MXFP4 V to F16 (half2) in shared memory.
template<int DV, int nwarps, int nbatch_fa, int stride_tile_V, int nbatch_V2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_load_V_f16(
        const char * const __restrict__ V_row_base,
        const int V_qs_head_off,
        const int V_e_head_off,
        const int V_pos_stride,
        half2 * const __restrict__ tile_V,
        const int k_VKQ_0,
        const int i0_start,
        const int k_VKQ_sup) {
    constexpr int nbyte_pairs = nbatch_V2 / 2;

#pragma unroll
    for (int t0 = 0; t0 < nbatch_fa; t0 += nwarps) {
        const int t = t0 + threadIdx.y;
        if (t0 + nwarps > nbatch_fa && t >= nbatch_fa) {
            break;
        }

        const char * row_t = V_row_base + int64_t(k_VKQ_0 + t) * V_pos_stride;

#pragma unroll
        for (int bp0 = 0; bp0 < nbyte_pairs; bp0 += WARP_SIZE) {
            const int bp = bp0 + threadIdx.x;
            if (bp0 + WARP_SIZE > nbyte_pairs && bp >= nbyte_pairs) {
                break;
            }

            const int blk_idx   = bp / 8;
            const int bp_in_blk = bp % 8;

            const int d_h2_lo = i0_start / 2 + blk_idx * 16 + bp_in_blk;
            const int d_h2_hi = d_h2_lo + 8;

            half2 val_lo, val_hi;
            if (oob_check && t >= k_VKQ_sup) {
                val_lo = make_half2(0.0f, 0.0f);
                val_hi = make_half2(0.0f, 0.0f);
            } else {
                const int qs_blk_off = (i0_start / 32 + blk_idx) * 16;

                const uint16_t pair = *reinterpret_cast<const uint16_t *>(
                    row_t + V_qs_head_off + qs_blk_off + 2 * bp_in_blk);
                const uint8_t b0 = pair & 0xFF;
                const uint8_t b1 = pair >> 8;

                const uint8_t e_val = *(row_t + V_e_head_off + i0_start / 32 + blk_idx);
                const half2 scale_h2 = __float2half2_rn(ggml_cuda_e8m0_to_fp32(e_val));

                __nv_fp4x2_e2m1 fp4_lo;
                fp4_lo.__x = (b0 & 0x0F) | ((b1 & 0x0F) << 4);
                val_lo = __hmul2(__float22half2_rn(float2(fp4_lo)), scale_h2);

                __nv_fp4x2_e2m1 fp4_hi;
                fp4_hi.__x = (b0 >> 4) | (b1 & 0xF0);
                val_hi = __hmul2(__float22half2_rn(float2(fp4_hi)), scale_h2);
            }
            tile_V[t * stride_tile_V + d_h2_lo] = val_lo;
            tile_V[t * stride_tile_V + d_h2_hi] = val_hi;
        }
    }
}

// Load mask for MXFP4 kernel (synchronous, no cp.async).
template<int ncols1, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_load_mask(
        const half * const __restrict__ mask_h,
        half * const __restrict__ tile_mask,
        const int stride_mask,
        const int k_VKQ_sup,
        const int j0,
        const uint3 ne01) {
    if constexpr (oob_check) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                tile_mask[j_sram * (nbatch_fa + 8) + i] = i < k_VKQ_sup ? mask_h[j_vram * stride_mask + i] : half(0.0f);
            }
        }
    } else if constexpr (nbatch_fa < 2 * WARP_SIZE) {
        constexpr int cols_per_warp = 2 * WARP_SIZE / nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y * cols_per_warp + threadIdx.x / (WARP_SIZE / cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (WARP_SIZE / cols_per_warp);
            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram * (nbatch_fa + 8) + 2 * i, mask_h + j_vram * stride_mask + 2 * i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2 * WARP_SIZE) {
                const int i = i0 + 2 * threadIdx.x;
                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram * (nbatch_fa + 8) + i, mask_h + j_vram * stride_mask + i);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Q quantization: F32 → MXFP4 in shared memory
// ------------------------------------------------------------------------------------------------------------------

// Quantize Q: F32 global -> MXFP4 nibbles + E8M0 scales in shared memory.
template<int DKQ, int ncols, int nwarps, int stride_q_qs, int stride_q_sc>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_quantize_Q(
        const float2 * const __restrict__ Q_f2,
        int      * const __restrict__ tile_Q_qs,
        uint32_t * const __restrict__ tile_Q_sc,
        const float scale,
        const int stride_Q1,
        const int stride_Q2,
        const int ncols1,
        const int ncols2,
        const int jt,
        const int zt_gqa,
        const int gqa_ratio,
        const uint3 ne01) {
    constexpr int vals_per_block = QK_MXFP4;  // 32
    constexpr int blocks_per_col = DKQ / vals_per_block;

    // No early break: all warp threads stay in sync for __shfl_xor_sync scale exchange.
    constexpr int threads_total = nwarps * WARP_SIZE;
    constexpr int total_blocks = ncols * blocks_per_col;

#pragma unroll
    for (int b0 = 0; b0 < total_blocks; b0 += threads_total) {
        const int b = b0 + threadIdx.y * WARP_SIZE + threadIdx.x;

        const bool active = (b0 + threads_total <= total_blocks || b < total_blocks);

        const int jc = b / blocks_per_col;
        const int block_idx = b % blocks_per_col;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        bool valid = active &&
                     (ncols1 == 1 || jt * ncols1 + j < int(ne01.z)) &&
                     (ncols2 == 1 || zt_gqa * ncols2 + c < gqa_ratio);

        float vals[vals_per_block];
        if (valid) {
            const float2 * Q_col = Q_f2 + (jt * ncols1 + j) * stride_Q1 + c * stride_Q2;
            const int val_start = block_idx * vals_per_block;
#pragma unroll
            for (int i = 0; i < vals_per_block / 2; ++i) {
                const float2 tmp = Q_col[val_start / 2 + i];
                vals[2 * i + 0] = tmp.x * scale;
                vals[2 * i + 1] = tmp.y * scale;
            }
        } else {
#pragma unroll
            for (int i = 0; i < vals_per_block; ++i) {
                vals[i] = 0.0f;
            }
        }

        // Walsh-Hadamard rotation to match K-side rotation: H(Q).H(K)^T = Q.K^T.
        hadamard_32_inplace(vals);

        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < vals_per_block; ++i) {
            amax = fmaxf(amax, fabsf(vals[i]));
        }

        uint8_t e;
        float inv_d;
        if (!(amax > 0.0f)) {
            e = 0;
            inv_d = 0.0f;
        } else {
            constexpr int FP4_E2M1_EMAX = 2;
            const int e_int = __float2int_rn(log2f(amax));
            int biased = e_int - FP4_E2M1_EMAX + 127;
            biased = max(biased, 0);
            biased = min(biased, 254);
            e = static_cast<uint8_t>(biased);
            inv_d = __frcp_rn(ggml_cuda_e8m0_to_fp32(e));
        }

        // Pack 32 values into 4 ints of nibble data (low 16 in low nibbles, high 16 in high).
        if (active) {
#pragma unroll
            for (int i = 0; i < vals_per_block / 4; i += 2) {
                const int int_idx = block_idx * (vals_per_block / 8) + i / 2;

                // __nv_fp4x4_e2m1(float4) packs 4 values into 2 nibble-pair bytes.
                __nv_fp4x4_e2m1 fp4_lo(make_float4(
                    vals[0               + 2 * i + 0] * inv_d,
                    vals[vals_per_block/2 + 2 * i + 0] * inv_d,
                    vals[0               + 2 * i + 1] * inv_d,
                    vals[vals_per_block/2 + 2 * i + 1] * inv_d
                ));
                const char2 lo = *reinterpret_cast<const char2 *>(&fp4_lo);

                __nv_fp4x4_e2m1 fp4_hi(make_float4(
                    vals[0               + 2 * (i + 1) + 0] * inv_d,
                    vals[vals_per_block/2 + 2 * (i + 1) + 0] * inv_d,
                    vals[0               + 2 * (i + 1) + 1] * inv_d,
                    vals[vals_per_block/2 + 2 * (i + 1) + 1] * inv_d
                ));
                const char2 hi = *reinterpret_cast<const char2 *>(&fp4_hi);

                const uint32_t lo_u16 = *reinterpret_cast<const uint16_t *>(&lo);
                const uint32_t hi_u16 = *reinterpret_cast<const uint16_t *>(&hi);
                tile_Q_qs[jc * stride_q_qs + int_idx] = lo_u16 | (hi_u16 << 16);
            }
        }

        // Exchange scales between even/odd block partners via XOR-1 shuffle.
        const uint8_t e_partner = __shfl_xor_sync(0xFFFFFFFF, e, 1);

        if (active && block_idx % 2 == 0) {
            const int scale_pair_idx = block_idx / 2;
            tile_Q_sc[jc * stride_q_sc + scale_pair_idx] = (uint32_t)e | ((uint32_t)e_partner << 8);
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Main iteration function
// ------------------------------------------------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool needs_fixup, bool is_fixup, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_iter(
        const float2         * const __restrict__ Q_f2,
        const char           * const __restrict__ K_row_base,
        const int K_qs_head_off,
        const int K_e_head_off,
        const int K_sign_head_off,
        const int K_res_e_head_off,
        const int K_pos_stride,
        const char           * const __restrict__ V_row_base,
        const int V_qs_head_off,
        const int V_e_head_off,
        const int V_pos_stride,
        const half           * const __restrict__ mask_h,
        float2               * const __restrict__ dstk,
        float2               * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_mask,
        int      * const __restrict__ tile_Q_qs,
        uint32_t * const __restrict__ tile_Q_sc,
        int      * const __restrict__ tile_K_qs,
        uint32_t * const __restrict__ tile_K_sc,
        int      * const __restrict__ tile_K_qs_next,
        uint32_t * const __restrict__ tile_K_sc_next,
        half2    * const __restrict__ tile_V,
        half     * const __restrict__ tile_mask,
        half     * const __restrict__ tile_mask_next,
        tile<16, 8, float>   * const __restrict__ VKQ_C,
        float                * const __restrict__ KQ_max,
        float                * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup,
        const bool last_iter,
        const int k_VKQ_sup_next) {
#ifdef BLACKWELL_MMA_AVAILABLE
    constexpr int ncols          = ncols1 * ncols2;
    constexpr int cols_per_warp  = 8;
    constexpr int np             = nwarps * cols_per_warp / ncols;
    constexpr int nbatch_fa      = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_fa;
    constexpr int nbatch_V2      = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_V2;

    constexpr int stride_q_qs    = DKQ / 8 + 4;
    constexpr int stride_q_sc    = DKQ / 64;
    constexpr int stride_k_qs    = DKQ / 8 + 4;
    constexpr int stride_k_sc    = DKQ / 64;
    constexpr int stride_tile_V  = nbatch_V2 + 4;

    using T_A_KQ  = tile<16, 8, int>;    // FP4 MMA operands.
    using T_B_KQ  = tile< 8, 8, int>;
    using T_C_KQ  = tile<16, 8, float>;

    using T_A_VKQ = tile<16, 8, half2>; // F16 MMA for VKQ.
    using T_B_VKQ = tile< 8, 8, half2>;
    using T_C_VKQ = tile<16, 8, float>;

    const int k_VKQ_0 = kb0 * nbatch_fa;

    constexpr int k_res_byte_offset = 2 * nbatch_fa * (stride_k_qs * (int)sizeof(int) + stride_k_sc * (int)sizeof(uint32_t));
    int      * tile_K_res_qs      = (int      *)((char *)tile_K_qs      + k_res_byte_offset);
    uint32_t * tile_K_res_sc      = (uint32_t *)((char *)tile_K_sc      + k_res_byte_offset);

    // ---- Phase 1: KQ MMA (K_curr and K_res_curr already loaded and synced by caller) ----

    T_C_KQ KQ_C[nbatch_fa / (np * T_C_KQ::I)];

    constexpr int kq_iters = DKQ / 64;

#pragma unroll
    for (int d0 = 0; d0 < kq_iters; ++d0) {
        T_B_KQ Q_B;
        load_ldmatrix(Q_B, tile_Q_qs + (threadIdx.y / np) * cols_per_warp * stride_q_qs + d0 * 8, stride_q_qs);

#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np * T_A_KQ::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np) * T_A_KQ::I;

            const int k_row = i_KQ_0 + (threadIdx.x / 4) + (threadIdx.x % 2) * 8;
            const int q_col = (threadIdx.y / np) * cols_per_warp + (threadIdx.x / 4);
            const uint32_t b_scale = tile_Q_sc[q_col * stride_q_sc + d0];

            // Primary K MMA
            {
                T_A_KQ K_A;
                load_ldmatrix(K_A, tile_K_qs + i_KQ_0 * stride_k_qs + d0 * 8, stride_k_qs);
                const uint32_t a_scale = tile_K_sc[k_row * stride_k_sc + d0];
                mma_block_scaled(KQ_C[i_KQ_00 / (np * T_A_KQ::I)], K_A, Q_B, a_scale, b_scale);
            }

            // Residual K MMA (same Q_B, same b_scale, accumulates into same KQ_C)
            {
                T_A_KQ K_res_A;
                load_ldmatrix(K_res_A, tile_K_res_qs + i_KQ_0 * stride_k_qs + d0 * 8, stride_k_qs);
                const uint32_t a_scale_res = tile_K_res_sc[k_row * stride_k_sc + d0];
                mma_block_scaled(KQ_C[i_KQ_00 / (np * T_A_KQ::I)], K_res_A, Q_B, a_scale_res, b_scale);
            }
        }
    }

    // ---- Phase 2a: Preload K[i+1], K_res[i+1], mask[i+1] + V[curr] ----

    if (!last_iter) {
        const int k_VKQ_next = (kb0 + 1) * nbatch_fa;

        flash_attn_ext_mxfp4_load_K<DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
            (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
             tile_K_qs_next, tile_K_sc_next, k_VKQ_next, k_VKQ_sup_next);

        flash_attn_ext_mxfp4_load_K_res_compact<DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
            (K_row_base, K_sign_head_off, K_res_e_head_off, K_pos_stride,
             (int *)((char *)tile_K_qs_next + k_res_byte_offset),
             (uint32_t *)((char *)tile_K_sc_next + k_res_byte_offset),
             k_VKQ_next, k_VKQ_sup_next);

        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_mxfp4_load_mask<ncols1, nwarps, nbatch_fa, oob_check>
                (mask_h + k_VKQ_next, tile_mask_next, stride_mask, k_VKQ_sup_next, jt * ncols1, ne01);
        }
    }

    // V load for current iteration: issue memory requests early so softmax hides latency.
    static_assert(DV == 2 * nbatch_V2, "V outer loop assumption: DV must equal 2*nbatch_V2");
    flash_attn_ext_mxfp4_load_V_f16<DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
        (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
         tile_V, k_VKQ_0, 0, k_VKQ_sup);

    // ---- Phase 2b: Softmax + VKQ rescale (reads mask_curr, pure register work) ----

    if (use_logit_softcap) {
#pragma unroll
        for (int i = 0; i < nbatch_fa / (np * T_C_KQ::I); ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap * tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[2];
    KQ_max_new[0] = KQ_max[0];
    KQ_max_new[1] = KQ_max[1];
    float KQ_rowsum_add[2] = {0.0f, 0.0f};

    if (ncols2 > 1 || mask_h) {
#pragma unroll
        for (int i00 = 0; i00 < nbatch_fa; i00 += np * T_C_KQ::I) {
            const int i0 = i00 + (threadIdx.y % np) * T_C_KQ::I;
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int i = i0 + T_C_KQ::get_i(l);
                const int j = ((threadIdx.y / np) * T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;
                KQ_C[i00 / (np * T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j * (nbatch_fa + 8) + i]);
            }
        }
    }

    static_assert(nbatch_fa % (np * T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np * T_C_KQ::I) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (!oob_check || k0 + (threadIdx.y % np) * T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                const int KQ_idx = l % 2;
                KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0 / (np * T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < 2; ++col) {
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
        }
    }

    static_assert(nbatch_fa % (np * T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np * T_C_KQ::I) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (!oob_check || k0 + (threadIdx.y % np) * T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                const int KQ_idx = l % 2;
                KQ_C[k0 / (np * T_C_KQ::I)].x[l] = fast_expf(KQ_C[k0 / (np * T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                KQ_rowsum_add[KQ_idx] += KQ_C[k0 / (np * T_C_KQ::I)].x[l];
            } else {
                KQ_C[k0 / (np * T_C_KQ::I)].x[l] = 0.0f;
            }
        }
    }

    {
        float KQ_max_scale[2];
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];
            *((uint32_t *)&KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#pragma unroll
        for (int i = 0; i < DV / T_C_VKQ::I; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale[l % 2];
            }
        }
    }

    T_B_VKQ B_VKQ[nbatch_fa / (np * 2 * T_B_VKQ::J)];
    static_assert(nbatch_fa % (np * 2 * T_B_VKQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k = 0; k < nbatch_fa / (np * 2 * T_B_VKQ::J); ++k) {
        B_VKQ[k] = get_transposed(get_half2(KQ_C[k]));
    }

    __syncthreads(); // Ensures K[next], mask[next], V writes complete.

    // ---- Phase 3: VKQ MMA ----

    {
        constexpr int i0_stride = T_C_VKQ::I;
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < DV; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa / 2) % (np * T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa / 2; k00 += np * T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np) * T_A_VKQ::J;

                T_A_VKQ A;
                load_ldmatrix_trans(A, tile_V + 2 * k0 * stride_tile_V + i_VKQ_0 / 2, stride_tile_V);
                mma(VKQ_C[i_VKQ_0 / i0_stride], A, B_VKQ[k00 / (np * T_A_VKQ::J)]);
            }
        }
    }

    __syncthreads(); // Ensures V reads complete before next iteration overwrites tile_V.

#else
    GGML_UNUSED_VARS(Q_f2, K_row_base, K_qs_head_off, K_e_head_off,
        K_sign_head_off, K_res_e_head_off, K_pos_stride,
        V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
        mask_h, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, stride_mask,
        tile_Q_qs, tile_Q_sc, tile_K_qs, tile_K_sc, tile_K_qs_next, tile_K_sc_next, tile_V, tile_mask, tile_mask_next,
        VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, last_iter, k_VKQ_sup_next);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------
// Process tile: orchestrates Q loading, iteration loop, and result writeback.
// ------------------------------------------------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_process_tile(
        const float2         * const __restrict__ Q_f2,
        const char           * const __restrict__ K_row_base,
        const int K_qs_head_off,
        const int K_e_head_off,
        const int K_pos_stride,
        const char           * const __restrict__ V_row_base,
        const int V_qs_head_off,
        const int V_e_head_off,
        const int V_pos_stride,
        const half           * const __restrict__ mask_h,
        const float          * const __restrict__ sinks_f,
        float2               * const __restrict__ dstk,
        float2               * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop) {
#ifdef BLACKWELL_MMA_AVAILABLE
    constexpr int ncols          = ncols1 * ncols2;
    constexpr int cols_per_warp  = 8;
    constexpr int np             = nwarps * cols_per_warp / ncols;
    constexpr int nbatch_fa      = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_fa;
    constexpr int nbatch_V2      = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_V2;
    constexpr int nbatch_combine = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_combine;

    using T_B_KQ      = tile< 8, 8, int>;
    using T_C_KQ      = tile<16, 8, float>;
    using T_C_VKQ     = tile<16, 8, float>;
    using T_C_VKQ_h2  = tile<16, 4, half2>;
    using T_B_combine = tile< 8, 8, half2>;

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp / ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_q_qs    = DKQ / 8 + 4;
    constexpr int stride_q_sc    = DKQ / 64;
    constexpr int stride_k_qs    = DKQ / 8 + 4;
    constexpr int stride_k_sc    = DKQ / 64;

    constexpr int stride_tile_V = nbatch_V2 + 4;

    extern __shared__ char smem_mxfp4[];

    int      * tile_Q_qs = (int      *) smem_mxfp4;
    uint32_t * tile_Q_sc = (uint32_t *)(tile_Q_qs + ncols * stride_q_qs);

    int      * tile_K_qs_A     = (int      *)(tile_Q_sc + ncols * stride_q_sc);
    uint32_t * tile_K_sc_A     = (uint32_t *)(tile_K_qs_A + nbatch_fa * stride_k_qs);
    int      * tile_K_qs_B     = (int      *)(tile_K_sc_A + nbatch_fa * stride_k_sc);
    uint32_t * tile_K_sc_B     = (uint32_t *)(tile_K_qs_B + nbatch_fa * stride_k_qs);
    int      * tile_K_res_qs_A = (int      *)(tile_K_sc_B + nbatch_fa * stride_k_sc);
    uint32_t * tile_K_res_sc_A = (uint32_t *)(tile_K_res_qs_A + nbatch_fa * stride_k_qs);
    int      * tile_K_res_qs_B = (int      *)(tile_K_res_sc_A + nbatch_fa * stride_k_sc);
    uint32_t * tile_K_res_sc_B = (uint32_t *)(tile_K_res_qs_B + nbatch_fa * stride_k_qs);
    half2    * tile_V      = (half2 *)(tile_K_res_sc_B + nbatch_fa * stride_k_sc);
    constexpr int nbytes_V = nbatch_fa * stride_tile_V * (int)sizeof(half2);
    constexpr int nbytes_mask = ncols1 * (nbatch_fa + 8) * (int)sizeof(half);
    half     * tile_mask_A = (half *)((char *)tile_V + nbytes_V);
    half     * tile_mask_B = (half *)((char *)tile_mask_A + nbytes_mask);

    T_C_VKQ VKQ_C[DV / T_C_VKQ::I];

    float KQ_rowsum[2] = {0.0f};
    float KQ_max[2];
    KQ_max[0] = -FLT_MAX / 2.0f;
    KQ_max[1] = -FLT_MAX / 2.0f;

    flash_attn_ext_mxfp4_quantize_Q<DKQ, ncols, nwarps, stride_q_qs, stride_q_sc>
        (Q_f2, tile_Q_qs, tile_Q_sc, scale, stride_Q1, stride_Q2, ncols1, ncols2, jt, zt_gqa, gqa_ratio, ne01);

    __syncthreads();

    // Double-buffered iteration: K[i+1] loads overlap with softmax on K[i].
    int kb0 = kb0_start;
    int      * tile_K_qs_curr = tile_K_qs_A;
    uint32_t * tile_K_sc_curr = tile_K_sc_A;
    int      * tile_K_qs_next = tile_K_qs_B;
    uint32_t * tile_K_sc_next = tile_K_sc_B;
    half     * tile_mask_curr = tile_mask_A;
    half     * tile_mask_next = tile_mask_B;

    constexpr int k_res_byte_offset = 2 * nbatch_fa * (stride_k_qs * (int)sizeof(int) + stride_k_sc * (int)sizeof(uint32_t));

    // Compact 1-bit sign residual offsets within SoA row.
    constexpr int blocks_per_head_K_local = DKQ / QK_MXFP4;
    const int n_head_kv_local = ne02 / gqa_ratio;
    const int blocks_per_row_primary = n_head_kv_local * blocks_per_head_K_local;
    const int compact_qs_start = blocks_per_row_primary * 16;
    const int head_block_start = K_qs_head_off / 16;
    const int K_sign_head_off  = compact_qs_start + head_block_start * 4;
    const int K_res_e_head_off = compact_qs_start + blocks_per_row_primary * 4 + head_block_start;

    if constexpr (ncols2 == 1) {
        constexpr bool oob_check = true;

        // Preload K[0], K_res[0], mask[0] into buffer A.
        {
            constexpr int k_VKQ_sup_v = nbatch_fa;
            const int k_VKQ_0 = kb0_start * nbatch_fa;

            if (mask_h) {
                flash_attn_ext_mxfp4_load_mask<ncols1, nwarps, nbatch_fa, oob_check>
                    (mask_h + k_VKQ_0, tile_mask_curr, stride_mask, k_VKQ_sup_v, jt * ncols1, ne01);
            }
            flash_attn_ext_mxfp4_load_K<DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
                (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
                 tile_K_qs_curr, tile_K_sc_curr, k_VKQ_0, k_VKQ_sup_v);
            flash_attn_ext_mxfp4_load_K_res_compact<DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
                (K_row_base, K_sign_head_off, K_res_e_head_off, K_pos_stride,
                 (int *)((char *)tile_K_qs_curr + k_res_byte_offset),
                 (uint32_t *)((char *)tile_K_sc_curr + k_res_byte_offset),
                 k_VKQ_0, k_VKQ_sup_v);
            __syncthreads();
        }

        for (; kb0 < kb0_stop - 1; ++kb0) {
            constexpr int k_VKQ_sup_v = nbatch_fa;
            const int k_VKQ_sup_next = (kb0 + 1 == kb0_stop - 1) ? (ne11 - (kb0 + 1) * nbatch_fa) : nbatch_fa;

            flash_attn_ext_mxfp4_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
                (Q_f2, K_row_base, K_qs_head_off, K_e_head_off,
                 K_sign_head_off, K_res_e_head_off, K_pos_stride,
                 V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
                 mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_mask,
                 tile_Q_qs, tile_Q_sc, tile_K_qs_curr, tile_K_sc_curr,
                 tile_K_qs_next, tile_K_sc_next,
                 tile_V, tile_mask_curr, tile_mask_next,
                 VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v,
                 false, k_VKQ_sup_next);

            // Swap double buffers.
            { int      * tmp = tile_K_qs_curr; tile_K_qs_curr = tile_K_qs_next; tile_K_qs_next = tmp; }
            { uint32_t * tmp = tile_K_sc_curr; tile_K_sc_curr = tile_K_sc_next; tile_K_sc_next = tmp; }
            { half     * tmp = tile_mask_curr; tile_mask_curr = tile_mask_next; tile_mask_next = tmp; }
        }
        {
            const int k_VKQ_sup_v = ne11 - kb0 * nbatch_fa;

            flash_attn_ext_mxfp4_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
                (Q_f2, K_row_base, K_qs_head_off, K_e_head_off,
                 K_sign_head_off, K_res_e_head_off, K_pos_stride,
                 V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
                 mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_mask,
                 tile_Q_qs, tile_Q_sc, tile_K_qs_curr, tile_K_sc_curr,
                 tile_K_qs_next, tile_K_sc_next,
                 tile_V, tile_mask_curr, tile_mask_next,
                 VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v,
                 true, 0);
        }
    } else {
        constexpr bool oob_check = false;

        // Preload K[0], K_res[0], mask[0] into buffer A.
        {
            constexpr int k_VKQ_sup_v = nbatch_fa;
            const int k_VKQ_0 = kb0_start * nbatch_fa;

            flash_attn_ext_mxfp4_load_mask<ncols1, nwarps, nbatch_fa, oob_check>
                (mask_h + k_VKQ_0, tile_mask_curr, stride_mask, k_VKQ_sup_v, jt * ncols1, ne01);
            flash_attn_ext_mxfp4_load_K<DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
                (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
                 tile_K_qs_curr, tile_K_sc_curr, k_VKQ_0, k_VKQ_sup_v);
            flash_attn_ext_mxfp4_load_K_res_compact<DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
                (K_row_base, K_sign_head_off, K_res_e_head_off, K_pos_stride,
                 (int *)((char *)tile_K_qs_curr + k_res_byte_offset),
                 (uint32_t *)((char *)tile_K_sc_curr + k_res_byte_offset),
                 k_VKQ_0, k_VKQ_sup_v);
            __syncthreads();
        }

        for (; kb0 < kb0_stop - 1; ++kb0) {
            constexpr int k_VKQ_sup_v = nbatch_fa;

            flash_attn_ext_mxfp4_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
                (Q_f2, K_row_base, K_qs_head_off, K_e_head_off,
                 K_sign_head_off, K_res_e_head_off, K_pos_stride,
                 V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
                 mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_mask,
                 tile_Q_qs, tile_Q_sc, tile_K_qs_curr, tile_K_sc_curr,
                 tile_K_qs_next, tile_K_sc_next,
                 tile_V, tile_mask_curr, tile_mask_next,
                 VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v,
                 false, nbatch_fa);

            // Swap double buffers.
            { int      * tmp = tile_K_qs_curr; tile_K_qs_curr = tile_K_qs_next; tile_K_qs_next = tmp; }
            { uint32_t * tmp = tile_K_sc_curr; tile_K_sc_curr = tile_K_sc_next; tile_K_sc_next = tmp; }
            { half     * tmp = tile_mask_curr; tile_mask_curr = tile_mask_next; tile_mask_next = tmp; }
        }
        {
            constexpr int k_VKQ_sup_v = nbatch_fa;

            flash_attn_ext_mxfp4_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
                (Q_f2, K_row_base, K_qs_head_off, K_e_head_off,
                 K_sign_head_off, K_res_e_head_off, K_pos_stride,
                 V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
                 mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_mask,
                 tile_Q_qs, tile_Q_sc, tile_K_qs_curr, tile_K_sc_curr,
                 tile_K_qs_next, tile_K_sc_next,
                 tile_V, tile_mask_curr, tile_mask_next,
                 VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v,
                 true, 0);
        }
    }

    {
#pragma unroll
        for (int col = 0; col < 2; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[2];
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const int jc = T_C_KQ::get_j(col);
            const float sink = sinks_f[jc % ncols2];
            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;
            *((uint32_t *)&KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;
            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_max_add;
        }

#pragma unroll
        for (int i = 0; i < DV / T_C_VKQ::I; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale[l % 2];
            }
        }
    }

    // Write results: re-use Q shared memory region for combining.
    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV / 2) % nbatch_combine == 0, "bad nbatch_combine");

    half2 * tile_combine = (half2 *)smem_mxfp4;

    {
        const int jc_cwmo = (threadIdx.x % (2 * T_C_VKQ_h2::J)) / T_C_VKQ_h2::J;
        const int jc_cwm  = threadIdx.y * (2 * T_C_VKQ_h2::J) + 2 * T_C_VKQ_h2::get_j(-1) + jc_cwmo;
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]);

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2 * T_C_VKQ_h2::J) {
            ((float2 *)tile_combine)[jc_cwm * (tile_stride / 2) + nbatch_combine / 2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            if (needs_fixup && threadIdx.x < cols_per_warp) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x * ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < cols_per_warp) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x) * ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1 && threadIdx.y % np == 0) {
        constexpr int nmeta = np * cols_per_warp >= WARP_SIZE ? np * cols_per_warp / WARP_SIZE : 1;
        const int jc_meta = threadIdx.y * cols_per_warp + (np * cols_per_warp < WARP_SIZE ? threadIdx.x % (np * cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *)tile_combine) + jc_meta * (tile_stride / 2) + nbatch_combine / 2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * tile_stride / 2];
        }

        float KQ_cmn = meta[0].x;
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np * cols_per_warp / 2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
            }
        }

        float KQ_cms[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0] * meta[0].y;
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta] * meta[imeta].y;
        }
#pragma unroll
        for (int offset = np * cols_per_warp / 2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
            }
        }

#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np * cols_per_warp >= WARP_SIZE || threadIdx.x < np * cols_per_warp) {
                meta_ptr[imeta * WARP_SIZE * tile_stride / 2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x * ncols;
            dstk_fixup_meta[(threadIdx.y / np) * cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x) * ncols;
            dstk_fixup_meta[(threadIdx.y / np) * cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV / 2; k00 += nbatch_combine) {
        {
            const int jc_cwd = threadIdx.y * T_B_combine::I + T_B_combine::get_i(-1);
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_combine::J) {
                const T_B_combine B_tmp = get_transposed(get_half2(VKQ_C[(k00 + k1) / T_B_combine::J]));
#pragma unroll
                for (int l = 0; l < T_B_combine::ne; ++l) {
                    const int k = k1 + T_B_combine::get_j(l);
                    tile_combine[jc_cwd * tile_stride + k] = B_tmp.x[l];
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x * (2 * ncols) + blockIdx.x * (ncols * (DV / 2));

#pragma unroll
            for (int stride_k : {WARP_SIZE, WARP_SIZE / 2, WARP_SIZE / 4}) {
                const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2 * stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1 * stride_k);
                const int stride_jc = WARP_SIZE / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps / np) * stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y / np) * stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps / np) * stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst / cols_per_warp) * (np * cols_per_warp) + jc_dst % cols_per_warp;
                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt * ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa * ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *)tile_combine + jc_tile_K * tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip * cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_combine[(jc_tile_K + ip * cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x * KQ_crs;
                            dstk_val.y += dstk_val_add.y * KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst * (DV / 2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt * ncols1 + j_dst) * ne02 + c_dst) * (DV / 2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
        V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
        mask_h, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_mask,
        jt, zt_gqa, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------
// Global kernel entry point
// ------------------------------------------------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols1*ncols2).nthreads,
                  ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols1*ncols2).occupancy)
static __global__ void flash_attn_ext_mxfp4(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef BLACKWELL_MMA_AVAILABLE
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_fa;
    constexpr int nthreads  = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nthreads;
    constexpr int nwarps    = nthreads / WARP_SIZE;

    const int gqa_ratio = ne02 / ne12;

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_mask = nb31 / sizeof(half);

    constexpr int blocks_per_head_K = DKQ / QK_MXFP4;
    constexpr int blocks_per_head_V = DV  / QK_MXFP4;
    const int stride_K_blocks = nb11 / sizeof(block_mxfp4);
    const int stride_V_blocks = nb21 / sizeof(block_mxfp4);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    int       kbc      = int64_t(blockIdx.x + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;

    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence =  kbc / (iter_k * iter_j * iter_z_gqa * ne12);
        const int z_KV     = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence) / (iter_k * iter_j * iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV) / (iter_k * iter_j);
        const int jt       = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV - iter_k * iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

        const float2 * Q_f2   = (const float2 *)(Q + nb03 * sequence + nb02 * zt_Q);
        const half   * mask_h  = ncols2 == 1 && !mask ? nullptr :
            (const half *)(mask + nb33 * (sequence % ne33));
        float2       * dstk    = ((float2 *)dst) + (sequence * ne01.z * ne02 + zt_Q) * (DV / 2);
        const float  * sinks_f = sinks ? (const float *)sinks + zt_Q : nullptr;

        const char * K_row_base    = K + nb13 * sequence;
        const int K_qs_head_off    = z_KV * blocks_per_head_K * 16;
        const int K_e_head_off     = stride_K_blocks * 16 + z_KV * blocks_per_head_K;
        const char * V_row_base    = V + nb23 * sequence;
        const int V_qs_head_off    = z_KV * blocks_per_head_V * 16;
        const int V_e_head_off     = stride_V_blocks * 16 + z_KV * blocks_per_head_V;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence * iter_j + jt] / nbatch_fa);
        }

        constexpr bool is_fixup = false;
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false;
            flash_attn_ext_mxfp4_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_row_base, K_qs_head_off, K_e_head_off, nb11,
                 V_row_base, V_qs_head_off, V_e_head_off, nb21,
                 mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true;
            flash_attn_ext_mxfp4_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_row_base, K_qs_head_off, K_e_head_off, nb11,
                 V_row_base, V_qs_head_off, V_e_head_off, nb21,
                 mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int sequence =  kbc / (iter_k * iter_j * iter_z_gqa * ne12);
    const int z_KV     = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence) / (iter_k * iter_j * iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV) / (iter_k * iter_j);
    const int jt       = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV - iter_k * iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

    const float2 * Q_f2   = (const float2 *)(Q + nb03 * sequence + nb02 * zt_Q);
    const half   * mask_h  = ncols2 == 1 && !mask ? nullptr :
        (const half *)(mask + nb33 * (sequence % ne33));
    float2       * dstk    = ((float2 *)dst) + (sequence * ne01.z * ne02 + zt_Q) * (DV / 2);
    const float  * sinks_f = sinks ? (const float *)sinks + zt_Q : nullptr;

    // SoA KV pointers: row base (sequence only, no head offset) + per-head qs/e offsets.
    const char * K_row_base    = K + nb13 * sequence;
    const int K_qs_head_off    = z_KV * blocks_per_head_K * 16;
    const int K_e_head_off     = stride_K_blocks * 16 + z_KV * blocks_per_head_K;
    const char * V_row_base    = V + nb23 * sequence;
    const int V_qs_head_off    = z_KV * blocks_per_head_V * 16;
    const int V_e_head_off     = stride_V_blocks * 16 + z_KV * blocks_per_head_V;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence * iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup    = true;
    constexpr bool needs_fixup = false;
    flash_attn_ext_mxfp4_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_row_base, K_qs_head_off, K_e_head_off, nb11,
         V_row_base, V_qs_head_off, V_e_head_off, nb21,
         mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------
// Host launch function
// ------------------------------------------------------------------------------------------------------------------

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_mxfp4_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();

    constexpr int ncols = ncols1 * ncols2;

    const fattn_mma_config config = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols);
    const int  nthreads       = config.nthreads;
    const int  nbatch_fa      = config.nbatch_fa;
    const int  nbatch_V2      = config.nbatch_V2;
    const int  nbatch_combine = config.nbatch_combine;

    constexpr int cols_per_warp = 8;
    const int nwarps = nthreads / WARP_SIZE;

    // Shared memory size calculation — MUST match device-side exactly.
    constexpr int stride_q_qs   = DKQ / 8 + 4;
    constexpr int stride_q_sc   = DKQ / 64;
    const     int stride_k_qs   = DKQ / 8 + 4;
    const     int stride_k_sc   = DKQ / 64;

    // F16 V layout stride (must match device-side).
    const int stride_tile_V = nbatch_V2 + 4;

    const size_t nbytes_Q_qs    = ncols     * stride_q_qs   * sizeof(int);
    const size_t nbytes_Q_sc    = ncols     * stride_q_sc   * sizeof(uint32_t);
    const size_t nbytes_K_qs    = nbatch_fa * stride_k_qs   * sizeof(int);
    const size_t nbytes_K_sc    = nbatch_fa * stride_k_sc   * sizeof(uint32_t);
    // V F16 region.
    const size_t nbytes_V       = nbatch_fa * stride_tile_V * sizeof(half2);
    const size_t nbytes_mask    = ncols1    * (nbatch_fa + 8) * sizeof(half);

    const size_t nbytes_Q_region    = nbytes_Q_qs + nbytes_Q_sc;
    const size_t nbytes_K_double    = 4 * (nbytes_K_qs + nbytes_K_sc);    // Double-buffered K + K_res (A + B × 2).
    const size_t nbytes_mask_double = 2 * nbytes_mask;                     // Double-buffered mask (A + B).
    const size_t nbytes_KV_region   = nbytes_K_double + nbytes_V + nbytes_mask_double;

    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, nbytes_Q_region + nbytes_KV_region);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *)KQV->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_mxfp4<DKQ, DV, ncols1, ncols2, use_logit_softcap>;

        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(
                reinterpret_cast<fattn_kernel_t>(fattn_kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_mxfp4<DKQ, DV, ncols1, ncols2, use_logit_softcap>;

        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(
                reinterpret_cast<fattn_kernel_t>(fattn_kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
    }

    // need_f16_K=false, need_f16_V=false: MXFP4 kernel reads raw block_mxfp4 data directly.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true);
}


#define DECL_FATTN_MMA_MXFP4_CASE(DKQ, DV, ncols1, ncols2)                             \
    template void ggml_cuda_flash_attn_ext_mma_mxfp4_case                              \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)       \

#define DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2(DKQ, DV, ncols)    \
    extern DECL_FATTN_MMA_MXFP4_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_MXFP4_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_MXFP4_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_MXFP4_CASE(DKQ, DV, (ncols)/ 8,  8); \

DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2( 64,  64,  8)
DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2(128, 128,  8)
DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2(256, 256,  8)

DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2( 64,  64, 16)
DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2(128, 128, 16)
DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2(256, 256, 16)

DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2( 64,  64, 32)
DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2(128, 128, 32)
DECL_FATTN_MMA_MXFP4_CASE_ALL_NCOLS2(256, 256, 32)
