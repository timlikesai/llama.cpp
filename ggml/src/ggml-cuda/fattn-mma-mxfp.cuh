#pragma once

// Unified MXFP flash attention MMA kernel for all OCP Microscaling (MX) formats.
//
// References:
//   - MX format specification: OCP Microscaling Formats (MX) Specification v1.0
//     https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
//     Rouhani et al., "Microscaling Data Formats for Deep Learning", arXiv:2310.10537
//   - Fast exp approximation: adapted from Flash Attention (Tri Dao, Dao-AILab/flash-attention).
//     Cubic Remez minimax polynomial for 2^x on [0,1]; avoids SFU bottleneck.
//     Building on: Schraudolph 1999, "A Fast, Compact Approximation of the Exponential Function"
//   - Walsh-Hadamard KV cache rotation:
//     Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arXiv:2404.00456
//     Zhang et al., "Block Rotation is All You Need for MXFP4 Quantization", arXiv:2511.04214
//       (block-32 Hadamard matching MX block size; global rotation hurts MXFP4 due to PoT E8M0)
//     Dao et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony
//       and Low-precision", arXiv:2407.08608 (incoherent processing with Hadamard for FP8 attention)
//   - KV cache quantization:
//     Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", arXiv:2402.02750
//       (per-channel K, per-token V — foundational asymmetric KV quantization)
//     Li et al., "KVTuner: Sensitivity-Aware Mixed-Precision KV Cache", arXiv:2502.04420
//       (theoretical analysis: K quantization errors more damaging than V; supports mixed K/V precision)
//     Saxena et al., "KVLinC: KV Cache Quantization with Hadamard Rotation", arXiv:2510.05373
//   - E8M0 scale and block formats:
//     Kim et al., "Bridging the Gap for Microscaling FP4 Quantization", arXiv:2509.23202
//       (outlier amplification and group asymmetry as root causes of MXFP4 degradation)
//     INT vs FP: Park et al., "Comprehensive Study of Fine-Grained Quantization", arXiv:2510.25602
//       (MXFP6 as accuracy sweet spot; MXINT8 outperforms MXFP8 at 8-bit without rotation)
//   - MLA attention:
//     Liu et al., "SnapMLA: Efficient Long-Context MLA Decoding", arXiv:2602.10718
//       (RoPE-aware per-token KV quantization; V scale fusion into P matrix for MLA)
//   - SageAttention3: Jia et al., "Microscaling FP4 Attention on Blackwell", arXiv:2505.11594
//       (NeurIPS 2025 Spotlight; FP4 QKV quantization inside attention, 5x over FlashAttention-3)

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "hadamard.cuh"
#include "mxfp-traits.cuh"

using namespace ggml_cuda_mma;

// MMA-specific traits:
// Drive the unified flash attention MMA kernel. All MX formats share
// identical tile register layouts (4 A regs, 2 B regs, 4 C/D regs).
// Differences: MMA instruction, k-dimension, scale pairing, Q packing.

template<ggml_type type> struct mxfp_mma_traits;

template<> struct mxfp_mma_traits<GGML_TYPE_MXFP4_E2M1> {
    static constexpr int k_per_mma      = 64;    // m16n8k64
    static constexpr int smem_k_qs_div  = 8;     // stride_k_qs = DKQ/8 + 4
    static constexpr int smem_k_sc_div  = 64;    // stride_k_sc = DKQ/64 (paired scales)
    static constexpr int emax           = MXFP4_E2M1_EMAX_OFFSET;
    static constexpr bool can_cp_async_k = true;
    static constexpr bool needs_smem_expand_k = false;

    static __device__ __forceinline__ void mma_kq(
            tile<16, 8, float> & D, const tile<16, 8, int> & A,
            const tile<8, 8, int> & B, uint32_t a_sc, uint32_t b_sc) {
        mma_block_scaled(D, A, B, a_sc, b_sc);
    }
};

template<> struct mxfp_mma_traits<GGML_TYPE_MXFP6_E2M3> {
    static constexpr int k_per_mma      = 32;    // m16n8k32
    static constexpr int smem_k_qs_div  = 4;     // stride_k_qs = DKQ/4 + 4
    static constexpr int smem_k_sc_div  = 32;    // stride_k_sc = DKQ/32 (individual)
    static constexpr int emax           = MXFP6_E2M3_EMAX_OFFSET;
    static constexpr bool can_cp_async_k = true;  // raw packed bytes loaded as 16B chunks
    static constexpr bool needs_smem_expand_k = true; // packed→expanded in-place after async load

    static __device__ __forceinline__ void mma_kq(
            tile<16, 8, float> & D, const tile<16, 8, int> & A,
            const tile<8, 8, int> & B, uint32_t a_sc, uint32_t b_sc) {
        mma_block_scaled_f6_e2m3(D, A, B, a_sc, b_sc);
    }
};

template<> struct mxfp_mma_traits<GGML_TYPE_MXFP6_E3M2> {
    static constexpr int k_per_mma      = 32;
    static constexpr int smem_k_qs_div  = 4;
    static constexpr int smem_k_sc_div  = 32;
    static constexpr int emax           = MXFP6_E3M2_EMAX_OFFSET;
    static constexpr bool can_cp_async_k = true;  // raw packed bytes loaded as 16B chunks
    static constexpr bool needs_smem_expand_k = true; // packed→expanded in-place after async load

    static __device__ __forceinline__ void mma_kq(
            tile<16, 8, float> & D, const tile<16, 8, int> & A,
            const tile<8, 8, int> & B, uint32_t a_sc, uint32_t b_sc) {
        mma_block_scaled_f6_e3m2(D, A, B, a_sc, b_sc);
    }
};

template<> struct mxfp_mma_traits<GGML_TYPE_MXFP8_E4M3> {
    static constexpr int k_per_mma      = 32;    // m16n8k32
    static constexpr int smem_k_qs_div  = 4;     // stride_k_qs = DKQ/4 + 4
    static constexpr int smem_k_sc_div  = 32;    // stride_k_sc = DKQ/32 (individual)
    static constexpr int emax           = MXFP8_E4M3_EMAX_OFFSET;
    static constexpr bool can_cp_async_k = true;  // 32-byte blocks = 2x 16B cp.async
    static constexpr bool needs_smem_expand_k = false;

    static __device__ __forceinline__ void mma_kq(
            tile<16, 8, float> & D, const tile<16, 8, int> & A,
            const tile<8, 8, int> & B, uint32_t a_sc, uint32_t b_sc) {
        mma_block_scaled_f8(D, A, B, a_sc, b_sc);
    }
};

template<> struct mxfp_mma_traits<GGML_TYPE_MXFP8_E5M2> {
    static constexpr int k_per_mma      = 32;
    static constexpr int smem_k_qs_div  = 4;
    static constexpr int smem_k_sc_div  = 32;
    static constexpr int emax           = MXFP8_E5M2_EMAX_OFFSET;
    static constexpr bool can_cp_async_k = true;
    static constexpr bool needs_smem_expand_k = false;

    static __device__ __forceinline__ void mma_kq(
            tile<16, 8, float> & D, const tile<16, 8, int> & A,
            const tile<8, 8, int> & B, uint32_t a_sc, uint32_t b_sc) {
        mma_block_scaled_f8_e5m2(D, A, B, a_sc, b_sc);
    }
};

// V type discriminant (runtime dispatch — avoids 5×5 K/V template explosion).
enum {
    MXFP_V_FP4       = 0,  // MXFP4 E2M1
    MXFP_V_FP8_E4M3  = 1,  // MXFP8 E4M3
    MXFP_V_FP8_E5M2  = 2,  // MXFP8 E5M2
    MXFP_V_FP6_E2M3  = 3,  // MXFP6 E2M3
    MXFP_V_FP6_E3M2  = 4,  // MXFP6 E3M2
};

// Fast exp(x) via cubic polynomial approximation (Dao-AILab/flash-attention).
// Splits into 2^floor(x*log2e) (bit manipulation) * poly(frac) (3 FMAs).
// Coefficients: Remez minimax for 2^x on [0,1]. Max relative error ~0.3% (x <= 0).
// Avoids SFU bottleneck by using CUDA cores instead of the Special Function Unit.
static __device__ __forceinline__ float fast_expf_mxfp(const float x) {
    const float x_log2e = x * 1.4426950408889634f;
    const float xi = floorf(x_log2e);
    if (xi < -126.0f) {
        return 0.0f;
    }
    const float r  = x_log2e - xi;
    const float poly = fmaf(fmaf(fmaf(0.07711909f, r, 0.22756439f), r, 0.69514614f), r, 1.0f);
    const int exp_bits = ((int)xi + 127) << 23;
    float pow2_floor;
    memcpy(&pow2_floor, &exp_bits, sizeof(float));
    return pow2_floor * poly;
}

// Unified MXFP MMA config: identical tuning for all MX formats at standard D values.
template<ggml_type mxfp_type>
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_mxfp_get_config(
        const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2,  64,  32,  32,  32, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 3,  64,  64,  64,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);

    // MLA: D_K=576, D_V=512. Single-buffered (K/V share smem), nbatch_fa=32.
    // K and V share the same physical smem region, loaded sequentially.
    // smem ≈ 55 KB → occupancy=2 on 128 KB SM (vs occupancy=1 with double-buffer).
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  32, 3,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 2,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 288, 256, 128, 1, false);

    // Computed fallback for arbitrary D values.
    if (DKQ % 32 == 0 && DV % 32 == 0 && DKQ > 0 && DV > 0 && ncols > 0) {
        const int nwarps_min      = (ncols + 7) / 8;
        const int nthreads_       = nwarps_min * 32;
        const int nbatch_fa_      = 32;
        const int nbatch_V2_      = DV / 2;
        const int nbatch_K2_      = DKQ / 2;
        const int nbatch_combine_ = DV / 2 <= 128 ? DV / 2 : 128;
        const int occupancy_      = 2;
        return fattn_mma_config{nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_,
                                nbatch_combine_, 1, false};
    }
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

template<ggml_type mxfp_type>
static __host__ fattn_mma_config ggml_cuda_fattn_mma_mxfp_get_config(
        const int DKQ, const int DV, const int ncols, const int /*cc*/) {
    return ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols);
}

// ------------------------------------------------------------------------------------------------------------------
// K loading helpers
// ------------------------------------------------------------------------------------------------------------------

// Load K from global to shared memory (SoA layout). All types use cp.async.
// FP4: 16 bytes/block, 1x cp.async per block, paired scales.
// FP6: 24 bytes/block, cp.async 16B chunks (packed), expand later via expand_K_fp6, individual scales.
// FP8: 32 bytes/block, 2x 16B cp.async per block, individual scales.
template<ggml_type mxfp_type, int DKQ, int nwarps, int nbatch_fa, int stride_k_qs, int stride_k_sc, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp_load_K(
        const char * const __restrict__ K_row_base,
        const int K_qs_head_off,
        const int K_e_head_off,
        const int K_pos_stride,
        int      * const __restrict__ tile_K_qs,
        uint32_t * const __restrict__ tile_K_sc,
        const int k_VKQ_0,
        const int k_VKQ_sup) {
    using traits = mxfp_mma_traits<mxfp_type>;
    constexpr int blocks_per_head = DKQ / 32;

    if constexpr (mxfp_type == GGML_TYPE_MXFP4_E2M1) {
        // FP4: 16 bytes per block, 1x cp.async per block.
        const unsigned int tile_K_qs_32 = ggml_cuda_cvta_generic_to_shared(tile_K_qs);

#pragma unroll
        for (int flat0 = 0; flat0 < nbatch_fa * blocks_per_head; flat0 += nwarps * WARP_SIZE) {
            const int flat = flat0 + threadIdx.y * WARP_SIZE + threadIdx.x;

            if (flat0 + nwarps * WARP_SIZE > nbatch_fa * blocks_per_head && flat >= nbatch_fa * blocks_per_head) {
                break;
            }

            const int i = flat / blocks_per_head;
            const int b = flat % blocks_per_head;

            if (oob_check && i >= k_VKQ_sup) {
                *reinterpret_cast<int4 *>(&tile_K_qs[i * stride_k_qs + b * 4]) = make_int4(0, 0, 0, 0);
            } else {
                const char * row_i = K_row_base + int64_t(k_VKQ_0 + i) * K_pos_stride;
                const unsigned int smem_dst = tile_K_qs_32 + (i * stride_k_qs + b * 4) * (int)sizeof(int);
                cp_async_cg_16<128>(smem_dst, row_i + K_qs_head_off + b * 16);
            }
        }

        // E8M0 scales: paired (scale_vec::2X).
#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps) {
            const int i = i0 + threadIdx.y;
            if (i0 + nwarps > nbatch_fa && i >= nbatch_fa) {
                break;
            }
#pragma unroll
            for (int s0 = 0; s0 < blocks_per_head / 2; s0 += WARP_SIZE) {
                const int s = s0 + threadIdx.x;
                if (s0 + WARP_SIZE > blocks_per_head / 2 && s >= blocks_per_head / 2) {
                    break;
                }
                if (oob_check && i >= k_VKQ_sup) {
                    tile_K_sc[i * stride_k_sc + s] = 0;
                } else {
                    const char * row_i = K_row_base + int64_t(k_VKQ_0 + i) * K_pos_stride;
                    const uint8_t e0 = *(row_i + K_e_head_off + 2 * s);
                    const uint8_t e1 = *(row_i + K_e_head_off + 2 * s + 1);
                    tile_K_sc[i * stride_k_sc + s] = (uint32_t)e0 | ((uint32_t)e1 << 8);
                }
            }
        }
    } else {
        // FP6/FP8: qs loading differs (FP6: 24B packed chunks, FP8: 2x 16B per block).
        const unsigned int tile_K_qs_32 = ggml_cuda_cvta_generic_to_shared(tile_K_qs);
        constexpr bool is_fp6 = (mxfp_type == GGML_TYPE_MXFP6_E2M3 || mxfp_type == GGML_TYPE_MXFP6_E3M2);

        if constexpr (is_fp6) {
            // FP6: cp.async 16-byte chunks of packed data. Total qs per row = blocks_per_head * 24 bytes,
            // always a multiple of 16. Expansion to fp6x4 words happens later via expand_K_fp6.
            constexpr int chunks_per_row = blocks_per_head * 24 / 16;  // = 3*DKQ/64
            constexpr int total_chunks = nbatch_fa * chunks_per_row;

#pragma unroll
            for (int flat0 = 0; flat0 < total_chunks; flat0 += nwarps * WARP_SIZE) {
                const int flat = flat0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (flat0 + nwarps * WARP_SIZE > total_chunks && flat >= total_chunks) {
                    break;
                }

                const int i = flat / chunks_per_row;
                const int c = flat % chunks_per_row;

                if (oob_check && i >= k_VKQ_sup) {
                    *reinterpret_cast<int4 *>((char *)tile_K_qs + i * stride_k_qs * (int)sizeof(int) + c * 16) = make_int4(0, 0, 0, 0);
                } else {
                    const char * row_i = K_row_base + int64_t(k_VKQ_0 + i) * K_pos_stride;
                    const unsigned int smem_dst = tile_K_qs_32 + i * stride_k_qs * (int)sizeof(int) + c * 16;
                    cp_async_cg_16<128>(smem_dst, row_i + K_qs_head_off + c * 16);
                }
            }
        } else {
            // FP8 (E4M3 or E5M2): 32 bytes/block, 2x 16B cp.async.
            constexpr int total_halves = nbatch_fa * blocks_per_head * 2;
#pragma unroll
            for (int flat0 = 0; flat0 < total_halves; flat0 += nwarps * WARP_SIZE) {
                const int flat = flat0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (flat0 + nwarps * WARP_SIZE > total_halves && flat >= total_halves) {
                    break;
                }

                const int i    = flat / (blocks_per_head * 2);
                const int bh   = flat % (blocks_per_head * 2);
                const int b    = bh / 2;
                const int half_idx = bh % 2;

                if (oob_check && i >= k_VKQ_sup) {
                    *reinterpret_cast<int4 *>(&tile_K_qs[i * stride_k_qs + b * 8 + half_idx * 4]) = make_int4(0, 0, 0, 0);
                } else {
                    const char * row_i = K_row_base + int64_t(k_VKQ_0 + i) * K_pos_stride;
                    const unsigned int smem_dst = tile_K_qs_32 + (i * stride_k_qs + b * 8 + half_idx * 4) * (int)sizeof(int);
                    cp_async_cg_16<128>(smem_dst, row_i + K_qs_head_off + b * QK_MXFP8 + half_idx * 16);
                }
            }
        }

        // E8M0 scales: individual (scale_vec::1X). Shared by FP6 and FP8.
#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps) {
            const int i = i0 + threadIdx.y;
            if (i0 + nwarps > nbatch_fa && i >= nbatch_fa) {
                break;
            }
#pragma unroll
            for (int s0 = 0; s0 < blocks_per_head; s0 += WARP_SIZE) {
                const int s = s0 + threadIdx.x;
                if (s0 + WARP_SIZE > blocks_per_head && s >= blocks_per_head) {
                    break;
                }
                if (oob_check && i >= k_VKQ_sup) {
                    tile_K_sc[i * stride_k_sc + s] = 0;
                } else {
                    const char * row_i = K_row_base + int64_t(k_VKQ_0 + i) * K_pos_stride;
                    const uint8_t e = *(row_i + K_e_head_off + s);
                    tile_K_sc[i * stride_k_sc + s] = (uint32_t)e;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// FP6 in-place smem expansion: packed 24B/block → expanded 32B/block (8 fp6x4 words)
// ------------------------------------------------------------------------------------------------------------------

// After cp.async loads raw packed FP6 bytes into the first (blocks_per_head * 24) bytes of each
// smem row, this function expands them in-place to the (blocks_per_head * 32) bytes of fp6x4 words
// that ldmatrix/MMA expects. Uses __syncthreads between read and write phases for cross-warp safety,
// since expanded blocks partially overlap packed blocks within each row.
template<int DKQ, int nwarps, int nbatch_fa, int stride_k_qs>
static __device__ __forceinline__ void flash_attn_ext_mxfp_expand_K_fp6(
        int * const __restrict__ tile_K_qs) {
    constexpr int blocks_per_head = DKQ / 32;
    constexpr int total_blocks = nbatch_fa * blocks_per_head;
    constexpr int nthreads = nwarps * WARP_SIZE;
    constexpr int max_bpt = (total_blocks + nthreads - 1) / nthreads;

    // Phase 1: Read packed bytes (6 × uint32_t per block = 24 bytes) from smem into registers.
    uint32_t raw[6 * max_bpt];

#pragma unroll
    for (int iter = 0; iter < max_bpt; ++iter) {
        const int flat = iter * nthreads + threadIdx.y * WARP_SIZE + threadIdx.x;
        if (flat < total_blocks) {
            const int i   = flat / blocks_per_head;
            const int blk = flat % blocks_per_head;
            const uint32_t * packed = (const uint32_t *)((const char *)tile_K_qs
                + i * stride_k_qs * (int)sizeof(int) + blk * 24);
            raw[iter * 6 + 0] = packed[0];
            raw[iter * 6 + 1] = packed[1];
            raw[iter * 6 + 2] = packed[2];
            raw[iter * 6 + 3] = packed[3];
            raw[iter * 6 + 4] = packed[4];
            raw[iter * 6 + 5] = packed[5];
        }
    }

    __syncthreads();  // All reads done before any writes (cross-warp safety).

    // Phase 2: Expand 24 packed bytes → 8 fp6x4 words (32 bytes) and write to final locations.
    // Two 12-byte cycles per block, each yielding 4 fp6x4 words.
#pragma unroll
    for (int iter = 0; iter < max_bpt; ++iter) {
        const int flat = iter * nthreads + threadIdx.y * WARP_SIZE + threadIdx.x;
        if (flat < total_blocks) {
            const int i   = flat / blocks_per_head;
            const int blk = flat % blocks_per_head;

#pragma unroll
            for (int cycle = 0; cycle < 2; ++cycle) {
                const uint32_t a = raw[iter * 6 + cycle * 3 + 0];
                const uint32_t b = raw[iter * 6 + cycle * 3 + 1];
                const uint32_t c = raw[iter * 6 + cycle * 3 + 2];

                const uint32_t r0 =  a                      & 0xFFFFFF;
                const uint32_t r1 = ((a >> 24) | (b <<  8)) & 0xFFFFFF;
                const uint32_t r2 = ((b >> 16) | (c << 16)) & 0xFFFFFF;
                const uint32_t r3 =  (c >>  8)              & 0xFFFFFF;

                // Expand 4 packed 6-bit values from a 24-bit word into byte-per-element format (00xxxxxx).
                auto fp6_expand_word = [] __device__ (uint32_t r) -> int {
                    return (r & 0x3F) | (((r >> 6) & 0x3F) << 8) | (((r >> 12) & 0x3F) << 16) | (((r >> 18) & 0x3F) << 24);
                };

                const int base = i * stride_k_qs + blk * 8 + cycle * 4;
                tile_K_qs[base + 0] = fp6_expand_word(r0);
                tile_K_qs[base + 1] = fp6_expand_word(r1);
                tile_K_qs[base + 2] = fp6_expand_word(r2);
                tile_K_qs[base + 3] = fp6_expand_word(r3);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// V loading helpers
// ------------------------------------------------------------------------------------------------------------------

// Load V: MXFP4 FP4 -> dequant to F16 half2 directly into shared tile.
template<int DV, int nwarps, int nbatch_fa, int stride_tile_V, int nbatch_V2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp_load_V_f16(
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

#if CUDART_VERSION >= 12080
                __nv_fp4x2_e2m1 fp4_lo;
                fp4_lo.__x = (b0 & 0x0F) | ((b1 & 0x0F) << 4);
                val_lo = __hmul2(__float22half2_rn(float2(fp4_lo)), scale_h2);

                __nv_fp4x2_e2m1 fp4_hi;
                fp4_hi.__x = (b0 >> 4) | (b1 & 0xF0);
                val_hi = __hmul2(__float22half2_rn(float2(fp4_hi)), scale_h2);
#else
                val_lo = __hmul2(make_half2(kvalues_mxfp4[b0 & 0x0F] * 0.5f,
                                            kvalues_mxfp4[b1 & 0x0F] * 0.5f), scale_h2);
                val_hi = __hmul2(make_half2(kvalues_mxfp4[b0 >> 4] * 0.5f,
                                            kvalues_mxfp4[(b1 >> 4) & 0x0F] * 0.5f), scale_h2);
#endif // CUDART_VERSION >= 12080
            }
            tile_V[t * stride_tile_V + d_h2_lo] = val_lo;
            tile_V[t * stride_tile_V + d_h2_hi] = val_hi;
        }
    }
}

// Load V: MXFP8 FP8 -> dequant to F16 half2 directly into shared tile.
template<ggml_type v_mxfp8_type, int DV, int nwarps, int nbatch_fa, int stride_tile_V, int nbatch_V2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp_load_V_mxfp8_f16(
        const char * const __restrict__ V_row_base,
        const int V_qs_head_off,
        const int V_e_head_off,
        const int V_pos_stride,
        half2 * const __restrict__ tile_V,
        const int k_VKQ_0,
        const int i0_start,
        const int k_VKQ_sup) {
    constexpr int nelem_pairs = nbatch_V2;

#pragma unroll
    for (int t0 = 0; t0 < nbatch_fa; t0 += nwarps) {
        const int t = t0 + threadIdx.y;
        if (t0 + nwarps > nbatch_fa && t >= nbatch_fa) {
            break;
        }

        const char * row_t = V_row_base + int64_t(k_VKQ_0 + t) * V_pos_stride;

#pragma unroll
        for (int ep0 = 0; ep0 < nelem_pairs; ep0 += WARP_SIZE) {
            const int ep = ep0 + threadIdx.x;
            if (ep0 + WARP_SIZE > nelem_pairs && ep >= nelem_pairs) {
                break;
            }

            const int blk_idx     = ep / 16;
            const int pair_in_blk = ep % 16;
            const int d_h2 = i0_start / 2 + blk_idx * 16 + pair_in_blk;

            half2 val;
            if (oob_check && t >= k_VKQ_sup) {
                val = make_half2(0.0f, 0.0f);
            } else {
                const int qs_blk_off = (i0_start / 32 + blk_idx) * QK_MXFP8;
                const uint16_t pair = *reinterpret_cast<const uint16_t *>(
                    row_t + V_qs_head_off + qs_blk_off + 2 * pair_in_blk);
                const uint8_t e_val = *(row_t + V_e_head_off + i0_start / 32 + blk_idx);
                const half2 scale_h2 = __float2half2_rn(ggml_cuda_e8m0_to_fp32(e_val));

#if CUDART_VERSION >= 12050
                if constexpr (v_mxfp8_type == GGML_TYPE_MXFP8_E4M3) {
                    __nv_fp8x2_e4m3 fp8; fp8.__x = pair;
                    val = __hmul2(half2(fp8), scale_h2);
                } else {
                    __nv_fp8x2_e5m2 fp8; fp8.__x = pair;
                    val = __hmul2(half2(fp8), scale_h2);
                }
#else
                const uint8_t fp8_0 = pair & 0xFF;
                const uint8_t fp8_1 = pair >> 8;
                const float scale = __half2float(scale_h2.x);
                const float v0 = mxfp_traits<v_mxfp8_type>::dequant_elem(fp8_0) * scale;
                const float v1 = mxfp_traits<v_mxfp8_type>::dequant_elem(fp8_1) * scale;
                val = make_half2(__float2half(v0), __float2half(v1));
#endif
            }
            tile_V[t * stride_tile_V + d_h2] = val;
        }
    }
}

// Load V: MXFP6 FP6 -> dequant to F16 half2 directly into shared tile.
template<ggml_type v_mxfp6_type, int DV, int nwarps, int nbatch_fa, int stride_tile_V, int nbatch_V2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp_load_V_mxfp6_f16(
        const char * const __restrict__ V_row_base,
        const int V_qs_head_off,
        const int V_e_head_off,
        const int V_pos_stride,
        half2 * const __restrict__ tile_V,
        const int k_VKQ_0,
        const int i0_start,
        const int k_VKQ_sup) {
    // MXFP6 V: 24 bytes per 32-element block (6 bits/elem).
    // Process pairs of elements -> half2. 16 half2 per 32-element block.
    constexpr int nelem_pairs = nbatch_V2;

#pragma unroll
    for (int t0 = 0; t0 < nbatch_fa; t0 += nwarps) {
        const int t = t0 + threadIdx.y;
        if (t0 + nwarps > nbatch_fa && t >= nbatch_fa) {
            break;
        }

        const char * row_t = V_row_base + int64_t(k_VKQ_0 + t) * V_pos_stride;

#pragma unroll
        for (int ep0 = 0; ep0 < nelem_pairs; ep0 += WARP_SIZE) {
            const int ep = ep0 + threadIdx.x;
            if (ep0 + WARP_SIZE > nelem_pairs && ep >= nelem_pairs) {
                break;
            }

            const int blk_idx     = ep / 16;
            const int pair_in_blk = ep % 16;
            const int d_h2 = i0_start / 2 + blk_idx * 16 + pair_in_blk;

            half2 val;
            if (oob_check && t >= k_VKQ_sup) {
                val = make_half2(0.0f, 0.0f);
            } else {
                // Each block has 24 bytes = 8 groups of 3 bytes (4 fp6 values per group).
                // pair_in_blk indexes a pair of elements (0..15).
                // Each group of 3 bytes contains 4 elements, so:
                const int elem0 = pair_in_blk * 2;
                const int group0 = elem0 / 4;
                const int within0 = elem0 % 4;

                const uint8_t * qs_src = (const uint8_t *)(row_t + V_qs_head_off + (i0_start / 32 + blk_idx) * 24);

                // Unpack group containing elem0.
                uint32_t packed0 = (uint32_t)qs_src[group0 * 3] |
                                   ((uint32_t)qs_src[group0 * 3 + 1] << 8) |
                                   ((uint32_t)qs_src[group0 * 3 + 2] << 16);
                uint8_t v0_fp6 = (packed0 >> (within0 * 6)) & 0x3F;

                // elem1 = elem0 + 1
                const int elem1 = elem0 + 1;
                const int group1 = elem1 / 4;
                const int within1 = elem1 % 4;

                uint8_t v1_fp6;
                if (group1 == group0) {
                    v1_fp6 = (packed0 >> (within1 * 6)) & 0x3F;
                } else {
                    uint32_t packed1 = (uint32_t)qs_src[group1 * 3] |
                                       ((uint32_t)qs_src[group1 * 3 + 1] << 8) |
                                       ((uint32_t)qs_src[group1 * 3 + 2] << 16);
                    v1_fp6 = (packed1 >> (within1 * 6)) & 0x3F;
                }

                const uint8_t e_val = *(row_t + V_e_head_off + i0_start / 32 + blk_idx);
                const half2 scale_h2 = __float2half2_rn(ggml_cuda_e8m0_to_fp32(e_val));

#if CUDART_VERSION >= 12080
                {
                    constexpr __nv_fp6_interpretation_t fp6_interp =
                        (v_mxfp6_type == GGML_TYPE_MXFP6_E2M3) ? __NV_E2M3 : __NV_E3M2;
                    const __nv_fp6x2_storage_t p = (uint16_t)v0_fp6 | ((uint16_t)v1_fp6 << 8);
                    const __half2_raw h2r = __nv_cvt_fp6x2_to_halfraw2(p, fp6_interp);
                    val = __hmul2(*reinterpret_cast<const half2 *>(&h2r), scale_h2);
                }
#else
                {
                    const float scale = __half2float(scale_h2.x);
                    const float f0 = mxfp_traits<v_mxfp6_type>::dequant_elem(v0_fp6) * scale;
                    const float f1 = mxfp_traits<v_mxfp6_type>::dequant_elem(v1_fp6) * scale;
                    val = make_half2(__float2half(f0), __float2half(f1));
                }
#endif
            }
            tile_V[t * stride_tile_V + d_h2] = val;
        }
    }
}

// Dispatch V loading by runtime v_type. Avoids 5×5 K/V template explosion by
// keeping V type as a runtime parameter instead of a template parameter.
template<int DV, int nwarps, int nbatch_fa, int stride_tile_V, int nbatch_V2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp_load_V_dispatch(
        const char * const __restrict__ V_row_base,
        const int V_qs_head_off,
        const int V_e_head_off,
        const int V_pos_stride,
        half2 * const __restrict__ tile_V,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int v_type) {
    switch (v_type) {
        case MXFP_V_FP4:
            flash_attn_ext_mxfp_load_V_f16<DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
                (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride, tile_V, k_VKQ_0, 0, k_VKQ_sup);
            break;
        case MXFP_V_FP8_E4M3:
            flash_attn_ext_mxfp_load_V_mxfp8_f16<GGML_TYPE_MXFP8_E4M3, DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
                (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride, tile_V, k_VKQ_0, 0, k_VKQ_sup);
            break;
        case MXFP_V_FP8_E5M2:
            flash_attn_ext_mxfp_load_V_mxfp8_f16<GGML_TYPE_MXFP8_E5M2, DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
                (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride, tile_V, k_VKQ_0, 0, k_VKQ_sup);
            break;
        case MXFP_V_FP6_E2M3:
            flash_attn_ext_mxfp_load_V_mxfp6_f16<GGML_TYPE_MXFP6_E2M3, DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
                (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride, tile_V, k_VKQ_0, 0, k_VKQ_sup);
            break;
        default: // MXFP_V_FP6_E3M2
            flash_attn_ext_mxfp_load_V_mxfp6_f16<GGML_TYPE_MXFP6_E3M2, DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
                (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride, tile_V, k_VKQ_0, 0, k_VKQ_sup);
            break;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Mask loading
// ------------------------------------------------------------------------------------------------------------------

template<int ncols1, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp_load_mask(
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
// Q quantization: F32 -> MXFP in shared memory (type determined by mxfp_type)
// ------------------------------------------------------------------------------------------------------------------

template<ggml_type mxfp_type, int DKQ, int ncols, int nwarps, int stride_q_qs, int stride_q_sc, bool apply_hadamard>
static __device__ __forceinline__ void flash_attn_ext_mxfp_quantize_Q(
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
    using traits = mxfp_mma_traits<mxfp_type>;
    constexpr int vals_per_block = 32;
    constexpr int blocks_per_col = DKQ / vals_per_block;

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

        // Walsh-Hadamard rotation to match K-side rotation (QuaRot, arXiv:2404.00456).
        // Block-32 matches MX block size for optimal quantization (BRQ, arXiv:2511.04214).
        // FlashAttention-3 independently validates this approach for FP8 (arXiv:2407.08608).
        if constexpr (apply_hadamard) {
            hadamard_32_inplace(vals);
        }

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
            using soa_traits = mxfp_traits<mxfp_type>;
            constexpr int EMAX = traits::emax;
            // E8M0 scale: round(log2(amax)) via IEEE-754 bit extraction (Schraudolph 1999).
            // floor(log2(x)) from exponent field, +1 if mantissa >= sqrt(2)-1 (0x3504F3).
            uint32_t amax_bits;
            memcpy(&amax_bits, &amax, sizeof(uint32_t));
            const int e_floor = (int)((amax_bits >> 23) & 0xFF) - 127;
            const int e_int = e_floor + ((amax_bits & 0x7FFFFF) >= 0x3504F3 ? 1 : 0);
            const int e_base = e_int - EMAX + 127;

            // MSE-optimal search: test ±R around estimate, pick lowest MSE.
            // Matches set-rows and CPU paths for consistent quantization quality.
            const int e_lo = max(1, min(254, e_base - MXFP_E8M0_MSE_RANGE));
            const int e_hi = max(1, min(254, e_base + MXFP_E8M0_MSE_RANGE));
            int best_e = max(0, min(254, e_base));
            float best_mse = 1e30f;

#pragma unroll
            for (int test_e = e_lo; test_e <= e_hi; ++test_e) {
                const float test_scale = ggml_cuda_e8m0_to_fp32((uint8_t)test_e);
                const float test_inv = 1.0f / test_scale;
                float mse = 0.0f;
#pragma unroll
                for (int i = 0; i < vals_per_block; ++i) {
                    mse += soa_traits::mse_error(vals[i], test_inv, test_scale);
                }
                if (mse < best_mse) {
                    best_mse = mse;
                    best_e = test_e;
                }
            }

            e = static_cast<uint8_t>(best_e);
            // Construct reciprocal power-of-2 via integer bit ops (avoids SFU-bound __frcp_rn).
            const uint32_t inv_bits = (uint32_t)(254 - best_e) << 23;
            memcpy(&inv_d, &inv_bits, sizeof(float));
        }

        // Type-specific packing.
        if constexpr (mxfp_type == GGML_TYPE_MXFP4_E2M1) {
            // Pack 32 values into 4 ints of nibble data.
            if (active) {
#pragma unroll
                for (int i = 0; i < vals_per_block / 4; i += 2) {
                    const int int_idx = block_idx * (vals_per_block / 8) + i / 2;

#if CUDART_VERSION >= 12080
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
#else
                    const float lo_0 = vals[0               + 2 * i + 0] * inv_d;
                    const float lo_1 = vals[vals_per_block/2 + 2 * i + 0] * inv_d;
                    const float lo_2 = vals[0               + 2 * i + 1] * inv_d;
                    const float lo_3 = vals[vals_per_block/2 + 2 * i + 1] * inv_d;
                    char2 lo;
                    lo.x = ggml_cuda_float_to_fp4_e2m1(lo_0, 1.0f)
                         | (ggml_cuda_float_to_fp4_e2m1(lo_1, 1.0f) << 4);
                    lo.y = ggml_cuda_float_to_fp4_e2m1(lo_2, 1.0f)
                         | (ggml_cuda_float_to_fp4_e2m1(lo_3, 1.0f) << 4);

                    const float hi_0 = vals[0               + 2 * (i + 1) + 0] * inv_d;
                    const float hi_1 = vals[vals_per_block/2 + 2 * (i + 1) + 0] * inv_d;
                    const float hi_2 = vals[0               + 2 * (i + 1) + 1] * inv_d;
                    const float hi_3 = vals[vals_per_block/2 + 2 * (i + 1) + 1] * inv_d;
                    char2 hi;
                    hi.x = ggml_cuda_float_to_fp4_e2m1(hi_0, 1.0f)
                         | (ggml_cuda_float_to_fp4_e2m1(hi_1, 1.0f) << 4);
                    hi.y = ggml_cuda_float_to_fp4_e2m1(hi_2, 1.0f)
                         | (ggml_cuda_float_to_fp4_e2m1(hi_3, 1.0f) << 4);
#endif // CUDART_VERSION >= 12080

                    const uint32_t lo_u16 = *reinterpret_cast<const uint16_t *>(&lo);
                    const uint32_t hi_u16 = *reinterpret_cast<const uint16_t *>(&hi);
                    tile_Q_qs[jc * stride_q_qs + int_idx] = lo_u16 | (hi_u16 << 16);
                }
            }

            // Exchange scales between even/odd block partners via XOR-1 shuffle.
            const uint8_t e_partner = __shfl_xor_sync(0xFFFFFFFF, e, 1, WARP_SIZE);

            if (active && block_idx % 2 == 0) {
                const int scale_pair_idx = block_idx / 2;
                tile_Q_sc[jc * stride_q_sc + scale_pair_idx] = (uint32_t)e | ((uint32_t)e_partner << 8);
            }
        } else {
            // FP6/FP8: pack 32 quantized values into 8 ints (byte-padded format for MMA).
            if (active) {
#pragma unroll
                for (int i = 0; i < vals_per_block / 4; ++i) {
                    const int int_idx = block_idx * (vals_per_block / 4) + i;
                    const float4 f4 = make_float4(
                        vals[4 * i + 0] * inv_d, vals[4 * i + 1] * inv_d,
                        vals[4 * i + 2] * inv_d, vals[4 * i + 3] * inv_d);
                    uint32_t packed;
#if CUDART_VERSION >= 12080
                    if constexpr (mxfp_type == GGML_TYPE_MXFP6_E2M3) {
                        // x4 vectorized: float4 → 4 byte-padded FP6 values in uint32.
                        __nv_fp6x4_e2m3 fp6(f4); packed = fp6.__x;
                    } else if constexpr (mxfp_type == GGML_TYPE_MXFP6_E3M2) {
                        __nv_fp6x4_e3m2 fp6(f4); packed = fp6.__x;
                    } else
#endif
#if CUDART_VERSION >= 12050
                    if constexpr (mxfp_type == GGML_TYPE_MXFP8_E4M3) {
                        // x4 vectorized: float4 → 4 FP8 values in uint32.
                        __nv_fp8x4_e4m3 fp8(f4); packed = fp8.__x;
                    } else if constexpr (mxfp_type == GGML_TYPE_MXFP8_E5M2) {
                        __nv_fp8x4_e5m2 fp8(f4); packed = fp8.__x;
                    } else
#endif
                    {
                        // Scalar fallback for older CUDA.
                        constexpr bool is_fp6 = (mxfp_type == GGML_TYPE_MXFP6_E2M3 || mxfp_type == GGML_TYPE_MXFP6_E3M2);
                        constexpr uint8_t elem_mask = is_fp6 ? 0x3F : 0xFF;
                        uint8_t bytes[4];
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            bytes[v] = mxfp_traits<mxfp_type>::quantize_elem((&f4.x)[v]) & elem_mask;
                        }
                        packed = *reinterpret_cast<uint32_t *>(bytes);
                    }
                    tile_Q_qs[jc * stride_q_qs + int_idx] = packed;
                }
            }

            // E8M0 scale: individual store.
            if (active) {
                tile_Q_sc[jc * stride_q_sc + block_idx] = (uint32_t)e;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Main iteration function
// ------------------------------------------------------------------------------------------------------------------

template<ggml_type mxfp_type, int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool needs_fixup, bool is_fixup, bool oob_check, bool single_buf = false>
static __device__ __forceinline__ void flash_attn_ext_mxfp_iter(
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
        half2    * const __restrict__ tile_V_curr,
        half2    * const __restrict__ tile_V_next,
        half     * const __restrict__ tile_mask,
        half     * const __restrict__ tile_mask_next,
        tile<16, 8, float>   * const __restrict__ VKQ_C,
        float                * const __restrict__ KQ_max,
        float                * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup,
        const bool last_iter,
        const int k_VKQ_sup_next,
        const int v_type) {
#ifdef BLACKWELL_MMA_AVAILABLE
    using mma_traits = mxfp_mma_traits<mxfp_type>;

    constexpr int ncols          = ncols1 * ncols2;
    constexpr int cols_per_warp  = 8;
    constexpr int np             = nwarps * cols_per_warp / ncols;
    constexpr int nbatch_fa      = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols).nbatch_fa;
    constexpr int nbatch_V2      = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols).nbatch_V2;

    constexpr int stride_q_qs    = DKQ / mma_traits::smem_k_qs_div + 4;
    constexpr int stride_q_sc    = DKQ / mma_traits::smem_k_sc_div;
    constexpr int stride_k_qs    = DKQ / mma_traits::smem_k_qs_div + 4;
    constexpr int stride_k_sc    = DKQ / mma_traits::smem_k_sc_div;
    constexpr int stride_tile_V  = nbatch_V2 + 4;

    using T_A_KQ  = tile<16, 8, int>;
    using T_B_KQ  = tile< 8, 8, int>;
    using T_C_KQ  = tile<16, 8, float>;

    using T_A_VKQ = tile<16, 8, half2>;
    using T_B_VKQ = tile< 8, 8, half2>;
    using T_C_VKQ = tile<16, 8, float>;

    // ---- Phase 0: Expand FP6 packed data in smem (if loaded via cp.async) ----

    if constexpr (mma_traits::needs_smem_expand_k) {
        flash_attn_ext_mxfp_expand_K_fp6<DKQ, nwarps, nbatch_fa, stride_k_qs>(tile_K_qs);
        __syncthreads();
    }

    // ---- Phase 1: KQ MMA ----

    T_C_KQ KQ_C[nbatch_fa / (np * T_C_KQ::I)];

    constexpr int kq_iters = DKQ / mma_traits::k_per_mma;

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

            {
                T_A_KQ K_A;
                load_ldmatrix(K_A, tile_K_qs + i_KQ_0 * stride_k_qs + d0 * 8, stride_k_qs);
                const uint32_t a_scale = tile_K_sc[k_row * stride_k_sc + d0];
                mma_traits::mma_kq(KQ_C[i_KQ_00 / (np * T_A_KQ::I)], K_A, Q_B, a_scale, b_scale);
            }
        }
    }

    // ---- Phase 2a: Load V (single-buffer) or preload next K/V/mask (double-buffer) ----

    if constexpr (single_buf) {
        // Single-buffer: K was consumed by KQ MMA. Barrier before V overwrites K.
        __syncthreads();

        // Load V for the current iteration into tile_V_curr (which aliases tile_K_qs memory).
        const int k_VKQ_curr = kb0 * nbatch_fa;
        flash_attn_ext_mxfp_load_V_dispatch<DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
            (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
             tile_V_curr, k_VKQ_curr, k_VKQ_sup, v_type);
    } else if (!last_iter) {
        const int k_VKQ_next = (kb0 + 1) * nbatch_fa;

        flash_attn_ext_mxfp_load_K<mxfp_type, DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
            (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
             tile_K_qs_next, tile_K_sc_next, k_VKQ_next, k_VKQ_sup_next);

        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_mxfp_load_mask<ncols1, nwarps, nbatch_fa, oob_check>
                (mask_h + k_VKQ_next, tile_mask_next, stride_mask, k_VKQ_sup_next, jt * ncols1, ne01);
        }

        flash_attn_ext_mxfp_load_V_dispatch<DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
            (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
             tile_V_next, k_VKQ_next, k_VKQ_sup_next, v_type);
    }

    // ---- Phase 2b: Softmax + VKQ rescale ----

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
                KQ_C[k0 / (np * T_C_KQ::I)].x[l] = fast_expf_mxfp(KQ_C[k0 / (np * T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
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

    // Barrier before VKQ MMA reads tile_V_curr.
    // Single-buffer: V was loaded in Phase 2a above (no cp.async for V).
    // Double-buffer: wait for cp.async pipeline from K preloading.
    if constexpr (!single_buf && mma_traits::can_cp_async_k) {
        cp_async_wait_all();
    }
    __syncthreads();

    // ---- Phase 3: VKQ MMA (reads tile_V_curr) ----

    {
        constexpr int i0_stride = T_C_VKQ::I;
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < DV; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa / 2) % (np * T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa / 2; k00 += np * T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np) * T_A_VKQ::J;

                T_A_VKQ A;
                load_ldmatrix_trans(A, tile_V_curr + 2 * k0 * stride_tile_V + i_VKQ_0 / 2, stride_tile_V);
                mma(VKQ_C[i_VKQ_0 / i0_stride], A, B_VKQ[k00 / (np * T_A_VKQ::J)]);
            }
        }
    }

    __syncthreads();

#else
    GGML_UNUSED_VARS(Q_f2, K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
        V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
        mask_h, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, stride_mask,
        tile_Q_qs, tile_Q_sc, tile_K_qs, tile_K_sc, tile_K_qs_next, tile_K_sc_next, tile_V_curr, tile_V_next, tile_mask, tile_mask_next,
        VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, last_iter, k_VKQ_sup_next, v_type);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------
// Process tile: orchestrates Q loading, iteration loop, and result writeback.
// ------------------------------------------------------------------------------------------------------------------

template<ggml_type mxfp_type, int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_mxfp_process_tile(
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
        const int kb0_stop,
        const int v_type) {
#ifdef BLACKWELL_MMA_AVAILABLE
    using mma_traits = mxfp_mma_traits<mxfp_type>;

    constexpr int ncols          = ncols1 * ncols2;
    constexpr int cols_per_warp  = 8;
    constexpr int np             = nwarps * cols_per_warp / ncols;
    constexpr int nbatch_fa      = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols).nbatch_fa;
    constexpr int nbatch_V2      = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols).nbatch_V2;
    constexpr int nbatch_combine = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols).nbatch_combine;

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

    constexpr int stride_q_qs    = DKQ / mma_traits::smem_k_qs_div + 4;
    constexpr int stride_q_sc    = DKQ / mma_traits::smem_k_sc_div;
    constexpr int stride_k_qs    = DKQ / mma_traits::smem_k_qs_div + 4;
    constexpr int stride_k_sc    = DKQ / mma_traits::smem_k_sc_div;
    constexpr int stride_tile_V  = nbatch_V2 + 4;

    extern __shared__ char smem_mxfp[];

    int      * tile_Q_qs = (int      *) smem_mxfp;
    uint32_t * tile_Q_sc = (uint32_t *)(tile_Q_qs + ncols * stride_q_qs);

    // Single-buffer: K and V share smem, halving usage (for large D like MLA D=576).
    // Double-buffer: separate A/B buffers for overlap (for D <= 256).
    constexpr bool single_buf = (DKQ > 256 || DV > 256);

    constexpr int nbytes_K = nbatch_fa * stride_k_qs * (int)sizeof(int)
                           + nbatch_fa * stride_k_sc * (int)sizeof(uint32_t);
    constexpr int nbytes_V = nbatch_fa * stride_tile_V * (int)sizeof(half2);
    constexpr int nbytes_KV_shared = nbytes_K > nbytes_V ? nbytes_K : nbytes_V;
    constexpr int nbytes_mask = ncols1 * (nbatch_fa + 8) * (int)sizeof(half);

    char * kv_region = (char *)(tile_Q_sc + ncols * stride_q_sc);

    int      * tile_K_qs_A, * tile_K_qs_B;
    uint32_t * tile_K_sc_A, * tile_K_sc_B;
    half2    * tile_V_A, * tile_V_B;
    half     * tile_mask_A, * tile_mask_B;

    if constexpr (single_buf) {
        // K and V aliased at same physical address; loaded sequentially.
        tile_K_qs_A = (int *)kv_region;
        tile_K_sc_A = (uint32_t *)(tile_K_qs_A + nbatch_fa * stride_k_qs);
        tile_V_A    = (half2 *)kv_region;
        tile_mask_A = (half *)(kv_region + nbytes_KV_shared);
        tile_K_qs_B = tile_K_qs_A;
        tile_K_sc_B = tile_K_sc_A;
        tile_V_B    = tile_V_A;
        tile_mask_B = tile_mask_A;
    } else {
        // Separate A/B buffers for pipelining.
        tile_K_qs_A = (int *)kv_region;
        tile_K_sc_A = (uint32_t *)(tile_K_qs_A + nbatch_fa * stride_k_qs);
        tile_K_qs_B = (int      *)(tile_K_sc_A + nbatch_fa * stride_k_sc);
        tile_K_sc_B = (uint32_t *)(tile_K_qs_B + nbatch_fa * stride_k_qs);
        tile_V_A    = (half2 *)(tile_K_sc_B + nbatch_fa * stride_k_sc);
        tile_V_B    = (half2 *)((char *)tile_V_A + nbytes_V);
        tile_mask_A = (half *)((char *)tile_V_B + nbytes_V);
        tile_mask_B = (half *)((char *)tile_mask_A + nbytes_mask);
    }

    T_C_VKQ VKQ_C[DV / T_C_VKQ::I];

    float KQ_rowsum[2] = {0.0f};
    float KQ_max[2];
    KQ_max[0] = -FLT_MAX / 2.0f;
    KQ_max[1] = -FLT_MAX / 2.0f;

    // Hadamard Q rotation must match K-side rotation applied during KV cache write.
    // Skipped for: MLA (DKQ != DV, V is a view of K), E5M2/E3M2 (no quality benefit).
    constexpr bool apply_hadamard = (DKQ == DV) && mxfp_use_hadamard_v<mxfp_type>;
    flash_attn_ext_mxfp_quantize_Q<mxfp_type, DKQ, ncols, nwarps, stride_q_qs, stride_q_sc, apply_hadamard>
        (Q_f2, tile_Q_qs, tile_Q_sc, scale, stride_Q1, stride_Q2, ncols1, ncols2, jt, zt_gqa, gqa_ratio, ne01);

    __syncthreads();

    // Macro to call flash_attn_ext_mxfp_iter with the many shared arguments.
    // Varies only: tile pointers (curr/next), k_VKQ_sup, last_iter, k_VKQ_sup_next, and optionally single_buf.
#define MXFP_CALL_ITER(oob_v, single_buf_v, qs_curr, sc_curr, qs_next, sc_next, v_curr, v_next, m_curr, m_next, sup, last, sup_next) \
    flash_attn_ext_mxfp_iter                                                                                   \
        <mxfp_type, DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_v, single_buf_v> \
        (Q_f2, K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,                                         \
         V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,                                               \
         mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,                                               \
         ne01, ne02, stride_mask,                                                                              \
         tile_Q_qs, tile_Q_sc, qs_curr, sc_curr, qs_next, sc_next,                                            \
         v_curr, v_next, m_curr, m_next,                                                                       \
         VKQ_C, KQ_max, KQ_rowsum, jt, kb0, sup, last, sup_next, v_type)

    // Swap double-buffer pointers after each iteration.
#define MXFP_SWAP_BUFFERS() do {                                                                                \
    { int      * tmp = tile_K_qs_curr; tile_K_qs_curr = tile_K_qs_next; tile_K_qs_next = tmp; }                \
    { uint32_t * tmp = tile_K_sc_curr; tile_K_sc_curr = tile_K_sc_next; tile_K_sc_next = tmp; }                \
    { half2    * tmp = tile_V_curr;    tile_V_curr    = tile_V_next;    tile_V_next    = tmp; }                 \
    { half     * tmp = tile_mask_curr; tile_mask_curr = tile_mask_next; tile_mask_next = tmp; }                 \
} while (0)

    if constexpr (single_buf) {
        // ---- Single-buffer iteration ----
        // Each iteration: load K+mask → sync → iter (KQ MMA → sync → load V → softmax → sync → VKQ MMA → sync).
        // V is loaded inside iter after KQ MMA consumes K (they share smem).

        if constexpr (ncols2 == 1) {
            constexpr bool oob_check_v = true;

            for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
                const int k_VKQ_0 = kb0 * nbatch_fa;
                const int k_VKQ_sup_v = (kb0 == kb0_stop - 1) ? (ne11 - kb0 * nbatch_fa) : nbatch_fa;

                flash_attn_ext_mxfp_load_K<mxfp_type, DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check_v>
                    (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
                     tile_K_qs_A, tile_K_sc_A, k_VKQ_0, k_VKQ_sup_v);
                if (mask_h) {
                    flash_attn_ext_mxfp_load_mask<ncols1, nwarps, nbatch_fa, oob_check_v>
                        (mask_h + k_VKQ_0, tile_mask_A, stride_mask, k_VKQ_sup_v, jt * ncols1, ne01);
                }
                if constexpr (mma_traits::can_cp_async_k) {
                    cp_async_wait_all();
                }
                __syncthreads();

                MXFP_CALL_ITER(oob_check_v, true,
                    tile_K_qs_A, tile_K_sc_A, tile_K_qs_A, tile_K_sc_A,
                    tile_V_A, tile_V_A, tile_mask_A, tile_mask_A,
                    k_VKQ_sup_v, true, 0);
            }
        } else {
            constexpr bool oob_check_v = false;

            for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
                const int k_VKQ_0 = kb0 * nbatch_fa;

                flash_attn_ext_mxfp_load_K<mxfp_type, DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check_v>
                    (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
                     tile_K_qs_A, tile_K_sc_A, k_VKQ_0, nbatch_fa);
                flash_attn_ext_mxfp_load_mask<ncols1, nwarps, nbatch_fa, oob_check_v>
                    (mask_h + k_VKQ_0, tile_mask_A, stride_mask, nbatch_fa, jt * ncols1, ne01);
                if constexpr (mma_traits::can_cp_async_k) {
                    cp_async_wait_all();
                }
                __syncthreads();

                MXFP_CALL_ITER(oob_check_v, true,
                    tile_K_qs_A, tile_K_sc_A, tile_K_qs_A, tile_K_sc_A,
                    tile_V_A, tile_V_A, tile_mask_A, tile_mask_A,
                    nbatch_fa, true, 0);
            }
        }
    } else {
        // ---- Double-buffered iteration ----
        int kb0 = kb0_start;
        int      * tile_K_qs_curr = tile_K_qs_A;
        uint32_t * tile_K_sc_curr = tile_K_sc_A;
        int      * tile_K_qs_next = tile_K_qs_B;
        uint32_t * tile_K_sc_next = tile_K_sc_B;
        half2    * tile_V_curr    = tile_V_A;
        half2    * tile_V_next    = tile_V_B;
        half     * tile_mask_curr = tile_mask_A;
        half     * tile_mask_next = tile_mask_B;

        if constexpr (ncols2 == 1) {
            constexpr bool oob_check_v = true;

            // Preload K[0], mask[0], V[0] into buffer A.
            {
                constexpr int k_VKQ_sup_v = nbatch_fa;
                const int k_VKQ_0 = kb0_start * nbatch_fa;

                if (mask_h) {
                    flash_attn_ext_mxfp_load_mask<ncols1, nwarps, nbatch_fa, oob_check_v>
                        (mask_h + k_VKQ_0, tile_mask_curr, stride_mask, k_VKQ_sup_v, jt * ncols1, ne01);
                }
                flash_attn_ext_mxfp_load_K<mxfp_type, DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check_v>
                    (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
                     tile_K_qs_curr, tile_K_sc_curr, k_VKQ_0, k_VKQ_sup_v);
                flash_attn_ext_mxfp_load_V_dispatch<DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check_v>
                    (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
                     tile_V_curr, k_VKQ_0, k_VKQ_sup_v, v_type);
                if constexpr (mma_traits::can_cp_async_k) {
                    cp_async_wait_all();
                }
                __syncthreads();
            }

            for (; kb0 < kb0_stop - 1; ++kb0) {
                constexpr int k_VKQ_sup_v = nbatch_fa;
                const int k_VKQ_sup_next = (kb0 + 1 == kb0_stop - 1) ? (ne11 - (kb0 + 1) * nbatch_fa) : nbatch_fa;

                MXFP_CALL_ITER(oob_check_v, false,
                    tile_K_qs_curr, tile_K_sc_curr, tile_K_qs_next, tile_K_sc_next,
                    tile_V_curr, tile_V_next, tile_mask_curr, tile_mask_next,
                    k_VKQ_sup_v, false, k_VKQ_sup_next);

                MXFP_SWAP_BUFFERS();
            }
            {
                const int k_VKQ_sup_v = ne11 - kb0 * nbatch_fa;

                MXFP_CALL_ITER(oob_check_v, false,
                    tile_K_qs_curr, tile_K_sc_curr, tile_K_qs_next, tile_K_sc_next,
                    tile_V_curr, tile_V_next, tile_mask_curr, tile_mask_next,
                    k_VKQ_sup_v, true, 0);
            }
        } else {
            constexpr bool oob_check_v = false;

            // Preload K[0], mask[0], V[0] into buffer A.
            {
                constexpr int k_VKQ_sup_v = nbatch_fa;
                const int k_VKQ_0 = kb0_start * nbatch_fa;

                flash_attn_ext_mxfp_load_mask<ncols1, nwarps, nbatch_fa, oob_check_v>
                    (mask_h + k_VKQ_0, tile_mask_curr, stride_mask, k_VKQ_sup_v, jt * ncols1, ne01);
                flash_attn_ext_mxfp_load_K<mxfp_type, DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check_v>
                    (K_row_base, K_qs_head_off, K_e_head_off, K_pos_stride,
                     tile_K_qs_curr, tile_K_sc_curr, k_VKQ_0, k_VKQ_sup_v);
                flash_attn_ext_mxfp_load_V_dispatch<DV, nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check_v>
                    (V_row_base, V_qs_head_off, V_e_head_off, V_pos_stride,
                     tile_V_curr, k_VKQ_0, k_VKQ_sup_v, v_type);
                if constexpr (mma_traits::can_cp_async_k) {
                    cp_async_wait_all();
                }
                __syncthreads();
            }

            for (; kb0 < kb0_stop - 1; ++kb0) {
                constexpr int k_VKQ_sup_v = nbatch_fa;

                MXFP_CALL_ITER(oob_check_v, false,
                    tile_K_qs_curr, tile_K_sc_curr, tile_K_qs_next, tile_K_sc_next,
                    tile_V_curr, tile_V_next, tile_mask_curr, tile_mask_next,
                    k_VKQ_sup_v, false, nbatch_fa);

                MXFP_SWAP_BUFFERS();
            }
            {
                constexpr int k_VKQ_sup_v = nbatch_fa;

                MXFP_CALL_ITER(oob_check_v, false,
                    tile_K_qs_curr, tile_K_sc_curr, tile_K_qs_next, tile_K_sc_next,
                    tile_V_curr, tile_V_next, tile_mask_curr, tile_mask_next,
                    k_VKQ_sup_v, true, 0);
            }
        }
    }

#undef MXFP_CALL_ITER
#undef MXFP_SWAP_BUFFERS

    // ---- Finalize: rowsum reduction, attention sinks, result writeback ----

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

    half2 * tile_combine = (half2 *)smem_mxfp;

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
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio, ne11,
        stride_Q1, stride_Q2, stride_mask,
        jt, zt_gqa, kb0_start, kb0_stop, v_type);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------
// Global kernel entry point
// ------------------------------------------------------------------------------------------------------------------

template<ggml_type mxfp_type, int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols1*ncols2).nthreads,
                  ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols1*ncols2).occupancy)
static __global__ void flash_attn_ext_mxfp(
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
    using mma_traits = mxfp_mma_traits<mxfp_type>;

    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols).nbatch_fa;
    constexpr int nthreads  = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols).nthreads;
    constexpr int nwarps    = nthreads / WARP_SIZE;

    const int gqa_ratio = ne02 / ne12;

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_mask = nb31 / sizeof(half);

    // K block parameters from MXFP type.
    constexpr int k_qs_per_block = mxfp_traits<mxfp_type>::qs_per_block;
    constexpr int blocks_per_head_K = DKQ / 32;
    constexpr int blocks_per_head_V = DV / 32;
    constexpr bool V_is_K_view = DKQ != DV;

    // Derive K stride from nb11 and the K block size.
    const int k_block_size = k_qs_per_block + 1;  // qs + 1 byte E8M0
    const int stride_K_blocks = nb11 / k_block_size;

    // V type detection from stride. Block sizes: mxfp4=17, mxfp6=25, mxfp8=33.
    const int expected_mxfp4_stride = ne12 * blocks_per_head_V * 17;
    const int expected_mxfp6_stride = ne12 * blocks_per_head_V * 25;

    int v_type;
    if (V_is_K_view) {
        // MLA: V reads first DV dims from K's row, same format as K.
        if constexpr (mxfp_type == GGML_TYPE_MXFP4_E2M1) {
            v_type = MXFP_V_FP4;
        } else if constexpr (mxfp_type == GGML_TYPE_MXFP6_E2M3) {
            v_type = MXFP_V_FP6_E2M3;
        } else if constexpr (mxfp_type == GGML_TYPE_MXFP6_E3M2) {
            v_type = MXFP_V_FP6_E3M2;
        } else if constexpr (mxfp_type == GGML_TYPE_MXFP8_E5M2) {
            v_type = MXFP_V_FP8_E5M2;
        } else {
            v_type = MXFP_V_FP8_E4M3;
        }
    } else if (nb21 == expected_mxfp4_stride) {
        v_type = MXFP_V_FP4;
    } else if (nb21 == expected_mxfp6_stride) {
        // FP6 — use same variant as K.
        if constexpr (mxfp_type == GGML_TYPE_MXFP6_E3M2) {
            v_type = MXFP_V_FP6_E3M2;
        } else {
            v_type = MXFP_V_FP6_E2M3;
        }
    } else {
        // MXFP8 — determine variant from K type.
        if constexpr (mxfp_type == GGML_TYPE_MXFP8_E5M2) {
            v_type = MXFP_V_FP8_E5M2;
        } else {
            v_type = MXFP_V_FP8_E4M3;
        }
    }

    // V block parameters for head offset computation.
    int v_qs_per_block_rt;
    int v_block_size_rt;
    switch (v_type) {
        case MXFP_V_FP4:       v_qs_per_block_rt = 16; v_block_size_rt = 17; break;
        case MXFP_V_FP6_E2M3:
        case MXFP_V_FP6_E3M2:  v_qs_per_block_rt = 24; v_block_size_rt = 25; break;
        default:                v_qs_per_block_rt = 32; v_block_size_rt = 33; break;  // MXFP8
    }
    const int stride_V_blocks = V_is_K_view ? stride_K_blocks : (nb21 / v_block_size_rt);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    // Decode kbc → tile coordinates and compute pointers for K/V/Q/mask/dst.
    // Used twice (main loop + fixup tail), factored into a lambda to avoid duplication.
    const float2 * tp_Q_f2;
    const half   * tp_mask_h;
    float2       * tp_dstk;
    const float  * tp_sinks_f;
    const char   * tp_K_row_base;
    int tp_K_qs_head_off, tp_K_e_head_off;
    const char   * tp_V_row_base;
    int tp_V_qs_head_off, tp_V_e_head_off;
    float tp_slope;
    int tp_jt, tp_zt_gqa, tp_sequence;

    auto compute_tile_ptrs = [&] __device__ (const int kbc_val) {
        tp_sequence =  kbc_val / (iter_k * iter_j * iter_z_gqa * ne12);
        const int z_KV   = (kbc_val - iter_k * iter_j * iter_z_gqa * ne12 * tp_sequence) / (iter_k * iter_j * iter_z_gqa);
        tp_zt_gqa = (kbc_val - iter_k * iter_j * iter_z_gqa * ne12 * tp_sequence - iter_k * iter_j * iter_z_gqa * z_KV) / (iter_k * iter_j);
        tp_jt     = (kbc_val - iter_k * iter_j * iter_z_gqa * ne12 * tp_sequence - iter_k * iter_j * iter_z_gqa * z_KV - iter_k * iter_j * tp_zt_gqa) / iter_k;
        const int zt_Q = z_KV * gqa_ratio + tp_zt_gqa * ncols2;

        tp_Q_f2      = (const float2 *)(Q + nb03 * tp_sequence + nb02 * zt_Q);
        tp_mask_h    = ncols2 == 1 && !mask ? nullptr : (const half *)(mask + nb33 * (tp_sequence % ne33));
        tp_dstk      = ((float2 *)dst) + (tp_sequence * ne01.z * ne02 + zt_Q) * (DV / 2);
        tp_sinks_f   = sinks ? (const float *)sinks + zt_Q : nullptr;

        tp_K_row_base    = K + nb13 * tp_sequence;
        tp_K_qs_head_off = z_KV * blocks_per_head_K * k_qs_per_block;
        tp_K_e_head_off  = stride_K_blocks * k_qs_per_block + z_KV * blocks_per_head_K;

        const int v_head_blks = V_is_K_view ? blocks_per_head_K : blocks_per_head_V;
        const int v_str_blks  = V_is_K_view ? stride_K_blocks   : stride_V_blocks;
        tp_V_row_base    = V_is_K_view ? tp_K_row_base : (V + nb23 * tp_sequence);
        tp_V_qs_head_off = z_KV * v_head_blks * v_qs_per_block_rt;
        tp_V_e_head_off  = v_str_blks * v_qs_per_block_rt + z_KV * v_head_blks;

        tp_slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;
    };

    // Macro: call process_tile with current tp_* state. Avoids repeating the 20-arg call.
    #define MXFP_CALL_PROCESS_TILE(needs_fixup_v, is_fixup_v) \
        flash_attn_ext_mxfp_process_tile<mxfp_type, DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup_v, is_fixup_v> \
            (tp_Q_f2, tp_K_row_base, tp_K_qs_head_off, tp_K_e_head_off, nb11, \
             tp_V_row_base, tp_V_qs_head_off, tp_V_e_head_off, nb21, \
             tp_mask_h, tp_sinks_f, tp_dstk, dst_meta, scale, tp_slope, logit_softcap, \
             ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_mask, tp_jt, tp_zt_gqa, kb0_start, kb0_stop, v_type)

    int       kbc      = int64_t(blockIdx.x + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;

    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        compute_tile_ptrs(kbc);

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[tp_sequence * iter_j + tp_jt] / nbatch_fa);
        }

        if (kb0_start == 0) {
            MXFP_CALL_PROCESS_TILE(false, false);
        } else {
            MXFP_CALL_PROCESS_TILE(true, false);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    compute_tile_ptrs(kbc);

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[tp_sequence * iter_j + tp_jt] / nbatch_fa);
    }

    MXFP_CALL_PROCESS_TILE(false, true);

    #undef MXFP_CALL_PROCESS_TILE
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

template <ggml_type mxfp_type, int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_mxfp_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();

    constexpr int ncols = ncols1 * ncols2;
    using mma_traits = mxfp_mma_traits<mxfp_type>;

    const fattn_mma_config config = ggml_cuda_fattn_mma_mxfp_get_config<mxfp_type>(DKQ, DV, ncols);
    const int  nthreads       = config.nthreads;
    const int  nbatch_fa      = config.nbatch_fa;
    const int  nbatch_V2      = config.nbatch_V2;
    const int  nbatch_combine = config.nbatch_combine;

    constexpr int cols_per_warp = 8;
    const int nwarps = nthreads / WARP_SIZE;

    // Shared memory size calculation — strides from traits.
    constexpr int stride_q_qs   = DKQ / mma_traits::smem_k_qs_div + 4;
    constexpr int stride_q_sc   = DKQ / mma_traits::smem_k_sc_div;
    const     int stride_k_qs   = DKQ / mma_traits::smem_k_qs_div + 4;
    const     int stride_k_sc   = DKQ / mma_traits::smem_k_sc_div;

    const int stride_tile_V = nbatch_V2 + 4;

    const size_t nbytes_Q_qs    = ncols     * stride_q_qs   * sizeof(int);
    const size_t nbytes_Q_sc    = ncols     * stride_q_sc   * sizeof(uint32_t);
    const size_t nbytes_K_qs    = nbatch_fa * stride_k_qs   * sizeof(int);
    const size_t nbytes_K_sc    = nbatch_fa * stride_k_sc   * sizeof(uint32_t);
    const size_t nbytes_V       = nbatch_fa * stride_tile_V * sizeof(half2);
    const size_t nbytes_mask    = ncols1    * (nbatch_fa + 8) * sizeof(half);

    const size_t nbytes_Q_region    = nbytes_Q_qs + nbytes_Q_sc;

    // Single-buffer: K and V share smem (only one loaded at a time). For large D (MLA).
    // Double-buffer: separate A/B buffers for K, V, mask pipelining.
    constexpr bool single_buf = (DKQ > 256 || DV > 256);

    size_t nbytes_KV_region;
    if constexpr (single_buf) {
        const size_t nbytes_K_single = nbytes_K_qs + nbytes_K_sc;
        const size_t nbytes_KV_shared = std::max(nbytes_K_single, nbytes_V);
        nbytes_KV_region = nbytes_KV_shared + nbytes_mask;
    } else {
        nbytes_KV_region = 2 * (nbytes_K_qs + nbytes_K_sc) + 2 * nbytes_V + 2 * nbytes_mask;
    }

    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, nbytes_Q_region + nbytes_KV_region);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *)KQV->op_params + 2, sizeof(float));

    const auto select_kernel = [&](fattn_kernel_t kernel, bool (& sml)[GGML_CUDA_MAX_DEVICES]) {
        if (!sml[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            sml[id] = true;
        }
        return kernel;
    };

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        static bool sml[GGML_CUDA_MAX_DEVICES] = {false};
        fattn_kernel = select_kernel(flash_attn_ext_mxfp<mxfp_type, DKQ, DV, ncols1, ncols2, false>, sml);
    } else {
        static bool sml[GGML_CUDA_MAX_DEVICES] = {false};
        fattn_kernel = select_kernel(flash_attn_ext_mxfp<mxfp_type, DKQ, DV, ncols1, ncols2, true>, sml);
    }

    // need_f16_K=false, need_f16_V=false: MXFP kernel reads raw block data directly.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true);
}


// ------------------------------------------------------------------------------------------------------------------
// Template instance declarations
// ------------------------------------------------------------------------------------------------------------------

#define DECL_FATTN_MMA_MXFP_CASE(MXFP_TYPE, DKQ, DV, ncols1, ncols2) \
    template void ggml_cuda_flash_attn_ext_mma_mxfp_case             \
    <MXFP_TYPE, DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

#define DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, DKQ, DV, ncols)    \
    extern DECL_FATTN_MMA_MXFP_CASE(MXFP_TYPE, DKQ, DV, (ncols)/1,  1); \
    extern DECL_FATTN_MMA_MXFP_CASE(MXFP_TYPE, DKQ, DV, (ncols)/2,  2); \
    extern DECL_FATTN_MMA_MXFP_CASE(MXFP_TYPE, DKQ, DV, (ncols)/4,  4); \
    extern DECL_FATTN_MMA_MXFP_CASE(MXFP_TYPE, DKQ, DV, (ncols)/8,  8);

#define DECL_FATTN_MMA_MXFP_STANDARD(MXFP_TYPE) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE,  64,  64,  8) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 128, 128,  8) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 256, 256,  8) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE,  64,  64, 16) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 128, 128, 16) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 256, 256, 16) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE,  64,  64, 32) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 128, 128, 32) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 256, 256, 32)

DECL_FATTN_MMA_MXFP_STANDARD(GGML_TYPE_MXFP4_E2M1)
DECL_FATTN_MMA_MXFP_STANDARD(GGML_TYPE_MXFP6_E2M3)
DECL_FATTN_MMA_MXFP_STANDARD(GGML_TYPE_MXFP8_E4M3)
#ifdef GGML_CUDA_MXFP_ALL_VARIANTS
DECL_FATTN_MMA_MXFP_STANDARD(GGML_TYPE_MXFP6_E3M2)
DECL_FATTN_MMA_MXFP_STANDARD(GGML_TYPE_MXFP8_E5M2)
#endif // GGML_CUDA_MXFP_ALL_VARIANTS

// MLA D=576/512
#define DECL_FATTN_MMA_MXFP_MLA(MXFP_TYPE) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 576, 512,  8) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 576, 512, 16) \
    DECL_FATTN_MMA_MXFP_ALL_NCOLS2(MXFP_TYPE, 576, 512, 32)

DECL_FATTN_MMA_MXFP_MLA(GGML_TYPE_MXFP4_E2M1)
DECL_FATTN_MMA_MXFP_MLA(GGML_TYPE_MXFP6_E2M3)
DECL_FATTN_MMA_MXFP_MLA(GGML_TYPE_MXFP8_E4M3)
#ifdef GGML_CUDA_MXFP_ALL_VARIANTS
DECL_FATTN_MMA_MXFP_MLA(GGML_TYPE_MXFP6_E3M2)
DECL_FATTN_MMA_MXFP_MLA(GGML_TYPE_MXFP8_E5M2)
#endif // GGML_CUDA_MXFP_ALL_VARIANTS
