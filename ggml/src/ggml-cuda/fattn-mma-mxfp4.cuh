#pragma once

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

// MXFP4 MMA config: Blackwell-only, cols_per_warp = 8 (FP4 B-tile width).
// Supported: DKQ ∈ {64, 128, 256}, DV = DKQ, ncols ∈ {8, 16, 32}
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_mxfp4_get_config(
        const int DKQ, const int DV, const int ncols) {
    // Config: DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, nstages_target, Q_in_reg
    // nbatch_K2 and nbatch_V2 are in half2 units for F16, but for MXFP4 we use them differently.
    // nbatch_K2 = DKQ/8 (int elements per K row in packed nibble form) - but actually we pass DKQ/2 for consistency.
    // nbatch_V2 = half2 count for dequantized V tile.

    // For MXFP4: K is loaded as packed nibbles (int), V is dequantized to half2.
    // nbatch_K2 is not used directly (K stride is computed from DKQ).
    // nbatch_V2 = number of half2 elements per V row in shared memory.

    // nstages_target=1 is unused by MXFP4 (no cp.async pipeline), but must be >=1 to pass macro validation.
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

// Load K data from global to shared memory. K is stored as block_mxfp4 structs (17 bytes each).
// We load the packed nibble data (qs) and scales (e) separately into two shared memory regions.
// cp.async CANNOT be used because block_mxfp4.qs is at byte offset 1, which is not 16-byte aligned.
// Typed loads (int/short) also fail: Blackwell enforces strict natural alignment on global memory.
template<int DKQ, int nwarps, int nbatch_fa, int stride_k_qs, int stride_k_sc, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_load_K(
        const block_mxfp4 * const __restrict__ K_mxfp4,
        int      * const __restrict__ tile_K_qs,
        uint32_t * const __restrict__ tile_K_sc,
        const int stride_K,  // In block_mxfp4 units (blocks per row).
        const int k_VKQ_sup) {
    // K has DKQ values per row = DKQ/QK_MXFP4 blocks per row = DKQ/32 blocks per row.
    // Each block has 16 bytes of qs (= DKQ/32 * 16 bytes = DKQ/2 bytes = DKQ/8 ints per row).
    // Scales: DKQ/32 blocks per row, packed as DKQ/64 uint32 pairs per row.
    constexpr int ints_per_row = DKQ / 8;    // Packed nibble ints per K row.
    constexpr int blocks_per_row = DKQ / 32; // MXFP4 blocks per K row.

#pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps) {
        const int i = i0 + threadIdx.y;

        if (i0 + nwarps > nbatch_fa && i >= nbatch_fa) {
            break;
        }

        // Load packed nibble data: each thread loads one int (4 bytes = 8 nibbles) at a time.
        // alignment=1: block_mxfp4.qs is at byte offset 1 → odd-aligned, Blackwell enforces strict alignment.
#pragma unroll
        for (int k0 = 0; k0 < ints_per_row; k0 += WARP_SIZE) {
            const int k = k0 + threadIdx.x;
            if (k0 + WARP_SIZE > ints_per_row && k >= ints_per_row) {
                break;
            }

            if (oob_check && i >= k_VKQ_sup) {
                tile_K_qs[i * stride_k_qs + k] = 0;
            } else {
                // k-th int corresponds to bytes [4*k .. 4*k+3] in the qs data.
                // These bytes span across block(s). Block index = byte_offset / 16.
                // Within block: byte_offset % 16.
                const int byte_offset = k * 4;
                const int block_idx = byte_offset / 16;
                const int byte_in_block = byte_offset % 16;

                int val;
                ggml_cuda_memcpy_1<sizeof(int), 1>(&val, K_mxfp4[i * stride_K + block_idx].qs + byte_in_block);
                tile_K_qs[i * stride_k_qs + k] = val;
            }
        }

        // Load scales: 4X packing — 4 E8M0 exponents per uint32_t (1 per 16 elements).
        // Each MXFP4 block has 1 scale for 32 values; duplicate for the 2 halves of 16.
#pragma unroll
        for (int s0 = 0; s0 < blocks_per_row / 2; s0 += WARP_SIZE) {
            const int s = s0 + threadIdx.x;
            if (s0 + WARP_SIZE > blocks_per_row / 2 && s >= blocks_per_row / 2) {
                break;
            }

            if (oob_check && i >= k_VKQ_sup) {
                tile_K_sc[i * stride_k_sc + s] = 0;
            } else {
                const uint8_t e0 = K_mxfp4[i * stride_K + 2 * s + 0].e;
                const uint8_t e1 = K_mxfp4[i * stride_K + 2 * s + 1].e;
                // 4X: duplicate each block's scale for both 16-element halves.
                // __byte_perm with selector 0x1100: byte0→byte0, byte0→byte1, byte1→byte2, byte1→byte3.
                tile_K_sc[i * stride_k_sc + s] = __byte_perm((uint32_t)e0 | ((uint32_t)e1 << 8), 0, 0x1100);
            }
        }
    }
}

// Single-pass V loader: Dequantize MXFP4 → F16 (half2) in shared memory.
// Each warp handles a batch of tokens, each thread handles 2 V dims (→ 1 half2).
// Layout matches F16 kernel: tile_V[token * stride_tile_V + dim_h2].
template<int nwarps, int nbatch_fa, int stride_tile_V, int nbatch_V2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_load_V_f16(
        const block_mxfp4 * const __restrict__ V_mxfp4,
        half2 * const __restrict__ tile_V,
        const int stride_V,
        const int i0_start,
        const int k_VKQ_sup) {
#pragma unroll
    for (int t0 = 0; t0 < nbatch_fa; t0 += nwarps) {
        const int t = t0 + threadIdx.y;
        if (t0 + nwarps > nbatch_fa && t >= nbatch_fa) {
            break;
        }

#pragma unroll
        for (int d0 = 0; d0 < nbatch_V2; d0 += WARP_SIZE) {
            const int d_h2 = d0 + threadIdx.x;  // half2 index
            if (d0 + WARP_SIZE > nbatch_V2 && d_h2 >= nbatch_V2) {
                break;
            }

            half2 val;
            if (oob_check && t >= k_VKQ_sup) {
                val = make_half2(0.0f, 0.0f);
            } else {
                const int abs_dim = i0_start + 2 * d_h2;  // Absolute V dimension (2 per half2).
                const int blk_idx = abs_dim / 32;
                const int d_in_blk = abs_dim % 32;
                const block_mxfp4 & blk = V_mxfp4[t * stride_V + blk_idx];

                // Nibble extraction: d_in_blk is always even, so both values are in the same byte half.
                const int half_idx = d_in_blk / 16;       // 0 = low nibbles, 1 = high nibbles
                const int byte_base = d_in_blk - half_idx * 16;
                const int shift = half_idx * 4;

                const uint8_t nib0 = (blk.qs[byte_base]     >> shift) & 0xF;
                const uint8_t nib1 = (blk.qs[byte_base + 1] >> shift) & 0xF;

                // Hardware FP4 dequantization: nibble → float via E2M1 type conversion.
                // Intrinsic E2M1 values {0,0.5,1,1.5,2,3,4,6} = kvalues_mxfp4 * 0.5,
                // so scale uses e8m0 directly (no * 0.5f).
                // E8M0 scales are power-of-two → F16 multiply is exact (just exponent add).
                const half2 scale_h2 = __float2half2_rn(ggml_cuda_e8m0_to_fp32(blk.e));
                __nv_fp4x2_e2m1 fp4_pair;
                fp4_pair.__x = nib0 | (nib1 << 4);
                val = __hmul2(__float22half2_rn(float2(fp4_pair)), scale_h2);
            }
            tile_V[t * stride_tile_V + d_h2] = val;
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

// Quantize Q from F32 (global memory) to MXFP4 packed nibbles + E8M0 scales in shared memory.
// Q is scaled by `scale` during quantization.
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

    // Each thread processes one block of 32 values within a Q column.
    // Threads stay warp-synchronized (no early break) so we can use __shfl_xor_sync
    // to exchange E8M0 scales between even/odd block partners, eliminating atomicOr.
    constexpr int threads_total = nwarps * WARP_SIZE;
    constexpr int total_blocks = ncols * blocks_per_col;

#pragma unroll
    for (int b0 = 0; b0 < total_blocks; b0 += threads_total) {
        const int b = b0 + threadIdx.y * WARP_SIZE + threadIdx.x;

        // Use active flag instead of break to keep all warp threads in sync for shuffle.
        const bool active = (b0 + threads_total <= total_blocks || b < total_blocks);

        const int jc = b / blocks_per_col;       // Column index (0..ncols-1).
        const int block_idx = b % blocks_per_col; // Block index within column.

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        // Check bounds. Inactive threads (b >= total_blocks) also get valid=false.
        bool valid = active &&
                     (ncols1 == 1 || jt * ncols1 + j < int(ne01.z)) &&
                     (ncols2 == 1 || zt_gqa * ncols2 + c < gqa_ratio);

        // Load 32 float values for this block from Q.
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

        // Compute E8M0 scale.
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < vals_per_block; ++i) {
            amax = fmaxf(amax, fabsf(vals[i]));
        }

        // Compute E8M0 scale for this block (inlined from compute_e8m0_scale in quantize.cu).
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

        // Pack 32 values into 16 bytes (4 ints) of nibble data.
        // Layout: low 16 values in low nibbles, high 16 values in high nibbles.
        // Process pairs of FP4x4 conversions to write full ints (no shared memory RMW).
        if (active) {
#pragma unroll
            for (int i = 0; i < vals_per_block / 4; i += 2) {
                const int int_idx = block_idx * (vals_per_block / 8) + i / 2;

                // Hardware FP4 conversion: 4 floats → 4 packed nibbles in one operation.
                // __nv_fp4x4_e2m1 layout: float4(a,b,c,d) → char2(.x = b<<4|a, .y = d<<4|c).
                // We need byte0 = low_half|high_half<<4, so order: low0, high0, low1, high1.

                // Low pair → bytes 0-1 of the int.
                __nv_fp4x4_e2m1 fp4_lo(make_float4(
                    vals[0               + 2 * i + 0] * inv_d,
                    vals[vals_per_block/2 + 2 * i + 0] * inv_d,
                    vals[0               + 2 * i + 1] * inv_d,
                    vals[vals_per_block/2 + 2 * i + 1] * inv_d
                ));
                const char2 lo = *reinterpret_cast<const char2 *>(&fp4_lo);

                // High pair → bytes 2-3 of the int.
                __nv_fp4x4_e2m1 fp4_hi(make_float4(
                    vals[0               + 2 * (i + 1) + 0] * inv_d,
                    vals[vals_per_block/2 + 2 * (i + 1) + 0] * inv_d,
                    vals[0               + 2 * (i + 1) + 1] * inv_d,
                    vals[vals_per_block/2 + 2 * (i + 1) + 1] * inv_d
                ));
                const char2 hi = *reinterpret_cast<const char2 *>(&fp4_hi);

                // Full int write — no read-modify-write on shared memory.
                // char2 is {x, y} = 2 bytes; reinterpret as uint16_t for natural packing.
                const uint32_t lo_u16 = *reinterpret_cast<const uint16_t *>(&lo);
                const uint32_t hi_u16 = *reinterpret_cast<const uint16_t *>(&hi);
                tile_Q_qs[jc * stride_q_qs + int_idx] = lo_u16 | (hi_u16 << 16);
            }
        }

        // Exchange scales between even/odd block partners via warp shuffle.
        // XOR-1 pairs adjacent lanes: lane 0↔1, 2↔3, etc.
        // blocks_per_col is always even (DKQ ∈ {64,128,256} / 32), so adjacent lanes
        // always process even/odd blocks within the same column.
        const uint8_t e_partner = __shfl_xor_sync(0xFFFFFFFF, e, 1);

        // Even block writes both scales as a single uint32 (no atomicOr needed).
        // 4X packing: duplicate each block's scale for both 16-element halves.
        if (active && block_idx % 2 == 0) {
            const int scale_pair_idx = block_idx / 2;
            // __byte_perm with selector 0x1100: byte0→byte0, byte0→byte1, byte1→byte2, byte1→byte3.
            tile_Q_sc[jc * stride_q_sc + scale_pair_idx] = __byte_perm((uint32_t)e | ((uint32_t)e_partner << 8), 0, 0x1100);
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
        const block_mxfp4    * const __restrict__ K_mxfp4,
        const block_mxfp4    * const __restrict__ V_mxfp4,
        const half           * const __restrict__ mask_h,
        float2               * const __restrict__ dstk,
        float2               * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        int      * const __restrict__ tile_Q_qs,
        uint32_t * const __restrict__ tile_Q_sc,
        int      * const __restrict__ tile_K_qs,
        uint32_t * const __restrict__ tile_K_sc,
        half2    * const __restrict__ tile_V,  // F16 V data in shared memory
        half     * const __restrict__ tile_mask,
        tile<16, 8, float>   * const __restrict__ VKQ_C,
        float                * const __restrict__ KQ_max,
        float                * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup) {
#ifdef BLACKWELL_MMA_AVAILABLE
    constexpr int ncols          = ncols1 * ncols2;
    constexpr int cols_per_warp  = 8;  // FP4 B-tile width.
    constexpr int np             = nwarps * cols_per_warp / ncols; // Parallel warps per Q column.
    constexpr int nbatch_fa      = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_fa;
    constexpr int nbatch_V2      = ggml_cuda_fattn_mma_mxfp4_get_config(DKQ, DV, ncols).nbatch_V2;

    constexpr int stride_q_qs    = DKQ / 8 + 4;
    constexpr int stride_q_sc    = DKQ / 64;
    constexpr int stride_k_qs    = DKQ / 8 + 4;
    constexpr int stride_k_sc    = DKQ / 64;

    // FP4 MMA: A=K (tile<16,8,int>), B=Q (tile<8,8,int>), C=KQ (tile<16,8,float>)
    // FP4 MMA processes 64 FP4 values (2 blocks of 32) per instruction.
    // K dimension: 16 rows × DKQ values → DKQ/64 MMA calls per K tile.
    // In int terms: A is 16×8 ints (16 rows × 8 ints = 16 rows × 64 nibbles).
    //               B is  8×8 ints ( 8 cols × 8 ints =  8 cols × 64 nibbles).
    using T_A_KQ  = tile<16, 8, int>;    // K tile (row-major, 16 K rows × 64 FP4 values)
    using T_B_KQ  = tile< 8, 8, int>;    // Q tile (column-major, 8 Q cols × 64 FP4 values)
    using T_C_KQ  = tile<16, 8, float>;  // KQ tile (row-major, 16 K rows × 8 Q cols)

    // F16 MMA for VKQ: V from shared memory (half2), softmax from registers (half2).
    using T_A_VKQ = tile<16, 8, half2>;  // V in F16 (transposed: 16 V-dims × 16 tokens)
    using T_B_VKQ = tile< 8, 8, half2>;  // Softmax in F16 (col-major: 16 tokens × 8 queries)
    using T_C_VKQ = tile<16, 8, float>;  // VKQ accumulator in F32 (16 V-dims × 8 queries)

    const int k_VKQ_0 = kb0 * nbatch_fa;

    // KQ accumulator tiles.
    T_C_KQ KQ_C[nbatch_fa / (np * T_C_KQ::I)];

    // Load mask.
    if (ncols2 > 1 || mask_h) {
        flash_attn_ext_mxfp4_load_mask<ncols1, nwarps, nbatch_fa, oob_check>
            (mask_h + k_VKQ_0, tile_mask, stride_mask, k_VKQ_sup, jt * ncols1, ne01);
    }

    // Load K tile.
    flash_attn_ext_mxfp4_load_K<DKQ, nwarps, nbatch_fa, stride_k_qs, stride_k_sc, oob_check>
        (K_mxfp4 + int64_t(k_VKQ_0) * stride_K, tile_K_qs, tile_K_sc, stride_K, k_VKQ_sup);

    __syncthreads();

    // Compute KQ = K × Q using FP4 block-scaled MMA.
    // Iterate over the D dimension in chunks of 64 FP4 values (= 8 ints = 2 MXFP4 blocks).
    constexpr int kq_iters = DKQ / 64;  // Number of MMA calls per KQ tile pair.

#pragma unroll
    for (int d0 = 0; d0 < kq_iters; ++d0) {
        // Load Q B-tile from shared memory.
        T_B_KQ Q_B;
        load_ldmatrix(Q_B, tile_Q_qs + (threadIdx.y / np) * cols_per_warp * stride_q_qs + d0 * 8, stride_q_qs);

#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np * T_A_KQ::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np) * T_A_KQ::I;

            // Load K A-tile from shared memory.
            T_A_KQ K_A;
            load_ldmatrix(K_A, tile_K_qs + i_KQ_0 * stride_k_qs + d0 * 8, stride_k_qs);

            // For the A matrix (K): each row has its own scale pair at dimension d0.
            // Based on NVIDIA block-scaling docs, 2 threads per quad supply the scale.
            // Thread mapping: row = threadIdx.x/4 + (threadIdx.x%2)*8  (from mmq.cuh reference).
            const int k_row = i_KQ_0 + (threadIdx.x / 4) + (threadIdx.x % 2) * 8;
            const uint32_t a_scale = tile_K_sc[k_row * stride_k_sc + d0];

            // For the B matrix (Q): each column has its own scale pair at dimension d0.
            // Thread mapping: thread t → column = t / 4 (for m16n8k64, B is 8 cols).
            // But actually we have cols_per_warp=8 columns per warp, and the warp offset is (threadIdx.y/np)*8.
            const int q_col = (threadIdx.y / np) * cols_per_warp + (threadIdx.x / 4);
            const uint32_t b_scale = tile_Q_sc[q_col * stride_q_sc + d0];

            mma_block_scaled_4x(KQ_C[i_KQ_00 / (np * T_A_KQ::I)], K_A, Q_B, a_scale, b_scale);
        }
    }

    // Apply logit softcap.
    if (use_logit_softcap) {
#pragma unroll
        for (int i = 0; i < nbatch_fa / (np * T_C_KQ::I); ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap * tanhf(KQ_C[i].x[l]);
            }
        }
    }

    // Apply mask and compute softmax.
    float KQ_max_new[2];  // cols_per_thread = 2 for 8-column Turing-style tiles.
    KQ_max_new[0] = KQ_max[0];
    KQ_max_new[1] = KQ_max[1];
    float KQ_rowsum_add[2] = {0.0f, 0.0f};

    // cols_per_warp == 8 path (FP4 MMA always uses 8-column tiles).
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

    // Find max for softmax.
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

    // Reduce max across threads within each KQ column (spread across 8 threads for cols_per_warp=8).
#pragma unroll
    for (int col = 0; col < 2; ++col) {
#pragma unroll
        for (int offset = 16; offset >= 4; offset >>= 1) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
        }
    }

    // Compute exp and rowsum.
    static_assert(nbatch_fa % (np * T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np * T_C_KQ::I) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (!oob_check || k0 + (threadIdx.y % np) * T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                const int KQ_idx = l % 2;
                KQ_C[k0 / (np * T_C_KQ::I)].x[l] = expf(KQ_C[k0 / (np * T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                KQ_rowsum_add[KQ_idx] += KQ_C[k0 / (np * T_C_KQ::I)].x[l];
            } else {
                KQ_C[k0 / (np * T_C_KQ::I)].x[l] = 0.0f;
            }
        }
    }

    // Rescale VKQ accumulators.
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

    // Convert softmax weights to half2 B operand in registers (no shared memory needed).
    T_B_VKQ B_VKQ[nbatch_fa / (np * 2 * T_B_VKQ::J)];
    static_assert(nbatch_fa % (np * 2 * T_B_VKQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k = 0; k < nbatch_fa / (np * 2 * T_B_VKQ::J); ++k) {
        B_VKQ[k] = get_transposed(get_half2(KQ_C[k]));
    }

    // VKQ with F16 MMA: V from shared memory (half2), softmax from registers (B_VKQ).
    // All configs intentionally set nbatch_V2 = DV/2 so this loop executes exactly once.
    // Unlike F16 which interleaves V loads across multiple iterations, MXFP4 does a single-pass
    // dequantize-and-compute. Multi-iteration interleaving was tested and hurt performance due to
    // the extra __syncthreads and shared memory pressure from partial V tiles.
    constexpr int stride_tile_V = nbatch_V2 + 4;  // half2 stride

#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2 * nbatch_V2) {
        static_assert(DV % (2 * nbatch_V2) == 0, "bad loop size");

        flash_attn_ext_mxfp4_load_V_f16<nwarps, nbatch_fa, stride_tile_V, nbatch_V2, oob_check>
            (V_mxfp4 + int64_t(k_VKQ_0) * stride_V, tile_V, stride_V,
             i0_start, k_VKQ_sup);

        __syncthreads();

        constexpr int i0_stride = T_C_VKQ::I;  // 16
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_start + 2 * nbatch_V2; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa / 2) % (np * T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa / 2; k00 += np * T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np) * T_A_VKQ::J;

                T_A_VKQ A;
                load_ldmatrix_trans(A, tile_V + 2 * k0 * stride_tile_V + (i_VKQ_0 - i0_start) / 2, stride_tile_V);
                mma(VKQ_C[i_VKQ_0 / i0_stride], A, B_VKQ[k00 / (np * T_A_VKQ::J)]);
            }
        }
    }

#else
    GGML_UNUSED_VARS(Q_f2, K_mxfp4, V_mxfp4, mask_h, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q_qs, tile_Q_sc, tile_K_qs, tile_K_sc, tile_V, tile_mask,
        VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------
// Process tile: orchestrates Q loading, iteration loop, and result writeback.
// ------------------------------------------------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_mxfp4_process_tile(
        const float2         * const __restrict__ Q_f2,
        const block_mxfp4    * const __restrict__ K_mxfp4,
        const block_mxfp4    * const __restrict__ V_mxfp4,
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
        const int stride_K,
        const int stride_V,
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
    using T_C_VKQ     = tile<16, 8, float>;  // VKQ accumulator in F32.
    using T_C_VKQ_h2  = tile<16, 4, half2>;  // Half2 version for combine section.
    using T_B_combine = tile< 8, 8, half2>;  // For output combine: get_transposed(T_C_VKQ_h2) → T_B_combine.

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp / ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_q_qs    = DKQ / 8 + 4;
    constexpr int stride_q_sc    = DKQ / 64;
    constexpr int stride_k_qs    = DKQ / 8 + 4;
    constexpr int stride_k_sc    = DKQ / 64;

    // F16 V layout stride.
    constexpr int stride_tile_V = nbatch_V2 + 4;  // half2 units

    // Shared memory layout.
    extern __shared__ char smem_mxfp4[];

    // Q region: Q_qs + Q_sc (persistent across iterations).
    int      * tile_Q_qs = (int      *) smem_mxfp4;
    uint32_t * tile_Q_sc = (uint32_t *)(tile_Q_qs + ncols * stride_q_qs);

    // KV region: K_qs + K_sc + V_f16 + mask (reloaded each iteration).
    int      * tile_K_qs = (int      *)(tile_Q_sc + ncols * stride_q_sc);
    uint32_t * tile_K_sc = (uint32_t *)(tile_K_qs + nbatch_fa * stride_k_qs);
    // V F16 data comes after K region.
    half2    * tile_V    = (half2 *)(tile_K_sc + nbatch_fa * stride_k_sc);
    constexpr int nbytes_V = nbatch_fa * stride_tile_V * (int)sizeof(half2);
    half     * tile_mask = (half *)((char *)tile_V + nbytes_V);

    // VKQ accumulators in registers.
    T_C_VKQ VKQ_C[DV / T_C_VKQ::I];

    float KQ_rowsum[2] = {0.0f};
    float KQ_max[2];
    KQ_max[0] = -FLT_MAX / 2.0f;
    KQ_max[1] = -FLT_MAX / 2.0f;

    // Quantize Q: F32 → MXFP4 in shared memory.
    flash_attn_ext_mxfp4_quantize_Q<DKQ, ncols, nwarps, stride_q_qs, stride_q_sc>
        (Q_f2, tile_Q_qs, tile_Q_sc, scale, stride_Q1, stride_Q2, ncols1, ncols2, jt, zt_gqa, gqa_ratio, ne01);

    __syncthreads();

    // Main iteration loop over K/V tiles.
    int kb0 = kb0_start;

    if constexpr (ncols2 == 1) {
        constexpr bool oob_check = true;
        for (; kb0 < kb0_stop - 1; ++kb0) {
            constexpr int k_VKQ_sup_v = nbatch_fa;
            flash_attn_ext_mxfp4_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
                (Q_f2, K_mxfp4, V_mxfp4, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask,
                 tile_Q_qs, tile_Q_sc, tile_K_qs, tile_K_sc, tile_V, tile_mask,
                 VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v);
        }
        const int k_VKQ_sup_v = ne11 - kb0 * nbatch_fa;
        flash_attn_ext_mxfp4_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
            (Q_f2, K_mxfp4, V_mxfp4, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask,
             tile_Q_qs, tile_Q_sc, tile_K_qs, tile_K_sc, tile_V, tile_mask,
             VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 < kb0_stop - 1; ++kb0) {
            constexpr int k_VKQ_sup_v = nbatch_fa;
            flash_attn_ext_mxfp4_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
                (Q_f2, K_mxfp4, V_mxfp4, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask,
                 tile_Q_qs, tile_Q_sc, tile_K_qs, tile_K_sc, tile_V, tile_mask,
                 VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v);
        }
        constexpr int k_VKQ_sup_v = nbatch_fa;
        flash_attn_ext_mxfp4_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, oob_check>
            (Q_f2, K_mxfp4, V_mxfp4, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask,
             tile_Q_qs, tile_Q_sc, tile_K_qs, tile_K_sc, tile_V, tile_mask,
             VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup_v);
    }

    // Sum up partial KQ rowsums (spread across 8 threads for cols_per_warp=8).
    {
#pragma unroll
        for (int col = 0; col < 2; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // Handle attention sinks.
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

    // VKQ_C will be converted F32→half2 incrementally in the combine loop below
    // to avoid allocating a separate VKQ_C_h2 array (saves ~16 registers).

    // Write results to shared memory and then to global memory.
    // Re-use Q shared memory region for combining results.
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

    // Write VKQ data to shared memory and then to global memory.
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
    GGML_UNUSED_VARS(Q_f2, K_mxfp4, V_mxfp4, mask_h, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
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
    const int stride_K    = nb11 / sizeof(block_mxfp4);  // Blocks per row.
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = nb21 / sizeof(block_mxfp4);

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

        const float2      * Q_f2    = (const float2      *)(Q + nb03 * sequence + nb02 * zt_Q);
        const block_mxfp4 * K_mxfp4 = (const block_mxfp4 *)(K + nb13 * sequence + nb12 * z_KV);
        const half        * mask_h   = ncols2 == 1 && !mask ? nullptr :
            (const half *)(mask + nb33 * (sequence % ne33));
        float2            * dstk     = ((float2 *)dst) + (sequence * ne01.z * ne02 + zt_Q) * (DV / 2);

        const block_mxfp4 * V_mxfp4 = (const block_mxfp4 *)(V + nb23 * sequence + nb22 * z_KV);
        const float       * sinks_f  = sinks ? (const float *)sinks + zt_Q : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence * iter_j + jt] / nbatch_fa);
        }

        constexpr bool is_fixup = false;
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false;
            flash_attn_ext_mxfp4_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_mxfp4, V_mxfp4, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true;
            flash_attn_ext_mxfp4_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_mxfp4, V_mxfp4, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
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

    const float2      * Q_f2    = (const float2      *)(Q + nb03 * sequence + nb02 * zt_Q);
    const block_mxfp4 * K_mxfp4 = (const block_mxfp4 *)(K + nb13 * sequence + nb12 * z_KV);
    const half        * mask_h   = ncols2 == 1 && !mask ? nullptr :
        (const half *)(mask + nb33 * (sequence % ne33));
    float2            * dstk     = ((float2 *)dst) + (sequence * ne01.z * ne02 + zt_Q) * (DV / 2);

    const block_mxfp4 * V_mxfp4 = (const block_mxfp4 *)(V + nb23 * sequence + nb22 * z_KV);
    const float       * sinks_f  = sinks ? (const float *)sinks + zt_Q : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence * iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup    = true;
    constexpr bool needs_fixup = false;
    flash_attn_ext_mxfp4_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_mxfp4, V_mxfp4, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
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

    const size_t nbytes_Q_region  = nbytes_Q_qs + nbytes_Q_sc;
    const size_t nbytes_KV_region = nbytes_K_qs + nbytes_K_sc + nbytes_V + nbytes_mask;

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
