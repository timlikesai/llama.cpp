#pragma once

#include "common.cuh"
#include "hadamard.cuh"

// ------------------------------------------------------------------------------------------------------------------
// MXFP Type Traits
// ------------------------------------------------------------------------------------------------------------------
// Single source of truth for all MX (Microscaling) format parameters.
// Adding a new MX variant = one new specialization.
//
// E8M0 scale computation: integer bit extraction with sqrt(2) rounding for MSE-optimal
// power-of-two scale selection (Bridging the Gap, arXiv:2509.23202; Four Over Six, arXiv:2512.02010).
// Block-32 matches OCP MX spec (arXiv:2310.10537) and BRQ rotation alignment (arXiv:2511.04214).
// All MX formats share: QK=32 (block size), E8M0 shared exponent.

template<ggml_type type> struct mxfp_traits;

// Compile-time check: is this an MXFP SoA type?
template<ggml_type type>
static constexpr bool is_mxfp_soa_v =
    type == GGML_TYPE_MXFP4_E2M1 || type == GGML_TYPE_MXFP8_E4M3 ||
    type == GGML_TYPE_MXFP6_E2M3 || type == GGML_TYPE_MXFP6_E3M2 ||
    type == GGML_TYPE_MXFP8_E5M2;

// Runtime check: ggml_is_type_mxfp() is now in ggml.h (shared across all backends).

// Runtime check: is this MXFP type compiled (respects GGML_CUDA_MXFP_ALL_VARIANTS gate)?
static __host__ __device__ __forceinline__ bool ggml_is_type_mxfp_enabled(ggml_type type) {
    if (type == GGML_TYPE_MXFP4_E2M1 || type == GGML_TYPE_MXFP8_E4M3 || type == GGML_TYPE_MXFP6_E2M3) {
        return true;
    }
#ifdef GGML_CUDA_MXFP_ALL_VARIANTS
    if (type == GGML_TYPE_MXFP6_E3M2 || type == GGML_TYPE_MXFP8_E5M2) {
        return true;
    }
#endif
    return false;
}

// SoA head offset calculation:
// Computes qs and E8M0 byte offsets for a given head in a SoA-layout MXFP row.
// Layout: [qs_block0 | qs_block1 | ... | e_block0 | e_block1 | ...]
// Defined outside CUDART guard so it works on HIP/MUSA too.
template<ggml_type type, int D>
static __device__ __forceinline__ void mxfp_soa_head_offsets(
        const int nb_row, const int head, const int gqa_ratio,
        int & qs_off, int & e_off) {
    // block_size and qs_per_block derived from ggml-common.h struct definitions:
    //   MXFP4: block=17 (1+16), qs=16    MXFP8: block=33 (1+32), qs=32    MXFP6: block=25 (1+24), qs=24
    constexpr int qs_per_block = (type == GGML_TYPE_MXFP4_E2M1) ? 16 :
                                 (type == GGML_TYPE_MXFP8_E4M3 || type == GGML_TYPE_MXFP8_E5M2) ? 32 : 24;
    constexpr int block_size   = qs_per_block + 1;  // +1 byte for E8M0 scale
    constexpr int blocks_per_head = D / 32;
    const int stride_blocks = nb_row / block_size;
    const int z = head / gqa_ratio;
    qs_off = z * blocks_per_head * qs_per_block;
    e_off  = stride_blocks * qs_per_block + z * blocks_per_head;
}

// The trait specializations below use NVIDIA CUDA 12.8+ intrinsics (__nv_cvt_float_to_fp4,
// __nv_cvt_float_to_fp6, __nv_fp8_e4m3, etc.) from cuda_fp4.h / cuda_fp8.h.
#if CUDART_VERSION >= 12080

// FP4 E2M1:
template<> struct mxfp_traits<GGML_TYPE_MXFP4_E2M1> {
    static constexpr int bits_per_elem = 4;
    static constexpr int qs_per_block  = 16;   // 32 * 4 / 8
    static constexpr int block_size    = sizeof(block_mxfp4);
    static constexpr int e8m0_offset   = 2;    // ceil(log2(6.0)) — max finite FP4 E2M1 value

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const uint8_t nibble = ggml_cuda_float_to_fp4_e2m1(val, inv_scale);
        const float recon = kvalues_mxfp4[nibble] * 0.5f * scale;
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            qs_dst[j] = __nv_cvt_float2_to_fp4x2(
                make_float2(src[j] * inv_d, src[QK_MXFP4/2 + j] * inv_d),
                __NV_E2M1, cudaRoundNearest);
        }
    }
};

// FP6 helpers:
namespace mxfp_detail {
    // FP6 E2M3 layout: [S(1) | E(2) | M(3)] — max normal = 7.5
    static __device__ __forceinline__ float fp6_e2m3_to_float(uint8_t v) {
        const float sign = (v & 0x20) ? -1.0f : 1.0f;
        const int exp  = (v >> 3) & 0x3;
        const int mant = v & 0x7;
        if (exp == 0) return sign * (float)mant * 0.125f;
        return sign * (1.0f + mant * 0.125f) * (float)(1 << (exp - 1));
    }

    // FP6 E3M2 layout: [S(1) | E(3) | M(2)] — max normal = 28.0, no NaN/Inf
    static __device__ __forceinline__ float fp6_e3m2_to_float(uint8_t v) {
        const float sign = (v & 0x20) ? -1.0f : 1.0f;
        const int exp  = (v >> 2) & 0x7;
        const int mant = v & 0x3;
        if (exp == 0) return sign * (float)mant * 0.0625f;  // 2^(-4)
        // MX E3M2 has no NaN/Inf — exp=7 is a valid normal value (max finite = 28.0).
        return sign * ldexpf(1.0f + mant * 0.25f, exp - 3);
    }

    // Pack 4 six-bit values into 3 bytes
    static __device__ __forceinline__ void pack_fp6x4(const uint8_t v[4], uint8_t out[3]) {
        uint32_t packed = (v[0] & 0x3F) | ((v[1] & 0x3F) << 6) |
                          ((v[2] & 0x3F) << 12) | ((v[3] & 0x3F) << 18);
        out[0] = (uint8_t)(packed);
        out[1] = (uint8_t)(packed >> 8);
        out[2] = (uint8_t)(packed >> 16);
    }

    // Unpack 3 bytes into 4 six-bit values
    static __device__ __forceinline__ void unpack_fp6x4(const uint8_t in[3], uint8_t v[4]) {
        uint32_t packed = (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16);
        v[0] = packed & 0x3F;
        v[1] = (packed >> 6) & 0x3F;
        v[2] = (packed >> 12) & 0x3F;
        v[3] = (packed >> 18) & 0x3F;
    }
} // namespace mxfp_detail

// FP6 E2M3:
template<> struct mxfp_traits<GGML_TYPE_MXFP6_E2M3> {
    static constexpr int bits_per_elem = 6;
    static constexpr int qs_per_block  = 24;   // 32 * 6 / 8
    static constexpr int block_size    = sizeof(block_mxfp6);
    static constexpr int e8m0_offset   = 3;    // ceil(log2(7.5)) — max finite FP6 E2M3 value

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const __nv_fp6_storage_t fp6 = __nv_cvt_float_to_fp6(val * inv_scale, __NV_E2M3, cudaRoundNearest);
        const __half_raw h = __nv_cvt_fp6_to_halfraw(fp6, __NV_E2M3);
        const float recon = __half2float(*reinterpret_cast<const __half *>(&h)) * scale;
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
        const __half_raw h = __nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)raw, __NV_E2M3);
        return __half2float(*reinterpret_cast<const __half *>(&h));
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
        return (uint8_t)__nv_cvt_float_to_fp6(val, __NV_E2M3, cudaRoundNearest);
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; j += 4) {
            uint8_t vals[4];
            for (int jj = 0; jj < 4; ++jj) {
                vals[jj] = (uint8_t)__nv_cvt_float_to_fp6(src[j + jj] * inv_d, __NV_E2M3, cudaRoundNearest);
            }
            mxfp_detail::pack_fp6x4(vals, &qs_dst[j * 3 / 4]);
        }
    }
};

// FP6 E3M2:
template<> struct mxfp_traits<GGML_TYPE_MXFP6_E3M2> {
    static constexpr int bits_per_elem = 6;
    static constexpr int qs_per_block  = 24;
    static constexpr int block_size    = sizeof(block_mxfp6);
    static constexpr int e8m0_offset   = 5;    // ceil(log2(28.0)) — max finite FP6 E3M2 value

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const __nv_fp6_storage_t fp6 = __nv_cvt_float_to_fp6(val * inv_scale, __NV_E3M2, cudaRoundNearest);
        const __half_raw h = __nv_cvt_fp6_to_halfraw(fp6, __NV_E3M2);
        const float recon = __half2float(*reinterpret_cast<const __half *>(&h)) * scale;
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
        const __half_raw h = __nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)raw, __NV_E3M2);
        return __half2float(*reinterpret_cast<const __half *>(&h));
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
        return (uint8_t)__nv_cvt_float_to_fp6(val, __NV_E3M2, cudaRoundNearest);
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; j += 4) {
            uint8_t vals[4];
            for (int jj = 0; jj < 4; ++jj) {
                vals[jj] = (uint8_t)__nv_cvt_float_to_fp6(src[j + jj] * inv_d, __NV_E3M2, cudaRoundNearest);
            }
            mxfp_detail::pack_fp6x4(vals, &qs_dst[j * 3 / 4]);
        }
    }
};

// FP8 E4M3:
template<> struct mxfp_traits<GGML_TYPE_MXFP8_E4M3> {
    static constexpr int bits_per_elem = 8;
    static constexpr int qs_per_block  = 32;   // 32 * 8 / 8
    static constexpr int block_size    = sizeof(block_mxfp8);
    static constexpr int e8m0_offset   = 8;    // ceil(log2(448)) — max finite FP8 E4M3 value

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const uint8_t fp8 = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3 *>(&fp8);
        const float recon = float(fp8_val) * scale;
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
        const __nv_fp8_e4m3 v = *reinterpret_cast<const __nv_fp8_e4m3 *>(&raw);
        return float(v);
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
        return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; ++j) {
            qs_dst[j] = __nv_cvt_float_to_fp8(src[j] * inv_d, __NV_SATFINITE, __NV_E4M3);
        }
    }
};

// FP8 E5M2:
template<> struct mxfp_traits<GGML_TYPE_MXFP8_E5M2> {
    static constexpr int bits_per_elem = 8;
    static constexpr int qs_per_block  = 32;
    static constexpr int block_size    = sizeof(block_mxfp8);
    static constexpr int e8m0_offset   = 16;   // ceil(log2(57344)) — max finite FP8 E5M2 value

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const uint8_t fp8 = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E5M2);
        const __nv_fp8_e5m2 fp8_val = *reinterpret_cast<const __nv_fp8_e5m2 *>(&fp8);
        const float recon = float(fp8_val) * scale;
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
        const __nv_fp8_e5m2 v = *reinterpret_cast<const __nv_fp8_e5m2 *>(&raw);
        return float(v);
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
        return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E5M2);
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; ++j) {
            qs_dst[j] = __nv_cvt_float_to_fp8(src[j] * inv_d, __NV_SATFINITE, __NV_E5M2);
        }
    }
};

// Unified SoA quantization:
// Shared Hadamard rotation + direct E8M0 scale + type-specific writes.
template<ggml_type mxfp_type, bool apply_hadamard>
static __device__ void quantize_f32_mxfp_block_soa(
        const float * __restrict__ x,
        char * __restrict__ row_base,
        const int block_idx,
        const int blocks_per_row_total) {
    using traits = mxfp_traits<mxfp_type>;
    constexpr int QK = 32;

    float vals[QK];
    const float * src = x;
    if constexpr (apply_hadamard) {
        for (int j = 0; j < QK; ++j) {
            vals[j] = x[j];
        }
        hadamard_32_inplace(vals);
        src = vals;
    }

    float amax = 0.0f;
    for (int j = 0; j < QK; ++j) {
        amax = fmaxf(amax, fabsf(src[j]));
    }

    uint8_t e_val = 0;
    float inv_d = 0.0f;
    if (amax != 0.0f) {
        // Base estimate via integer bit extraction (no SFU).
        uint32_t amax_bits;
        memcpy(&amax_bits, &amax, sizeof(uint32_t));
        const int floor_log2 = (int)((amax_bits >> 23) & 0xFF) - 127;
        // Round log2: add 1 if mantissa >= sqrt(2)-1 (0x3504F3 in IEEE-754 23-bit mantissa).
        const int round_log2 = floor_log2 + ((amax_bits & 0x7FFFFF) >= 0x3504F3 ? 1 : 0);
        const int e_base = round_log2 - traits::e8m0_offset + 127;

        // MSE-optimal search: test ±1 around estimate, pick lowest MSE.
        const int e_lo = max(1, min(255, e_base - 1));
        const int e_hi = max(1, min(255, e_base + 1));
        int best_e = max(0, min(255, e_base));
        float best_mse = 1e30f;

        for (int test_e = e_lo; test_e <= e_hi; ++test_e) {
            const float test_scale = ggml_cuda_e8m0_to_fp32((uint8_t)test_e);
            const float test_inv = 1.0f / test_scale;
            float mse = 0.0f;
            for (int j = 0; j < QK; ++j) {
                mse += traits::mse_error(src[j], test_inv, test_scale);
            }
            if (mse < best_mse) {
                best_mse = mse;
                best_e = test_e;
            }
        }

        e_val = (uint8_t)best_e;

        inv_d = 1.0f / ggml_cuda_e8m0_to_fp32(e_val);
    }

    // Write quantized values to SoA qs region.
    traits::write_qs(src, row_base, block_idx, blocks_per_row_total, inv_d);

    // Write E8M0 scale byte to SoA E8M0 region.
    *(row_base + blocks_per_row_total * traits::qs_per_block + block_idx) = e_val;
}

#endif // CUDART_VERSION >= 12080

