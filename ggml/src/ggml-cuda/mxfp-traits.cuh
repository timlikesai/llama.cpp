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

// FP4 E2M1 traits: uses LUT + ggml_cuda_float_to_fp4_e2m1 (has non-CUDART fallback).
// Defined outside CUDART guard so MXFP4 works on all backends (CUDA, HIP, MUSA).
template<> struct mxfp_traits<GGML_TYPE_MXFP4_E2M1> {
    static constexpr int bits_per_elem = 4;
    static constexpr int qs_per_block  = 16;   // 32 * 4 / 8
    static constexpr int block_size    = sizeof(block_mxfp4);
    static constexpr int e8m0_offset   = MXFP4_E2M1_EMAX_OFFSET;

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
        return kvalues_mxfp4[raw & 0xF] * 0.5f;
    }

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
            const uint8_t lo = ggml_cuda_float_to_fp4_e2m1(src[j], inv_d);
            const uint8_t hi = ggml_cuda_float_to_fp4_e2m1(src[QK_MXFP4/2 + j], inv_d);
            qs_dst[j] = lo | (hi << 4);
        }
    }
};

// ------------------------------------------------------------------------------------------------------------------
// Portable MXFP helpers — IEEE bit manipulation, no CUDA 12.8+ intrinsics needed.
// Ported from our Metal shader implementation (ggml-metal.metal) to work on all backends.
// ------------------------------------------------------------------------------------------------------------------
namespace mxfp_detail {

    // --- FP6 dequantization ---

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

    // --- FP6 quantization (round-to-nearest-even) ---

    static __device__ __forceinline__ uint8_t float_to_fp6_e2m3(float x) {
        uint8_t sign = 0;
        if (x < 0) { sign = 0x20; x = -x; }
        if (x == 0) return sign;
        if (x >= 7.5f) return sign | 0x1F;  // max finite

        uint32_t bits;
        memcpy(&bits, &x, sizeof(uint32_t));
        int f32_exp = (int)((bits >> 23) & 0xFF) - 127;

        if (f32_exp < 0) {
            // Subnormal in E2M3: mant * 2^(-3)
            float scaled = x * 8.0f;
            int mant = (int)(scaled + 0.5f);
            if (mant > 7) return sign | 0x08;  // smallest normal
            return sign | (uint8_t)mant;
        }
        if (f32_exp > 2) f32_exp = 2;

        float mantf = (x / (float)(1 << f32_exp)) - 1.0f;
        int mant = (int)(mantf * 8.0f + 0.5f);
        if (mant > 7) { mant = 0; f32_exp++; }
        if (f32_exp > 2) return sign | 0x1F;
        return sign | (uint8_t)(((f32_exp + 1) << 3) | mant);
    }

    static __device__ __forceinline__ uint8_t float_to_fp6_e3m2(float x) {
        uint8_t sign = 0;
        if (x < 0) { sign = 0x20; x = -x; }
        if (x == 0) return sign;
        if (x >= 28.0f) return sign | 0x1F;  // max finite

        uint32_t bits;
        memcpy(&bits, &x, sizeof(uint32_t));
        int f32_exp = (int)((bits >> 23) & 0xFF) - 127;
        int biased_exp = f32_exp + 3;

        if (biased_exp <= 0) {
            // Subnormal in E3M2: mant * 2^(-4)
            float scaled = x * 16.0f;
            int mant = (int)(scaled + 0.5f);
            if (mant > 3) return sign | 0x04;  // smallest normal
            return sign | (uint8_t)mant;
        }
        if (biased_exp > 7) return sign | 0x1F;

        float pow2 = (f32_exp >= 0) ? (float)(1 << f32_exp) : 1.0f / (float)(1 << (-f32_exp));
        float mantf = (x / pow2) - 1.0f;
        int mant = (int)(mantf * 4.0f + 0.5f);
        if (mant > 3) { mant = 0; biased_exp++; }
        if (biased_exp > 7) return sign | 0x1F;
        return sign | (uint8_t)((biased_exp << 2) | mant);
    }

    // --- FP8 dequantization (IEEE bit manipulation) ---

    // FP8 E4M3: [S(1) | E(4) | M(3)] — bias=7, max finite=448
    static __device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) {
        uint32_t sign = ((uint32_t)(v & 0x80)) << 24;
        uint32_t exp  = (v >> 3) & 0xF;
        uint32_t mant = v & 0x7;

        if (exp == 0) {
            if (mant == 0) { float r; uint32_t s = sign; memcpy(&r, &s, 4); return r; }
            // Subnormal: mant * 2^(1-7) * 2^(-3) = mant * 2^(-9)
            float val = (float)mant * (1.0f / 512.0f);
            uint32_t vb; memcpy(&vb, &val, 4);
            vb = (vb & 0x7FFFFFFFu) | sign;
            memcpy(&val, &vb, 4);
            return val;
        }
        if (exp == 15 && mant == 7) {
            uint32_t nan_bits = sign | 0x7FC00000u;
            float r; memcpy(&r, &nan_bits, 4); return r;
        }
        // Normal: (-1)^S * 2^(E-7) * (1 + M/8) → F32 exp = E-7+127 = E+120
        uint32_t f32_bits = sign | ((exp + 120) << 23) | (mant << 20);
        float r; memcpy(&r, &f32_bits, 4); return r;
    }

    // FP8 E5M2: [S(1) | E(5) | M(2)] — bias=15, max finite=57344
    static __device__ __forceinline__ float fp8_e5m2_to_float(uint8_t v) {
        uint32_t sign = ((uint32_t)(v & 0x80)) << 24;
        uint32_t exp  = (v >> 2) & 0x1F;
        uint32_t mant = v & 0x3;

        if (exp == 0) {
            if (mant == 0) { float r; uint32_t s = sign; memcpy(&r, &s, 4); return r; }
            // Subnormal: mant * 2^(1-15) * 2^(-2) = mant/4 * 2^(-14)
            float val = (float)mant * 0.25f * (1.0f / 16384.0f);
            uint32_t vb; memcpy(&vb, &val, 4);
            vb = (vb & 0x7FFFFFFFu) | sign;
            memcpy(&val, &vb, 4);
            return val;
        }
        if (exp == 31) {
            uint32_t inf_nan = sign | 0x7F800000u | (mant ? 0x400000u : 0);
            float r; memcpy(&r, &inf_nan, 4); return r;
        }
        // Normal: F32 exp = E-15+127 = E+112
        uint32_t f32_bits = sign | ((exp + 112) << 23) | (mant << 21);
        float r; memcpy(&r, &f32_bits, 4); return r;
    }

    // --- FP8 quantization (round-to-nearest-even, saturate-to-finite) ---

    static __device__ __forceinline__ uint8_t float_to_fp8_e4m3(float x) {
        uint32_t bits;
        memcpy(&bits, &x, sizeof(uint32_t));
        uint8_t sign = (bits >> 24) & 0x80;
        bits &= 0x7FFFFFFFu;
        if (bits == 0) return sign;

        uint32_t f32_exp  = (bits >> 23) & 0xFF;
        uint32_t f32_mant = bits & 0x7FFFFF;
        int e4m3_exp = (int)f32_exp - 120;

        if (e4m3_exp < 0) {
            // Subnormal in E4M3
            int shift = 1 - e4m3_exp;
            uint32_t full_mant = (1u << 23) | f32_mant;
            int total_shift = 20 + shift;
            if (total_shift >= 32) return sign;
            uint32_t mant3 = full_mant >> total_shift;
            if (total_shift > 0 && total_shift < 32) {
                uint32_t round_bit = (full_mant >> (total_shift - 1)) & 1;
                uint32_t sticky = (total_shift > 1) ? (full_mant & ((1u << (total_shift - 1)) - 1)) : 0;
                if (round_bit && (sticky || (mant3 & 1))) mant3++;
            }
            if (mant3 > 7) return sign | 0x08;
            return sign | (uint8_t)mant3;
        }

        uint32_t round_bit = (f32_mant >> 19) & 1;
        uint32_t sticky = f32_mant & ((1u << 19) - 1);
        uint32_t mant3 = f32_mant >> 20;
        if (round_bit && (sticky || (mant3 & 1))) {
            mant3++;
            if (mant3 > 7) { mant3 = 0; e4m3_exp++; }
        }
        if (e4m3_exp > 15 || (e4m3_exp == 15 && mant3 >= 7)) return sign | 0x7E; // max finite
        return sign | (uint8_t)((e4m3_exp << 3) | mant3);
    }

    static __device__ __forceinline__ uint8_t float_to_fp8_e5m2(float x) {
        uint32_t bits;
        memcpy(&bits, &x, sizeof(uint32_t));
        uint8_t sign = (bits >> 24) & 0x80;
        bits &= 0x7FFFFFFFu;
        if (bits == 0) return sign;

        uint32_t f32_exp  = (bits >> 23) & 0xFF;
        uint32_t f32_mant = bits & 0x7FFFFF;
        int e5m2_exp = (int)f32_exp - 112;

        if (e5m2_exp < 0) {
            int shift = 1 - e5m2_exp;
            uint32_t full_mant = (1u << 23) | f32_mant;
            int total_shift = 21 + shift;
            if (total_shift >= 32) return sign;
            uint32_t mant2 = full_mant >> total_shift;
            if (total_shift > 0 && total_shift < 32) {
                uint32_t round_bit = (full_mant >> (total_shift - 1)) & 1;
                uint32_t sticky = (total_shift > 1) ? (full_mant & ((1u << (total_shift - 1)) - 1)) : 0;
                if (round_bit && (sticky || (mant2 & 1))) mant2++;
            }
            if (mant2 > 3) return sign | 0x04;
            return sign | (uint8_t)mant2;
        }

        uint32_t round_bit = (f32_mant >> 20) & 1;
        uint32_t sticky = f32_mant & ((1u << 20) - 1);
        uint32_t mant2 = f32_mant >> 21;
        if (round_bit && (sticky || (mant2 & 1))) {
            mant2++;
            if (mant2 > 3) { mant2 = 0; e5m2_exp++; }
        }
        if (e5m2_exp >= 31) return sign | 0x7B; // max finite
        return sign | (uint8_t)((e5m2_exp << 2) | mant2);
    }

    // --- FP6 packing/unpacking ---

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
    static constexpr int e8m0_offset   = MXFP6_E2M3_EMAX_OFFSET;

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
#if CUDART_VERSION >= 12080
        const __nv_fp6_storage_t fp6 = __nv_cvt_float_to_fp6(val * inv_scale, __NV_E2M3, cudaRoundNearest);
        const __half_raw h = __nv_cvt_fp6_to_halfraw(fp6, __NV_E2M3);
        const float recon = __half2float(*reinterpret_cast<const __half *>(&h)) * scale;
#else
        const uint8_t fp6 = mxfp_detail::float_to_fp6_e2m3(val * inv_scale);
        const float recon = mxfp_detail::fp6_e2m3_to_float(fp6) * scale;
#endif
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
#if CUDART_VERSION >= 12080
        const __half_raw h = __nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)raw, __NV_E2M3);
        return __half2float(*reinterpret_cast<const __half *>(&h));
#else
        return mxfp_detail::fp6_e2m3_to_float(raw);
#endif
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
#if CUDART_VERSION >= 12080
        return (uint8_t)__nv_cvt_float_to_fp6(val, __NV_E2M3, cudaRoundNearest);
#else
        return mxfp_detail::float_to_fp6_e2m3(val);
#endif
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; j += 4) {
            uint8_t vals[4];
            for (int jj = 0; jj < 4; ++jj) {
#if CUDART_VERSION >= 12080
                vals[jj] = (uint8_t)__nv_cvt_float_to_fp6(src[j + jj] * inv_d, __NV_E2M3, cudaRoundNearest);
#else
                vals[jj] = mxfp_detail::float_to_fp6_e2m3(src[j + jj] * inv_d);
#endif
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
    static constexpr int e8m0_offset   = MXFP6_E3M2_EMAX_OFFSET;

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
#if CUDART_VERSION >= 12080
        const __nv_fp6_storage_t fp6 = __nv_cvt_float_to_fp6(val * inv_scale, __NV_E3M2, cudaRoundNearest);
        const __half_raw h = __nv_cvt_fp6_to_halfraw(fp6, __NV_E3M2);
        const float recon = __half2float(*reinterpret_cast<const __half *>(&h)) * scale;
#else
        const uint8_t fp6 = mxfp_detail::float_to_fp6_e3m2(val * inv_scale);
        const float recon = mxfp_detail::fp6_e3m2_to_float(fp6) * scale;
#endif
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
#if CUDART_VERSION >= 12080
        const __half_raw h = __nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)raw, __NV_E3M2);
        return __half2float(*reinterpret_cast<const __half *>(&h));
#else
        return mxfp_detail::fp6_e3m2_to_float(raw);
#endif
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
#if CUDART_VERSION >= 12080
        return (uint8_t)__nv_cvt_float_to_fp6(val, __NV_E3M2, cudaRoundNearest);
#else
        return mxfp_detail::float_to_fp6_e3m2(val);
#endif
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; j += 4) {
            uint8_t vals[4];
            for (int jj = 0; jj < 4; ++jj) {
#if CUDART_VERSION >= 12080
                vals[jj] = (uint8_t)__nv_cvt_float_to_fp6(src[j + jj] * inv_d, __NV_E3M2, cudaRoundNearest);
#else
                vals[jj] = mxfp_detail::float_to_fp6_e3m2(src[j + jj] * inv_d);
#endif
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
    static constexpr int e8m0_offset   = MXFP8_E4M3_EMAX_OFFSET;

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
#if CUDART_VERSION >= 12050
        const uint8_t fp8 = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3 *>(&fp8);
        const float recon = float(fp8_val) * scale;
#else
        const uint8_t fp8 = mxfp_detail::float_to_fp8_e4m3(val * inv_scale);
        const float recon = mxfp_detail::fp8_e4m3_to_float(fp8) * scale;
#endif
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
#if CUDART_VERSION >= 12050
        const __nv_fp8_e4m3 v = *reinterpret_cast<const __nv_fp8_e4m3 *>(&raw);
        return float(v);
#else
        return mxfp_detail::fp8_e4m3_to_float(raw);
#endif
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
#if CUDART_VERSION >= 12050
        return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
#else
        return mxfp_detail::float_to_fp8_e4m3(val);
#endif
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; ++j) {
#if CUDART_VERSION >= 12050
            qs_dst[j] = __nv_cvt_float_to_fp8(src[j] * inv_d, __NV_SATFINITE, __NV_E4M3);
#else
            qs_dst[j] = mxfp_detail::float_to_fp8_e4m3(src[j] * inv_d);
#endif
        }
    }
};

// FP8 E5M2:
template<> struct mxfp_traits<GGML_TYPE_MXFP8_E5M2> {
    static constexpr int bits_per_elem = 8;
    static constexpr int qs_per_block  = 32;
    static constexpr int block_size    = sizeof(block_mxfp8);
    static constexpr int e8m0_offset   = MXFP8_E5M2_EMAX_OFFSET;

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
#if CUDART_VERSION >= 12050
        const uint8_t fp8 = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E5M2);
        const __nv_fp8_e5m2 fp8_val = *reinterpret_cast<const __nv_fp8_e5m2 *>(&fp8);
        const float recon = float(fp8_val) * scale;
#else
        const uint8_t fp8 = mxfp_detail::float_to_fp8_e5m2(val * inv_scale);
        const float recon = mxfp_detail::fp8_e5m2_to_float(fp8) * scale;
#endif
        const float err = val - recon;
        return err * err;
    }

    static __device__ __forceinline__ float dequant_elem(uint8_t raw) {
#if CUDART_VERSION >= 12050
        const __nv_fp8_e5m2 v = *reinterpret_cast<const __nv_fp8_e5m2 *>(&raw);
        return float(v);
#else
        return mxfp_detail::fp8_e5m2_to_float(raw);
#endif
    }

    static __device__ __forceinline__ uint8_t quantize_elem(float val) {
#if CUDART_VERSION >= 12050
        return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E5M2);
#else
        return mxfp_detail::float_to_fp8_e5m2(val);
#endif
    }

    static __device__ __forceinline__ void write_qs(
            const float * __restrict__ src, char * __restrict__ row_base,
            int block_idx, int /*blocks_per_row_total*/, float inv_d) {
        uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * qs_per_block);
        for (int j = 0; j < 32; ++j) {
#if CUDART_VERSION >= 12050
            qs_dst[j] = __nv_cvt_float_to_fp8(src[j] * inv_d, __NV_SATFINITE, __NV_E5M2);
#else
            qs_dst[j] = mxfp_detail::float_to_fp8_e5m2(src[j] * inv_d);
#endif
        }
    }
};

// Unified SoA quantization:
// Shared Hadamard rotation + direct E8M0 scale + type-specific writes.
// All MXFP traits use portable IEEE bit manipulation — works on CUDA, HIP, MUSA.
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
    if (amax != 0.0f && isfinite(amax)) {
        // Base estimate via integer bit extraction (no SFU).
        uint32_t amax_bits;
        memcpy(&amax_bits, &amax, sizeof(uint32_t));
        const int floor_log2 = (int)((amax_bits >> 23) & 0xFF) - 127;
        // Round log2: add 1 if mantissa >= sqrt(2)-1 (0x3504F3 in IEEE-754 23-bit mantissa).
        const int round_log2 = floor_log2 + ((amax_bits & 0x7FFFFF) >= 0x3504F3 ? 1 : 0);
        const int e_base = round_log2 - traits::e8m0_offset + 127;

        // MSE-optimal search: test ±R around estimate, pick lowest MSE.
        const int e_lo = max(1, min(255, e_base - MXFP_E8M0_MSE_RANGE));
        const int e_hi = max(1, min(255, e_base + MXFP_E8M0_MSE_RANGE));
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

