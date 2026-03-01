#pragma once

#include "common.cuh"
#include "hadamard.cuh"

// ── MXFP Type Traits ──────────────────────────────────────────────────
// Single source of truth for all MX (Microscaling) format parameters.
// Adding a new MX variant = one new specialization.
// All MX formats share: QK=32 (block size), E8M0 shared exponent.

template<ggml_type type> struct mxfp_traits;

// ── FP4 E2M1 ────────────────────────────────────────────────────────
template<> struct mxfp_traits<GGML_TYPE_MXFP4> {
    static constexpr int bits_per_elem = 4;
    static constexpr int qs_per_block  = 16;   // 32 * 4 / 8
    static constexpr int e8m0_offset   = 2;    // ceil(log2(6.0))

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

// ── FP6 E2M3 ────────────────────────────────────────────────────────
// Software FP6 conversion — hardware intrinsics require cuda_fp6.h (CUDA 13.0+)
namespace mxfp_detail {
    static __device__ __forceinline__ uint8_t float_to_fp6_e2m3(float x) {
        uint8_t sign = 0;
        if (x < 0) { sign = 0x20; x = -x; }
        if (x == 0) return sign;
        if (x >= 7.5f) return sign | 0x1F;  // saturate to max finite

        // E2M3: bias=1. Subnormal when biased_exp=0: val = mant * 2^(-3)
        int exp = 0;
        float frac = frexpf(x, &exp);
        exp -= 1;  // x = (1+m) * 2^exp
        int biased_exp = exp + 1;

        if (biased_exp <= 0) {
            // Subnormal: val = mant/8
            int mant = __float2int_rn(x * 8.0f);
            mant = min(mant, 7);
            return sign | (uint8_t)mant;
        }
        if (biased_exp > 3) biased_exp = 3;

        float mantf = (x / (float)(1 << exp)) - 1.0f;
        int mant = __float2int_rn(mantf * 8.0f);
        if (mant > 7) { mant = 0; biased_exp++; }
        if (biased_exp > 3) return sign | 0x1F;
        return sign | (uint8_t)((biased_exp << 3) | mant);

        GGML_UNUSED(frac);
    }

    static __device__ __forceinline__ float fp6_e2m3_to_float(uint8_t v) {
        const float sign = (v & 0x20) ? -1.0f : 1.0f;
        const int exp  = (v >> 3) & 0x3;
        const int mant = v & 0x7;
        if (exp == 0) return sign * (float)mant * 0.125f;
        return sign * (1.0f + mant * 0.125f) * (float)(1 << (exp - 1));
    }

    static __device__ __forceinline__ uint8_t float_to_fp6_e3m2(float x) {
        uint8_t sign = 0;
        if (x < 0) { sign = 0x20; x = -x; }
        if (x == 0) return sign;
        if (x >= 28.0f) return sign | 0x1F;  // saturate to max finite (exp=7, mant=3)

        // E3M2: bias=3. Subnormal when biased_exp=0: val = mant * 2^(-4)
        int exp = 0;
        float frac = frexpf(x, &exp);
        exp -= 1;  // x = (1+m) * 2^exp
        int biased_exp = exp + 3;

        if (biased_exp <= 0) {
            // Subnormal: val = mant/16
            int mant = __float2int_rn(x * 16.0f);
            mant = min(mant, 3);
            return sign | (uint8_t)mant;
        }
        if (biased_exp > 7) biased_exp = 7;

        float mantf = (x / (float)(1 << exp)) - 1.0f;
        int mant = __float2int_rn(mantf * 4.0f);
        if (mant > 3) { mant = 0; biased_exp++; }
        if (biased_exp > 7) return sign | 0x1F;
        return sign | (uint8_t)((biased_exp << 2) | mant);

        GGML_UNUSED(frac);
    }

    static __device__ __forceinline__ float fp6_e3m2_to_float(uint8_t v) {
        const float sign = (v & 0x20) ? -1.0f : 1.0f;
        const int exp  = (v >> 2) & 0x7;
        const int mant = v & 0x3;
        if (exp == 0) return sign * (float)mant * 0.0625f;  // 2^(-4)
        // MX E3M2 has no NaN/Inf — exp=7 is a valid normal value (max finite = 28.0).
        return sign * (1.0f + mant * 0.25f) * (float)(1 << (exp - 3));
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

template<> struct mxfp_traits<GGML_TYPE_MXFP6_E2M3> {
    static constexpr int bits_per_elem = 6;
    static constexpr int qs_per_block  = 24;   // 32 * 6 / 8
    static constexpr int e8m0_offset   = 3;    // ceil(log2(7.5))

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const __nv_fp6_storage_t fp6 = __nv_cvt_float_to_fp6(val * inv_scale, __NV_E2M3, cudaRoundNearest);
        const __half_raw h = __nv_cvt_fp6_to_halfraw(fp6, __NV_E2M3);
        const float recon = __half2float(*reinterpret_cast<const __half *>(&h)) * scale;
        const float err = val - recon;
        return err * err;
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

// ── FP6 E3M2 ────────────────────────────────────────────────────────
template<> struct mxfp_traits<GGML_TYPE_MXFP6_E3M2> {
    static constexpr int bits_per_elem = 6;
    static constexpr int qs_per_block  = 24;
    static constexpr int e8m0_offset   = 5;    // ceil(log2(28.0))

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const __nv_fp6_storage_t fp6 = __nv_cvt_float_to_fp6(val * inv_scale, __NV_E3M2, cudaRoundNearest);
        const __half_raw h = __nv_cvt_fp6_to_halfraw(fp6, __NV_E3M2);
        const float recon = __half2float(*reinterpret_cast<const __half *>(&h)) * scale;
        const float err = val - recon;
        return err * err;
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

// ── FP8 E4M3 ────────────────────────────────────────────────────────
template<> struct mxfp_traits<GGML_TYPE_MXFP8> {
    static constexpr int bits_per_elem = 8;
    static constexpr int qs_per_block  = 32;   // 32 * 8 / 8
    static constexpr int e8m0_offset   = 8;    // ceil(log2(448))

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const uint8_t fp8 = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3 *>(&fp8);
        const float recon = float(fp8_val) * scale;
        const float err = val - recon;
        return err * err;
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

// ── FP8 E5M2 ────────────────────────────────────────────────────────
template<> struct mxfp_traits<GGML_TYPE_MXFP8_E5M2> {
    static constexpr int bits_per_elem = 8;
    static constexpr int qs_per_block  = 32;
    static constexpr int e8m0_offset   = 16;   // ceil(log2(57344))

    static __device__ __forceinline__ float mse_error(float val, float inv_scale, float scale) {
        const uint8_t fp8 = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E5M2);
        const __nv_fp8_e5m2 fp8_val = *reinterpret_cast<const __nv_fp8_e5m2 *>(&fp8);
        const float recon = float(fp8_val) * scale;
        const float err = val - recon;
        return err * err;
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

// ── Unified SoA quantization ────────────────────────────────────────
// Shared Hadamard rotation + MSE-optimal E8M0 search + type-specific writes.
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

    // MSE-optimal E8M0 search: test +-1 around amax estimate, pick lowest MSE.
    float amax = 0.0f;
    for (int j = 0; j < QK; ++j) {
        amax = fmaxf(amax, fabsf(src[j]));
    }

    const int e_base = (amax == 0.0f) ? 0 : __float2int_rn(log2f(amax)) - traits::e8m0_offset + 127;
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

    const uint8_t e_val = (amax == 0.0f) ? (uint8_t)0 : (uint8_t)best_e;
    const float inv_d = (amax == 0.0f) ? 0.0f : 1.0f / ggml_cuda_e8m0_to_fp32(e_val);

    // Write quantized values to SoA qs region.
    traits::write_qs(src, row_base, block_idx, blocks_per_row_total, inv_d);

    // Write E8M0 scale byte to SoA E8M0 region.
    *(row_base + blocks_per_row_total * traits::qs_per_block + block_idx) = e_val;
}

