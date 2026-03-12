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

// Compile-time check: should Hadamard rotation be applied for this MXFP type?
// References centralized MXFP_USE_HADAMARD_* defines from ggml-common.h.
template<ggml_type type>
static constexpr bool mxfp_use_hadamard_v =
    (type == GGML_TYPE_MXFP4_E2M1 && MXFP_USE_HADAMARD_E2M1) ||
    (type == GGML_TYPE_MXFP8_E4M3 && MXFP_USE_HADAMARD_E4M3) ||
    (type == GGML_TYPE_MXFP8_E5M2 && MXFP_USE_HADAMARD_E5M2) ||
    (type == GGML_TYPE_MXFP6_E2M3 && MXFP_USE_HADAMARD_E2M3) ||
    (type == GGML_TYPE_MXFP6_E3M2 && MXFP_USE_HADAMARD_E3M2);

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
    // qs_per_block from centralized MXFP_QS_PER_BLOCK_* defines in ggml-common.h.
    constexpr int qs_per_block = (type == GGML_TYPE_MXFP4_E2M1) ? MXFP_QS_PER_BLOCK_E2M1 :
                                 (type == GGML_TYPE_MXFP8_E4M3 || type == GGML_TYPE_MXFP8_E5M2) ?
                                     MXFP_QS_PER_BLOCK_E4M3 : MXFP_QS_PER_BLOCK_E2M3;
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
    static constexpr int bits_per_elem = MXFP_BITS_PER_ELEM_E2M1;
    static constexpr int qs_per_block  = MXFP_QS_PER_BLOCK_E2M1;
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
#if CUDART_VERSION >= 12080
        // Vectorized: quantize lo+hi pair → byte via x2 intrinsic (lo→low nibble, hi→high nibble).
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            __nv_fp4x2_e2m1 fp4(make_float2(src[j] * inv_d, src[QK_MXFP4/2 + j] * inv_d));
            qs_dst[j] = fp4.__x;
        }
#else
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t lo = ggml_cuda_float_to_fp4_e2m1(src[j], inv_d);
            const uint8_t hi = ggml_cuda_float_to_fp4_e2m1(src[QK_MXFP4/2 + j], inv_d);
            qs_dst[j] = lo | (hi << 4);
        }
#endif
    }
};

// Portable MXFP element converters are now in ggml-common.h (canonical source: ggml_mxfp_*).
// Thin wrappers for backward compatibility with existing call sites using mxfp_detail:: prefix.
namespace mxfp_detail {
    static __device__ __forceinline__ float fp4_e2m1_to_float(uint8_t v) { return ggml_mxfp_fp4_e2m1_to_float(v); }
    static __device__ __forceinline__ uint8_t float_to_fp4_e2m1(float x) { return ggml_mxfp_float_to_fp4_e2m1(x); }
    static __device__ __forceinline__ float fp6_e2m3_to_float(uint8_t v) { return ggml_mxfp_fp6_e2m3_to_float(v); }
    static __device__ __forceinline__ float fp6_e3m2_to_float(uint8_t v) { return ggml_mxfp_fp6_e3m2_to_float(v); }
    static __device__ __forceinline__ uint8_t float_to_fp6_e2m3(float x) { return ggml_mxfp_float_to_fp6_e2m3(x); }
    static __device__ __forceinline__ uint8_t float_to_fp6_e3m2(float x) { return ggml_mxfp_float_to_fp6_e3m2(x); }
    static __device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) { return ggml_mxfp_fp8_e4m3_to_float(v); }
    static __device__ __forceinline__ float fp8_e5m2_to_float(uint8_t v) { return ggml_mxfp_fp8_e5m2_to_float(v); }
    static __device__ __forceinline__ uint8_t float_to_fp8_e4m3(float x) { return ggml_mxfp_float_to_fp8_e4m3(x); }
    static __device__ __forceinline__ uint8_t float_to_fp8_e5m2(float x) { return ggml_mxfp_float_to_fp8_e5m2(x); }
    static __device__ __forceinline__ void pack_fp6x4(const uint8_t v[4], uint8_t out[3]) { ggml_mxfp_pack_fp6x4(v, out); }
    static __device__ __forceinline__ void unpack_fp6x4(const uint8_t in[3], uint8_t v[4]) { ggml_mxfp_unpack_fp6x4(in, v); }
} // namespace mxfp_detail

// FP6 E2M3:
template<> struct mxfp_traits<GGML_TYPE_MXFP6_E2M3> {
    static constexpr int bits_per_elem = MXFP_BITS_PER_ELEM_E2M3;
    static constexpr int qs_per_block  = MXFP_QS_PER_BLOCK_E2M3;
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
#if CUDART_VERSION >= 12080
            // Vectorized: quantize 4 floats → 4 byte-padded FP6 in one x4 intrinsic.
            __nv_fp6x4_e2m3 fp6(make_float4(
                src[j + 0] * inv_d, src[j + 1] * inv_d,
                src[j + 2] * inv_d, src[j + 3] * inv_d));
            memcpy(vals, &fp6.__x, 4);
#else
            for (int jj = 0; jj < 4; ++jj) {
                vals[jj] = mxfp_detail::float_to_fp6_e2m3(src[j + jj] * inv_d);
            }
#endif
            mxfp_detail::pack_fp6x4(vals, &qs_dst[j * 3 / 4]);
        }
    }
};

// FP6 E3M2:
template<> struct mxfp_traits<GGML_TYPE_MXFP6_E3M2> {
    static constexpr int bits_per_elem = MXFP_BITS_PER_ELEM_E3M2;
    static constexpr int qs_per_block  = MXFP_QS_PER_BLOCK_E3M2;
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
#if CUDART_VERSION >= 12080
            // Vectorized: quantize 4 floats → 4 byte-padded FP6 in one x4 intrinsic.
            __nv_fp6x4_e3m2 fp6(make_float4(
                src[j + 0] * inv_d, src[j + 1] * inv_d,
                src[j + 2] * inv_d, src[j + 3] * inv_d));
            memcpy(vals, &fp6.__x, 4);
#else
            for (int jj = 0; jj < 4; ++jj) {
                vals[jj] = mxfp_detail::float_to_fp6_e3m2(src[j + jj] * inv_d);
            }
#endif
            mxfp_detail::pack_fp6x4(vals, &qs_dst[j * 3 / 4]);
        }
    }
};

// FP8 E4M3:
template<> struct mxfp_traits<GGML_TYPE_MXFP8_E4M3> {
    static constexpr int bits_per_elem = MXFP_BITS_PER_ELEM_E4M3;
    static constexpr int qs_per_block  = MXFP_QS_PER_BLOCK_E4M3;
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
        uint32_t * qs_dst = reinterpret_cast<uint32_t *>(row_base + block_idx * qs_per_block);
#if CUDART_VERSION >= 12050
        // Vectorized: quantize 4 floats → uint32 in one x4 intrinsic.
        for (int j = 0; j < 32; j += 4) {
            __nv_fp8x4_e4m3 fp8(make_float4(
                src[j + 0] * inv_d, src[j + 1] * inv_d,
                src[j + 2] * inv_d, src[j + 3] * inv_d));
            qs_dst[j / 4] = fp8.__x;
        }
#else
        uint8_t * qs_bytes = reinterpret_cast<uint8_t *>(qs_dst);
        for (int j = 0; j < 32; ++j) {
            qs_bytes[j] = mxfp_detail::float_to_fp8_e4m3(src[j] * inv_d);
        }
#endif
    }
};

// FP8 E5M2:
template<> struct mxfp_traits<GGML_TYPE_MXFP8_E5M2> {
    static constexpr int bits_per_elem = MXFP_BITS_PER_ELEM_E5M2;
    static constexpr int qs_per_block  = MXFP_QS_PER_BLOCK_E5M2;
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
        uint32_t * qs_dst = reinterpret_cast<uint32_t *>(row_base + block_idx * qs_per_block);
#if CUDART_VERSION >= 12050
        // Vectorized: quantize 4 floats → uint32 in one x4 intrinsic.
        for (int j = 0; j < 32; j += 4) {
            __nv_fp8x4_e5m2 fp8(make_float4(
                src[j + 0] * inv_d, src[j + 1] * inv_d,
                src[j + 2] * inv_d, src[j + 3] * inv_d));
            qs_dst[j / 4] = fp8.__x;
        }
#else
        uint8_t * qs_bytes = reinterpret_cast<uint8_t *>(qs_dst);
        for (int j = 0; j < 32; ++j) {
            qs_bytes[j] = mxfp_detail::float_to_fp8_e5m2(src[j] * inv_d);
        }
#endif
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
        const int e_lo = max(1, min(254, e_base - MXFP_E8M0_MSE_RANGE));
        const int e_hi = max(1, min(254, e_base + MXFP_E8M0_MSE_RANGE));
        int best_e = max(0, min(254, e_base));
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

