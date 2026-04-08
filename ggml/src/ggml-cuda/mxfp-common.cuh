#pragma once

#include "common.cuh"

// CUDA __device__ helpers for MXFP SoA flash attention and set_rows.
// Covers MXFP4 (E2M1), MXFP6 (E2M3), MXFP8 (E4M3).
//
// Element converters below mirror the host versions in ggml-mxfp.h but use CUDA
// intrinsics (__float_as_uint, __uint_as_float) instead of fp32_to_bits/fp32_from_bits.
// On CUDA 12.8+ they are unused — the NVIDIA fp4/fp6/fp8 intrinsics take over.

// ============================================================================
// 3-way MXFP type dispatch macro
// ============================================================================

#define MXFP_DISPATCH(runtime_type, ...) do {               \
    switch (runtime_type) {                                 \
        case GGML_TYPE_MXFP4: { constexpr ggml_type mxfp_type = GGML_TYPE_MXFP4; __VA_ARGS__; } break; \
        case GGML_TYPE_MXFP8: { constexpr ggml_type mxfp_type = GGML_TYPE_MXFP8; __VA_ARGS__; } break; \
        case GGML_TYPE_MXFP6: { constexpr ggml_type mxfp_type = GGML_TYPE_MXFP6; __VA_ARGS__; } break; \
        default: GGML_ABORT("unsupported MXFP type"); break; \
    }                                                       \
} while (0)

// ============================================================================
// Type traits: compile-time constants per MXFP type
// ============================================================================

template <ggml_type type> struct mxfp_type_traits;

template <> struct mxfp_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int emax_offset = MXFP4_E2M1_EMAX_OFFSET;
    static constexpr int qs_per_blk  = MXFP4_SOA_QS_PER_BLOCK;
};

template <> struct mxfp_type_traits<GGML_TYPE_MXFP8> {
    static constexpr int emax_offset = MXFP8_E4M3_EMAX_OFFSET;
    static constexpr int qs_per_blk  = MXFP8_SOA_QS_PER_BLOCK;
};

template <> struct mxfp_type_traits<GGML_TYPE_MXFP6> {
    static constexpr int emax_offset = MXFP6_E2M3_EMAX_OFFSET;
    static constexpr int qs_per_blk  = MXFP6_SOA_QS_PER_BLOCK;
};

// ============================================================================
// Element converters — __device__ equivalents of ggml-mxfp.h host functions.
// Only compiled when CUDART_VERSION < 12080 (fallback path).
// ============================================================================

// FP4 E2M1: [S(1) | E(2) | M(1)], max normal = 6.0
// cf. ggml_mxfp_best_index_e2m1 (host uses doubled-integer LUT, device uses float LUT)
static __device__ __forceinline__ float mxfp_fp4_e2m1_to_float(uint8_t v) {
    const float sign = (v & 0x8) ? -1.0f : 1.0f;
    const int exp  = (v >> 1) & 0x3;
    const int man  = v & 0x1;
    if (exp == 0) { return sign * (float)man * 0.5f; }
    return sign * (1.0f + (float)man) * (float)(1 << (exp - 1));
}

static __device__ __forceinline__ uint8_t mxfp_float_to_fp4_e2m1(float x) {
    uint8_t sign = 0;
    if (x < 0) { sign = 0x8; x = -x; }
    if (x >= 6.0f) { return sign | 0x7; }  // saturate

    // LUT-based best fit (same as CPU ggml_mxfp_best_index_e2m1 but with inline LUT)
    static constexpr float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    int best = 0;
    float best_err = fabsf(x - pos_lut[0]);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        float err = fabsf(x - pos_lut[i]);
        if (err < best_err) { best = i; best_err = err; }
    }
    return sign | (uint8_t)best;
}

// FP6 E2M3: [S(1) | E(2) | M(3)], bias=1, max finite=7.5  (cf. ggml_mxfp_fp6_e2m3_to_float)
static __device__ __forceinline__ float mxfp_fp6_e2m3_to_float(uint8_t v) {
    const float sign = (v & 0x20) ? -1.0f : 1.0f;
    const int exp  = (v >> 3) & 0x3;
    const int man  = v & 0x7;
    if (exp == 0) { return sign * (float)man * 0.125f; }
    return sign * (1.0f + man * 0.125f) * (float)(1 << (exp - 1));
}

static __device__ __forceinline__ uint8_t mxfp_float_to_fp6_e2m3(float x) {
    uint8_t sign = 0;
    if (x < 0) { sign = 0x20; x = -x; }
    if (x == 0) return sign;
    if (x >= 7.5f) return sign | 0x1F;

    uint32_t bits = __float_as_uint(x);
    int fp32_exp = (int)((bits >> 23) & 0xFF) - 127;

    if (fp32_exp < 0) {
        int e2m3_man = (int)(x * 8.0f + 0.5f);
        if (e2m3_man > 7) return sign | 0x08;
        return sign | (uint8_t)e2m3_man;
    }
    if (fp32_exp > 2) fp32_exp = 2;

    float frac = (x / (float)(1 << fp32_exp)) - 1.0f;
    int e2m3_man = (int)(frac * 8.0f + 0.5f);
    if (e2m3_man > 7) { e2m3_man = 0; fp32_exp++; }
    if (fp32_exp > 2) return sign | 0x1F;
    return sign | (uint8_t)(((fp32_exp + 1) << 3) | e2m3_man);
}

// FP8 E4M3: [S(1) | E(4) | M(3)], bias=7, NaN=0x7F/0xFF, max finite=448  (cf. ggml_mxfp_fp8_e4m3_to_float)
static __device__ __forceinline__ float mxfp_fp8_e4m3_to_float(uint8_t v) {
    uint32_t sign = ((uint32_t)(v & 0x80)) << 24;
    uint32_t exp  = (v >> 3) & 0xF;
    uint32_t man  = v & 0x7;
    if (exp == 0) {
        if (man == 0) return __uint_as_float(sign);
        float val = (float)man * (1.0f / 512.0f);
        uint32_t bits = __float_as_uint(val);
        return __uint_as_float((bits & 0x7FFFFFFFu) | sign);
    }
    if (exp == 15 && man == 7) return __uint_as_float(sign | 0x7FC00000u); // NaN
    return __uint_as_float(sign | ((exp + 120) << 23) | (man << 20));
}

static __device__ __forceinline__ uint8_t mxfp_float_to_fp8_e4m3(float x) {
    uint32_t bits = __float_as_uint(x);
    uint8_t sign = (bits >> 24) & 0x80;
    bits &= 0x7FFFFFFFu;
    if (bits == 0) return sign;

    uint32_t fp32_exp = (bits >> 23) & 0xFF;
    uint32_t fp32_man = bits & 0x7FFFFF;
    int e4m3_exp = (int)fp32_exp - 120;

    if (e4m3_exp <= 0) {
        int shift = 1 - e4m3_exp;
        uint32_t full_man = (1u << 23) | fp32_man;
        int total_shift = 20 + shift;
        if (total_shift >= 32) return sign;
        uint32_t e4m3_man = full_man >> total_shift;
        if (total_shift > 0 && total_shift < 32) {
            uint32_t round_bit = (full_man >> (total_shift - 1)) & 1;
            uint32_t sticky = (total_shift > 1) ? (full_man & ((1u << (total_shift - 1)) - 1)) : 0;
            if (round_bit && (sticky || (e4m3_man & 1))) e4m3_man++;
        }
        if (e4m3_man > 7) return sign | 0x08;
        return sign | (uint8_t)e4m3_man;
    }

    uint32_t round_bit = (fp32_man >> 19) & 1;
    uint32_t sticky = fp32_man & ((1u << 19) - 1);
    uint32_t e4m3_man = fp32_man >> 20;
    if (round_bit && (sticky || (e4m3_man & 1))) {
        e4m3_man++;
        if (e4m3_man > 7) { e4m3_man = 0; e4m3_exp++; }
    }
    if (e4m3_exp > 15 || (e4m3_exp == 15 && e4m3_man >= 7)) return sign | 0x7E;
    return sign | (uint8_t)((e4m3_exp << 3) | e4m3_man);
}

// FP6 packing: 4 six-bit values <-> 3 bytes  (cf. ggml_mxfp_pack_fp6x4)
static __device__ __forceinline__ void mxfp_pack_fp6x4(const uint8_t v[4], uint8_t out[3]) {
    uint32_t packed = (v[0] & 0x3F) | ((v[1] & 0x3F) << 6) |
                      ((v[2] & 0x3F) << 12) | ((v[3] & 0x3F) << 18);
    out[0] = (uint8_t)(packed);
    out[1] = (uint8_t)(packed >> 8);
    out[2] = (uint8_t)(packed >> 16);
}

// ============================================================================
// E8M0 scale computation  (cf. ggml_mxfp_compute_e8m0 in ggml-mxfp.h)
// ============================================================================

template <ggml_type type>
static __device__ __forceinline__ uint8_t mxfp_compute_e8m0(float amax) {
    if (!(amax > 0.0f)) return 0;
    const uint32_t amax_bits = __float_as_uint(amax);
    const int floor_log2 = (int)((amax_bits >> 23) & 0xFF) - 127;
    const int biased_log2 = floor_log2 + ((amax_bits & 0x7FFFFF) >= 0x3504F3 ? 1 : 0);
    const int e_base = biased_log2 - mxfp_type_traits<type>::emax_offset + 127;
    return (uint8_t)(e_base < 0 ? 0 : (e_base > 254 ? 254 : e_base));
}

// CUDA 12.8+ intrinsic helpers for half_raw ↔ float conversion.
#if CUDART_VERSION >= 12080
static __device__ __forceinline__ float halfraw_to_float(__half_raw hr) {
    return __half2float(*reinterpret_cast<__half *>(&hr));
}
static __device__ __forceinline__ float2 half2raw_to_float2(__half2_raw hr2) {
    __half2 h2 = *reinterpret_cast<__half2 *>(&hr2);
    return make_float2(__low2float(h2), __high2float(h2));
}
#endif

// Per-element quantize and dequant (uses intrinsics on 12.8+, fallback converters otherwise).
template <ggml_type type>
static __device__ __forceinline__ uint8_t mxfp_quantize_elem(float val, uint8_t e8m0) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0);
    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
    const float scaled = val * inv_d;
#if CUDART_VERSION >= 12080
    if constexpr (type == GGML_TYPE_MXFP4) { return (__nv_fp4_storage_t)__nv_cvt_float_to_fp4(scaled, __NV_E2M1, cudaRoundNearest); }
    if constexpr (type == GGML_TYPE_MXFP8) { return (__nv_fp8_storage_t)__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3); }
    if constexpr (type == GGML_TYPE_MXFP6) { return (__nv_fp6_storage_t)__nv_cvt_float_to_fp6(scaled, __NV_E2M3, cudaRoundNearest); }
#else
    if constexpr (type == GGML_TYPE_MXFP4) { return mxfp_float_to_fp4_e2m1(scaled); }
    if constexpr (type == GGML_TYPE_MXFP8) { return mxfp_float_to_fp8_e4m3(scaled); }
    if constexpr (type == GGML_TYPE_MXFP6) { return mxfp_float_to_fp6_e2m3(scaled); }
#endif
}

template <ggml_type type>
static __device__ __forceinline__ float mxfp_dequant_raw(uint8_t raw) {
#if CUDART_VERSION >= 12080
    if constexpr (type == GGML_TYPE_MXFP4) { return halfraw_to_float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)raw, __NV_E2M1)); }
    if constexpr (type == GGML_TYPE_MXFP8) { return halfraw_to_float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)raw, __NV_E4M3)); }
    if constexpr (type == GGML_TYPE_MXFP6) { return halfraw_to_float(__nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)raw, __NV_E2M3)); }
#else
    if constexpr (type == GGML_TYPE_MXFP4) { return mxfp_fp4_e2m1_to_float(raw); }
    if constexpr (type == GGML_TYPE_MXFP8) { return mxfp_fp8_e4m3_to_float(raw); }
    if constexpr (type == GGML_TYPE_MXFP6) { return mxfp_fp6_e2m3_to_float(raw); }
#endif
}

// MXFP4 branchless nibble-pair extraction from SoA layout.
static __device__ __forceinline__ void mxfp4_extract_nibble_pair(
        const uint8_t * qs, int pos0, uint8_t & nib0, uint8_t & nib1) {
    const int bi0 = pos0 & 15;
    uint16_t pair;
    memcpy(&pair, qs + bi0, 2);
    if (pos0 < 16) {
        nib0 = pair & 0x0F;
        nib1 = (pair >> 8) & 0x0F;
    } else {
        nib0 = (pair >> 4) & 0x0F;
        nib1 = (pair >> 12) & 0x0F;
    }
}

// MXFP6 pair extraction from SoA layout.
static __device__ __forceinline__ void mxfp6_unpack_pair(
        const uint8_t * qs_block, int pos0, uint8_t & v0, uint8_t & v1) {
    const int grp  = pos0 / 4;
    const int slot = pos0 % 4;  // 0 or 2
    const uint8_t * base = qs_block + grp * 3;
    uint32_t packed;
    memcpy(&packed, base, 3);
    packed &= 0x00FFFFFFu;
    const int shift = slot * 6;
    v0 = (packed >> shift) & 0x3F;
    v1 = (packed >> (shift + 6)) & 0x3F;
}

// Unified SoA pair dequant: two consecutive elements from one 32-element block.
// pos0 MUST be even. Returns float2{elem[pos0], elem[pos0+1]}.
template <ggml_type type>
static __device__ __forceinline__ float2 mxfp_dequant_elem_pair(
        const uint8_t * qs_block, uint8_t e8m0, int pos0) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0);

    if constexpr (type == GGML_TYPE_MXFP4) {
#if CUDART_VERSION >= 12080
        const int bi0 = pos0 & 15;
        uint16_t pair;
        memcpy(&pair, qs_block + bi0, 2);
        uint8_t packed;
        if (pos0 < 16) {
            packed = (pair & 0x0F) | ((pair >> 4) & 0xF0);
        } else {
            packed = ((pair >> 4) & 0x0F) | ((pair >> 8) & 0xF0);
        }
        const float2 raw = half2raw_to_float2(__nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1));
        return make_float2(raw.x * d, raw.y * d);
#else
        uint8_t nib0, nib1;
        mxfp4_extract_nibble_pair(qs_block, pos0, nib0, nib1);
        return make_float2(mxfp_dequant_raw<type>(nib0) * d, mxfp_dequant_raw<type>(nib1) * d);
#endif
    } else if constexpr (type == GGML_TYPE_MXFP8) {
#if CUDART_VERSION >= 12080
        const __nv_fp8x2_storage_t x2 = __ldg(reinterpret_cast<const __nv_fp8x2_storage_t *>(qs_block + pos0));
        const float2 raw = half2raw_to_float2(__nv_cvt_fp8x2_to_halfraw2(x2, __NV_E4M3));
        return make_float2(raw.x * d, raw.y * d);
#else
        return make_float2(mxfp_dequant_raw<type>(__ldg(qs_block + pos0)) * d,
                           mxfp_dequant_raw<type>(__ldg(qs_block + pos0 + 1)) * d);
#endif
    } else {
        // 6-bit: E2M3
        uint8_t v0, v1;
        mxfp6_unpack_pair(qs_block, pos0, v0, v1);
#if CUDART_VERSION >= 12080
        const __nv_fp6x2_storage_t x2 = (__nv_fp6x2_storage_t)(v0 | ((uint16_t)v1 << 8));
        const float2 raw = half2raw_to_float2(__nv_cvt_fp6x2_to_halfraw2(x2, __NV_E2M3));
        return make_float2(raw.x * d, raw.y * d);
#else
        return make_float2(mxfp_dequant_raw<type>(v0) * d, mxfp_dequant_raw<type>(v1) * d);
#endif
    }
}

// Single-element SoA dequant (used by k_mxfp_soa_to_f16 for MMA prefill conversion).
template <ggml_type type>
static __device__ __forceinline__ float mxfp_dequant_elem(
        const uint8_t * qs_base, const uint8_t * e8m0_base,
        int blk, int pos, int qs_per_blk) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0_base[blk]);
    const uint8_t * qs = qs_base + blk * qs_per_blk;
    uint8_t raw;
    if constexpr (type == GGML_TYPE_MXFP4) {
        const int byte_idx = pos < 16 ? pos : pos - 16;
        raw = (pos < 16) ? (qs[byte_idx] & 0x0F) : (qs[byte_idx] >> 4);
    } else if constexpr (type == GGML_TYPE_MXFP8) {
        raw = qs[pos];
    } else {
        const int grp  = pos / 4;
        const int slot = pos % 4;
        const uint8_t * base = qs + grp * 3;
        uint32_t packed;
        memcpy(&packed, base, 3);
        packed &= 0x00FFFFFFu;
        raw = (packed >> (slot * 6)) & 0x3F;
    }
    return mxfp_dequant_raw<type>(raw) * d;
}

// MXFP SoA → F16 dequant kernel (MMA flash attention pre-conversion).
// Each head's SoA chunk is addressed via standard tensor strides.
template <ggml_type mxfp_type>
static __global__ void k_mxfp_soa_to_f16(
        const char * __restrict__ src,
        half * __restrict__ dst,
        const int D,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3,
        const int blocks_per_head) {

    constexpr int qs_per_blk = mxfp_type_traits<mxfp_type>::qs_per_blk;

    const int64_t flat_row = blockIdx.x;
    const int elem = threadIdx.x;
    const int64_t nrows = ne1 * ne2 * ne3;
    if (flat_row >= nrows || elem >= D) return;

    const int64_t i3 = flat_row / (ne1 * ne2);
    const int64_t i2 = (flat_row / ne1) % ne2;
    const int64_t i1 = flat_row % ne1;

    const char * src_row = src + i1*nb1 + i2*nb2 + i3*nb3;
    const uint8_t * qs_base   = (const uint8_t *)src_row;
    const uint8_t * e8m0_base = qs_base + blocks_per_head * qs_per_blk;

    const int blk = elem / 32;
    const int pos = elem % 32;
    const float val = mxfp_dequant_elem<mxfp_type>(qs_base, e8m0_base, blk, pos, qs_per_blk);
    dst[flat_row * D + elem] = __float2half(val);
}

// Host dispatch for MXFP SoA → F16 conversion.
static void mxfp_soa_to_f16_cuda(
        const char * src, half * dst, ggml_type type,
        int64_t D, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, cudaStream_t stream) {

    const int blocks_per_head = (int)(D / 32);
    const int64_t nrows = ne1 * ne2 * ne3;
    const int threads = (int)D;
    const int grid    = (int)nrows;
    if (grid <= 0 || threads <= 0) return;

    MXFP_DISPATCH(type, {
        k_mxfp_soa_to_f16<mxfp_type><<<grid, threads, 0, stream>>>(
            src, dst, (int)D, ne1, ne2, ne3,
            (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
            blocks_per_head);
    });
}
