#pragma once

// MXFP element format converters and packing utilities (host/CPU).
// Ref: OCP Microscaling Formats (MX) Specification v1.0
// Requires: ggml-common.h included before this header.
// CUDA __device__ equivalents: ggml-cuda/mxfp-common.cuh

#include "ggml-impl.h"

#define MXFP4_E2M1_MAX_FINITE  6.0f   // 2^2 * (1 + 1/2)
#define MXFP6_E2M3_MAX_FINITE  7.5f   // 2^2 * (1 + 7/8)
#define MXFP8_E4M3_MAX_FINITE  448.0f // 2^8 * (1 + 6/8)

// EMAX_OFFSET constants are in ggml-common.h (shared with CUDA).

#ifdef __cplusplus
extern "C" {
#endif

// FP6 E2M3: [S(1) | E(2) | M(3)], bias=1, no Inf/NaN
// Note: FP4 E2M1 uses the kvalues_mxfp4 LUT instead of element converters.

static inline float ggml_mxfp_fp6_e2m3_to_float(uint8_t v) {
    const float sign = (v & 0x20) ? -1.0f : 1.0f;
    const int   exp  = (v >> 3) & 0x3;
    const int   man  = v & 0x7;
    if (exp == 0) {
        return sign * (float)man * 0.125f; // subnormal: M * 2^(1-bias-3)
    }
    return sign * (1.0f + man * 0.125f) * (float)(1 << (exp - 1)); // normal: (1 + M/8) * 2^(E-1)
}

static inline uint8_t ggml_mxfp_float_to_fp6_e2m3(float x) {
    uint8_t sign = 0;
    if (x < 0) { sign = 0x20; x = -x; }
    if (x == 0) return sign;
    if (x >= MXFP6_E2M3_MAX_FINITE) return sign | 0x1F; // saturate to 7.5

    uint32_t bits     = fp32_to_bits(x);
    int      fp32_exp = (int)((bits >> 23) & 0xFF) - 127;

    if (fp32_exp < 0) {
        // subnormal in E2M3: value = M * 2^(-3), so M = round(x * 8)
        int e2m3_man = (int)(x * 8.0f + 0.5f);
        if (e2m3_man > 7) {
            return sign | 0x08; // overflow to smallest normal (E=1, M=0)
        }
        return sign | (uint8_t)e2m3_man;
    }
    if (fp32_exp > 2) {
        fp32_exp = 2; // clamp to max E2M3 exponent
    }

    // normal: extract 3-bit mantissa from fractional part
    float frac     = (x / (float)(1 << fp32_exp)) - 1.0f; // fractional part of significand
    int   e2m3_man = (int)(frac * 8.0f + 0.5f);           // round to 3-bit mantissa
    if (e2m3_man > 7) {
        e2m3_man = 0;
        fp32_exp++;
    }
    if (fp32_exp > 2) {
        return sign | 0x1F; // overflow to max
    }
    // encode: biased_exp = fp32_exp + bias = fp32_exp + 1
    return sign | (uint8_t)(((fp32_exp + 1) << 3) | e2m3_man);
}

// FP8 E4M3: [S(1) | E(4) | M(3)], bias=7, NaN=0x7F/0xFF, max finite=448

static inline float ggml_mxfp_fp8_e4m3_to_float(uint8_t v) {
    const uint32_t sign = ((uint32_t)(v & 0x80)) << 24; // F32 sign bit position
    const uint32_t exp  = (v >> 3) & 0xF;
    const uint32_t man  = v & 0x7;

    if (exp == 0) {
        if (man == 0) {
            return fp32_from_bits(sign); // +/-0
        }
        // subnormal: M * 2^(1-bias) * 2^(-3) = M * 2^(-9)
        float val     = (float)man * (1.0f / 512.0f);
        uint32_t bits = fp32_to_bits(val);
        bits = (bits & 0x7FFFFFFFu) | sign;
        return fp32_from_bits(bits);
    }
    if (exp == 15 && man == 7) {
        return fp32_from_bits(sign | 0x7FC00000u); // NaN
    }
    // normal: (1 + M/8) * 2^(E-7) -> F32 biased exp = E - 7 + 127 = E + 120
    return fp32_from_bits(sign | ((exp + 120) << 23) | (man << 20));
}

static inline uint8_t ggml_mxfp_float_to_fp8_e4m3(float x) {
    uint32_t bits = fp32_to_bits(x);
    uint8_t  sign = (bits >> 24) & 0x80;
    bits &= 0x7FFFFFFFu; // abs
    if (bits == 0) {
        return sign;
    }

    uint32_t fp32_exp = (bits >> 23) & 0xFF;
    uint32_t fp32_man = bits & 0x7FFFFF;
    int      e4m3_exp = (int)fp32_exp - 120; // F32 bias(127) - E4M3 bias(7) = 120

    if (e4m3_exp <= 0) {
        // subnormal in E4M3: denormalize the F32 significand
        int      shift     = 1 - e4m3_exp;
        uint32_t full_man  = (1u << 23) | fp32_man; // implicit leading 1
        int      total_shift = 20 + shift;          // 23 - 3 + shift
        if (total_shift >= 32) {
            return sign;  // too small to represent
        }
        uint32_t e4m3_man = full_man >> total_shift;
        // round-to-nearest-even
        if (total_shift > 0 && total_shift < 32) {
            uint32_t round_bit = (full_man >> (total_shift - 1)) & 1;
            uint32_t sticky    = (total_shift > 1) ? (full_man & ((1u << (total_shift - 1)) - 1)) : 0;
            if (round_bit && (sticky || (e4m3_man & 1))) {
                e4m3_man++;
            }
        }
        if (e4m3_man > 7) {
            return sign | 0x08; // overflow to smallest normal
        }
        return sign | (uint8_t)e4m3_man;
    }

    // normal: truncate F32 23-bit mantissa to 3-bit with round-to-nearest-even
    uint32_t round_bit = (fp32_man >> 19) & 1;
    uint32_t sticky    = fp32_man & ((1u << 19) - 1);
    uint32_t e4m3_man  = fp32_man >> 20;
    if (round_bit && (sticky || (e4m3_man & 1))) {
        e4m3_man++;
        if (e4m3_man > 7) {
            e4m3_man = 0;
            e4m3_exp++;
        }
    }
    if (e4m3_exp > 15 || (e4m3_exp == 15 && e4m3_man >= 7)) {
        return sign | 0x7E; // saturate to max finite (avoid NaN at 0x7F)
    }
    return sign | (uint8_t)((e4m3_exp << 3) | e4m3_man);
}

static inline void ggml_mxfp_pack_fp6x4(const uint8_t v[4], uint8_t out[3]) {
    uint32_t packed = (v[0] & 0x3F) | ((v[1] & 0x3F) << 6) |
                      ((v[2] & 0x3F) << 12) | ((v[3] & 0x3F) << 18);
    out[0] = (uint8_t)(packed);
    out[1] = (uint8_t)(packed >> 8);
    out[2] = (uint8_t)(packed >> 16);
}

static inline void ggml_mxfp_unpack_fp6x4(const uint8_t in[3], uint8_t v[4]) {
    uint32_t packed = (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16);
    v[0] = packed & 0x3F;
    v[1] = (packed >> 6) & 0x3F;
    v[2] = (packed >> 12) & 0x3F;
    v[3] = (packed >> 18) & 0x3F;
}

// Best-fit E2M1 quantization using doubled integer LUT (kvalues_mxfp4) with half-scale.
static inline int ggml_mxfp_best_index_e2m1(float x, float d) {
    int best_index = 0;
    float best_err = fabsf(kvalues_mxfp4[0]*d - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i]*d - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return best_index;
}

static inline float ggml_mxfp_block_amax(const float * x, int n) {
    float amax = 0.0f;
    for (int j = 0; j < n; j++) {
        const float a = fabsf(x[j]);
        if (a > amax) amax = a;
    }
    return amax;
}

// round_log2: false = floor(log2) per MX spec, true = round(log2) for KV cache.
static inline uint8_t ggml_mxfp_compute_e8m0(const float * x, int qk, int emax_offset, bool round_log2) {
    const float amax = ggml_mxfp_block_amax(x, qk);
    if (amax == 0.0f) return 0;

    const uint32_t amax_bits = fp32_to_bits(amax);
    int biased_log2 = (int)((amax_bits >> 23) & 0xFF) - 127;

    if (round_log2) {
        // IEEE 754 mantissa of sqrt(2) ≈ 1.4142: fractional bits = 0x3504F3
        biased_log2 += ((amax_bits & 0x7FFFFF) >= 0x3504F3 ? 1 : 0);
    }

    const int e = biased_log2 - emax_offset + 127;
    return (uint8_t)(e < 0 ? 0 : (e > 254 ? 254 : e));
}

static inline int ggml_mxfp_emax_offset(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP4: return MXFP4_E2M1_EMAX_OFFSET;
        case GGML_TYPE_MXFP6: return MXFP6_E2M3_EMAX_OFFSET;
        case GGML_TYPE_MXFP8: return MXFP8_E4M3_EMAX_OFFSET;
        default: GGML_ABORT("unsupported MXFP type");
    }
}

// all block_mxfp* structs are [uint8_t e][uint8_t qs[N]], so qs bytes = block size - 1
static inline int ggml_mxfp_qs_per_block(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP4: return sizeof(block_mxfp4) - 1;
        case GGML_TYPE_MXFP6: return sizeof(block_mxfp6) - 1;
        case GGML_TYPE_MXFP8: return sizeof(block_mxfp8) - 1;
        default: GGML_ABORT("unsupported MXFP type");
    }
}

static inline void ggml_mxfp_quantize_block(enum ggml_type type, const float * GGML_RESTRICT src,
                                             uint8_t * GGML_RESTRICT qs, uint8_t * e_out, bool round_log2) {
    const uint8_t e = ggml_mxfp_compute_e8m0(src, 32, ggml_mxfp_emax_offset(type), round_log2);
    *e_out = e;

    switch (type) {
        case GGML_TYPE_MXFP4: {
            const float d = GGML_E8M0_TO_FP32_HALF(e);
            for (int j = 0; j < 16; ++j) {
                const uint8_t x0 = ggml_mxfp_best_index_e2m1(src[j],      d);
                const uint8_t x1 = ggml_mxfp_best_index_e2m1(src[j + 16], d);
                qs[j] = x0 | (x1 << 4);
            }
            break;
        }
        case GGML_TYPE_MXFP6: {
            const float d = ggml_e8m0_to_fp32(e);
            const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
            for (int j = 0, qi = 0; j < 32; j += 4, qi += 3) {
                uint8_t vals[4];
                for (int jj = 0; jj < 4; jj++) {
                    vals[jj] = ggml_mxfp_float_to_fp6_e2m3(src[j + jj] * inv_d);
                }
                ggml_mxfp_pack_fp6x4(vals, &qs[qi]);
            }
            break;
        }
        case GGML_TYPE_MXFP8: {
            const float d = ggml_e8m0_to_fp32(e);
            const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
            for (int j = 0; j < 32; ++j) {
                qs[j] = ggml_mxfp_float_to_fp8_e4m3(src[j] * inv_d);
            }
            break;
        }
        default:
            GGML_ABORT("unsupported MXFP type");
    }
}

static inline void ggml_mxfp_dequantize_block(enum ggml_type type, const uint8_t * GGML_RESTRICT qs,
                                               uint8_t e, float * GGML_RESTRICT dst) {
    switch (type) {
        case GGML_TYPE_MXFP4: {
            const float d = GGML_E8M0_TO_FP32_HALF(e);
            for (int j = 0; j < 16; ++j) {
                dst[j]      = kvalues_mxfp4[qs[j] & 0x0F] * d;
                dst[j + 16] = kvalues_mxfp4[qs[j] >> 4]   * d;
            }
            break;
        }
        case GGML_TYPE_MXFP6: {
            const float d = ggml_e8m0_to_fp32(e);
            for (int j = 0, qi = 0; j < 32; j += 4, qi += 3) {
                uint8_t vals[4];
                ggml_mxfp_unpack_fp6x4(&qs[qi], vals);
                for (int jj = 0; jj < 4; jj++) {
                    dst[j + jj] = ggml_mxfp_fp6_e2m3_to_float(vals[jj]) * d;
                }
            }
            break;
        }
        case GGML_TYPE_MXFP8: {
            const float d = ggml_e8m0_to_fp32(e);
            for (int j = 0; j < 32; ++j) {
                dst[j] = ggml_mxfp_fp8_e4m3_to_float(qs[j]) * d;
            }
            break;
        }
        default:
            GGML_ABORT("unsupported MXFP type");
    }
}

static_assert(offsetof(block_mxfp4, e) == 0 && offsetof(block_mxfp4, qs) == 1, "block_mxfp4 layout mismatch");
static_assert(offsetof(block_mxfp6, e) == 0 && offsetof(block_mxfp6, qs) == 1, "block_mxfp6 layout mismatch");
static_assert(offsetof(block_mxfp8, e) == 0 && offsetof(block_mxfp8, qs) == 1, "block_mxfp8 layout mismatch");

// AoS: [e0 qs0...] [e1 qs1...] ... (standard block layout)
static inline void ggml_mxfp_quantize_row(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k,
                                           enum ggml_type type) {
    assert(k % 32 == 0);
    const int nb  = k / 32;
    const int bsz = ggml_mxfp_qs_per_block(type) + 1; // [e][qs...]
    uint8_t * dst = (uint8_t *)y;

    for (int i = 0; i < nb; i++) {
        uint8_t * block = dst + i * bsz;
        ggml_mxfp_quantize_block(type, &x[i*32], block + 1, block, false); // floor(log2) per OCP MX spec
    }
}

static inline void ggml_mxfp_dequantize_row(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k,
                                             enum ggml_type type) {
    assert(k % 32 == 0);
    const int nb  = k / 32;
    const int bsz = ggml_mxfp_qs_per_block(type) + 1;
    const uint8_t * src = (const uint8_t *)x;

    for (int i = 0; i < nb; i++) {
        const uint8_t * block = src + i * bsz;
        ggml_mxfp_dequantize_block(type, block + 1, block[0], &y[i*32]);
    }
}

// SoA: [qs0 qs1 ... qsN | e0 e1 ... eN] (flash attention KV cache layout)
static inline void ggml_mxfp_quantize_soa_row(enum ggml_type type, const float * GGML_RESTRICT src,
                                               void * GGML_RESTRICT dst, int64_t k) {
    assert(k % 32 == 0);
    const int nb  = k / 32;
    const int qpb = ggml_mxfp_qs_per_block(type);
    uint8_t * qs   = (uint8_t *)dst;
    uint8_t * e8m0 = qs + nb * qpb;

    for (int i = 0; i < nb; i++) {
        ggml_mxfp_quantize_block(type, &src[i*32], &qs[i * qpb], &e8m0[i], true); // round(log2) reduces softmax bias
    }
}

static inline void ggml_mxfp_dequantize_soa_row(enum ggml_type type, const void * GGML_RESTRICT src,
                                                  float * GGML_RESTRICT dst, int64_t k) {
    assert(k % 32 == 0);
    const int nb  = k / 32;
    const int qpb = ggml_mxfp_qs_per_block(type);
    const uint8_t * qs   = (const uint8_t *)src;
    const uint8_t * e8m0 = qs + nb * qpb;

    for (int i = 0; i < nb; i++) {
        ggml_mxfp_dequantize_block(type, &qs[i * qpb], e8m0[i], &dst[i*32]);
    }
}

#ifdef __cplusplus
}
#endif
