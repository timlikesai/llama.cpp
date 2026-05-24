#pragma once

// OCP Microscaling Formats (MX) Specification v1.0
// MXFP4 E2M1: block-level quantize/dequantize for KV cache.
// Supports AoS (standard) and SoA (flash attention KV cache) layouts.

#include "ggml-impl.h"

#define MXFP4_EMAX_OFFSET   2  // floor(log2(6.0))

static inline float ggml_mxfp_block_amax(const float * x, int n) {
    float amax = 0.0f;
    for (int j = 0; j < n; j++) {
        const float a = fabsf(x[j]);
        if (a > amax) amax = a;
    }
    return amax;
}

// Best-fit E2M1 quantization using doubled integer LUT (kvalues_mxfp4) with half-scale.
static inline int best_index_mxfp4(float x, float d) {
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

// block_mxfp4 layout: [uint8_t e][uint8_t qs[16]]
static_assert(offsetof(block_mxfp4, e) == 0 && offsetof(block_mxfp4, qs) == 1, "block_mxfp4 layout mismatch");

static inline void ggml_mxfp_quantize_block(enum ggml_type type, const float * GGML_RESTRICT src,
                                             uint8_t * GGML_RESTRICT qs, uint8_t * e_out, bool round_log2) {
    const uint8_t e = ggml_mxfp_compute_e8m0(src, 32, MXFP4_EMAX_OFFSET, round_log2);
    *e_out = e;

    GGML_ASSERT(type == GGML_TYPE_MXFP4);
    const float d = GGML_E8M0_TO_FP32_HALF(e);
    for (int j = 0; j < 16; ++j) {
        const uint8_t x0 = best_index_mxfp4(src[j],      d);
        const uint8_t x1 = best_index_mxfp4(src[j + 16], d);
        qs[j] = x0 | (x1 << 4);
    }
}

static inline void ggml_mxfp_dequantize_block(enum ggml_type type, const uint8_t * GGML_RESTRICT qs,
                                               uint8_t e, float * GGML_RESTRICT dst) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4);
    const float d = GGML_E8M0_TO_FP32_HALF(e);
    for (int j = 0; j < 16; ++j) {
        dst[j]      = kvalues_mxfp4[qs[j] & 0x0F] * d;
        dst[j + 16] = kvalues_mxfp4[qs[j] >> 4]   * d;
    }
}

// AoS: [e0 qs0...] [e1 qs1...] ... (standard block layout)
static inline void ggml_mxfp_quantize_row(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k,
                                           enum ggml_type type) {
    assert(k % 32 == 0);
    const int nb  = k / 32;
    const int bsz = sizeof(block_mxfp4);
    uint8_t * dst = (uint8_t *)y;

    for (int i = 0; i < nb; i++) {
        uint8_t * block = dst + i * bsz;
        ggml_mxfp_quantize_block(type, &x[i*32], block + 1, block, false); // floor(log2) per OCP MX spec
    }
}

// SoA: [qs0 qs1 ... qsN | e0 e1 ... eN] (flash attention KV cache layout)
static inline void ggml_mxfp_quantize_soa_row(enum ggml_type type, const float * GGML_RESTRICT src,
                                               void * GGML_RESTRICT dst, int64_t k) {
    assert(k % 32 == 0);
    const int nb   = k / 32;
    const int qpb  = sizeof(block_mxfp4) - 1; // qs bytes per block (16)
    uint8_t * qs   = (uint8_t *)dst;
    uint8_t * e8m0 = qs + nb * qpb;

    for (int i = 0; i < nb; i++) {
        ggml_mxfp_quantize_block(type, &src[i*32], &qs[i * qpb], &e8m0[i], true); // round(log2) reduces softmax bias
    }
}

static inline void ggml_mxfp_dequantize_soa_row(enum ggml_type type, const void * GGML_RESTRICT src,
                                                  float * GGML_RESTRICT dst, int64_t k) {
    assert(k % 32 == 0);
    const int nb   = k / 32;
    const int qpb  = sizeof(block_mxfp4) - 1;
    const uint8_t * qs   = (const uint8_t *)src;
    const uint8_t * e8m0 = qs + nb * qpb;

    for (int i = 0; i < nb; i++) {
        ggml_mxfp_dequantize_block(type, &qs[i * qpb], e8m0[i], &dst[i*32]);
    }
}
