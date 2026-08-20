#pragma once

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// used from both the host (ggml, backends) and CUDA device code
#if defined(__CUDACC__)
#  define GGML_MXFP_INLINE static __host__ __device__ inline
#else
#  define GGML_MXFP_INLINE static inline
#endif

// MXFP formats (OCP Microscaling): blocks of 32 values, one ue8m0 scale per block.
// This is the reference implementation of the mxfp math. The CUDA intrinsic paths
// must produce identical results (see the tie tests in test-backend-ops.cpp), and
// other backends should implement against these functions.

// UOS (data-free) Qmax per format (arXiv:2605.20402, generalized E(q) = 8*E(q/2)):
#define GGML_MXFP_QMAX_E2M1 7.25f
#define GGML_MXFP_QMAX_E2M3 7.875f
#define GGML_MXFP_QMAX_E4M3 464.0f
// e8m0 helpers

// UOS scale: scale = 2^ceil(log2(amax / qmax)), so the normalized max falls in (qmax/2, qmax]
GGML_MXFP_INLINE uint8_t ggml_mxfp_uos_exponent(float amax, float qmax) {
    if (amax <= 0.0f) {
        return 0;
    }
    // frexpf is exact (no libm log2 rounding): amax/qmax = r * 2^e2, r in [0.5, 1)
    int e2;
    const float r = frexpf(amax / qmax, &e2);
    const int ei = (r == 0.5f) ? e2 - 1 : e2;
    const int e_byte = 127 + ei;
    return (uint8_t) (e_byte < 0 ? 0 : (e_byte > 254 ? 254 : e_byte));
}

// 1/scale of an e8m0 exponent, exact (power of 2)
GGML_MXFP_INLINE float ggml_mxfp_e8m0_inv_scale(uint8_t e) {
    return ldexpf(1.0f, 127 - (int) e);
}

// code -> float

// e2m3: e-field 00 -> m/8 (subnormal), e-field >= 1 -> (1+m/8)*2^(e-1), max 7.5
GGML_MXFP_INLINE float ggml_mxfp_e2m3_to_f32(uint8_t c) {
    const float sign = (c & 0x20) ? -1.0f : 1.0f;
    const int e = (c >> 3) & 0x3;
    const int m = c & 0x7;
    if (e == 0) {
        return sign * m / 8.0f;
    }
    return sign * (1.0f + m / 8.0f) * ldexpf(1.0f, e - 1);
}

// e4m3: e-field 0000 -> m/512 (subnormal), e-field >= 1 -> (1+m/8)*2^(e-7), max 448, 0x7F/0xFF = NaN -> 0
GGML_MXFP_INLINE float ggml_mxfp_e4m3_to_f32(uint8_t c) {
    if (c == 0x7F || c == 0xFF) {
        return 0.0f;
    }
    const float sign = (c & 0x80) ? -1.0f : 1.0f;
    const int e = (c >> 3) & 0xF;
    const int m = c & 0x7;
    if (e == 0) {
        return sign * m / 512.0f;
    }
    return sign * (1.0f + m / 8.0f) * ldexpf(1.0f, e - 7);
}

// float -> code, round to nearest even (RNE), matches the __nv_* satfinite intrinsics
// x is compared against the format grid scaled by d; NaN -> 0
GGML_MXFP_INLINE uint8_t ggml_mxfp_e2m3_rne(float x, float d) {
    const float ax = fabsf(x);
    int best_index = 0;
    float best_err = ax;
    for (int i = 1; i < 32; i++) {
        const float err = fabsf(ggml_mxfp_e2m3_to_f32((uint8_t) i) * d - ax);
        if (err < best_err || (err == best_err && (i & 1) == 0 && (best_index & 1) == 1)) {
            best_index = i;
            best_err = err;
        }
    }
    if (x < 0.0f) {
        best_index |= 0x20;
    }
    return (uint8_t) best_index;
}

GGML_MXFP_INLINE uint8_t ggml_mxfp_e4m3_rne(float x, float d) {
    const float ax = fabsf(x);
    int best_index = 0;
    float best_err = ax;
    for (int i = 1; i < 127; i++) {
        const float err = fabsf(ggml_mxfp_e4m3_to_f32((uint8_t) i) * d - ax);
        if (err < best_err || (err == best_err && (i & 1) == 0 && (best_index & 1) == 1)) {
            best_index = i;
            best_err = err;
        }
    }
    if (x < 0.0f) {
        best_index |= 0x80;
    }
    return (uint8_t) best_index;
}

// mxfp6 bitstream access: 32 x 6-bit codes in 24 little-endian bytes, value i is bits [6i, 6i+6)
GGML_MXFP_INLINE uint8_t ggml_mxfp6_code_get(const uint8_t * qs, int i) {
    const int pos = 6 * i;
    const int j = pos >> 3;
    const int off = pos & 7;
    return (uint8_t) ((qs[j] >> off) | (off ? (qs[j+1] << (8 - off)) & 0x3F : 0));
}

// qs must be zero-initialized
GGML_MXFP_INLINE void ggml_mxfp6_code_set(uint8_t * qs, int i, uint8_t c) {
    const int pos = 6 * i;
    const int j = pos >> 3;
    const int off = pos & 7;
    qs[j] |= (uint8_t) (c << off);
    if (off) {
        qs[j+1] |= (uint8_t) (c >> (8 - off));
    }
}

#ifdef __cplusplus
}
#endif
