#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// absolute max of a 32-element block, then the data-free e8m0 scale for the grid
static __device__ __forceinline__ ggml_mxfp_scale_result ggml_cuda_mxfp_blk_scale(const float * x, float qmax) {
    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    return ggml_mxfp_scale(amax, qmax);
}

static __device__ void quantize_f32_mxfp4_block(const float * __restrict__ x, block_mxfp4 * __restrict__ y) {
    const auto s = ggml_cuda_mxfp_blk_scale(x, GGML_MXFP_QMAX_E2M1);
    y->e = s.e;
#if CUDART_VERSION >= 12080
    for (int k = 0; k < QK_MXFP4/4; ++k) {
        __nv_fp4x4_e2m1 q(make_float4(
            x[2*k]*s.inv, x[2*k+QK_MXFP4/2]*s.inv,
            x[2*k+1]*s.inv, x[2*k+1+QK_MXFP4/2]*s.inv));
        uint16_t p = q.__x;
        y->qs[2*k]   = (uint8_t) p;
        y->qs[2*k+1] = (uint8_t)(p >> 8);
    }
#else
    for (int j = 0; j < QK_MXFP4/2; ++j) {
        y->qs[j] = ggml_cuda_float_to_fp4_e2m1(x[j]*s.inv, 1.0f) | (ggml_cuda_float_to_fp4_e2m1(x[QK_MXFP4/2+j]*s.inv, 1.0f) << 4);
    }
#endif // CUDART_VERSION >= 12080
}

static __device__ void quantize_f32_mxfp6_block(const float * __restrict__ x, block_mxfp6 * __restrict__ y) {
    const auto s = ggml_cuda_mxfp_blk_scale(x, GGML_MXFP_QMAX_E2M3);
    y->e = s.e;
#if CUDART_VERSION >= 12080
    for (int k = 0; k < QK_MXFP6/4; ++k) {
        const uint32_t p = __nv_fp6x4_e2m3(make_float4(
            x[4*k+0]*s.inv, x[4*k+1]*s.inv, x[4*k+2]*s.inv, x[4*k+3]*s.inv)).__x;
        // one code per byte in K order, pack into the 6-bit bitstream
        for (int j = 0; j < 4; ++j) {
            ggml_mxfp6_set_code(y->qs, 4*k+j, (uint8_t) (p >> (8*j)));
        }
    }
#else
    for (int j = 0; j < QK_MXFP6; ++j) {
        ggml_mxfp6_set_code(y->qs, j, ggml_fp32_to_e2m3(x[j]*s.inv));
    }
#endif // CUDART_VERSION >= 12080
}

static __device__ void quantize_f32_mxfp8_block(const float * __restrict__ x, block_mxfp8 * __restrict__ y) {
    const auto s = ggml_cuda_mxfp_blk_scale(x, GGML_MXFP_QMAX_E4M3);
    y->e = s.e;
#if defined(FP8_AVAILABLE)
    for (int k = 0; k < QK_MXFP8/4; ++k) {
        const uint32_t p = __nv_fp8x4_e4m3(make_float4(
            x[4*k+0]*s.inv, x[4*k+1]*s.inv, x[4*k+2]*s.inv, x[4*k+3]*s.inv)).__x;
        for (int j = 0; j < 4; ++j) {
            y->qs[4*k+j] = (int8_t) (p >> (8*j));
        }
    }
#else
    for (int j = 0; j < QK_MXFP8; ++j) {
        y->qs[j] = (int8_t) ggml_fp32_to_e4m3(x[j]*s.inv);
    }
#endif // defined(FP8_AVAILABLE)
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

static __device__ void cpy_blck_f32_mxfp4(const char * cxi, char * cdsti) {
    quantize_f32_mxfp4_block((const float *)cxi, (block_mxfp4 *)cdsti);
}

static __device__ void cpy_blck_f32_mxfp6(const char * cxi, char * cdsti) {
    quantize_f32_mxfp6_block((const float *)cxi, (block_mxfp6 *)cdsti);
}

static __device__ void cpy_blck_f32_mxfp8(const char * cxi, char * cdsti) {
    quantize_f32_mxfp8_block((const float *)cxi, (block_mxfp8 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
