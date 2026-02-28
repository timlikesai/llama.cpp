#pragma once

#include "ggml-common.h"
#include "convert.cuh"
#include "hadamard.cuh"

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

// SoA MXFP4 quantization: per-row layout [all_qs][all_e].
// When apply_hadamard=true (K cache): rotates, quantizes primary, then writes compact
// 1-bit sign residual (sign bits + E8M0) in extra blocks after the primary region.
template<bool apply_hadamard>
static __device__ void quantize_f32_mxfp4_block_soa(
        const float * __restrict__ x,
        char * __restrict__ row_base,
        const int block_idx,
        const int blocks_per_row_total,
        const int blocks_per_primary) {
    float vals[QK_MXFP4];
    const float * src = x;
    if constexpr (apply_hadamard) {
        for (int j = 0; j < QK_MXFP4; ++j) {
            vals[j] = x[j];
        }
        hadamard_32_inplace(vals);
        src = vals;
    }

    // MSE-optimal E8M0 search: test +-1 around amax estimate, pick lowest MSE.
    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4; ++j) {
        amax = fmaxf(amax, fabsf(src[j]));
    }

    const int e_base = (amax == 0.0f) ? 0 : __float2int_rn(log2f(amax)) - 2 + 127;
    const int e_lo = max(1, min(255, e_base - 1));
    const int e_hi = max(1, min(255, e_base + 1));

    int best_e = max(0, min(255, e_base));
    float best_mse = 1e30f;

    for (int test_e = e_lo; test_e <= e_hi; ++test_e) {
        const float test_scale = ggml_cuda_e8m0_to_fp32((uint8_t)test_e);
        const float test_inv = 1.0f / test_scale;
        float mse = 0.0f;
        for (int j = 0; j < QK_MXFP4; ++j) {
            const uint8_t nibble = ggml_cuda_float_to_fp4_e2m1(src[j], test_inv);
            const float recon = kvalues_mxfp4[nibble] * 0.5f * test_scale;
            const float err = src[j] - recon;
            mse += err * err;
        }
        if (mse < best_mse) {
            best_mse = mse;
            best_e = test_e;
        }
    }

    const uint8_t e_val = (amax == 0.0f) ? (uint8_t)0 : (uint8_t)best_e;
    const float inv_d = (amax == 0.0f) ? 0.0f : 1.0f / ggml_cuda_e8m0_to_fp32(e_val);

    uint8_t qs_bytes[QK_MXFP4/2];
    uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * 16);
    for (int j = 0; j < QK_MXFP4/2; ++j) {
        const uint8_t byte = __nv_cvt_float2_to_fp4x2(
            make_float2(src[j] * inv_d, src[QK_MXFP4/2 + j] * inv_d),
            __NV_E2M1, cudaRoundNearest);
        qs_dst[j] = byte;
        if constexpr (apply_hadamard) {
            qs_bytes[j] = byte;
        }
    }

    *(row_base + blocks_per_row_total * 16 + block_idx) = e_val;

    // Compact 1-bit sign residual: sign bits + per-block mean_abs E8M0.
    if constexpr (apply_hadamard) {
        const float scale = ggml_cuda_e8m0_to_fp32(e_val);

        float res[QK_MXFP4];
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t byte = qs_bytes[j];
            const float recon0 = kvalues_mxfp4[byte & 0xF] * 0.5f * scale;
            const float recon1 = kvalues_mxfp4[byte >> 4]  * 0.5f * scale;
            res[j]              = vals[j]              - recon0;
            res[j + QK_MXFP4/2] = vals[j + QK_MXFP4/2] - recon1;
        }

        uint32_t sign_bits = 0;
        for (int j = 0; j < QK_MXFP4; ++j) {
            if (res[j] < 0.0f) {
                sign_bits |= (1u << j);
            }
        }

        // E8M0 for residual: no FP4_E2M1_EMAX offset since sign values are +-1.0.
        float sum_abs = 0.0f;
        for (int j = 0; j < QK_MXFP4; ++j) {
            sum_abs += fabsf(res[j]);
        }
        const float mean_abs = sum_abs * (1.0f / QK_MXFP4);
        const int res_e = (mean_abs == 0.0f) ? 0 : __float2int_rn(log2f(mean_abs)) + 127;
        const uint8_t res_e_val = (uint8_t) max(0, min(255, res_e));

        const int compact_qs_off = blocks_per_primary * 16;
        *reinterpret_cast<uint32_t *>(row_base + compact_qs_off + block_idx * 4) = sign_bits;
        *(row_base + compact_qs_off + blocks_per_primary * 4 + block_idx) = res_e_val;
    }
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
