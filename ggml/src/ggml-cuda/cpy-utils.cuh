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

static __device__ void quantize_f32_mxfp4_block(const float * __restrict__ x, block_mxfp4 * __restrict__ y) {
    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }

    // Compute E8M0 exponent: biased so that max value maps into the FP4 range
    // Use round-to-nearest (__float2int_rn) to match reference compute_e8m0_scale in quantize.cu
    const int e = (amax == 0.0f) ? 0 : __float2int_rn(log2f(amax)) - 2 + 127;
    y->e = (uint8_t) max(0, min(255, e));

    // inv_d = 1/e8m0_scale — ggml_cuda_float_to_fp4_e2m1 uses actual E2M1 values (not doubled kvalues)
    const float inv_d = (amax == 0.0f) ? 0.0f : 1.0f / ggml_cuda_e8m0_to_fp32(y->e);

    for (int j = 0; j < QK_MXFP4/2; ++j) {
        const uint8_t xi0 = ggml_cuda_float_to_fp4_e2m1(x[0          + j], inv_d);
        const uint8_t xi1 = ggml_cuda_float_to_fp4_e2m1(x[QK_MXFP4/2 + j], inv_d);
        y->qs[j] = xi0 | (xi1 << 4);
    }
}

// SoA version: writes qs to row_base + block_idx*16, e to row_base + blocks_per_row_total*16 + block_idx.
// Same quantization math as quantize_f32_mxfp4_block, different write layout.
// Per-row SoA: all qs bytes contiguous at offset 0, all e bytes contiguous after qs.
//
// When apply_hadamard=true (K cache only):
//   1. Applies Walsh-Hadamard rotation before quantization to equalize block value magnitudes
//   2. Writes primary MXFP4 quantization at block_idx
//   3. Dequantizes primary, computes residual (error), quantizes residual to second MXFP4 block
//      at block_idx + blocks_per_row_total/2. The residual is NOT Hadamard-rotated — it's
//      already in the Hadamard domain (error of H(K)). The FA kernel accumulates both:
//      H(Q) · (K_primary + K_residual)^T = H(Q) · H(K)^T ≈ Q · K^T
//   Ref: BRQ (arxiv 2511.04214), MR-GPTQ (arxiv 2509.23202).
//
// blocks_per_row_total: total SoA blocks in the row (doubled for K cache with residual).
//   For V cache: equals blocks_per_primary (no doubling).
//   For K cache: equals 2 * blocks_per_primary (primary + residual regions).
template<bool apply_hadamard>
static __device__ void quantize_f32_mxfp4_block_soa(
        const float * __restrict__ x,
        char * __restrict__ row_base,
        const int block_idx,
        const int blocks_per_row_total) {
    // Conditionally buffer and rotate, or use input directly.
    float vals[QK_MXFP4];
    const float * src = x;
    if constexpr (apply_hadamard) {
        for (int j = 0; j < QK_MXFP4; ++j) {
            vals[j] = x[j];
        }
        hadamard_32_inplace(vals);
        src = vals;
    }

    // --- Primary quantization (same for K and V) ---

    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4; ++j) {
        amax = fmaxf(amax, fabsf(src[j]));
    }

    const int e = (amax == 0.0f) ? 0 : __float2int_rn(log2f(amax)) - 2 + 127;
    const uint8_t e_val = (uint8_t) max(0, min(255, e));
    const float inv_d = (amax == 0.0f) ? 0.0f : 1.0f / ggml_cuda_e8m0_to_fp32(e_val);

    // Write primary qs to SoA qs region (contiguous, starts at offset 0).
    uint8_t qs_bytes[QK_MXFP4/2];
    uint8_t * qs_dst = (uint8_t *)(row_base + block_idx * 16);
    for (int j = 0; j < QK_MXFP4/2; ++j) {
        const uint8_t xi0 = ggml_cuda_float_to_fp4_e2m1(src[0          + j], inv_d);
        const uint8_t xi1 = ggml_cuda_float_to_fp4_e2m1(src[QK_MXFP4/2 + j], inv_d);
        const uint8_t byte = xi0 | (xi1 << 4);
        qs_dst[j] = byte;
        if constexpr (apply_hadamard) {
            qs_bytes[j] = byte;  // Save for residual dequant
        }
    }

    // Write primary e to SoA e region (after all qs bytes in the row).
    *(row_base + blocks_per_row_total * 16 + block_idx) = e_val;

    // --- Residual quantization (K cache only) ---
    // Dequant primary, compute error, re-quantize to second MXFP4 block.
    // The residual is NOT Hadamard-rotated (it's already in Hadamard domain).
    if constexpr (apply_hadamard) {
        const int blocks_per_primary = blocks_per_row_total / 2;
        const float scale = ggml_cuda_e8m0_to_fp32(e_val);

        // Compute residual: src[j] - dequant(primary[j])
        // kvalues_mxfp4 values are doubled, so dequant = kvalues_mxfp4[nibble] * 0.5f * scale
        float res[QK_MXFP4];
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t byte = qs_bytes[j];
            const float recon0 = kvalues_mxfp4[byte & 0xF] * 0.5f * scale;
            const float recon1 = kvalues_mxfp4[byte >> 4]  * 0.5f * scale;
            res[j]              = vals[j]              - recon0;
            res[j + QK_MXFP4/2] = vals[j + QK_MXFP4/2] - recon1;
        }

        // Quantize residual with its own E8M0 scale
        float res_amax = 0.0f;
        for (int j = 0; j < QK_MXFP4; ++j) {
            res_amax = fmaxf(res_amax, fabsf(res[j]));
        }

        const int res_e = (res_amax == 0.0f) ? 0 : __float2int_rn(log2f(res_amax)) - 2 + 127;
        const uint8_t res_e_val = (uint8_t) max(0, min(255, res_e));
        const float res_inv_d = (res_amax == 0.0f) ? 0.0f : 1.0f / ggml_cuda_e8m0_to_fp32(res_e_val);

        // Write residual qs at block_idx + blocks_per_primary
        uint8_t * res_qs_dst = (uint8_t *)(row_base + (blocks_per_primary + block_idx) * 16);
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const uint8_t xi0 = ggml_cuda_float_to_fp4_e2m1(res[0          + j], res_inv_d);
            const uint8_t xi1 = ggml_cuda_float_to_fp4_e2m1(res[QK_MXFP4/2 + j], res_inv_d);
            res_qs_dst[j] = xi0 | (xi1 << 4);
        }

        // Write residual e at blocks_per_primary + block_idx in the e region
        *(row_base + blocks_per_row_total * 16 + blocks_per_primary + block_idx) = res_e_val;
    }
}

static __device__ void cpy_blck_f32_mxfp4(const char * cxi, char * cdsti) {
    quantize_f32_mxfp4_block((const float *)cxi, (block_mxfp4 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
