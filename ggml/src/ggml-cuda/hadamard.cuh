#pragma once

// Block-32 Walsh-Hadamard Transform for MXFP KV cache quantization.
// Spreads outlier energy across all block elements, improving E8M0 scale fit.
//
// References:
//   Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arXiv:2404.00456
//   Zhang et al., "Block Rotation is All You Need for MXFP4 Quantization", arXiv:2511.04214
//   Dao et al., "FlashAttention-3", arXiv:2407.08608 (incoherent processing for FP8 attention)

static constexpr float HADAMARD_32_NORM = 0.17677669529663689f; // 1/sqrt(32)

// Single-thread in-place Hadamard transform over 32 values.
static __device__ __forceinline__ void hadamard_32_inplace(float vals[32]) {
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
        for (int i = 0; i < 32; i += 2 * stride) {
#pragma unroll
            for (int j = 0; j < stride; ++j) {
                const float a = vals[i + j];
                const float b = vals[i + j + stride];
                vals[i + j]          = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        vals[i] *= HADAMARD_32_NORM;
    }
}

// Distributed Hadamard transform: 8 threads each hold 4 of 32 elements.
// Stages 1-2 are intra-thread butterflies, stages 3-5 use __shfl_xor_sync.
static __device__ __forceinline__ void hadamard_32_q8_1(float vals[4], const int lane) {
    {
        const float a0 = vals[0], b0 = vals[1];
        const float a1 = vals[2], b1 = vals[3];
        vals[0] = a0 + b0;
        vals[1] = a0 - b0;
        vals[2] = a1 + b1;
        vals[3] = a1 - b1;
    }

    {
        const float a0 = vals[0], b0 = vals[2];
        const float a1 = vals[1], b1 = vals[3];
        vals[0] = a0 + b0;
        vals[2] = a0 - b0;
        vals[1] = a1 + b1;
        vals[3] = a1 - b1;
    }

    // Cross-thread butterfly stages via warp shuffles.
#pragma unroll
    for (int xor_mask = 1; xor_mask <= 4; xor_mask *= 2) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float partner = __shfl_xor_sync(0xFFFFFFFF, vals[l], xor_mask);
            if (lane & xor_mask) {
                vals[l] = partner - vals[l];
            } else {
                vals[l] = vals[l] + partner;
            }
        }
    }

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        vals[l] *= HADAMARD_32_NORM;
    }
}
