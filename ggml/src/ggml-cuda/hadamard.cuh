#pragma once

// Block-32 Walsh-Hadamard Transform for MXFP KV cache quantization.
// ggml_mxfp_hadamard_32_inplace (single-thread) lives in ggml-common.h as a
// GGML_MXFP_FUNC — shared between CPU and CUDA device code.
// Only the distributed warp-shuffle variant is GPU-specific and lives here.

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
            const float partner = __shfl_xor_sync(0xFFFFFFFF, vals[l], xor_mask, WARP_SIZE);
            if (lane & xor_mask) {
                vals[l] = partner - vals[l];
            } else {
                vals[l] = vals[l] + partner;
            }
        }
    }

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        vals[l] *= MXFP_HADAMARD_32_NORM;
    }
}
