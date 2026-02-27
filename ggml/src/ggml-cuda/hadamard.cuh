// Block-32 Walsh-Hadamard Transform for MXFP4 KV cache quantization.
//
// Applying a Hadamard rotation before MXFP4 quantization spreads outlier energy
// uniformly across all elements in a 32-element block, making the E8M0 shared
// exponent a tighter fit and reducing quantization error.
//
// References:
//   - "Block Rotation is All You Need for MXFP4 Quantization" (BRQ, arxiv 2511.04214)
//     Proves block-wise rotation with block_size=32 outperforms global rotation for MXFP4.
//   - "Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization"
//     (MR-GPTQ, arxiv 2509.23202)
//     Proves analytically that rotations improve MXFP4 (E8M0 scales) but hurt NVFP4 (E4M3).
//   - "Tensor Core Accelerated Hadamard Transform Kernel" (HadaCore, arxiv 2412.08832)
//     Warp-shuffle butterfly pattern for GPU-efficient Hadamard transforms.

#pragma once

// 1/sqrt(32) normalization constant for the 32-element Hadamard transform.
// H_norm = H / sqrt(32) satisfies H_norm * H_norm^T = I (orthonormal).
static constexpr float HADAMARD_32_NORM = 0.17677669529663689f;

// Register-level 32-element Walsh-Hadamard Transform (in-place).
// Used when a single thread holds all 32 values (K quantization, MMA Q quantization).
//
// Algorithm: 5 butterfly stages (strides 1, 2, 4, 8, 16) followed by 1/sqrt(32) normalization.
// Cost: 80 add + 80 sub + 32 mul = 192 FP32 ops. Negligible vs memory traffic.
__device__ __forceinline__ void hadamard_32_inplace(float vals[32]) {
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

// Hybrid register+shuffle 32-element Walsh-Hadamard Transform for Q8_1 quantization.
// Used in the VEC flash attention kernel where 8 threads each hold 4 elements of a
// 32-element block (QI8_1 = 8 threads per Q8_1 block, 4 values per thread).
//
// Thread mapping: thread with lane index `lane` (0..7 within the Q8_1 group) holds
// elements [4*lane, 4*lane+1, 4*lane+2, 4*lane+3] of the 32-element block.
//
// Stages 1-2: intra-thread butterfly (element strides 1, 2) — pure register ops.
// Stages 3-5: cross-thread butterfly (element strides 4, 8, 16) via __shfl_xor_sync
//             with XOR masks 1, 2, 4 in thread-space.
//
// The XOR masks stay within each 8-thread group because they are all < 8, so groups
// {0-7}, {8-15}, {16-23}, {24-31} transform independently in parallel.
//
// Parameters:
//   vals[4] — the 4 float values held by this thread (modified in-place)
//   lane    — this thread's index within the 8-thread Q8_1 group (threadIdx.x % QI8_1)
__device__ __forceinline__ void hadamard_32_q8_1(float vals[4], const int lane) {
    // Stage 1: stride 1 in element-space (within each thread's 4 values).
    {
        const float a0 = vals[0], b0 = vals[1];
        const float a1 = vals[2], b1 = vals[3];
        vals[0] = a0 + b0;
        vals[1] = a0 - b0;
        vals[2] = a1 + b1;
        vals[3] = a1 - b1;
    }

    // Stage 2: stride 2 in element-space (within each thread's 4 values).
    {
        const float a0 = vals[0], b0 = vals[2];
        const float a1 = vals[1], b1 = vals[3];
        vals[0] = a0 + b0;
        vals[2] = a0 - b0;
        vals[1] = a1 + b1;
        vals[3] = a1 - b1;
    }

    // Stages 3-5: cross-thread butterfly via warp shuffles.
    // Element stride 4  → XOR mask 1 (swap with neighboring thread in 8-group)
    // Element stride 8  → XOR mask 2
    // Element stride 16 → XOR mask 4
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

    // Normalize for orthonormality: H_norm * H_norm^T = I.
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        vals[l] *= HADAMARD_32_NORM;
    }
}
