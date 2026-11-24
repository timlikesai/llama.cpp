        }
    }
}

// mxfp weights (e2m1/e2m3/e4m3) x e4m3 (mxfp8) activations block-scaled MMA path for Blackwell (m16n8k32, scale_vec::1X ue8m0).
// x rows: iter_k weight codes in 8-bit containers (packed in ggml_cuda_mmq_load_tiles_mxfp), then one ue8m0 scale byte per 32 values;
// y rows: 4 scale bytes (one per 32 values), then e4m3 values.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_mxfp_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8,  8, int>   tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;
    constexpr int iter_k        = ggml_cuda_mmq_get_K_vram(type, J, fallback);

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + iter_k / 4);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    const int g = threadIdx.x / 4;
    const int t = threadIdx.x % 4;

    // 2 threads per quad supply the scale register to the block_scale MMA, 0x7F is a dummy scale for the other lanes,
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
    const int tidx_A = g + (t % 2) * 8;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            // x_qs is an int pointer, so kx offsets are in 4-byte units: thread t reads 8 container bytes of its 32-byte block
            const int kx = k0 + 2*t;
            const int2 a0 = *(const int2 *) (x_qs + (i0 + n*tile_A::I + g    )*sram_stride + kx);
            const int2 a1 = *(const int2 *) (x_qs + (i0 + n*tile_A::I + g + 8)*sram_stride + kx);
            // register layout per mma_block_scaled: x[0],x[1] are rows g/g+8 k 8t..8t+3, x[2],x[3] are k 8t+4..8t+7
            A[n][frag].x[0] = a0.x;
            A[n][frag].x[2] = a0.y;
            A[n][frag].x[1] = a1.x;
            A[n][frag].x[3] = a1.y;
            scaleA[n][frag] = t < 2 ? x_sc[(i0 + n * tile_A::I + tidx_A) * sram_stride + k0 / tile_A::J] : 0x7F;
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            const int2 b0 = *(const int2 *) (y_qs + (j0 + g)*MMQ_TILE_Y_K + k0 + 2*t);
            B[frag].x[0] = b0.x;
            B[frag].x[1] = b0.y;
            scaleB[frag] = t == 0 ? y_sc[(j0 + g) * MMQ_TILE_Y_K + frag] : 0x7F;
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                tile_C C = {};
                mma_block_scaled<type>(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}
