#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + k0 + 0]        = v.x;
        x_qs[i*sram_stride + k0 + QI_MXFP4] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]        = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI_MXFP4;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#else
        x_df[i*(MMQ_TILE_NE_K/QI_MXFP4) + i/QI_MXFP4 + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mxfp weight tiles for the block-scaled mma: one thread per 32-element block, one 8-bit container
// byte per value, then one e8m0 scale byte per 32 values.
// mxf8f6f4 container packing (PTX ISA): e2m1 in bits [2..5], e2m3 in bits [0..5], e4m3 in bits [0..7].
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int *      x_qs = (int *) x_tile;
    uint32_t * x_sc = (uint32_t *) (x_qs + MMQ_ITER_K_FP4 / 4);

    constexpr int threads_per_row = MMQ_ITER_K_FP4 / QK_MXFP4;  // one thread per block
    constexpr int rows_per_warp = warp_size / threads_per_row;
    const int kbx = threadIdx.x % threads_per_row;
    const int row_in_warp = threadIdx.x / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (fallback) {
            i = min(i, i_max);
        }

        if constexpr (type == GGML_TYPE_MXFP4) {
            const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i * stride + kbx;
            // qs byte j holds elements {j, j+16} in its low/high nibble: bytes[j] = lo, bytes[j+16] = hi
            uint8_t * out = (uint8_t *) (x_qs + i*sram_stride + kbx*8);
#pragma unroll
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                out[j]      = (bxi->qs[j] & 0x0F) << 2;
                out[j + 16] = (bxi->qs[j] >> 2) & 0x3C;
            }
            x_sc[i*sram_stride + kbx] = bxi->e;
        } else if constexpr (type == GGML_TYPE_MXFP8) {
            // e4m3 codes are one per byte in K order; the scale byte makes qs unaligned,
            // so load bytes and assemble the 4-byte words in registers
            const block_mxfp8 * bxi = (const block_mxfp8 *) x + kbx0 + i * stride + kbx;
            const uint8_t * qsb = (const uint8_t *) bxi->qs;
            uint32_t * out = (uint32_t *) (x_qs + i*sram_stride + kbx*8);
#pragma unroll
            for (int w = 0; w < QK_MXFP8/4; ++w) {
                const int w0 = 4*w;
                out[w] = (uint32_t) qsb[w0] | ((uint32_t) qsb[w0+1] << 8) | ((uint32_t) qsb[w0+2] << 16) | ((uint32_t) qsb[w0+3] << 24);
            }
            x_sc[i*sram_stride + kbx] = bxi->e;
        } else {
            // e2m3 (mxfp6): 6-bit packed bitstream, every 4 codes fill exactly 3 bytes (code 4*k is byte-aligned),
            // unpack 4 at a time into the one-byte-per-code container
            const block_mxfp6 * bxi = (const block_mxfp6 *) x + kbx0 + i * stride + kbx;
            const uint8_t * qsb = (const uint8_t *) bxi->qs;
            uint8_t * out = (uint8_t *) (x_qs + i*sram_stride + kbx*8);
#pragma unroll
            for (int k = 0; k < QK_MXFP6/4; ++k) {
                const uint32_t w = (uint32_t) qsb[3*k] | ((uint32_t) qsb[3*k+1] << 8) | ((uint32_t) qsb[3*k+2] << 16);
                out[4*k + 0] = (uint8_t)  (w & 0x3F);
                out[4*k + 1] = (uint8_t) ((w >> 6) & 0x3F);
                out[4*k + 2] = (uint8_t) ((w >> 12) & 0x3F);
                out[4*k + 3] = (uint8_t) ((w >> 18) & 0x3F);
            }
            x_sc[i*sram_stride + kbx] = bxi->e;
        }
    }
}


template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kb0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_NVFP4, I);
    int   * x_qs = (int   *) x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / QK_NVFP4;
    constexpr int rows_per_warp = warp_size / threads_per_row;
    const int kbx = threadIdx.x % threads_per_row;
    const int row_in_warp = threadIdx.x / threads_per_row;

#pragma unroll
