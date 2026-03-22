#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr bool is_mxfp_k = (type_K == GGML_TYPE_MXFP4 || type_K == GGML_TYPE_MXFP8 || type_K == GGML_TYPE_MXFP6);
    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = !is_mxfp_k && type_K != GGML_TYPE_F16;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread, D>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread, D>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;

    // MXFP multihead detection (matches CPU mxfp_kv_params_init):
    // When nb[2] == ggml_row_size(type, D), heads are contiguous within one
    // KV-position stride, so the SoA region spans all heads. In this case,
    // do NOT add the per-head offset here — mxfp_dequant_head handles it.
    // Row size = type_size * D / blck_size (same as ggml_row_size).
    constexpr int mxfp_K_row_bytes = is_mxfp_k ?
        (type_K == GGML_TYPE_MXFP4 ? (int)(sizeof(block_mxfp4) * D / QK_MXFP4) :
         type_K == GGML_TYPE_MXFP8 ? (int)(sizeof(block_mxfp8) * D / QK_MXFP8) :
                                      (int)(sizeof(block_mxfp6) * D / QK_MXFP6)) : 0;
    const bool mxfp_K_multihead = is_mxfp_k && (nb12 == mxfp_K_row_bytes);
    constexpr bool is_mxfp_v = (type_V == GGML_TYPE_MXFP4 || type_V == GGML_TYPE_MXFP8 || type_V == GGML_TYPE_MXFP6);
    constexpr int mxfp_V_row_bytes = is_mxfp_v ?
        (type_V == GGML_TYPE_MXFP4 ? (int)(sizeof(block_mxfp4) * D / QK_MXFP4) :
         type_V == GGML_TYPE_MXFP8 ? (int)(sizeof(block_mxfp8) * D / QK_MXFP8) :
                                      (int)(sizeof(block_mxfp6) * D / QK_MXFP6)) : 0;
    const bool mxfp_V_multihead = is_mxfp_v && (nb22 == mxfp_V_row_bytes);

    if (mxfp_K_multihead) {
        K += nb13*sequence;  // no head offset — mxfp_dequant_head extracts per-head data
    } else {
        K += nb13*sequence + nb12*(head / gqa_ratio);
    }
    if (mxfp_V_multihead) {
        V += nb23*sequence;
    } else {
        V += nb23*sequence + nb22*(head / gqa_ratio);
    }

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
    constexpr int ne_shmem   = ne_KQ > ne_combine ? ne_KQ : ne_combine;
#ifdef V_DOT2_F32_F16_AVAILABLE
    // MXFP Q preprocessing needs D floats in shmem. KQ is half[] here, so need enough half elements.
    constexpr int ne_mxfp_q  = is_mxfp_k ? D * (int)(sizeof(float) / sizeof(half)) : 0;
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_shmem > ne_mxfp_q ? ne_shmem : ne_mxfp_q];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_shmem];
#endif // V_DOT2_F32_F16_AVAILABLE

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else if constexpr (is_mxfp_k) {
        // MXFP Q path: fuse Hadamard + quantize roundtrip via warp shuffles + shared memory.
        // K has Hadamard from set_rows — Q MUST have the same rotation for correct Q·K dot product.
        // Follows CPU scalar implementation step-by-step (ops.cpp mxfp_fa_params_init).
        float * Q_shmem = (float *)&KQ[0];

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float * Q_f = (const float *)(Q + j*nb01);
            const bool valid = (ncols == 1 || ic0 + j < int(ne01.z));

            // Step 1: Load Q into shared memory (all threads cooperate)
#pragma unroll
            for (int i = tid; i < D; i += nthreads) {
                Q_shmem[i] = valid ? Q_f[i] : 0.0f;
            }
            __syncthreads();

            // Step 2: Hadamard + roundtrip per 32-element block.
            // Strided loop handles D > nwarps*32 (e.g., D=256 with 4 warps).
            constexpr int nblocks_q = D / 32;
            const int warp_id = threadIdx.y;
            const int lane_id = threadIdx.x;

#pragma unroll
            for (int b = warp_id; b < nblocks_q; b += nwarps) {
                float val = Q_shmem[b * 32 + lane_id];

                // Fused Hadamard via warp shuffles — matches ggml_mxfp_hadamard_32_inplace
                val = mxfp_hadamard_warp(val);

                // Fused quantize roundtrip: amax → E8M0 → quantize → dequant
                // Uses shared constructs from ggml-common.h
                const float amax = mxfp_warp_amax(val);
                if constexpr (type_K == GGML_TYPE_MXFP4) {
                    uint8_t e = 0;
                    if (amax > 0.0f) {
                        const int e_base = ggml_mxfp_e8m0_base_estimate(amax, MXFP4_E2M1_EMAX_OFFSET);
                        e = (uint8_t)(e_base < 0 ? 0 : (e_base > 254 ? 254 : e_base));
                    }
                    e = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)e, 0);
                    const float d = ggml_mxfp_e8m0_to_fp32_half(e);
                    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
                    const uint8_t idx = mxfp4_quantize_elem(val, inv_d);
                    val = kvalues_mxfp4[idx] * d;
                } else if constexpr (type_K == GGML_TYPE_MXFP8) {
                    uint8_t e = 0;
                    if (amax > 0.0f) {
                        const int e_base = ggml_mxfp_e8m0_base_estimate(amax, MXFP8_E4M3_EMAX_OFFSET);
                        e = (uint8_t)(e_base < 0 ? 0 : (e_base > 254 ? 254 : e_base));
                    }
                    e = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)e, 0);
                    const float d = ggml_mxfp_e8m0_to_fp32(e);
                    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
                    val = ggml_mxfp_fp8_e4m3_to_float(ggml_mxfp_float_to_fp8_e4m3(val * inv_d)) * d;
                } else {
                    uint8_t e = 0;
                    if (amax > 0.0f) {
                        const int e_base = ggml_mxfp_e8m0_base_estimate(amax, MXFP6_E2M3_EMAX_OFFSET);
                        e = (uint8_t)(e_base < 0 ? 0 : (e_base > 254 ? 254 : e_base));
                    }
                    e = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)e, 0);
                    const float d = ggml_mxfp_e8m0_to_fp32(e);
                    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
                    val = ggml_mxfp_fp6_e2m3_to_float(ggml_mxfp_float_to_fp6_e2m3(val * inv_d)) * d;
                }

                Q_shmem[b * 32 + lane_id] = val;
            }
            __syncthreads();

            // Step 3: Load from shmem into Q_reg as consecutive float2 pairs with scale.
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                const float q0 = Q_shmem[2*i]     * scale;
                const float q1 = Q_shmem[2*i + 1] * scale;
#ifdef V_DOT2_F32_F16_AVAILABLE
                Q_reg[j][i0/nthreads_KQ] = make_half2(__float2half(q0), __float2half(q1));
#else
                Q_reg[j][i0/nthreads_KQ] = make_float2(q0, q1);
#endif
            }
            __syncthreads();
        }
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                if constexpr (is_mxfp_k) {
                    // MXFP multihead-aware KQ dot — matches CPU mxfp_row_ptr + mxfp_dequant_head.
                    // For multihead, the SoA row at K + i_KQ*nb11 spans all heads.
                    // We must read this head's qs and e8m0 from the correct offsets.
                    const char * K_row = K + i_KQ*nb11;
                    const int kv_head = head / gqa_ratio;

                    constexpr int blocks_per_head = D / 32;
                    constexpr int qs_per_blk = (type_K == GGML_TYPE_MXFP4) ? MXFP4_SOA_QS_PER_BLOCK :
                                               (type_K == GGML_TYPE_MXFP8) ? MXFP8_SOA_QS_PER_BLOCK :
                                                                               MXFP6_SOA_QS_PER_BLOCK;
                    constexpr int head_qs_bytes = blocks_per_head * qs_per_blk;

                    // In multihead mode, qs for this head starts at head*head_qs_bytes,
                    // and e8m0 starts at total_blocks*qs_per_blk + head*blocks_per_head.
                    // In non-multihead mode, the row IS one head's SoA data.
                    const int qs_off = mxfp_K_multihead ? kv_head * head_qs_bytes : 0;
                    const int total_blocks = mxfp_K_multihead ? ne12 * blocks_per_head : blocks_per_head;
                    const int e8m0_off = total_blocks * qs_per_blk + (mxfp_K_multihead ? kv_head * blocks_per_head : 0);

                    const uint8_t * qs_base   = (const uint8_t *)K_row + qs_off;
                    const uint8_t * e8m0_base = (const uint8_t *)K_row + e8m0_off;

                    // Inline dot product with Q — same math as vec_dot_fattn_vec_KQ_mxfp*
                    sum = 0.0f;
#pragma unroll
                    for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ) {
                        const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                        const int elem = 2 * i;

                        float k0, k1;
                        if constexpr (type_K == GGML_TYPE_MXFP4) {
                            const int blk0 = elem / 32, pos0 = elem % 32;
                            const int blk1 = (elem+1) / 32, pos1 = (elem+1) % 32;
                            const float d0 = ggml_mxfp_e8m0_to_fp32_half(e8m0_base[blk0]);
                            const float d1 = ggml_mxfp_e8m0_to_fp32_half(e8m0_base[blk1]);
                            const uint8_t * qs0 = qs_base + blk0 * qs_per_blk;
                            const uint8_t * qs1 = qs_base + blk1 * qs_per_blk;
                            const int bi0 = pos0 < 16 ? pos0 : pos0 - 16;
                            const int bi1 = pos1 < 16 ? pos1 : pos1 - 16;
                            k0 = kvalues_mxfp4[(pos0 < 16) ? (qs0[bi0] & 0x0F) : (qs0[bi0] >> 4)] * d0;
                            k1 = kvalues_mxfp4[(pos1 < 16) ? (qs1[bi1] & 0x0F) : (qs1[bi1] >> 4)] * d1;
                        } else if constexpr (type_K == GGML_TYPE_MXFP8) {
                            const int blk0 = elem / 32, pos0 = elem % 32;
                            const int blk1 = (elem+1) / 32, pos1 = (elem+1) % 32;
                            k0 = ggml_mxfp_fp8_e4m3_to_float(qs_base[blk0*qs_per_blk + pos0]) * ggml_mxfp_e8m0_to_fp32(e8m0_base[blk0]);
                            k1 = ggml_mxfp_fp8_e4m3_to_float(qs_base[blk1*qs_per_blk + pos1]) * ggml_mxfp_e8m0_to_fp32(e8m0_base[blk1]);
                        } else {
                            auto unpack = [&](int e) -> float {
                                const int blk = e / 32, pos = e % 32;
                                const int grp = pos / 4, slot = pos % 4;
                                const uint8_t * packed = qs_base + blk*qs_per_blk + grp*3;
                                uint8_t vals[4];
                                ggml_mxfp_unpack_fp6x4(packed, vals);
                                return ggml_mxfp_fp6_e2m3_to_float(vals[slot]) * ggml_mxfp_e8m0_to_fp32(e8m0_base[blk]);
                            };
                            k0 = unpack(elem);
                            k1 = unpack(elem + 1);
                        }

                        const float2 Q_f2 = ((const float2 *)Q_reg[j])[i0/nthreads_KQ];
                        sum += k0 * Q_f2.x + k1 * Q_f2.y;
                    }
                } else {
                    sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                }
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            // V dequant — for MXFP, compute multihead-aware pointers matching
            // CPU mxfp_row_ptr + mxfp_dequant_head exactly.
            const char * V_row = V + k*nb21;
            const char * V_dequant_src = V_row;

            // MXFP multihead V head extraction (matches CPU mxfp_dequant_head).
            // For multihead: extract this head's qs and e8m0 into a contiguous per-head buffer.
            // For non-multihead: V_row IS already per-head SoA data, use directly.
            // Buffer lives in registers/local memory — small (max D=256 → 264 bytes for MXFP8).
            char v_head_soa_buf[is_mxfp_v ? mxfp_V_row_bytes : 1];
            if constexpr (is_mxfp_v) {
                if (mxfp_V_multihead) {
                    constexpr int v_blocks_per_head = D / 32;
                    constexpr int v_qs_per_blk = (type_V == GGML_TYPE_MXFP4) ? MXFP4_SOA_QS_PER_BLOCK :
                                                 (type_V == GGML_TYPE_MXFP8) ? MXFP8_SOA_QS_PER_BLOCK :
                                                                                MXFP6_SOA_QS_PER_BLOCK;
                    constexpr int v_head_qs_bytes = v_blocks_per_head * v_qs_per_blk;
                    const int v_kv_head = head / gqa_ratio;
                    const int v_qs_off = v_kv_head * v_head_qs_bytes;
                    const int v_total_blocks = ne12 * v_blocks_per_head;  // ne12 = n_kv_heads
                    const int v_e8m0_off = v_total_blocks * v_qs_per_blk + v_kv_head * v_blocks_per_head;
                    // Copy this head's qs and e8m0 into contiguous buffer — exactly like CPU memcpy
                    for (int b = 0; b < v_head_qs_bytes; ++b) {
                        v_head_soa_buf[b] = V_row[v_qs_off + b];
                    }
                    for (int b = 0; b < v_blocks_per_head; ++b) {
                        v_head_soa_buf[v_head_qs_bytes + b] = V_row[v_e8m0_off + b];
                    }
                    V_dequant_src = v_head_soa_buf;
                }
            }

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                dequantize_V(V_dequant_src, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V_dequant_src, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)

// MXFP SoA flash attention — same-type and mixed K/V pairs.
#define EXTERN_DECL_FATTN_VEC_MXFP(D, type_KV) \
    extern DECL_FATTN_VEC_CASE(D, type_KV, type_KV);

#define EXTERN_DECL_FATTN_VEC_MXFP_MIXED(D, type_K, type_V) \
    extern DECL_FATTN_VEC_CASE(D, type_K, type_V);

EXTERN_DECL_FATTN_VEC_MXFP( 64, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP( 64, GGML_TYPE_MXFP8)
EXTERN_DECL_FATTN_VEC_MXFP( 64, GGML_TYPE_MXFP6)
EXTERN_DECL_FATTN_VEC_MXFP(128, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP(128, GGML_TYPE_MXFP8)
EXTERN_DECL_FATTN_VEC_MXFP(128, GGML_TYPE_MXFP6)
EXTERN_DECL_FATTN_VEC_MXFP(256, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP(256, GGML_TYPE_MXFP8)
EXTERN_DECL_FATTN_VEC_MXFP(256, GGML_TYPE_MXFP6)

EXTERN_DECL_FATTN_VEC_MXFP_MIXED( 64, GGML_TYPE_MXFP8, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP_MIXED( 64, GGML_TYPE_MXFP6, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP_MIXED(128, GGML_TYPE_MXFP8, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP_MIXED(128, GGML_TYPE_MXFP6, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP_MIXED(256, GGML_TYPE_MXFP8, GGML_TYPE_MXFP4)
EXTERN_DECL_FATTN_VEC_MXFP_MIXED(256, GGML_TYPE_MXFP6, GGML_TYPE_MXFP4)
