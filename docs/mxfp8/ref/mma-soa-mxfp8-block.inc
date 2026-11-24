#ifdef AMD_MFMA_AVAILABLE
        using floatx4_t = __attribute__((ext_vector_type(4))) float;
        floatx4_t& acc_frag = reinterpret_cast<floatx4_t&>(D.x[0]);
#if defined(CDNA3)
        using floatx2_t = __attribute__((ext_vector_type(2))) float;
        const floatx2_t& a_frag = reinterpret_cast<const floatx2_t&>(A.x[0]);
        const floatx2_t& b_frag = reinterpret_cast<const floatx2_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_mfma_f32_16x16x8_xf32(a_frag, b_frag, acc_frag, 0, 0, 0);
#elif defined(CDNA2) || defined(CDNA1)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            acc_frag = __builtin_amdgcn_mfma_f32_16x16x4f32(A.x[i], B.x[i], acc_frag, 0, 0, 0);
        }
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // defined(CDNA3)
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma_block_scaled(tile<16, 8, float> &     D,
                                                            const tile<16, 8, int> & A,
                                                            const tile<8, 8, int> &  B,
                                                            uint32_t                 a_scale,
                                                            uint32_t                 b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
#else
        GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
#endif  // BLACKWELL_MMA_AVAILABLE
    }

    // MXFP8 block-scaled MMA: kind::mxf8f6f4, m16n8k32, FP8 E4M3
    // A: 4 registers, B: 2 registers, scale_vec::1X: 1 E8M0 scale per 32 elements
    static __device__ __forceinline__ void mma_block_scaled_f8(tile<16, 8, float> &     D,
                                                               const tile<16, 8, int> & A,
                                                               const tile<8, 8, int> &  B,
                                                               uint32_t                 a_scale,
                                                               uint32_t                 b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;

        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
#else
        GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
#endif  // BLACKWELL_MMA_AVAILABLE
    }

    // MXFP8 E5M2 block-scaled MMA: kind::mxf8f6f4, m16n8k32, FP8 E5M2
    static __device__ __forceinline__ void mma_block_scaled_f8_e5m2(tile<16, 8, float> &     D,
                                                                    const tile<16, 8, int> & A,
                                                                    const tile<8, 8, int> &  B,
                                                                    uint32_t                 a_scale,
                                                                    uint32_t                 b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;

        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e5m2.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
#else
        GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
#endif  // BLACKWELL_MMA_AVAILABLE
    }

    // MXFP6 E2M3 block-scaled MMA: kind::mxf8f6f4, m16n8k32, FP6 E2M3
    // Data in fp6x4 format: 4 six-bit elements packed in low 24 bits of each 32-bit register.
    static __device__ __forceinline__ void mma_block_scaled_f6_e2m3(tile<16, 8, float> &     D,
                                                                    const tile<16, 8, int> & A,
                                                                    const tile<8, 8, int> &  B,
                                                                    uint32_t                 a_scale,
                                                                    uint32_t                 b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;

        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e2m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
#else
        GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
#endif  // BLACKWELL_MMA_AVAILABLE
    }

    // MXFP6 E3M2 block-scaled MMA: kind::mxf8f6f4, m16n8k32, FP6 E3M2
    static __device__ __forceinline__ void mma_block_scaled_f6_e3m2(tile<16, 8, float> &     D,
                                                                    const tile<16, 8, int> & A,
                                                                    const tile<8, 8, int> &  B,
                                                                    uint32_t                 a_scale,
                                                                    uint32_t                 b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;

        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
#else
        GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
