#pragma once

#include "common.cuh"

// CUDA warp-cooperative helpers for MXFP SoA flash attention.
// Used by fattn-vec.cuh, fattn-common.cuh, and set-rows.cu.
// Per-element math via ggml-common.h (GGML_MXFP_FUNC).

// 3-way MXFP type dispatch: expands EXPR once per type with `mxfp_type` as the template arg.
// Use in switch(runtime_type) bodies to eliminate repetitive MXFP4/8/6 case blocks.
#define MXFP_DISPATCH(runtime_type, ...) do {               \
    switch (runtime_type) {                                 \
        case GGML_TYPE_MXFP4: { constexpr ggml_type mxfp_type = GGML_TYPE_MXFP4; __VA_ARGS__; } break; \
        case GGML_TYPE_MXFP8: { constexpr ggml_type mxfp_type = GGML_TYPE_MXFP8; __VA_ARGS__; } break; \
        case GGML_TYPE_MXFP6: { constexpr ggml_type mxfp_type = GGML_TYPE_MXFP6; __VA_ARGS__; } break; \
        default: GGML_ABORT("unsupported MXFP type"); break; \
    }                                                       \
} while (0)

// Type traits: compile-time constants per MXFP type.

template <ggml_type type> struct mxfp_type_traits;

template <> struct mxfp_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int emax_offset = MXFP4_E2M1_EMAX_OFFSET;
    static constexpr int qs_per_blk  = MXFP4_SOA_QS_PER_BLOCK;
};

template <> struct mxfp_type_traits<GGML_TYPE_MXFP8> {
    static constexpr int emax_offset = MXFP8_E4M3_EMAX_OFFSET;
    static constexpr int qs_per_blk  = MXFP8_SOA_QS_PER_BLOCK;
};

template <> struct mxfp_type_traits<GGML_TYPE_MXFP6> {
    static constexpr int emax_offset = MXFP6_E2M3_EMAX_OFFSET;
    static constexpr int qs_per_blk  = MXFP6_SOA_QS_PER_BLOCK;
};

// Row bytes = AoS block size * blocks_per_row = (qs_per_blk + 1) * (D / 32).
// Used for multihead SoA detection: multihead = (nb2 == MXFP_ROW_BYTES_EXPLICIT(type, D)).
// Uses explicit constants (not traits) to avoid incomplete-type errors for non-MXFP types.
#define MXFP_ROW_BYTES_EXPLICIT(type, D) (                                      \
    ((type) == GGML_TYPE_MXFP4) ? ((MXFP4_SOA_QS_PER_BLOCK + 1) * ((D) / 32)) : \
    ((type) == GGML_TYPE_MXFP8) ? ((MXFP8_SOA_QS_PER_BLOCK + 1) * ((D) / 32)) : \
    ((type) == GGML_TYPE_MXFP6) ? ((MXFP6_SOA_QS_PER_BLOCK + 1) * ((D) / 32)) : 0)

// Compute multihead-aware qs_base and e8m0_base pointers from a SoA row.
// Deduplicates the K and V multihead offset logic in fattn-vec.cuh.
template <ggml_type type, int D>
static __device__ __forceinline__ void mxfp_multihead_ptrs(
        const char * row, int kv_head, int n_heads, bool multihead,
        const uint8_t * & qs_base, const uint8_t * & e8m0_base) {
    constexpr int blocks_per_head = D / 32;
    constexpr int qs_per_blk = mxfp_type_traits<type>::qs_per_blk;
    constexpr int head_qs_bytes = blocks_per_head * qs_per_blk;
    if (multihead) {
        const int total_blocks = n_heads * blocks_per_head;
        qs_base   = (const uint8_t *)row + kv_head * head_qs_bytes;
        e8m0_base = (const uint8_t *)row + total_blocks * qs_per_blk + kv_head * blocks_per_head;
    } else {
        qs_base   = (const uint8_t *)row;
        e8m0_base = qs_base + blocks_per_head * qs_per_blk;
    }
}

// Warp-cooperative primitives.

// Warp-wide max absolute value via __shfl_xor_sync.
static __device__ __forceinline__ float mxfp_warp_amax(float val) {
    val = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp-cooperative block-32 Walsh-Hadamard transform.
// GPU warp-shuffle equivalent of ggml_mxfp_hadamard_32_inplace (ggml-common.h).
static __device__ __forceinline__ float mxfp_hadamard_warp(float val) {
    #pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, stride);
        val = (threadIdx.x & stride) ? (other - val) : (val + other);
    }
    return val * MXFP_HADAMARD_32_NORM;
}

// E8M0 scale from amax: ggml_mxfp_e8m0_base_estimate (ggml-common.h) + clamp.
template <ggml_type type>
static __device__ __forceinline__ uint8_t mxfp_compute_e8m0(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }
    const int e_base = ggml_mxfp_e8m0_base_estimate(amax, mxfp_type_traits<type>::emax_offset);
    return (uint8_t)(e_base < 0 ? 0 : (e_base > 254 ? 254 : e_base));
}

// MXFP intrinsic helpers (CUDA 12.8+ / sm_100+).
// E8M0 encode intrinsic (__nv_cvt_float_to_e8m0) is BROKEN — never use it.
#if CUDART_VERSION >= 12080

// halfraw → float / half2raw → float2 conversion helpers.
static __device__ __forceinline__ float  halfraw_to_float(__half_raw hr) {
    return __half2float(*reinterpret_cast<__half *>(&hr));
}
static __device__ __forceinline__ float2 half2raw_to_float2(__half2_raw hr2) {
    __half2 h2 = *reinterpret_cast<__half2 *>(&hr2);
    return make_float2(__low2float(h2), __high2float(h2));
}

#endif // CUDART_VERSION >= 12080

// Quantize one float element given its E8M0 scale byte → uint8_t.
template <ggml_type type>
static __device__ __forceinline__ uint8_t mxfp_quantize_elem(float val, uint8_t e8m0) {
    const float d     = ggml_cuda_e8m0_to_fp32(e8m0);
    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
    const float scaled = val * inv_d;
#if CUDART_VERSION >= 12080
    if constexpr (type == GGML_TYPE_MXFP4) { return (__nv_fp4_storage_t)__nv_cvt_float_to_fp4(scaled, __NV_E2M1, cudaRoundNearest); }
    if constexpr (type == GGML_TYPE_MXFP8) { return (__nv_fp8_storage_t)__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3); }
    if constexpr (type == GGML_TYPE_MXFP6) { return (__nv_fp6_storage_t)__nv_cvt_float_to_fp6(scaled, __NV_E2M3, cudaRoundNearest); }
#else
    if constexpr (type == GGML_TYPE_MXFP4) { return ggml_mxfp_float_to_fp4_e2m1(scaled); }
    if constexpr (type == GGML_TYPE_MXFP8) { return ggml_mxfp_float_to_fp8_e4m3(scaled); }
    if constexpr (type == GGML_TYPE_MXFP6) { return ggml_mxfp_float_to_fp6_e2m3(scaled); }
#endif
}

// Dequant one raw element value (after type-specific extraction).
template <ggml_type type>
static __device__ __forceinline__ float mxfp_dequant_raw(uint8_t raw) {
#if CUDART_VERSION >= 12080
    if constexpr (type == GGML_TYPE_MXFP4) { return halfraw_to_float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)raw, __NV_E2M1)); }
    if constexpr (type == GGML_TYPE_MXFP8) { return halfraw_to_float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)raw, __NV_E4M3)); }
    if constexpr (type == GGML_TYPE_MXFP6) { return halfraw_to_float(__nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)raw, __NV_E2M3)); }
#else
    if constexpr (type == GGML_TYPE_MXFP4) { return ggml_mxfp_fp4_e2m1_to_float(raw); }
    if constexpr (type == GGML_TYPE_MXFP8) { return ggml_mxfp_fp8_e4m3_to_float(raw); }
    if constexpr (type == GGML_TYPE_MXFP6) { return ggml_mxfp_fp6_e2m3_to_float(raw); }
#endif
}

// Quantize-roundtrip: float → quantize → dequant → float.
// Used by Q Hadamard roundtrip (MMA and VEC paths).
template <ggml_type type>
static __device__ __forceinline__ float mxfp_quantize_roundtrip(float val, uint8_t e8m0) {
    const float d   = ggml_cuda_e8m0_to_fp32(e8m0);
    const uint8_t q = mxfp_quantize_elem<type>(val, e8m0);
    return mxfp_dequant_raw<type>(q) * d;
}

// Complete Q preprocessing: Hadamard + amax + E8M0 + roundtrip.

template <ggml_type type>
static __device__ __forceinline__ float mxfp_hadamard_roundtrip(float val) {
    val = mxfp_hadamard_warp(val);
    const float amax = mxfp_warp_amax(val);
    uint8_t e8m0 = mxfp_compute_e8m0<type>(amax);
    e8m0 = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)e8m0, 0);
    return mxfp_quantize_roundtrip<type>(val, e8m0);
}

// MXFP4 branchless nibble-pair extraction from SoA layout.
// pos0 is always even → both in same half (low or high nibbles).

static __device__ __forceinline__ void mxfp4_extract_nibble_pair(
        const uint8_t * qs, int pos0, uint8_t & nib0, uint8_t & nib1) {
    // pos0 is always even, pos1 = pos0+1. Both are in the same half (low or high nibbles).
    const int shift = (pos0 >= 16) ? 4 : 0;
    const int bi0 = pos0 & 15;  // byte index: equivalent to pos < 16 ? pos : pos - 16
    const int bi1 = bi0 + 1;    // pos1 = pos0+1, always same half, so bi1 = bi0+1
    nib0 = (qs[bi0] >> shift) & 0x0F;
    nib1 = (qs[bi1] >> shift) & 0x0F;
}

// MXFP6 pair extraction from SoA layout.
// pos0 is always even → pos0%4 ∈ {0,2} → always same group.

static __device__ __forceinline__ void mxfp6_unpack_pair(
        const uint8_t * qs_block, int pos0, uint8_t & v0, uint8_t & v1) {
    // pos0 is always even → pos0%4 ∈ {0, 2} → grp0 == grp1 always
    const int grp  = pos0 / 4;
    const int slot = pos0 % 4;  // 0 or 2
    uint8_t vals[4];
    ggml_mxfp_unpack_fp6x4(qs_block + grp * 3, vals);
    v0 = vals[slot];
    v1 = vals[slot + 1];
}

// Unified SoA pair dequant: two consecutive elements from one 32-element block.
// pos0 MUST be even. Returns float2{elem[pos0], elem[pos0+1]}.

template <ggml_type type>
static __device__ __forceinline__ float2 mxfp_dequant_elem_pair(
        const uint8_t * qs_block, uint8_t e8m0, int pos0) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0);

    // Type-specific extraction + x2 intrinsic dequant when available.
    if constexpr (type == GGML_TYPE_MXFP4) {
#if CUDART_VERSION >= 12080
        const int shift = (pos0 >= 16) ? 4 : 0;
        const int bi0 = pos0 & 15;
        const uint8_t packed = ((qs_block[bi0] >> shift) & 0x0F) | (((qs_block[bi0 + 1] >> shift) & 0x0F) << 4);
        const float2 raw = half2raw_to_float2(__nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1));
        return make_float2(raw.x * d, raw.y * d);
#else
        uint8_t nib0, nib1;
        mxfp4_extract_nibble_pair(qs_block, pos0, nib0, nib1);
        return make_float2(mxfp_dequant_raw<type>(nib0) * d, mxfp_dequant_raw<type>(nib1) * d);
#endif
    } else if constexpr (type == GGML_TYPE_MXFP8) {
#if CUDART_VERSION >= 12080
        const __nv_fp8x2_storage_t x2 = *reinterpret_cast<const __nv_fp8x2_storage_t *>(qs_block + pos0);
        const float2 raw = half2raw_to_float2(__nv_cvt_fp8x2_to_halfraw2(x2, __NV_E4M3));
        return make_float2(raw.x * d, raw.y * d);
#else
        return make_float2(mxfp_dequant_raw<type>(qs_block[pos0]) * d,
                           mxfp_dequant_raw<type>(qs_block[pos0 + 1]) * d);
#endif
    } else {
        uint8_t v0, v1;
        mxfp6_unpack_pair(qs_block, pos0, v0, v1);
#if CUDART_VERSION >= 12080
        const __nv_fp6x2_storage_t x2 = (__nv_fp6x2_storage_t)(v0 | ((uint16_t)v1 << 8));
        const float2 raw = half2raw_to_float2(__nv_cvt_fp6x2_to_halfraw2(x2, __NV_E2M3));
        return make_float2(raw.x * d, raw.y * d);
#else
        return make_float2(mxfp_dequant_raw<type>(v0) * d, mxfp_dequant_raw<type>(v1) * d);
#endif
    }
}

// Single-element SoA dequant (arbitrary position, used by SoA→F16 conversion).
template <ggml_type type>
static __device__ __forceinline__ float mxfp_dequant_elem(
        const uint8_t * qs_base, const uint8_t * e8m0_base,
        int blk, int pos, int qs_per_blk) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0_base[blk]);
    const uint8_t * qs = qs_base + blk * qs_per_blk;
    uint8_t raw;
    if constexpr (type == GGML_TYPE_MXFP4) {
        const int byte_idx = pos < 16 ? pos : pos - 16;
        raw = (pos < 16) ? (qs[byte_idx] & 0x0F) : (qs[byte_idx] >> 4);
    } else if constexpr (type == GGML_TYPE_MXFP8) {
        raw = qs[pos];
    } else {
        const int grp  = pos / 4;
        const int slot = pos % 4;
        uint8_t vals[4];
        ggml_mxfp_unpack_fp6x4(qs + grp * 3, vals);
        raw = vals[slot];
    }
    return mxfp_dequant_raw<type>(raw) * d;
}

// MXFP SoA → F16 dequant kernel for MMA flash attention pre-conversion.

template <ggml_type mxfp_type>
static __global__ void k_mxfp_soa_to_f16(
        const char * __restrict__ src,
        half * __restrict__ dst,
        const int D,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t nb1,    // byte stride between KV positions
        const int64_t nb2,    // byte stride between heads
        const int64_t nb3,    // byte stride between sequences
        const int blocks_per_head,     // D / 32
        const bool multihead,          // multihead SoA detection (matches CPU mxfp_kv_params_init)
        const int head_qs_bytes,       // blocks_per_head * qs_per_blk
        const int head_e8m0_offset) {  // total_blocks * qs_per_blk (total across all heads if multihead)

    constexpr int qs_per_blk = mxfp_type_traits<mxfp_type>::qs_per_blk;

    const int64_t flat_row = blockIdx.x;
    const int elem = threadIdx.x;  // element within row (0..D-1)
    const int64_t nrows = ne1 * ne2 * ne3;
    if (flat_row >= nrows || elem >= D) return;

    // Decompose flat row → (i1, i2, i3) using strides
    const int64_t i3 = flat_row / (ne1 * ne2);
    const int64_t i2 = (flat_row / ne1) % ne2;
    const int64_t i1 = flat_row % ne1;

    // Multihead: row spans all heads, extract per-head qs/e8m0 offsets.
    const char * src_row;
    const uint8_t * qs_base;
    const uint8_t * e8m0_base;

    if (multihead) {
        // Row spans all heads. Head extraction matches CPU mxfp_dequant_head.
        src_row = src + i1*nb1 + i3*nb3;  // no i2*nb2 — head offset handled below
        const int head_idx = (int)i2;
        qs_base   = (const uint8_t *)src_row + head_idx * head_qs_bytes;
        e8m0_base = (const uint8_t *)src_row + head_e8m0_offset + head_idx * blocks_per_head;
    } else {
        src_row = src + i1*nb1 + i2*nb2 + i3*nb3;
        qs_base   = (const uint8_t *)src_row;
        e8m0_base = qs_base + blocks_per_head * qs_per_blk;
    }

    const int blk = elem / 32;
    const int pos = elem % 32;

    const float val = mxfp_dequant_elem<mxfp_type>(qs_base, e8m0_base, blk, pos, qs_per_blk);
    dst[flat_row * D + elem] = __float2half(val);
}

// Host dispatch for MXFP SoA → F16 conversion.
static void mxfp_soa_to_f16_cuda(
        const char * src, half * dst, ggml_type type,
        int64_t D, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, cudaStream_t stream) {

    const int blocks_per_head = (int)(D / 32);

    // Multihead detection:
    // multihead = (nb2 == ggml_row_size(type, D))
    const bool multihead = (nb2 == (size_t)ggml_row_size(type, D));

    const int64_t nrows = ne1 * ne2 * ne3;
    const int threads = (int)D;
    const int grid    = (int)nrows;
    if (grid <= 0 || threads <= 0) return;

    MXFP_DISPATCH(type, {
        constexpr int qs_per_blk = mxfp_type_traits<mxfp_type>::qs_per_blk;
        const int head_qs_bytes = blocks_per_head * qs_per_blk;
        const int total_blocks = multihead ? (int)ne2 * blocks_per_head : blocks_per_head;
        const int head_e8m0_offset = total_blocks * qs_per_blk;
        k_mxfp_soa_to_f16<mxfp_type><<<grid, threads, 0, stream>>>(
            src, dst, (int)D, ne1, ne2, ne3,
            (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
            blocks_per_head, multihead, head_qs_bytes, head_e8m0_offset);
    });
}

// Q Hadamard + roundtrip kernel for MMA flash attention pre-processing.
// One warp per 32-element block, operates on F32 Q rows in-place.

template <ggml_type mxfp_type>
static __global__ void k_mxfp_q_hadamard_roundtrip(
        float * __restrict__ Q,
        const int D,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s01,    // stride in floats (nb01/sizeof(float))
        const int64_t s02,
        const int64_t s03) {

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;

    const int64_t global_warp = (int64_t)blockIdx.x * warps_per_block + warp;

    const int nblocks_per_row = D / 32;
    const int64_t nrows = ne1 * ne2 * ne3;
    const int64_t total_blocks = nblocks_per_row * nrows;
    if (global_warp >= total_blocks) return;

    const int64_t flat_row = global_warp / nblocks_per_row;
    const int blk = (int)(global_warp % nblocks_per_row);

    // Decompose flat row → (i1, i2, i3) and use actual strides
    const int64_t i3 = flat_row / (ne1 * ne2);
    const int64_t i2 = (flat_row / ne1) % ne2;
    const int64_t i1 = flat_row % ne1;

    float * q_block = Q + i1*s01 + i2*s02 + i3*s03 + blk * 32;

    float val = q_block[lane];
    val = mxfp_hadamard_roundtrip<mxfp_type>(val);
    q_block[lane] = val;
}

// Host dispatch for Q Hadamard+roundtrip.
static void mxfp_q_hadamard_roundtrip_cuda(
        float * Q, ggml_type k_type, int64_t D,
        int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb01, size_t nb02, size_t nb03, cudaStream_t stream) {

    const int nblocks_per_row = (int)(D / 32);
    const int64_t nrows = ne1 * ne2 * ne3;
    const int64_t total_blocks = nblocks_per_row * nrows;
    if (total_blocks == 0) return;

    const int warps_per_cuda_block = 4;
    const int threads = warps_per_cuda_block * 32;
    const int grid = (int)((total_blocks + warps_per_cuda_block - 1) / warps_per_cuda_block);

    const int64_t s01 = (int64_t)(nb01 / sizeof(float));
    const int64_t s02 = (int64_t)(nb02 / sizeof(float));
    const int64_t s03 = (int64_t)(nb03 / sizeof(float));

    MXFP_DISPATCH(k_type,
        k_mxfp_q_hadamard_roundtrip<mxfp_type><<<grid, threads, 0, stream>>>(
            Q, (int)D, ne1, ne2, ne3, s01, s02, s03)
    );
}
