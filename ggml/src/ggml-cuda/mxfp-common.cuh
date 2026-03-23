#pragma once

#include "common.cuh"

// CUDA warp-cooperative helpers for MXFP SoA flash attention.
//
// Shared type traits, E8M0 computation, quantize/dequant, and roundtrip
// templates live here — used by fattn-vec.cuh, fattn-common.cuh, and set-rows.cu.
// All per-element math calls shared portable functions from ggml-common.h
// (available as __device__ __forceinline__ via GGML_MXFP_FUNC), with
// ggml_cuda_e8m0_to_fp32 (common.cuh) for intrinsic E8M0 decode.

// ============================================================================
// Type traits: compile-time constants per MXFP type.
// ============================================================================

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

// ============================================================================
// Warp-cooperative primitives (must be defined first — used by everything below).
// ============================================================================

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
// Matches ggml_mxfp_hadamard_32_inplace exactly:
//   CPU: vals[i+j] = a + b;  vals[i+j+stride] = a - b;
//   GPU: lower lane (bit clear) = val + other;  upper lane (bit set) = other - val.
static __device__ __forceinline__ float mxfp_hadamard_warp(float val) {
    #pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, stride);
        val = (threadIdx.x & stride) ? (other - val) : (val + other);
    }
    return val * MXFP_HADAMARD_32_NORM;
}

// MXFP4 E2M1 quantize: float → 4-bit index (0-15, high bit = sign).
// Same decision boundaries as CPU best_index_mxfp4.
// Takes inv_scale (pre-computed 1/scale) rather than scale, to avoid
// redundant division across 32 lanes.
static __device__ __forceinline__ uint8_t mxfp4_quantize_elem(float x, float inv_scale) {
    const float normalized = fabsf(x) * inv_scale;
    int idx;
    if      (normalized < 0.5f)  idx = 0;
    else if (normalized < 1.5f)  idx = 1;
    else if (normalized < 2.5f)  idx = 2;
    else if (normalized < 3.5f)  idx = 3;
    else if (normalized < 5.0f)  idx = 4;
    else if (normalized < 7.0f)  idx = 5;
    else if (normalized < 10.0f) idx = 6;
    else                         idx = 7;
    return (uint8_t)((x < 0.0f) ? (idx + 8) : idx);
}

// ============================================================================
// Shared E8M0 scale computation from amax.
// Calls ggml_mxfp_e8m0_base_estimate (ggml-common.h) + clamp.
// ============================================================================

template <ggml_type type>
static __device__ __forceinline__ uint8_t mxfp_compute_e8m0(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }
    const int e_base = ggml_mxfp_e8m0_base_estimate(amax, mxfp_type_traits<type>::emax_offset);
    return (uint8_t)(e_base < 0 ? 0 : (e_base > 254 ? 254 : e_base));
}

// ============================================================================
// MXFP element dequant/quant via hardware intrinsics (CUDA 12.8+ / sm_100+).
// These replace the portable LUT lookups with single-instruction conversions.
// Safe decode only for E8M0 — encode intrinsics (float→E8M0) have rounding bugs.
//
// Both single-element and x2 paired dequant variants provided. The x2 variants
// convert two elements in one instruction, halving CVT instruction count.
// Quantize intrinsics for fp8/fp6/fp4 encode are also provided.
// ============================================================================

#if CUDART_VERSION >= 12080

// --- Single-element dequant intrinsics ---

static __device__ __forceinline__ float mxfp8_dequant_intrinsic(uint8_t x) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)x, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half *>(&hr));
}

// Returns raw E2M1 value (NOT doubled), multiply by full scale, not half-scale.
static __device__ __forceinline__ float mxfp4_dequant_intrinsic(uint8_t nibble) {
    __half_raw hr = __nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)nibble, __NV_E2M1);
    return __half2float(*reinterpret_cast<__half *>(&hr));
}

static __device__ __forceinline__ float mxfp6_dequant_intrinsic(uint8_t x) {
    __half_raw hr = __nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)x, __NV_E2M3);
    return __half2float(*reinterpret_cast<__half *>(&hr));
}

// --- Paired (x2) dequant intrinsics: convert 2 elements in 1 instruction ---

static __device__ __forceinline__ float2 mxfp8_dequant_x2_intrinsic(__nv_fp8x2_storage_t x2) {
    __half2_raw hr2 = __nv_cvt_fp8x2_to_halfraw2(x2, __NV_E4M3);
    __half2 h2 = *reinterpret_cast<__half2 *>(&hr2);
    return make_float2(__low2float(h2), __high2float(h2));
}

static __device__ __forceinline__ float2 mxfp4_dequant_x2_intrinsic(__nv_fp4x2_storage_t x2) {
    __half2_raw hr2 = __nv_cvt_fp4x2_to_halfraw2(x2, __NV_E2M1);
    __half2 h2 = *reinterpret_cast<__half2 *>(&hr2);
    return make_float2(__low2float(h2), __high2float(h2));
}

static __device__ __forceinline__ float2 mxfp6_dequant_x2_intrinsic(__nv_fp6x2_storage_t x2) {
    __half2_raw hr2 = __nv_cvt_fp6x2_to_halfraw2(x2, __NV_E2M3);
    __half2 h2 = *reinterpret_cast<__half2 *>(&hr2);
    return make_float2(__low2float(h2), __high2float(h2));
}

// --- Quantize intrinsics: float → fp8/fp6/fp4 in one instruction ---

static __device__ __forceinline__ uint8_t mxfp8_quantize_intrinsic(float x) {
    return (__nv_fp8_storage_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

static __device__ __forceinline__ uint8_t mxfp6_quantize_intrinsic(float x) {
    return (__nv_fp6_storage_t)__nv_cvt_float_to_fp6(x, __NV_E2M3, cudaRoundNearest);
}

static __device__ __forceinline__ uint8_t mxfp4_quantize_intrinsic(float x) {
    return (__nv_fp4_storage_t)__nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest);
}

#endif // CUDART_VERSION >= 12080

// ============================================================================
// Shared quantize-roundtrip: float → quantize → dequant → float.
// Used by Q Hadamard roundtrip (MMA and VEC paths).
// Caller provides the pre-broadcast E8M0 scale byte.
// ============================================================================

template <ggml_type type>
static __device__ __forceinline__ float mxfp_quantize_roundtrip(float val, uint8_t e8m0);

template <>
__device__ __forceinline__ float mxfp_quantize_roundtrip<GGML_TYPE_MXFP4>(float val, uint8_t e8m0) {
#if CUDART_VERSION >= 12080
    const float d     = ggml_cuda_e8m0_to_fp32(e8m0);
    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q   = mxfp4_quantize_intrinsic(val * inv_d);
    return mxfp4_dequant_intrinsic(q) * d;
#else
    const float d     = ggml_mxfp_e8m0_to_fp32_half(e8m0);
    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t idx = mxfp4_quantize_elem(val, inv_d);
    return kvalues_mxfp4[idx] * d;
#endif
}

template <>
__device__ __forceinline__ float mxfp_quantize_roundtrip<GGML_TYPE_MXFP8>(float val, uint8_t e8m0) {
    const float d     = ggml_cuda_e8m0_to_fp32(e8m0);
    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
#if CUDART_VERSION >= 12080
    const uint8_t q = mxfp8_quantize_intrinsic(val * inv_d);
    return mxfp8_dequant_intrinsic(q) * d;
#else
    return ggml_mxfp_fp8_e4m3_to_float(ggml_mxfp_float_to_fp8_e4m3(val * inv_d)) * d;
#endif
}

template <>
__device__ __forceinline__ float mxfp_quantize_roundtrip<GGML_TYPE_MXFP6>(float val, uint8_t e8m0) {
    const float d     = ggml_cuda_e8m0_to_fp32(e8m0);
    const float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
#if CUDART_VERSION >= 12080
    const uint8_t q = mxfp6_quantize_intrinsic(val * inv_d);
    return mxfp6_dequant_intrinsic(q) * d;
#else
    return ggml_mxfp_fp6_e2m3_to_float(ggml_mxfp_float_to_fp6_e2m3(val * inv_d)) * d;
#endif
}

// ============================================================================
// Shared Hadamard + E8M0 + roundtrip: the complete Q preprocessing step.
// Fuses Hadamard transform, warp-wide amax, E8M0 computation, and roundtrip.
// Used by both VEC and MMA Q paths.
// ============================================================================

template <ggml_type type>
static __device__ __forceinline__ float mxfp_hadamard_roundtrip(float val) {
    val = mxfp_hadamard_warp(val);
    const float amax = mxfp_warp_amax(val);
    uint8_t e8m0 = mxfp_compute_e8m0<type>(amax);
    e8m0 = (uint8_t)__shfl_sync(0xFFFFFFFF, (int)e8m0, 0);
    return mxfp_quantize_roundtrip<type>(val, e8m0);
}

// ============================================================================
// MXFP4 branchless nibble extraction from SoA layout.
// SoA layout: 16 bytes per 32-element block. Positions 0-15 are low nibbles
// of bytes 0-15, positions 16-31 are high nibbles of bytes 0-15.
// Given an even pos0 and pos1 = pos0+1 (always same half), extract both nibbles
// from one byte load (if pos0 is odd in a byte pair) or two adjacent byte loads.
//
// Key insight: since elem is always even, pos0 is always even.
// Even pos → both pos0 and pos1 are in the same nibble-half (both low or both high).
// We can extract with: byte = qs[pos0 & 15], shift = (pos0 >= 16) ? 4 : 0.
// ============================================================================

static __device__ __forceinline__ void mxfp4_extract_nibble_pair(
        const uint8_t * qs, int pos0, uint8_t & nib0, uint8_t & nib1) {
    // pos0 is always even, pos1 = pos0+1. Both are in the same half (low or high nibbles).
    const int shift = (pos0 >= 16) ? 4 : 0;
    const int bi0 = pos0 & 15;  // byte index: equivalent to pos < 16 ? pos : pos - 16
    const int bi1 = bi0 + 1;    // pos1 = pos0+1, always same half, so bi1 = bi0+1
    nib0 = (qs[bi0] >> shift) & 0x0F;
    nib1 = (qs[bi1] >> shift) & 0x0F;
}

// ============================================================================
// MXFP6 optimized pair dequant from SoA layout.
// 32 elements packed as 8 groups of 4, each group = 3 bytes.
// For an even pos0, pos0 and pos1=pos0+1 are always in the same group
// (since even/odd pairs only cross groups at pos0=3→pos1=4, 7→8, etc.,
// but pos0 is always even so this never happens: even%4 ∈ {0,2}, never 3).
// This means we ALWAYS unpack one group and extract two elements.
// ============================================================================

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

// ============================================================================
// Unified MXFP SoA pair dequant: extract + scale + convert two consecutive
// elements from a single 32-element block.  Encapsulates all type-specific
// logic (nibble extraction, fp6 unpacking, byte reads) and the intrinsic vs
// portable fallback selection.
//
// qs_block: pointer to this block's qs region (qs_base + blk * qs_per_blk)
// e8m0:     the raw E8M0 scale byte for this block
// pos0:     position of first element within the block (MUST be even)
// Returns:  float2{elem[pos0], elem[pos0+1]} fully dequantized
// ============================================================================

template <ggml_type type>
static __device__ __forceinline__ float2 mxfp_dequant_elem_pair(
        const uint8_t * qs_block, uint8_t e8m0, int pos0);

template <>
__device__ __forceinline__ float2 mxfp_dequant_elem_pair<GGML_TYPE_MXFP4>(
        const uint8_t * qs_block, uint8_t e8m0, int pos0) {
#if CUDART_VERSION >= 12080
    // x2 intrinsic: extract both nibbles as a packed byte, convert in one instruction.
    // SoA layout: pos0 < 16 → low nibbles, pos0 >= 16 → high nibbles.
    // nib0 is in bits [0:3] or [4:7], nib1 in the adjacent byte's same half.
    // Pack nib0 in low nibble, nib1 in high nibble for fp4x2 intrinsic.
    const int shift = (pos0 >= 16) ? 4 : 0;
    const int bi0 = pos0 & 15;
    const uint8_t packed = ((qs_block[bi0] >> shift) & 0x0F) | (((qs_block[bi0 + 1] >> shift) & 0x0F) << 4);
    const float d = ggml_cuda_e8m0_to_fp32(e8m0);
    const float2 raw = mxfp4_dequant_x2_intrinsic(packed);
    return make_float2(raw.x * d, raw.y * d);
#else
    uint8_t nib0, nib1;
    mxfp4_extract_nibble_pair(qs_block, pos0, nib0, nib1);
    const float d = ggml_mxfp_e8m0_to_fp32_half(e8m0);
    return make_float2(kvalues_mxfp4[nib0] * d,
                       kvalues_mxfp4[nib1] * d);
#endif
}

template <>
__device__ __forceinline__ float2 mxfp_dequant_elem_pair<GGML_TYPE_MXFP8>(
        const uint8_t * qs_block, uint8_t e8m0, int pos0) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0);
#if CUDART_VERSION >= 12080
    // x2 intrinsic: load two consecutive bytes as uint16, convert in one instruction.
    const __nv_fp8x2_storage_t x2 = *reinterpret_cast<const __nv_fp8x2_storage_t *>(qs_block + pos0);
    const float2 raw = mxfp8_dequant_x2_intrinsic(x2);
    return make_float2(raw.x * d, raw.y * d);
#else
    return make_float2(ggml_mxfp_fp8_e4m3_to_float(qs_block[pos0])     * d,
                       ggml_mxfp_fp8_e4m3_to_float(qs_block[pos0 + 1]) * d);
#endif
}

template <>
__device__ __forceinline__ float2 mxfp_dequant_elem_pair<GGML_TYPE_MXFP6>(
        const uint8_t * qs_block, uint8_t e8m0, int pos0) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0);
    uint8_t v0, v1;
    mxfp6_unpack_pair(qs_block, pos0, v0, v1);
#if CUDART_VERSION >= 12080
    // x2 intrinsic: pack two 6-bit values into uint16, convert in one instruction.
    const __nv_fp6x2_storage_t x2 = (__nv_fp6x2_storage_t)(v0 | ((uint16_t)v1 << 8));
    const float2 raw = mxfp6_dequant_x2_intrinsic(x2);
    return make_float2(raw.x * d, raw.y * d);
#else
    return make_float2(ggml_mxfp_fp6_e2m3_to_float(v0) * d,
                       ggml_mxfp_fp6_e2m3_to_float(v1) * d);
#endif
}

// ============================================================================
// MXFP SoA single-element dequant for SoA→F16 conversion.
// Unlike the pair variant, this handles arbitrary positions (not just even).
// ============================================================================

template <ggml_type type>
static __device__ __forceinline__ float mxfp_dequant_elem(
        const uint8_t * qs_base, const uint8_t * e8m0_base,
        int blk, int pos, int qs_per_blk);

template <>
__device__ __forceinline__ float mxfp_dequant_elem<GGML_TYPE_MXFP4>(
        const uint8_t * qs_base, const uint8_t * e8m0_base,
        int blk, int pos, int qs_per_blk) {
    const uint8_t * qs = qs_base + blk * qs_per_blk;
    const int byte_idx = pos < 16 ? pos : pos - 16;
    const uint8_t nibble = (pos < 16) ? (qs[byte_idx] & 0x0F) : (qs[byte_idx] >> 4);
#if CUDART_VERSION >= 12080
    const float d = ggml_cuda_e8m0_to_fp32(e8m0_base[blk]);
    return mxfp4_dequant_intrinsic(nibble) * d;
#else
    const float d = ggml_mxfp_e8m0_to_fp32_half(e8m0_base[blk]);
    return kvalues_mxfp4[nibble] * d;
#endif
}

template <>
__device__ __forceinline__ float mxfp_dequant_elem<GGML_TYPE_MXFP8>(
        const uint8_t * qs_base, const uint8_t * e8m0_base,
        int blk, int pos, int qs_per_blk) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0_base[blk]);
#if CUDART_VERSION >= 12080
    return mxfp8_dequant_intrinsic(qs_base[blk * qs_per_blk + pos]) * d;
#else
    return ggml_mxfp_fp8_e4m3_to_float(qs_base[blk * qs_per_blk + pos]) * d;
#endif
}

template <>
__device__ __forceinline__ float mxfp_dequant_elem<GGML_TYPE_MXFP6>(
        const uint8_t * qs_base, const uint8_t * e8m0_base,
        int blk, int pos, int qs_per_blk) {
    const float d = ggml_cuda_e8m0_to_fp32(e8m0_base[blk]);
    const int grp  = pos / 4;
    const int slot = pos % 4;
    uint8_t vals[4];
    ggml_mxfp_unpack_fp6x4(qs_base + blk * qs_per_blk + grp * 3, vals);
#if CUDART_VERSION >= 12080
    return mxfp6_dequant_intrinsic(vals[slot]) * d;
#else
    return ggml_mxfp_fp6_e2m3_to_float(vals[slot]) * d;
#endif
}

// ============================================================================
// MXFP SoA → F16 dequant kernel for MMA flash attention pre-conversion.
// Templatized on MXFP type for compile-time optimization.
// Reads SoA layout [qs0..qsN | e0..eN] per row, writes contiguous F16 output.
// ============================================================================

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

    // Matches CPU mxfp_row_ptr: multihead skips head offset in row pointer,
    // and mxfp_dequant_head extracts per-head data within the row.
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
// Matches CPU mxfp_kv_params_init for multihead detection.
static void mxfp_soa_to_f16_cuda(
        const char * src, half * dst, ggml_type type,
        int64_t D, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, cudaStream_t stream) {

    const int blocks_per_head = (int)(D / 32);

    // Multihead detection — matches CPU mxfp_kv_params_init exactly:
    // multihead = (nb2 == ggml_row_size(type, D))
    const bool multihead = (nb2 == (size_t)ggml_row_size(type, D));

    int qs_per_blk;
    switch (type) {
        case GGML_TYPE_MXFP4: qs_per_blk = MXFP4_SOA_QS_PER_BLOCK; break;
        case GGML_TYPE_MXFP8: qs_per_blk = MXFP8_SOA_QS_PER_BLOCK; break;
        case GGML_TYPE_MXFP6: qs_per_blk = MXFP6_SOA_QS_PER_BLOCK; break;
        default: GGML_ABORT("unsupported MXFP type"); return;
    }

    const int head_qs_bytes = blocks_per_head * qs_per_blk;
    const int total_blocks = multihead ? (int)ne2 * blocks_per_head : blocks_per_head;
    const int head_e8m0_offset = total_blocks * qs_per_blk;

    const int64_t nrows = ne1 * ne2 * ne3;

    // One CUDA block per row, D threads per block (D <= 256 for FA)
    const int threads = (int)D;
    const int grid    = (int)nrows;
    if (grid <= 0 || threads <= 0) return;

    switch (type) {
        case GGML_TYPE_MXFP4:
            k_mxfp_soa_to_f16<GGML_TYPE_MXFP4><<<grid, threads, 0, stream>>>(
                src, dst, (int)D, ne1, ne2, ne3,
                (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
                blocks_per_head, multihead, head_qs_bytes, head_e8m0_offset);
            break;
        case GGML_TYPE_MXFP8:
            k_mxfp_soa_to_f16<GGML_TYPE_MXFP8><<<grid, threads, 0, stream>>>(
                src, dst, (int)D, ne1, ne2, ne3,
                (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
                blocks_per_head, multihead, head_qs_bytes, head_e8m0_offset);
            break;
        case GGML_TYPE_MXFP6:
            k_mxfp_soa_to_f16<GGML_TYPE_MXFP6><<<grid, threads, 0, stream>>>(
                src, dst, (int)D, ne1, ne2, ne3,
                (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
                blocks_per_head, multihead, head_qs_bytes, head_e8m0_offset);
            break;
        default:
            break;
    }
}

// ============================================================================
// MXFP Q Hadamard + roundtrip kernel for MMA flash attention pre-processing.
// Templatized on MXFP type. Uses shared mxfp_hadamard_roundtrip template.
// Operates on contiguous F32 Q rows in-place.
// One warp per 32-element block (same pattern as VEC kernel Q path and set_rows).
// ============================================================================

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

    switch (k_type) {
        case GGML_TYPE_MXFP4:
            k_mxfp_q_hadamard_roundtrip<GGML_TYPE_MXFP4><<<grid, threads, 0, stream>>>(
                Q, (int)D, ne1, ne2, ne3, s01, s02, s03);
            break;
        case GGML_TYPE_MXFP8:
            k_mxfp_q_hadamard_roundtrip<GGML_TYPE_MXFP8><<<grid, threads, 0, stream>>>(
                Q, (int)D, ne1, ne2, ne3, s01, s02, s03);
            break;
        case GGML_TYPE_MXFP6:
            k_mxfp_q_hadamard_roundtrip<GGML_TYPE_MXFP6><<<grid, threads, 0, stream>>>(
                Q, (int)D, ne1, ne2, ne3, s01, s02, s03);
            break;
        default:
            GGML_ABORT("unsupported MXFP type");
            break;
    }
}
