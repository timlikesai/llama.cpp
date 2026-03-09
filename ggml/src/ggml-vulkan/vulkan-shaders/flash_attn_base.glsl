
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;

// V type ID for mixed K/V MXFP types (scalar/cm1 paths).
// 0 = matched (V same as K), 1=mxfp4, 2=e4m3, 3=e5m2, 4=e2m3, 5=e3m2
const uint V_TYPE_ID = (Flags >> 4u) & 7u;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1
#if defined(DATA_A_F32)
layout (binding = 1) readonly buffer K_PACKED {vec4 k_data_packed[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED {vec4 v_data_packed[];} v_packed;
#elif defined(DATA_A_MXFP4) || defined(DATA_A_MXFP8_E4M3) || defined(DATA_A_MXFP8_E5M2) || defined(DATA_A_MXFP6_E2M3) || defined(DATA_A_MXFP6_E3M2)
layout (binding = 1) readonly buffer K_RAW {A_TYPE k_data[];} k_raw;
layout (binding = 1) readonly buffer K_U32 {uint k_u32[];} k_raw_u32;
// V buffer uses raw uint access to support any MXFP V type via runtime dispatch
layout (binding = 2) readonly buffer V_U32 {uint v_u32[];} v_raw_u32;
#elif defined(A_TYPE_PACKED16)
layout (binding = 1) readonly buffer K_PACKED16 {A_TYPE_PACKED16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {A_TYPE_PACKED16 v_data_packed16[];} v_packed;
#if defined(MIXED_Q_DEQUANT)
// Raw V access for mixed standard quant K/V (e.g. q8_0 K + q4_0 V)
layout (binding = 2) readonly buffer V_U32_Q {uint v_u32[];} v_raw_u32;
#endif
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

// Branchless E2M1 dequant via constant IEEE-754 bit patterns.
// No shared memory, no branches — just index + OR + uintBitsToFloat.
#if defined(DATA_A_MXFP4) || defined(MXFP_ALL_DEQUANT)
float fp4_e2m1_to_float(uint n) {
    // IEEE-754 bits for the 8 positive E2M1 values:
    // 0→0.0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
    const uint fp4_bits[8] = uint[8](
        0x00000000u, 0x3F000000u, 0x3F800000u, 0x3FC00000u,
        0x40000000u, 0x40400000u, 0x40800000u, 0x40C00000u
    );
    return uintBitsToFloat(fp4_bits[n & 0x7u] | ((n & 0x8u) << 28));
}
#endif

#if defined(DATA_A_F32)
#undef BLOCK_SIZE
#define BLOCK_SIZE 4
#define BLOCK_BYTE_SIZE 16

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // iqs is currently always zero in the flash attention shaders
    if (binding_idx == BINDING_IDX_K) {
        return FLOAT_TYPEV4(k_packed.k_data_packed[a_offset + ib]);
    } else {
        return FLOAT_TYPEV4(v_packed.v_data_packed[a_offset + ib]);
    }
}
#endif

#if defined(DATA_A_Q4_0)
#define BLOCK_BYTE_SIZE 18

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - FLOAT_TYPE(8.0f));
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - FLOAT_TYPE(8.0f));
    }
}
#endif

#if defined(DATA_A_Q8_0)
#define BLOCK_BYTE_SIZE 34
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        const i8vec2 v0 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    } else {
        const i8vec2 v0 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    }
}
#endif

#if defined(DATA_A_MXFP4)
#define BLOCK_BYTE_SIZE 17
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    const uint idx = a_offset + ib;
    // iqs is element index (0-31 in steps of 4).
    const uint iqs0 = iqs & 0xFu;           // byte index (wraps at 16)
    const uint shift = (iqs & 0x10u) >> 2;   // 0 for elements 0-15, 4 for 16-31
    // Load 4 qs bytes as one uint32 via unaligned read (1-2 loads vs 4 byte loads).
    // Scale byte at struct offset 0, qs at offset 1.
    const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + iqs0;
    if (binding_idx == BINDING_IDX_K) {
        const float d = e8m0_to_fp32(k_raw.k_data[idx].e);
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint packed = k_raw_u32.k_u32[w] >> s;
        if (s != 0u) packed |= k_raw_u32.k_u32[w + 1u] << (32u - s);
        return FLOAT_TYPEV4(
            fp4_e2m1_to_float((packed >>  0) >> shift & 0xFu) * d,
            fp4_e2m1_to_float((packed >>  8) >> shift & 0xFu) * d,
            fp4_e2m1_to_float((packed >> 16) >> shift & 0xFu) * d,
            fp4_e2m1_to_float((packed >> 24) >> shift & 0xFu) * d
        );
    } else {
        const uint v_byte_off = idx * BLOCK_BYTE_SIZE;
        const uint ew = v_byte_off >> 2u;
        const uint es = (v_byte_off & 3u) << 3u;
        const float d = e8m0_to_fp32(uint8_t((v_raw_u32.v_u32[ew] >> es) & 0xFFu));
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint packed = v_raw_u32.v_u32[w] >> s;
        if (s != 0u) packed |= v_raw_u32.v_u32[w + 1u] << (32u - s);
        return FLOAT_TYPEV4(
            fp4_e2m1_to_float((packed >>  0) >> shift & 0xFu) * d,
            fp4_e2m1_to_float((packed >>  8) >> shift & 0xFu) * d,
            fp4_e2m1_to_float((packed >> 16) >> shift & 0xFu) * d,
            fp4_e2m1_to_float((packed >> 24) >> shift & 0xFu) * d
        );
    }
}
#endif

#if defined(DATA_A_MXFP8_E4M3)
#define BLOCK_BYTE_SIZE 33
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    const uint idx = a_offset + ib;
    if (binding_idx == BINDING_IDX_K) {
        const float d = e8m0_to_fp32(k_raw.k_data[idx].e);
        // Load 4 consecutive qs bytes as one uint32
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + iqs;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint packed = k_raw_u32.k_u32[w] >> s;
        if (s != 0u) packed |= k_raw_u32.k_u32[w + 1u] << (32u - s);
        return FLOAT_TYPEV4(
            d * fp8_e4m3_to_float(packed & 0xFFu),
            d * fp8_e4m3_to_float((packed >> 8) & 0xFFu),
            d * fp8_e4m3_to_float((packed >> 16) & 0xFFu),
            d * fp8_e4m3_to_float((packed >> 24) & 0xFFu)
        );
    } else {
        const uint v_byte_off = idx * BLOCK_BYTE_SIZE;
        const uint ew = v_byte_off >> 2u;
        const uint es = (v_byte_off & 3u) << 3u;
        const float d = e8m0_to_fp32(uint8_t((v_raw_u32.v_u32[ew] >> es) & 0xFFu));
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + iqs;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint packed = v_raw_u32.v_u32[w] >> s;
        if (s != 0u) packed |= v_raw_u32.v_u32[w + 1u] << (32u - s);
        return FLOAT_TYPEV4(
            d * fp8_e4m3_to_float(packed & 0xFFu),
            d * fp8_e4m3_to_float((packed >> 8) & 0xFFu),
            d * fp8_e4m3_to_float((packed >> 16) & 0xFFu),
            d * fp8_e4m3_to_float((packed >> 24) & 0xFFu)
        );
    }
}
#endif

#if defined(DATA_A_MXFP8_E5M2)
#define BLOCK_BYTE_SIZE 33
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    const uint idx = a_offset + ib;
    if (binding_idx == BINDING_IDX_K) {
        const float d = e8m0_to_fp32(k_raw.k_data[idx].e);
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + iqs;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint packed = k_raw_u32.k_u32[w] >> s;
        if (s != 0u) packed |= k_raw_u32.k_u32[w + 1u] << (32u - s);
        return FLOAT_TYPEV4(
            d * fp8_e5m2_to_float(packed & 0xFFu),
            d * fp8_e5m2_to_float((packed >> 8) & 0xFFu),
            d * fp8_e5m2_to_float((packed >> 16) & 0xFFu),
            d * fp8_e5m2_to_float((packed >> 24) & 0xFFu)
        );
    } else {
        const uint v_byte_off = idx * BLOCK_BYTE_SIZE;
        const uint ew = v_byte_off >> 2u;
        const uint es = (v_byte_off & 3u) << 3u;
        const float d = e8m0_to_fp32(uint8_t((v_raw_u32.v_u32[ew] >> es) & 0xFFu));
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + iqs;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint packed = v_raw_u32.v_u32[w] >> s;
        if (s != 0u) packed |= v_raw_u32.v_u32[w + 1u] << (32u - s);
        return FLOAT_TYPEV4(
            d * fp8_e5m2_to_float(packed & 0xFFu),
            d * fp8_e5m2_to_float((packed >> 8) & 0xFFu),
            d * fp8_e5m2_to_float((packed >> 16) & 0xFFu),
            d * fp8_e5m2_to_float((packed >> 24) & 0xFFu)
        );
    }
}
#endif

#if defined(DATA_A_MXFP6_E2M3)
#define BLOCK_BYTE_SIZE 25
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    const uint idx = a_offset + ib;
    const uint group = iqs / 4u;
    const uint base = group * 3u;
    if (binding_idx == BINDING_IDX_K) {
        const float d = e8m0_to_fp32(k_raw.k_data[idx].e);
        // Load 3 packed bytes as uint32 and extract
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + base;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint raw = k_raw_u32.k_u32[w] >> s;
        if (s > 8u) raw |= k_raw_u32.k_u32[w + 1u] << (32u - s);
        uint v0, v1, v2, v3;
        unpack_fp6x4(raw & 0xFFu, (raw >> 8) & 0xFFu, (raw >> 16) & 0xFFu, v0, v1, v2, v3);
        return FLOAT_TYPEV4(
            d * fp6_e2m3_to_float(v0),
            d * fp6_e2m3_to_float(v1),
            d * fp6_e2m3_to_float(v2),
            d * fp6_e2m3_to_float(v3)
        );
    } else {
        const uint v_byte_off = idx * BLOCK_BYTE_SIZE;
        const uint ew = v_byte_off >> 2u;
        const uint es = (v_byte_off & 3u) << 3u;
        const float d = e8m0_to_fp32(uint8_t((v_raw_u32.v_u32[ew] >> es) & 0xFFu));
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + base;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint raw = v_raw_u32.v_u32[w] >> s;
        if (s > 8u) raw |= v_raw_u32.v_u32[w + 1u] << (32u - s);
        uint v0, v1, v2, v3;
        unpack_fp6x4(raw & 0xFFu, (raw >> 8) & 0xFFu, (raw >> 16) & 0xFFu, v0, v1, v2, v3);
        return FLOAT_TYPEV4(
            d * fp6_e2m3_to_float(v0),
            d * fp6_e2m3_to_float(v1),
            d * fp6_e2m3_to_float(v2),
            d * fp6_e2m3_to_float(v3)
        );
    }
}
#endif

#if defined(DATA_A_MXFP6_E3M2)
#define BLOCK_BYTE_SIZE 25
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    const uint idx = a_offset + ib;
    const uint group = iqs / 4u;
    const uint base = group * 3u;
    if (binding_idx == BINDING_IDX_K) {
        const float d = e8m0_to_fp32(k_raw.k_data[idx].e);
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + base;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint raw = k_raw_u32.k_u32[w] >> s;
        if (s > 8u) raw |= k_raw_u32.k_u32[w + 1u] << (32u - s);
        uint v0, v1, v2, v3;
        unpack_fp6x4(raw & 0xFFu, (raw >> 8) & 0xFFu, (raw >> 16) & 0xFFu, v0, v1, v2, v3);
        return FLOAT_TYPEV4(
            d * fp6_e3m2_to_float(v0),
            d * fp6_e3m2_to_float(v1),
            d * fp6_e3m2_to_float(v2),
            d * fp6_e3m2_to_float(v3)
        );
    } else {
        const uint v_byte_off = idx * BLOCK_BYTE_SIZE;
        const uint ew = v_byte_off >> 2u;
        const uint es = (v_byte_off & 3u) << 3u;
        const float d = e8m0_to_fp32(uint8_t((v_raw_u32.v_u32[ew] >> es) & 0xFFu));
        const uint byte_off = idx * BLOCK_BYTE_SIZE + 1u + base;
        const uint w = byte_off >> 2u;
        const uint s = (byte_off & 3u) << 3u;
        uint raw = v_raw_u32.v_u32[w] >> s;
        if (s > 8u) raw |= v_raw_u32.v_u32[w + 1u] << (32u - s);
        uint v0, v1, v2, v3;
        unpack_fp6x4(raw & 0xFFu, (raw >> 8) & 0xFFu, (raw >> 16) & 0xFFu, v0, v1, v2, v3);
        return FLOAT_TYPEV4(
            d * fp6_e3m2_to_float(v0),
            d * fp6_e3m2_to_float(v1),
            d * fp6_e3m2_to_float(v2),
            d * fp6_e3m2_to_float(v3)
        );
    }
}
#endif

// ===== Mixed V type runtime dispatch (scalar/cm1 paths) =====
#if defined(MXFP_ALL_DEQUANT)

// Helper: read E8M0 scale byte from v_raw_u32 at a given byte offset
float v_e8m0_scale(uint byte_off) {
    uint w = byte_off >> 2u;
    uint s = (byte_off & 3u) << 3u;
    return e8m0_to_fp32(uint8_t((v_raw_u32.v_u32[w] >> s) & 0xFFu));
}

// Helper: read uint32 from v_raw_u32 at unaligned byte offset
uint v_read_u32(uint byte_off) {
    uint w = byte_off >> 2u;
    uint s = (byte_off & 3u) << 3u;
    uint val = v_raw_u32.v_u32[w] >> s;
    if (s != 0u) val |= v_raw_u32.v_u32[w + 1u] << (32u - s);
    return val;
}

FLOAT_TYPEV4 dequantize4_v_mxfp4(uint idx, uint iqs) {
    const uint iqs0 = iqs & 0xFu;
    const uint shift = (iqs & 0x10u) >> 2;
    const float d = v_e8m0_scale(idx * 17u);
    uint packed = v_read_u32(idx * 17u + 1u + iqs0);
    return FLOAT_TYPEV4(
        fp4_e2m1_to_float((packed >>  0) >> shift & 0xFu) * d,
        fp4_e2m1_to_float((packed >>  8) >> shift & 0xFu) * d,
        fp4_e2m1_to_float((packed >> 16) >> shift & 0xFu) * d,
        fp4_e2m1_to_float((packed >> 24) >> shift & 0xFu) * d
    );
}

FLOAT_TYPEV4 dequantize4_v_mxfp8_e4m3(uint idx, uint iqs) {
    const float d = v_e8m0_scale(idx * 33u);
    uint packed = v_read_u32(idx * 33u + 1u + iqs);
    return FLOAT_TYPEV4(
        d * fp8_e4m3_to_float(packed & 0xFFu),
        d * fp8_e4m3_to_float((packed >> 8) & 0xFFu),
        d * fp8_e4m3_to_float((packed >> 16) & 0xFFu),
        d * fp8_e4m3_to_float((packed >> 24) & 0xFFu)
    );
}

FLOAT_TYPEV4 dequantize4_v_mxfp8_e5m2(uint idx, uint iqs) {
    const float d = v_e8m0_scale(idx * 33u);
    uint packed = v_read_u32(idx * 33u + 1u + iqs);
    return FLOAT_TYPEV4(
        d * fp8_e5m2_to_float(packed & 0xFFu),
        d * fp8_e5m2_to_float((packed >> 8) & 0xFFu),
        d * fp8_e5m2_to_float((packed >> 16) & 0xFFu),
        d * fp8_e5m2_to_float((packed >> 24) & 0xFFu)
    );
}

FLOAT_TYPEV4 dequantize4_v_mxfp6_e2m3(uint idx, uint iqs) {
    const uint group = iqs / 4u;
    const uint base = group * 3u;
    const float d = v_e8m0_scale(idx * 25u);
    const uint byte_off = idx * 25u + 1u + base;
    uint w = byte_off >> 2u;
    uint s = (byte_off & 3u) << 3u;
    uint raw = v_raw_u32.v_u32[w] >> s;
    if (s > 8u) raw |= v_raw_u32.v_u32[w + 1u] << (32u - s);
    uint v0, v1, v2, v3;
    unpack_fp6x4(raw & 0xFFu, (raw >> 8) & 0xFFu, (raw >> 16) & 0xFFu, v0, v1, v2, v3);
    return FLOAT_TYPEV4(
        d * fp6_e2m3_to_float(v0),
        d * fp6_e2m3_to_float(v1),
        d * fp6_e2m3_to_float(v2),
        d * fp6_e2m3_to_float(v3)
    );
}

FLOAT_TYPEV4 dequantize4_v_mxfp6_e3m2(uint idx, uint iqs) {
    const uint group = iqs / 4u;
    const uint base = group * 3u;
    const float d = v_e8m0_scale(idx * 25u);
    const uint byte_off = idx * 25u + 1u + base;
    uint w = byte_off >> 2u;
    uint s = (byte_off & 3u) << 3u;
    uint raw = v_raw_u32.v_u32[w] >> s;
    if (s > 8u) raw |= v_raw_u32.v_u32[w + 1u] << (32u - s);
    uint v0, v1, v2, v3;
    unpack_fp6x4(raw & 0xFFu, (raw >> 8) & 0xFFu, (raw >> 16) & 0xFFu, v0, v1, v2, v3);
    return FLOAT_TYPEV4(
        d * fp6_e3m2_to_float(v0),
        d * fp6_e3m2_to_float(v1),
        d * fp6_e3m2_to_float(v2),
        d * fp6_e3m2_to_float(v3)
    );
}

// V dequant dispatcher: routes to the correct V type based on V_TYPE_ID spec constant.
FLOAT_TYPEV4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    const uint idx = a_offset + ib;
    switch (V_TYPE_ID) {
        case 1u: return dequantize4_v_mxfp4(idx, iqs);
        case 2u: return dequantize4_v_mxfp8_e4m3(idx, iqs);
        case 3u: return dequantize4_v_mxfp8_e5m2(idx, iqs);
        case 4u: return dequantize4_v_mxfp6_e2m3(idx, iqs);
        case 5u: return dequantize4_v_mxfp6_e3m2(idx, iqs);
        default: return dequantize4(ib, iqs, a_offset, BINDING_IDX_V);
    }
}

// V block byte size for offset calculations, based on V_TYPE_ID.
uint get_v_block_byte_size() {
    switch (V_TYPE_ID) {
        case 1u: return 17u;  // mxfp4
        case 2u: return 33u;  // mxfp8_e4m3
        case 3u: return 33u;  // mxfp8_e5m2
        case 4u: return 25u;  // mxfp6_e2m3
        case 5u: return 25u;  // mxfp6_e3m2
        default: return BLOCK_BYTE_SIZE;
    }
}

#endif // MXFP_ALL_DEQUANT

// Mixed standard quant K/V: standalone V dequant for q4_0 and q8_0.
// These read from raw uint32 V binding, independent of the K type.
#if defined(MIXED_Q_DEQUANT)

FLOAT_TYPEV4 dequantize4_v_q4_0(uint ib, uint iqs) {
    // q4_0 block: 2 bytes scale (f16) + 16 bytes qs = 18 bytes
    const uint byte_off = ib * 18u;
    // Read f16 scale from first 2 bytes
    const uint scale_word = v_raw_u32.v_u32[byte_off >> 2u];
    const uint scale_shift = (byte_off & 3u) << 3u;
    const uint16_t scale_u16 = uint16_t((scale_word >> scale_shift) & 0xFFFFu);
    const FLOAT_TYPE d = FLOAT_TYPE(unpackHalf2x16(uint(scale_u16)).x);
    // Read qs nibbles: offset 2 bytes into block
    const uint qs_byte_off = byte_off + 2u + (iqs & 0xFu);
    const uint shift = (iqs & 0x10u) >> 2u;
    const uint w0 = qs_byte_off >> 2u;
    const uint s0 = (qs_byte_off & 3u) << 3u;
    uint vui_lo = (v_raw_u32.v_u32[w0] >> s0);
    if (s0 != 0u) vui_lo |= (v_raw_u32.v_u32[w0 + 1u] << (32u - s0));
    const uint w1 = (qs_byte_off + 2u) >> 2u;
    const uint s1 = ((qs_byte_off + 2u) & 3u) << 3u;
    uint vui_hi = (v_raw_u32.v_u32[w1] >> s1);
    if (s1 != 0u) vui_hi |= (v_raw_u32.v_u32[w1 + 1u] << (32u - s1));
    vui_lo >>= shift;
    vui_hi >>= shift;
    return d * (FLOAT_TYPEV4(vui_lo & 0xFu, (vui_lo >> 8u) & 0xFu, vui_hi & 0xFu, (vui_hi >> 8u) & 0xFu) - FLOAT_TYPE(8.0));
}

FLOAT_TYPEV4 dequantize4_v_q8_0(uint ib, uint iqs) {
    // q8_0 block: 2 bytes scale (f16) + 32 bytes qs = 34 bytes
    const uint byte_off = ib * 34u;
    // Read f16 scale
    const uint scale_word = v_raw_u32.v_u32[byte_off >> 2u];
    const uint scale_shift = (byte_off & 3u) << 3u;
    const uint16_t scale_u16 = uint16_t((scale_word >> scale_shift) & 0xFFFFu);
    const FLOAT_TYPE d = FLOAT_TYPE(unpackHalf2x16(uint(scale_u16)).x);
    // Read qs bytes: offset 2 bytes into block
    const uint qs_byte_off = byte_off + 2u + iqs;
    const uint w = qs_byte_off >> 2u;
    const uint s = (qs_byte_off & 3u) << 3u;
    uint packed = v_raw_u32.v_u32[w] >> s;
    if (s != 0u) packed |= v_raw_u32.v_u32[w + 1u] << (32u - s);
    const i8vec2 v0 = unpack8(int32_t(packed & 0xFFFFu)).xy;
    const i8vec2 v1 = unpack8(int32_t((packed >> 16u) & 0xFFFFu)).xy;
    return d * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
}

// V dequant dispatcher for mixed standard quant K/V.
FLOAT_TYPEV4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    const uint idx = a_offset + ib;
    switch (V_TYPE_ID) {
        case 6u: return dequantize4_v_q4_0(idx, iqs);
        case 7u: return dequantize4_v_q8_0(idx, iqs);
        default: return dequantize4(ib, iqs, a_offset, BINDING_IDX_V);
    }
}

uint get_v_block_byte_size() {
    switch (V_TYPE_ID) {
        case 6u: return 18u;  // q4_0
        case 7u: return 34u;  // q8_0
        default: return BLOCK_BYTE_SIZE;
    }
}

#endif // MIXED_Q_DEQUANT

// K and V may have different block byte sizes when using mixed K/V types.
#ifdef BLOCK_BYTE_SIZE
#define K_BLOCK_BYTE_SIZE BLOCK_BYTE_SIZE
// V_BLOCK_BYTE_SIZE: for matched types use K's block size.
// For mixed types with MXFP_ALL_DEQUANT or MIXED_Q_DEQUANT, get_v_block_byte_size() is used at runtime.
#define V_BLOCK_BYTE_SIZE BLOCK_BYTE_SIZE
#endif

// ===== MXFP Q preprocessing for flash attention =====
// Hadamard rotation + quantize/dequantize round-trip on Q values.
// This improves perplexity when using MXFP KV cache quantization.
#if defined(DATA_A_MXFP4) || defined(DATA_A_MXFP8_E4M3) || defined(DATA_A_MXFP8_E5M2) || defined(DATA_A_MXFP6_E2M3) || defined(DATA_A_MXFP6_E3M2)
#define MXFP_Q_PREPROCESS

// In-place Hadamard transform on 32 elements in shared memory.
// shmem points to the start of the 32-element block (as 8 vec4s).
// Each thread in the workgroup processes assigned elements.
void hadamard_32_shmem(inout float vals[32]) {
    // 5 stages of butterfly operations for 32-point Hadamard
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
        for (uint i = 0u; i < 32u; i++) {
            if ((i & stride) == 0u) {
                float a = vals[i];
                float b = vals[i | stride];
                vals[i]          = a + b;
                vals[i | stride] = a - b;
            }
        }
    }
    const float norm = 0.17677669529663689; // 1/sqrt(32)
    for (uint i = 0u; i < 32u; i++) {
        vals[i] *= norm;
    }
}

// MXFP4 quantize/dequantize round-trip for a single value given a scale
float mxfp4_roundtrip_val(float val, float scale) {
    if (scale == 0.0) return 0.0;
    float av = abs(val) / scale;
    // Decision boundary quantization with direct dequant (no array lookup)
    float dq;
    if      (av < 0.25)  dq = 0.0;
    else if (av < 0.75)  dq = 0.5;
    else if (av < 1.25)  dq = 1.0;
    else if (av < 1.75)  dq = 1.5;
    else if (av < 2.5)   dq = 2.0;
    else if (av < 3.5)   dq = 3.0;
    else if (av < 5.0)   dq = 4.0;
    else                  dq = 6.0;
    dq *= scale;
    return val < 0.0 ? -dq : dq;
}

// MXFP8 E4M3 quantize/dequantize round-trip
float mxfp8_e4m3_roundtrip_val(float val, float scale) {
    if (scale == 0.0) return 0.0;
    float scaled = val / scale;
    // Clamp to E4M3 range: [-448, 448]
    scaled = clamp(scaled, -448.0, 448.0);
    // Round to nearest representable E4M3 value via float conversion
    // E4M3: 4-bit exponent (bias 7), 3-bit mantissa
    uint bits = floatBitsToUint(scaled);
    uint sign = bits & 0x80000000u;
    uint abs_bits = bits & 0x7FFFFFFFu;
    // Extract and rebias exponent from float32 (bias 127) to E4M3 (bias 7)
    int exp32 = int((abs_bits >> 23) & 0xFFu) - 127;
    uint mant32 = abs_bits & 0x7FFFFFu; // 23-bit mantissa
    int exp8 = exp32 + 7;
    if (exp8 < 0) return 0.0; // underflow
    if (exp8 > 15) exp8 = 15; // overflow clamp
    // Round mantissa to 3 bits (from 23)
    uint mant8 = (mant32 + (1u << 19)) >> 20; // round to nearest
    if (mant8 >= 8u) { mant8 = 0u; exp8++; if (exp8 > 15) exp8 = 15; }
    // Dequantize back: reconstruct float from E4M3 bits
    float result;
    if (exp8 == 0) {
        result = float(mant8) * exp2(-9.0); // subnormal: 2^(1-7) * mant/8 = 2^-6 * mant/8
    } else {
        result = (1.0 + float(mant8) / 8.0) * exp2(float(exp8 - 7));
    }
    return (sign != 0u ? -result : result) * scale;
}

// MXFP8 E5M2 quantize/dequantize round-trip
float mxfp8_e5m2_roundtrip_val(float val, float scale) {
    if (scale == 0.0) return 0.0;
    float scaled = val / scale;
    scaled = clamp(scaled, -57344.0, 57344.0);
    uint bits = floatBitsToUint(scaled);
    uint sign = bits & 0x80000000u;
    uint abs_bits = bits & 0x7FFFFFFFu;
    int exp32 = int((abs_bits >> 23) & 0xFFu) - 127;
    uint mant32 = abs_bits & 0x7FFFFFu;
    int exp8 = exp32 + 15;
    if (exp8 < 0) return 0.0;
    if (exp8 > 30) exp8 = 30; // E5M2 max normal exponent
    uint mant8 = (mant32 + (1u << 20)) >> 21;
    if (mant8 >= 4u) { mant8 = 0u; exp8++; if (exp8 > 30) exp8 = 30; }
    float result;
    if (exp8 == 0) {
        result = float(mant8) * exp2(-16.0);
    } else {
        result = (1.0 + float(mant8) / 4.0) * exp2(float(exp8 - 15));
    }
    return (sign != 0u ? -result : result) * scale;
}

// MXFP6 E2M3 quantize/dequantize round-trip
float mxfp6_e2m3_roundtrip_val(float val, float scale) {
    if (scale == 0.0) return 0.0;
    float scaled = val / scale;
    scaled = clamp(scaled, -7.5, 7.5);
    uint sign = floatBitsToUint(scaled) & 0x80000000u;
    float av = abs(scaled);
    // E2M3: 2-bit exponent (bias 1), 3-bit mantissa
    // Subnormal: value = mant/8, range [0, 0.875]
    // Normal:    value = (1 + mant/8) * 2^(exp-1), exp 1-3
    float result;
    if (av < 0.0625) {
        // Below smallest subnormal (0.125), round to 0
        result = 0.0;
    } else if (av < 1.0) {
        // Subnormal range: round to nearest mant/8
        uint mant6 = uint(av * 8.0 + 0.5);
        if (mant6 >= 8u) { mant6 = 0u; result = 1.0; }  // rounds up to smallest normal
        else { result = float(mant6) / 8.0; }
    } else {
        int exp32 = int((floatBitsToUint(av) >> 23) & 0xFFu) - 127;
        uint mant32 = floatBitsToUint(av) & 0x7FFFFFu;
        int exp6 = exp32 + 1;
        if (exp6 > 3) exp6 = 3;
        uint mant6 = (mant32 + (1u << 19)) >> 20;
        if (mant6 >= 8u) { mant6 = 0u; exp6++; if (exp6 > 3) exp6 = 3; }
        result = (1.0 + float(mant6) / 8.0) * exp2(float(exp6 - 1));
    }
    return (sign != 0u ? -result : result) * scale;
}

// MXFP6 E3M2 quantize/dequantize round-trip
float mxfp6_e3m2_roundtrip_val(float val, float scale) {
    if (scale == 0.0) return 0.0;
    float scaled = val / scale;
    scaled = clamp(scaled, -28.0, 28.0);
    uint sign = floatBitsToUint(scaled) & 0x80000000u;
    float av = abs(scaled);
    // E3M2: 3-bit exponent (bias 3), 2-bit mantissa
    // Subnormal: value = mant * 2^(-2) / 4 = mant/16, range [0, 0.1875]
    // Normal:    value = (1 + mant/4) * 2^(exp-3), exp 1-7
    float result;
    if (av < 0.03125) {
        // Below smallest subnormal (0.0625), round to 0
        result = 0.0;
    } else if (av < 0.25) {
        // Subnormal range: round to nearest mant/16
        uint mant6 = uint(av * 16.0 + 0.5);
        if (mant6 >= 4u) { mant6 = 0u; result = 0.25; }  // rounds up to smallest normal
        else { result = float(mant6) / 16.0; }
    } else {
        int exp32 = int((floatBitsToUint(av) >> 23) & 0xFFu) - 127;
        uint mant32 = floatBitsToUint(av) & 0x7FFFFFu;
        int exp6 = exp32 + 3;
        if (exp6 > 7) exp6 = 7;
        uint mant6 = (mant32 + (1u << 20)) >> 21;
        if (mant6 >= 4u) { mant6 = 0u; exp6++; if (exp6 > 7) exp6 = 7; }
        result = (1.0 + float(mant6) / 4.0) * exp2(float(exp6 - 3));
    }
    return (sign != 0u ? -result : result) * scale;
}

// MXFP quantize/dequantize round-trip dispatcher
float mxfp_roundtrip_val(float val, float scale) {
#if defined(DATA_A_MXFP4)
    return mxfp4_roundtrip_val(val, scale);
#elif defined(DATA_A_MXFP8_E4M3)
    return mxfp8_e4m3_roundtrip_val(val, scale);
#elif defined(DATA_A_MXFP8_E5M2)
    return mxfp8_e5m2_roundtrip_val(val, scale);
#elif defined(DATA_A_MXFP6_E2M3)
    return mxfp6_e2m3_roundtrip_val(val, scale);
#elif defined(DATA_A_MXFP6_E3M2)
    return mxfp6_e3m2_roundtrip_val(val, scale);
#endif
}

// MSE-optimal E8M0 scale for Q preprocessing.
// Tests ±R candidates around round(log2(amax)), picks lowest round-trip MSE.
// Matches CUDA MMA Q and CPU paths for consistent quantization quality.
// Cost: (2R+1) × 32 roundtrip evals per block — negligible vs attention compute.
// Constants (MXFP_E8M0_MSE_RANGE, *_EMAX_OFFSET) injected from vulkan-shaders-gen.cpp.
#if defined(DATA_A_MXFP4)
    #define MXFP_EMAX_OFFSET MXFP4_E2M1_EMAX_OFFSET
#elif defined(DATA_A_MXFP8_E4M3)
    #define MXFP_EMAX_OFFSET MXFP8_E4M3_EMAX_OFFSET
#elif defined(DATA_A_MXFP8_E5M2)
    #define MXFP_EMAX_OFFSET MXFP8_E5M2_EMAX_OFFSET
#elif defined(DATA_A_MXFP6_E2M3)
    #define MXFP_EMAX_OFFSET MXFP6_E2M3_EMAX_OFFSET
#elif defined(DATA_A_MXFP6_E3M2)
    #define MXFP_EMAX_OFFSET MXFP6_E3M2_EMAX_OFFSET
#endif

float compute_e8m0_scale(float vals[32]) {
    float amax = 0.0;
    for (uint i = 0u; i < 32u; i++) {
        amax = max(amax, abs(vals[i]));
    }
    if (amax == 0.0) return 0.0;

    uint amax_bits = floatBitsToUint(amax);
    int floor_log2 = int((amax_bits >> 23) & 0xFFu) - 127;
    int round_log2 = floor_log2 + (((amax_bits & 0x7FFFFFu) >= 0x3504F3u) ? 1 : 0);

    const int emax_offset = MXFP_EMAX_OFFSET;

    int e_base = round_log2 - emax_offset + 127;
    int e_lo = clamp(e_base - MXFP_E8M0_MSE_RANGE, 1, 254);
    int e_hi = clamp(e_base + MXFP_E8M0_MSE_RANGE, 1, 254);
    int best_e = clamp(e_base, 0, 254);
    float best_mse = 1.0e30;

    for (int test_e = e_lo; test_e <= e_hi; ++test_e) {
        float test_scale = (test_e == 0) ? 0.0 : uintBitsToFloat(uint(test_e) << 23);
        float mse = 0.0;
        for (uint j = 0u; j < 32u; ++j) {
            float recon = mxfp_roundtrip_val(vals[j], test_scale);
            float err = vals[j] - recon;
            mse += err * err;
        }
        if (mse < best_mse) {
            best_mse = mse;
            best_e = test_e;
        }
    }

    return (best_e == 0) ? 0.0 : uintBitsToFloat(uint(best_e) << 23);
}

// Subgroup-parallel MSE-optimal E8M0 scale: each lane holds one element of a 32-element block.
// Uses shuffle-tree reduction for MSE sum — matches CPU/CUDA/Metal MSE search quality.
// amax must already be reduced across the subgroup (all lanes hold the same value).
float compute_e8m0_scale_subgroup(float val, float amax) {
    if (amax == 0.0) return 0.0;

    uint amax_bits = floatBitsToUint(amax);
    int floor_log2 = int((amax_bits >> 23) & 0xFFu) - 127;
    int round_log2 = floor_log2 + (((amax_bits & 0x7FFFFFu) >= 0x3504F3u) ? 1 : 0);

    int e_base = round_log2 - MXFP_EMAX_OFFSET + 127;
    int e_lo = clamp(e_base - MXFP_E8M0_MSE_RANGE, 1, 254);
    int e_hi = clamp(e_base + MXFP_E8M0_MSE_RANGE, 1, 254);
    int best_e = clamp(e_base, 0, 254);
    float best_mse = 1.0e30;

    for (int test_e = e_lo; test_e <= e_hi; ++test_e) {
        float test_scale = uintBitsToFloat(uint(test_e) << 23);
        float recon = mxfp_roundtrip_val(val, test_scale);
        float err = val - recon;
        // Shuffle-tree sum across 32 lanes for total MSE
        float mse = err * err;
        mse += subgroupShuffleXor(mse, 1u);
        mse += subgroupShuffleXor(mse, 2u);
        mse += subgroupShuffleXor(mse, 4u);
        mse += subgroupShuffleXor(mse, 8u);
        mse += subgroupShuffleXor(mse, 16u);
        if (mse < best_mse) {
            best_mse = mse;
            best_e = test_e;
        }
    }

    return (best_e == 0) ? 0.0 : uintBitsToFloat(uint(best_e) << 23);
}

#endif // MXFP_Q_PREPROCESS

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
void gqaStore(const in uint32_t r, const in uint32_t c, const in FLOAT_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = D_TYPEV4(elems);
}
