
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
layout (binding = 2) readonly buffer V_RAW {A_TYPE v_data[];} v_raw;
#elif defined(A_TYPE_PACKED16)
layout (binding = 1) readonly buffer K_PACKED16 {A_TYPE_PACKED16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {A_TYPE_PACKED16 v_data_packed16[];} v_packed;
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
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
    A_TYPE blk;
    if (binding_idx == BINDING_IDX_K) {
        blk = k_raw.k_data[a_offset + ib];
    } else {
        blk = v_raw.v_data[a_offset + ib];
    }
    const float d = e8m0_to_fp32(blk.e) * 0.5;
    // CPU layout: lower nibbles of bytes 0-15 → elements 0-15,
    //             upper nibbles of bytes 0-15 → elements 16-31.
    // iqs is element index (0-31 in steps of 4).
    const uint iqs0 = iqs & 0xFu;           // byte index (wraps at 16)
    const uint shift = (iqs & 0x10u) >> 2;   // 0 for elements 0-15, 4 for 16-31
    return FLOAT_TYPEV4(
        kvalues_mxfp4[(uint(blk.qs[iqs0 + 0u]) >> shift) & 0xFu] * d,
        kvalues_mxfp4[(uint(blk.qs[iqs0 + 1u]) >> shift) & 0xFu] * d,
        kvalues_mxfp4[(uint(blk.qs[iqs0 + 2u]) >> shift) & 0xFu] * d,
        kvalues_mxfp4[(uint(blk.qs[iqs0 + 3u]) >> shift) & 0xFu] * d
    );
}
#endif

#if defined(DATA_A_MXFP8_E4M3)
#define BLOCK_BYTE_SIZE 33
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    A_TYPE blk;
    if (binding_idx == BINDING_IDX_K) {
        blk = k_raw.k_data[a_offset + ib];
    } else {
        blk = v_raw.v_data[a_offset + ib];
    }
    const float d = e8m0_to_fp32(blk.e);
    return FLOAT_TYPEV4(
        d * fp8_e4m3_to_float(uint(blk.qs[iqs + 0u])),
        d * fp8_e4m3_to_float(uint(blk.qs[iqs + 1u])),
        d * fp8_e4m3_to_float(uint(blk.qs[iqs + 2u])),
        d * fp8_e4m3_to_float(uint(blk.qs[iqs + 3u]))
    );
}
#endif

#if defined(DATA_A_MXFP8_E5M2)
#define BLOCK_BYTE_SIZE 33
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    A_TYPE blk;
    if (binding_idx == BINDING_IDX_K) {
        blk = k_raw.k_data[a_offset + ib];
    } else {
        blk = v_raw.v_data[a_offset + ib];
    }
    const float d = e8m0_to_fp32(blk.e);
    return FLOAT_TYPEV4(
        d * fp8_e5m2_to_float(uint(blk.qs[iqs + 0u])),
        d * fp8_e5m2_to_float(uint(blk.qs[iqs + 1u])),
        d * fp8_e5m2_to_float(uint(blk.qs[iqs + 2u])),
        d * fp8_e5m2_to_float(uint(blk.qs[iqs + 3u]))
    );
}
#endif

#if defined(DATA_A_MXFP6_E2M3)
#define BLOCK_BYTE_SIZE 25
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    A_TYPE blk;
    if (binding_idx == BINDING_IDX_K) {
        blk = k_raw.k_data[a_offset + ib];
    } else {
        blk = v_raw.v_data[a_offset + ib];
    }
    const float d = e8m0_to_fp32(blk.e);
    // iqs is element index within block; each group of 4 elements = 3 bytes
    uint group = iqs / 4u;
    uint base = group * 3u;
    uint v0, v1, v2, v3;
    unpack_fp6x4(uint(blk.qs[base]), uint(blk.qs[base + 1u]), uint(blk.qs[base + 2u]), v0, v1, v2, v3);
    return FLOAT_TYPEV4(
        d * fp6_e2m3_to_float(v0),
        d * fp6_e2m3_to_float(v1),
        d * fp6_e2m3_to_float(v2),
        d * fp6_e2m3_to_float(v3)
    );
}
#endif

#if defined(DATA_A_MXFP6_E3M2)
#define BLOCK_BYTE_SIZE 25
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    A_TYPE blk;
    if (binding_idx == BINDING_IDX_K) {
        blk = k_raw.k_data[a_offset + ib];
    } else {
        blk = v_raw.v_data[a_offset + ib];
    }
    const float d = e8m0_to_fp32(blk.e);
    uint group = iqs / 4u;
    uint base = group * 3u;
    uint v0, v1, v2, v3;
    unpack_fp6x4(uint(blk.qs[base]), uint(blk.qs[base + 1u]), uint(blk.qs[base + 2u]), v0, v1, v2, v3);
    return FLOAT_TYPEV4(
        d * fp6_e3m2_to_float(v0),
        d * fp6_e3m2_to_float(v1),
        d * fp6_e3m2_to_float(v2),
        d * fp6_e3m2_to_float(v3)
    );
}
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
    // Normalize
    const float norm = 1.0 / sqrt(32.0);
    for (uint i = 0u; i < 32u; i++) {
        vals[i] *= norm;
    }
}

// MXFP4 quantize/dequantize round-trip for a single value given a scale
float mxfp4_roundtrip_val(float val, float scale) {
    if (scale == 0.0) return 0.0;
    float inv_scale = 1.0 / scale;
    float av = abs(val) * inv_scale;
    // Decision boundary quantization (matching CPU/Metal)
    uint idx;
    if      (av < 0.25)  idx = 0u;  // -> 0.0
    else if (av < 0.75)  idx = 1u;  // -> 0.5
    else if (av < 1.25)  idx = 2u;  // -> 1.0
    else if (av < 1.75)  idx = 3u;  // -> 1.5
    else if (av < 2.5)   idx = 4u;  // -> 2.0
    else if (av < 3.5)   idx = 5u;  // -> 3.0
    else if (av < 5.0)   idx = 6u;  // -> 4.0
    else                  idx = 7u;  // -> 6.0
    const float kvalues[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    float dq = kvalues[idx] * scale;
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

// Compute E8M0 shared exponent for a 32-element block (MSE-optimal with ±1 search).
// Matches CPU mxfp_compute_e8m0_mse: round(log2(amax)) - emax_offset + 127, then ±1 MSE search.
// emax_offset accounts for the element format's dynamic range (max representable value).
float compute_e8m0_scale(float vals[32]) {
    float amax = 0.0;
    for (uint i = 0u; i < 32u; i++) {
        amax = max(amax, abs(vals[i]));
    }
    if (amax == 0.0) return 0.0;

    // round(log2(amax)) via integer bit extraction — matches CPU.
    uint amax_bits = floatBitsToUint(amax);
    int floor_log2 = int((amax_bits >> 23) & 0xFFu) - 127;
    // Round up if mantissa >= sqrt(2)-1 threshold (0x3504F3 in 23-bit IEEE mantissa)
    int round_log2 = floor_log2 + (((amax_bits & 0x7FFFFFu) >= 0x3504F3u) ? 1 : 0);

    // emax_offset: floor(log2(max_element_value)) for each format.
    // Shifts the E8M0 search center to account for the format's dynamic range.
    // Must match CPU's mxfp_elem_traits_t.emax_offset values.
#if defined(DATA_A_MXFP4)
    const int emax_offset = 2;   // max kvalue = 6.0 (after 0.5 factor), 2^2.58
#elif defined(DATA_A_MXFP8_E4M3)
    const int emax_offset = 8;   // max = 448, 2^8.8
#elif defined(DATA_A_MXFP8_E5M2)
    const int emax_offset = 16;  // max = 57344, 2^15.8
#elif defined(DATA_A_MXFP6_E2M3)
    const int emax_offset = 3;   // max = 7.5, 2^2.9
#elif defined(DATA_A_MXFP6_E3M2)
    const int emax_offset = 5;   // max = 28, 2^4.8
#endif

    int e_base = round_log2 - emax_offset + 127;

    // ±1 MSE search: test e_base-1, e_base, e_base+1, pick lowest total MSE.
    float best_mse = 1e30;
    int best_e = clamp(e_base, 1, 254);
    for (int delta = -1; delta <= 1; delta++) {
        int e_try = e_base + delta;
        if (e_try < 1 || e_try > 254) continue;
        float scale_try = uintBitsToFloat(uint(e_try) << 23);
        float mse = 0.0;
        for (uint i = 0u; i < 32u; i++) {
            float recon = mxfp_roundtrip_val(vals[i], scale_try);
            float err = vals[i] - recon;
            mse += err * err;
        }
        if (mse < best_mse) {
            best_mse = mse;
            best_e = e_try;
        }
    }
    return uintBitsToFloat(uint(best_e) << 23);
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
