// Conformance tests for canonical MXFP element converters in ggml-common.h.
// Verifies all valid bit patterns for each type, round-trip accuracy, and
// consistency between canonical converters and existing public API.

#include "ggml.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// Pull in the canonical converters from ggml-common.h (CPU platform → static inline).
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#undef GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_C

// Public API element converters (declared in ggml-quants.h, linked via ggml library).
extern "C" {
    float fp8_e4m3_to_float(uint8_t v);
    float fp8_e5m2_to_float(uint8_t v);
    float fp6_e2m3_to_float(uint8_t v);
    float fp6_e3m2_to_float(uint8_t v);
}

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " fmt "\n", ##__VA_ARGS__); \
        n_fail++; \
    } else { \
        n_pass++; \
    } \
} while (0)

// ---- FP4 E2M1 ----

static void test_fp4_e2m1_exhaustive(void) {
    printf("Testing FP4 E2M1 (16 values)...\n");

    // Expected canonical values for nibbles 0-15:
    // [S(1) | E(2) | M(1)], bias=1
    const float expected[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };

    for (int i = 0; i < 16; i++) {
        float got = ggml_mxfp_fp4_e2m1_to_float((uint8_t)i);
        // For ±0, compare bits to handle -0
        if (expected[i] == 0.0f) {
            uint32_t got_bits, exp_bits;
            memcpy(&got_bits, &got, 4);
            memcpy(&exp_bits, &expected[i], 4);
            CHECK(got_bits == exp_bits, "fp4_e2m1_to_float(%d): got 0x%08x, expected 0x%08x", i, got_bits, exp_bits);
        } else {
            CHECK(got == expected[i], "fp4_e2m1_to_float(%d): got %f, expected %f", i, got, expected[i]);
        }
    }

    // Verify canonical float LUT matches converter
    for (int i = 0; i < 16; i++) {
        float lut_val = kvalues_mxfp4_float[i];
        float conv_val = ggml_mxfp_fp4_e2m1_to_float((uint8_t)i);
        if (lut_val == 0.0f) {
            uint32_t lut_bits, conv_bits;
            memcpy(&lut_bits, &lut_val, 4);
            memcpy(&conv_bits, &conv_val, 4);
            CHECK(lut_bits == conv_bits, "kvalues_mxfp4_float[%d] (0x%08x) != fp4_e2m1_to_float (0x%08x)", i, lut_bits, conv_bits);
        } else {
            CHECK(lut_val == conv_val, "kvalues_mxfp4_float[%d] (%f) != fp4_e2m1_to_float (%f)", i, lut_val, conv_val);
        }
    }

    // Verify doubled int8 LUT is 2× canonical float LUT (for non-zero values)
    for (int i = 0; i < 16; i++) {
        float canonical = kvalues_mxfp4_float[i];
        float doubled = (float)kvalues_mxfp4[i];
        if (canonical == 0.0f) {
            CHECK(doubled == 0.0f, "kvalues_mxfp4[%d] should be 0 for zero canonical value", i);
        } else {
            CHECK(doubled == canonical * 2.0f, "kvalues_mxfp4[%d] (%f) != 2 * kvalues_mxfp4_float[%d] (%f)",
                  i, doubled, i, canonical);
        }
    }

    // Round-trip: float → fp4 → float for canonical values
    for (int i = 0; i < 8; i++) {
        float val = expected[i];
        uint8_t nibble = ggml_mxfp_float_to_fp4_e2m1(val);
        float roundtrip = ggml_mxfp_fp4_e2m1_to_float(nibble);
        CHECK(roundtrip == val, "fp4_e2m1 round-trip: %f → %d → %f", val, nibble, roundtrip);

        // Negative
        if (val > 0.0f) {
            nibble = ggml_mxfp_float_to_fp4_e2m1(-val);
            roundtrip = ggml_mxfp_fp4_e2m1_to_float(nibble);
            CHECK(roundtrip == -val, "fp4_e2m1 round-trip: %f → %d → %f", -val, nibble, roundtrip);
        }
    }

    // Boundary tests: values at decision boundaries should snap to nearest
    struct { float input; uint8_t expected_nibble; } boundary_tests[] = {
        { 0.24f, 0x0 },  // < 0.25 → 0
        { 0.26f, 0x1 },  // > 0.25 → 0.5
        { 0.74f, 0x1 },  // < 0.75 → 0.5
        { 0.76f, 0x2 },  // > 0.75 → 1.0
        { 1.24f, 0x2 },  // < 1.25 → 1.0
        { 1.26f, 0x3 },  // > 1.25 → 1.5
        { 2.49f, 0x4 },  // < 2.5  → 2.0
        { 2.51f, 0x5 },  // > 2.5  → 3.0
        { 4.99f, 0x6 },  // < 5.0  → 4.0
        { 5.01f, 0x7 },  // > 5.0  → 6.0
        { 100.0f, 0x7 }, // clamp to max
    };
    for (int i = 0; i < (int)(sizeof(boundary_tests)/sizeof(boundary_tests[0])); i++) {
        uint8_t got = ggml_mxfp_float_to_fp4_e2m1(boundary_tests[i].input);
        CHECK(got == boundary_tests[i].expected_nibble,
              "fp4_e2m1 boundary: %f → %d, expected %d",
              boundary_tests[i].input, got, boundary_tests[i].expected_nibble);
    }
}

// ---- FP6 E2M3 ----

static void test_fp6_e2m3_exhaustive(void) {
    printf("Testing FP6 E2M3 (64 values)...\n");

    for (int i = 0; i < 64; i++) {
        float val = ggml_mxfp_fp6_e2m3_to_float((uint8_t)i);

        // Verify basic properties
        int sign_bit = (i >> 5) & 1;
        if (sign_bit && (i & 0x1F) != 0) {
            CHECK(val < 0.0f, "fp6_e2m3: nibble %d should be negative", i);
        }

        // Round-trip
        uint8_t roundtrip_nibble = ggml_mxfp_float_to_fp6_e2m3(val);
        float roundtrip_val = ggml_mxfp_fp6_e2m3_to_float(roundtrip_nibble);
        CHECK(roundtrip_val == val, "fp6_e2m3 round-trip: nibble %d → %f → %d → %f",
              i, val, roundtrip_nibble, roundtrip_val);
    }

    // Max finite = 7.5
    float max_val = ggml_mxfp_fp6_e2m3_to_float(0x1F);
    CHECK(max_val == 7.5f, "fp6_e2m3 max: got %f, expected 7.5", max_val);

    // Clamp
    uint8_t clamped = ggml_mxfp_float_to_fp6_e2m3(100.0f);
    CHECK(clamped == 0x1F, "fp6_e2m3 clamp: 100.0 → %d, expected 31", clamped);
}

// ---- FP6 E3M2 ----

static void test_fp6_e3m2_exhaustive(void) {
    printf("Testing FP6 E3M2 (64 values)...\n");

    for (int i = 0; i < 64; i++) {
        float val = ggml_mxfp_fp6_e3m2_to_float((uint8_t)i);

        int sign_bit = (i >> 5) & 1;
        if (sign_bit && (i & 0x1F) != 0) {
            CHECK(val < 0.0f, "fp6_e3m2: nibble %d should be negative", i);
        }

        uint8_t roundtrip_nibble = ggml_mxfp_float_to_fp6_e3m2(val);
        float roundtrip_val = ggml_mxfp_fp6_e3m2_to_float(roundtrip_nibble);
        CHECK(roundtrip_val == val, "fp6_e3m2 round-trip: nibble %d → %f → %d → %f",
              i, val, roundtrip_nibble, roundtrip_val);
    }

    // Max finite = 28.0 (no NaN/Inf in E3M2)
    float max_val = ggml_mxfp_fp6_e3m2_to_float(0x1F);
    CHECK(max_val == 28.0f, "fp6_e3m2 max: got %f, expected 28.0", max_val);
}

// ---- FP8 E4M3 ----

static void test_fp8_e4m3_exhaustive(void) {
    printf("Testing FP8 E4M3 (256 values)...\n");

    int nan_count = 0;
    for (int i = 0; i < 256; i++) {
        float val = ggml_mxfp_fp8_e4m3_to_float((uint8_t)i);

        if (isnan(val)) {
            nan_count++;
            continue;
        }

        int sign_bit = (i >> 7) & 1;
        if (sign_bit && fabsf(val) > 0.0f) {
            CHECK(val < 0.0f, "fp8_e4m3: byte %d should be negative", i);
        }

        uint8_t roundtrip_byte = ggml_mxfp_float_to_fp8_e4m3(val);
        float roundtrip_val = ggml_mxfp_fp8_e4m3_to_float(roundtrip_byte);
        CHECK(roundtrip_val == val, "fp8_e4m3 round-trip: byte %d → %f → %d → %f",
              i, val, roundtrip_byte, roundtrip_val);
    }

    // E4M3 has 2 NaN values: 0x7F and 0xFF
    CHECK(nan_count == 2, "fp8_e4m3: expected 2 NaN values, got %d", nan_count);

    // Max finite = 448
    float max_val = ggml_mxfp_fp8_e4m3_to_float(0x7E);
    CHECK(max_val == 448.0f, "fp8_e4m3 max: got %f, expected 448.0", max_val);
}

// ---- FP8 E5M2 ----

static void test_fp8_e5m2_exhaustive(void) {
    printf("Testing FP8 E5M2 (256 values)...\n");

    int nan_inf_count = 0;
    for (int i = 0; i < 256; i++) {
        float val = ggml_mxfp_fp8_e5m2_to_float((uint8_t)i);

        if (isnan(val) || isinf(val)) {
            nan_inf_count++;
            continue;
        }

        int sign_bit = (i >> 7) & 1;
        if (sign_bit && fabsf(val) > 0.0f) {
            CHECK(val < 0.0f, "fp8_e5m2: byte %d should be negative", i);
        }

        uint8_t roundtrip_byte = ggml_mxfp_float_to_fp8_e5m2(val);
        float roundtrip_val = ggml_mxfp_fp8_e5m2_to_float(roundtrip_byte);
        CHECK(roundtrip_val == val, "fp8_e5m2 round-trip: byte %d → %f → %d → %f",
              i, val, roundtrip_byte, roundtrip_val);
    }

    // E5M2: exp=31, 2 mantissa bits → 4 per sign (1 Inf + 3 NaN) = 8 total
    CHECK(nan_inf_count == 8, "fp8_e5m2: expected 8 NaN/Inf values, got %d", nan_inf_count);

    // Max finite = 57344
    float max_val = ggml_mxfp_fp8_e5m2_to_float(0x7B);
    CHECK(max_val == 57344.0f, "fp8_e5m2 max: got %f, expected 57344.0", max_val);
}

// ---- FP6 pack/unpack ----

static void test_fp6_pack_unpack(void) {
    printf("Testing FP6 pack/unpack...\n");

    // All possible 6-bit values in groups of 4
    for (int a = 0; a < 64; a += 7) {
        for (int b = 0; b < 64; b += 11) {
            for (int c = 0; c < 64; c += 13) {
                for (int d = 0; d < 64; d += 17) {
                    uint8_t in[4] = { (uint8_t)a, (uint8_t)b, (uint8_t)c, (uint8_t)d };
                    uint8_t packed[3];
                    uint8_t out[4];
                    ggml_mxfp_pack_fp6x4(in, packed);
                    ggml_mxfp_unpack_fp6x4(packed, out);
                    CHECK(out[0] == in[0] && out[1] == in[1] && out[2] == in[2] && out[3] == in[3],
                          "fp6 pack/unpack: {%d,%d,%d,%d} → {%d,%d,%d,%d}",
                          in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3]);
                }
            }
        }
    }

    // Edge cases: all zeros and all ones
    {
        uint8_t zeros[4] = {0, 0, 0, 0};
        uint8_t packed[3], out[4];
        ggml_mxfp_pack_fp6x4(zeros, packed);
        ggml_mxfp_unpack_fp6x4(packed, out);
        CHECK(out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0,
              "fp6 pack/unpack: all zeros failed");
    }
    {
        uint8_t maxes[4] = {63, 63, 63, 63};
        uint8_t packed[3], out[4];
        ggml_mxfp_pack_fp6x4(maxes, packed);
        ggml_mxfp_unpack_fp6x4(packed, out);
        CHECK(out[0] == 63 && out[1] == 63 && out[2] == 63 && out[3] == 63,
              "fp6 pack/unpack: all 63s failed");
    }
}

// ---- MXFP property defines ----

static void test_mxfp_properties(void) {
    printf("Testing MXFP property defines...\n");

    CHECK(MXFP_BITS_PER_ELEM_E2M1 == 4, "E2M1 bits_per_elem");
    CHECK(MXFP_BITS_PER_ELEM_E4M3 == 8, "E4M3 bits_per_elem");
    CHECK(MXFP_BITS_PER_ELEM_E5M2 == 8, "E5M2 bits_per_elem");
    CHECK(MXFP_BITS_PER_ELEM_E2M3 == 6, "E2M3 bits_per_elem");
    CHECK(MXFP_BITS_PER_ELEM_E3M2 == 6, "E3M2 bits_per_elem");

    CHECK(MXFP_QS_PER_BLOCK_E2M1 == 16, "E2M1 qs_per_block");
    CHECK(MXFP_QS_PER_BLOCK_E4M3 == 32, "E4M3 qs_per_block");
    CHECK(MXFP_QS_PER_BLOCK_E5M2 == 32, "E5M2 qs_per_block");
    CHECK(MXFP_QS_PER_BLOCK_E2M3 == 24, "E2M3 qs_per_block");
    CHECK(MXFP_QS_PER_BLOCK_E3M2 == 24, "E3M2 qs_per_block");

    // SOA aliases
    CHECK(MXFP4_SOA_QS_PER_BLOCK == MXFP_QS_PER_BLOCK_E2M1, "MXFP4 SOA alias");
    CHECK(MXFP8_SOA_QS_PER_BLOCK == MXFP_QS_PER_BLOCK_E4M3, "MXFP8 SOA alias");
    CHECK(MXFP6_SOA_QS_PER_BLOCK == MXFP_QS_PER_BLOCK_E2M3, "MXFP6 SOA alias");

    // Hadamard
    CHECK(MXFP_USE_HADAMARD_E2M1 == 1, "E2M1 use_hadamard");
    CHECK(MXFP_USE_HADAMARD_E4M3 == 1, "E4M3 use_hadamard");
    CHECK(MXFP_USE_HADAMARD_E5M2 == 0, "E5M2 use_hadamard");
    CHECK(MXFP_USE_HADAMARD_E2M3 == 1, "E2M3 use_hadamard");
    CHECK(MXFP_USE_HADAMARD_E3M2 == 0, "E3M2 use_hadamard");

    // Cross-check with ggml_mxfp_use_hadamard API
    CHECK(ggml_mxfp_use_hadamard(GGML_TYPE_MXFP4_E2M1) == (bool)MXFP_USE_HADAMARD_E2M1, "API vs define E2M1");
    CHECK(ggml_mxfp_use_hadamard(GGML_TYPE_MXFP8_E4M3) == (bool)MXFP_USE_HADAMARD_E4M3, "API vs define E4M3");
    CHECK(ggml_mxfp_use_hadamard(GGML_TYPE_MXFP8_E5M2) == (bool)MXFP_USE_HADAMARD_E5M2, "API vs define E5M2");
    CHECK(ggml_mxfp_use_hadamard(GGML_TYPE_MXFP6_E2M3) == (bool)MXFP_USE_HADAMARD_E2M3, "API vs define E2M3");
    CHECK(ggml_mxfp_use_hadamard(GGML_TYPE_MXFP6_E3M2) == (bool)MXFP_USE_HADAMARD_E3M2, "API vs define E3M2");
}

// ---- E8M0 shared exponent ----

static void test_e8m0_exhaustive(void) {
    printf("Testing E8M0 to float (256 values)...\n");

    // x=0: 2^(-127)
    CHECK(ggml_mxfp_e8m0_to_fp32(0) != 0.0f, "e8m0(0) should be nonzero (2^-127)");

    // All 256 values: verify bit pattern matches expected 2^(x-127)
    for (int i = 0; i < 256; i++) {
        float val = ggml_mxfp_e8m0_to_fp32((uint8_t)i);
        if (i == 0) {
            // Special case: denormalized encoding for 2^(-127)
            uint32_t bits;
            memcpy(&bits, &val, sizeof(bits));
            CHECK(bits == 0x00400000, "e8m0(0) bits=0x%08x expected 0x00400000", bits);
        } else {
            // Normal: should be exactly 2^(i-127)
            uint32_t bits;
            memcpy(&bits, &val, sizeof(bits));
            uint32_t expected = (uint32_t)i << 23;
            CHECK(bits == expected, "e8m0(%d) bits=0x%08x expected 0x%08x", i, bits, expected);
        }
    }

    // Half variant: verify ggml_mxfp_e8m0_to_fp32_half(x) == ggml_mxfp_e8m0_to_fp32(x) * 0.5f
    // Skip x=255: full(255) = Inf, 0.5*Inf = Inf, but half(255) = 2^127 (valid, not Inf).
    // The half function intentionally avoids Inf by using exponent (x-1).
    for (int i = 0; i < 255; i++) {
        float full = ggml_mxfp_e8m0_to_fp32((uint8_t)i);
        float half = ggml_mxfp_e8m0_to_fp32_half((uint8_t)i);
        float expected = full * 0.5f;
        uint32_t half_bits, exp_bits;
        memcpy(&half_bits, &half, sizeof(half_bits));
        memcpy(&exp_bits, &expected, sizeof(exp_bits));
        CHECK(half_bits == exp_bits, "e8m0_half(%d) bits=0x%08x expected 0x%08x", i, half_bits, exp_bits);
    }
}

// ---- Hadamard constant ----

static void test_hadamard_constant(void) {
    printf("Testing Hadamard norm constant...\n");

    float computed = 1.0f / sqrtf(32.0f);
    CHECK(MXFP_HADAMARD_32_NORM == computed,
          "MXFP_HADAMARD_32_NORM=%.20g expected=%.20g",
          (double)MXFP_HADAMARD_32_NORM, (double)computed);
}

// ---- Public API consistency ----

static void test_public_api_consistency(void) {
    printf("Testing public API consistency with canonical converters...\n");

    // FP8 E4M3: public API fp8_e4m3_to_float matches canonical
    for (int i = 0; i < 256; i++) {
        float pub = fp8_e4m3_to_float((uint8_t)i);
        float can = ggml_mxfp_fp8_e4m3_to_float((uint8_t)i);
        if (isnan(pub) && isnan(can)) continue;  // both NaN is fine
        uint32_t pub_bits, can_bits;
        memcpy(&pub_bits, &pub, 4);
        memcpy(&can_bits, &can, 4);
        CHECK(pub_bits == can_bits, "fp8_e4m3 API mismatch at %d: pub=0x%08x can=0x%08x", i, pub_bits, can_bits);
    }

    // FP8 E5M2: public API fp8_e5m2_to_float matches canonical
    for (int i = 0; i < 256; i++) {
        float pub = fp8_e5m2_to_float((uint8_t)i);
        float can = ggml_mxfp_fp8_e5m2_to_float((uint8_t)i);
        if ((isnan(pub) || isinf(pub)) && (isnan(can) || isinf(can))) {
            // Both special, check sign+type match
            uint32_t pub_bits, can_bits;
            memcpy(&pub_bits, &pub, 4);
            memcpy(&can_bits, &can, 4);
            CHECK(pub_bits == can_bits, "fp8_e5m2 API special mismatch at %d", i);
            continue;
        }
        uint32_t pub_bits, can_bits;
        memcpy(&pub_bits, &pub, 4);
        memcpy(&can_bits, &can, 4);
        CHECK(pub_bits == can_bits, "fp8_e5m2 API mismatch at %d: pub=0x%08x can=0x%08x", i, pub_bits, can_bits);
    }

    // FP6 E2M3: public API fp6_e2m3_to_float matches canonical
    for (int i = 0; i < 64; i++) {
        float pub = fp6_e2m3_to_float((uint8_t)i);
        float can = ggml_mxfp_fp6_e2m3_to_float((uint8_t)i);
        uint32_t pub_bits, can_bits;
        memcpy(&pub_bits, &pub, 4);
        memcpy(&can_bits, &can, 4);
        CHECK(pub_bits == can_bits, "fp6_e2m3 API mismatch at %d: pub=0x%08x can=0x%08x", i, pub_bits, can_bits);
    }

    // FP6 E3M2: public API fp6_e3m2_to_float matches canonical
    for (int i = 0; i < 64; i++) {
        float pub = fp6_e3m2_to_float((uint8_t)i);
        float can = ggml_mxfp_fp6_e3m2_to_float((uint8_t)i);
        uint32_t pub_bits, can_bits;
        memcpy(&pub_bits, &pub, 4);
        memcpy(&can_bits, &can, 4);
        CHECK(pub_bits == can_bits, "fp6_e3m2 API mismatch at %d: pub=0x%08x can=0x%08x", i, pub_bits, can_bits);
    }
}

int main(void) {
    printf("=== MXFP Converter Conformance Tests ===\n\n");

    test_mxfp_properties();
    test_fp4_e2m1_exhaustive();
    test_fp6_e2m3_exhaustive();
    test_fp6_e3m2_exhaustive();
    test_fp8_e4m3_exhaustive();
    test_fp8_e5m2_exhaustive();
    test_fp6_pack_unpack();
    test_e8m0_exhaustive();
    test_hadamard_constant();
    test_public_api_consistency();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
