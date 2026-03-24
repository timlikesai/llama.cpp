// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 = 0.0024f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 = 0.0008f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 = 0.0008f;
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP4 = 0.0040f;
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP6 = 0.0010f;
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP8 = 0.0008f;

constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_MXFP = 0.012f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Canonical E2M1 values — ground truth for FP4 converter validation.
static const float kvalues_mxfp4_float[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// FP6 E2M3 dequantization LUT — ground truth for converter validation.
static const float kvalues_mxfp6_e2m3[64] = {
     0.0f,  0.125f,   0.25f,  0.375f,    0.5f,  0.625f,   0.75f,  0.875f,
     1.0f,  1.125f,   1.25f,  1.375f,    1.5f,  1.625f,   1.75f,  1.875f,
     2.0f,   2.25f,    2.5f,   2.75f,    3.0f,   3.25f,    3.5f,   3.75f,
     4.0f,    4.5f,    5.0f,    5.5f,    6.0f,    6.5f,    7.0f,    7.5f,
    -0.0f, -0.125f,  -0.25f, -0.375f,   -0.5f, -0.625f,  -0.75f, -0.875f,
    -1.0f, -1.125f,  -1.25f, -1.375f,   -1.5f, -1.625f,  -1.75f, -1.875f,
    -2.0f,  -2.25f,   -2.5f,  -2.75f,   -3.0f,  -3.25f,   -3.5f,  -3.75f,
    -4.0f,   -4.5f,   -5.0f,   -5.5f,   -6.0f,   -6.5f,   -7.0f,   -7.5f,
};

// FP8 E4M3 dequantization LUT — ground truth for converter validation.
// 0x7E/0xFE = +-448 (max finite), 0x7F/0xFF = NaN.
static const float kvalues_mxfp8_e4m3[256] = {
           0.0f, 0.001953125f,  0.00390625f, 0.005859375f,   0.0078125f, 0.009765625f,  0.01171875f, 0.013671875f,
      0.015625f, 0.017578125f,  0.01953125f, 0.021484375f,   0.0234375f, 0.025390625f,  0.02734375f, 0.029296875f,
       0.03125f,  0.03515625f,   0.0390625f,  0.04296875f,    0.046875f,  0.05078125f,   0.0546875f,  0.05859375f,
        0.0625f,   0.0703125f,    0.078125f,   0.0859375f,     0.09375f,   0.1015625f,    0.109375f,   0.1171875f,
         0.125f,    0.140625f,     0.15625f,    0.171875f,      0.1875f,    0.203125f,     0.21875f,    0.234375f,
          0.25f,     0.28125f,      0.3125f,     0.34375f,       0.375f,     0.40625f,      0.4375f,     0.46875f,
           0.5f,      0.5625f,       0.625f,      0.6875f,        0.75f,      0.8125f,       0.875f,      0.9375f,
           1.0f,       1.125f,        1.25f,       1.375f,         1.5f,       1.625f,        1.75f,       1.875f,
           2.0f,        2.25f,         2.5f,        2.75f,         3.0f,        3.25f,         3.5f,        3.75f,
           4.0f,         4.5f,         5.0f,         5.5f,         6.0f,         6.5f,         7.0f,         7.5f,
           8.0f,         9.0f,        10.0f,        11.0f,        12.0f,        13.0f,        14.0f,        15.0f,
          16.0f,        18.0f,        20.0f,        22.0f,        24.0f,        26.0f,        28.0f,        30.0f,
          32.0f,        36.0f,        40.0f,        44.0f,        48.0f,        52.0f,        56.0f,        60.0f,
          64.0f,        72.0f,        80.0f,        88.0f,        96.0f,       104.0f,       112.0f,       120.0f,
         128.0f,       144.0f,       160.0f,       176.0f,       192.0f,       208.0f,       224.0f,       240.0f,
         256.0f,       288.0f,       320.0f,       352.0f,       384.0f,       416.0f,       448.0f,        NAN,
          -0.0f,-0.001953125f, -0.00390625f,-0.005859375f,  -0.0078125f,-0.009765625f, -0.01171875f,-0.013671875f,
     -0.015625f,-0.017578125f, -0.01953125f,-0.021484375f,  -0.0234375f,-0.025390625f, -0.02734375f,-0.029296875f,
      -0.03125f, -0.03515625f,  -0.0390625f, -0.04296875f,   -0.046875f, -0.05078125f,  -0.0546875f, -0.05859375f,
       -0.0625f,  -0.0703125f,   -0.078125f,  -0.0859375f,    -0.09375f,  -0.1015625f,   -0.109375f,  -0.1171875f,
        -0.125f,   -0.140625f,    -0.15625f,   -0.171875f,     -0.1875f,   -0.203125f,    -0.21875f,   -0.234375f,
         -0.25f,    -0.28125f,     -0.3125f,    -0.34375f,      -0.375f,    -0.40625f,     -0.4375f,    -0.46875f,
          -0.5f,     -0.5625f,      -0.625f,     -0.6875f,       -0.75f,     -0.8125f,      -0.875f,     -0.9375f,
          -1.0f,      -1.125f,       -1.25f,      -1.375f,        -1.5f,      -1.625f,       -1.75f,      -1.875f,
          -2.0f,       -2.25f,        -2.5f,       -2.75f,        -3.0f,       -3.25f,        -3.5f,       -3.75f,
          -4.0f,        -4.5f,        -5.0f,        -5.5f,        -6.0f,        -6.5f,        -7.0f,        -7.5f,
          -8.0f,        -9.0f,       -10.0f,       -11.0f,       -12.0f,       -13.0f,       -14.0f,       -15.0f,
         -16.0f,       -18.0f,       -20.0f,       -22.0f,       -24.0f,       -26.0f,       -28.0f,       -30.0f,
         -32.0f,       -36.0f,       -40.0f,       -44.0f,       -48.0f,       -52.0f,       -56.0f,       -60.0f,
         -64.0f,       -72.0f,       -80.0f,       -88.0f,       -96.0f,      -104.0f,      -112.0f,      -120.0f,
        -128.0f,      -144.0f,      -160.0f,      -176.0f,      -192.0f,      -208.0f,      -224.0f,      -240.0f,
        -256.0f,      -288.0f,      -320.0f,      -352.0f,      -384.0f,      -416.0f,      -448.0f,        NAN,
};

// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    ggml_cpu_init();

    // --- MXFP unified format table ---
    struct mxfp_test_format {
        ggml_type     type;
        const char  * name;
        int           qs_per_block;
        int           n_codes;           // 16 (fp4), 64 (fp6), 256 (fp8)
        uint8_t       nan_bits;          // bit pattern for NaN (0 = no NaN in format)
        float         max_finite;
        int           emax_offset;
        float         max_quant_error;
        float         max_pipeline_error;
        const float * lut;               // canonical dequant LUT
        float       (*to_float)(uint8_t);
        uint8_t     (*to_quant)(float);
    };

    const mxfp_test_format mxfp_fmts[] = {
        { GGML_TYPE_MXFP4, "mxfp4", MXFP_QS_PER_BLOCK_E2M1,  16, 0,
          MXFP4_E2M1_MAX_FINITE, MXFP4_E2M1_EMAX_OFFSET,
          MAX_QUANTIZATION_TOTAL_ERROR_MXFP4, MAX_MXFP_PIPELINE_ERROR_MXFP4,
          kvalues_mxfp4_float, ggml_mxfp_fp4_e2m1_to_float, ggml_mxfp_float_to_fp4_e2m1 },
        { GGML_TYPE_MXFP6, "mxfp6", MXFP_QS_PER_BLOCK_E2M3,  64, 0,
          MXFP6_E2M3_MAX_FINITE, MXFP6_E2M3_EMAX_OFFSET,
          MAX_QUANTIZATION_TOTAL_ERROR_MXFP6, MAX_MXFP_PIPELINE_ERROR_MXFP6,
          kvalues_mxfp6_e2m3, ggml_mxfp_fp6_e2m3_to_float, ggml_mxfp_float_to_fp6_e2m3 },
        { GGML_TYPE_MXFP8, "mxfp8", MXFP_QS_PER_BLOCK_E4M3, 256, 0x7F,
          MXFP8_E4M3_MAX_FINITE, MXFP8_E4M3_EMAX_OFFSET,
          MAX_QUANTIZATION_TOTAL_ERROR_MXFP8, MAX_MXFP_PIPELINE_ERROR_MXFP8,
          kvalues_mxfp8_e4m3, ggml_mxfp_fp8_e4m3_to_float, ggml_mxfp_float_to_fp8_e4m3 },
    };
    const int n_mxfp_fmts = sizeof(mxfp_fmts) / sizeof(mxfp_fmts[0]);

    // --- check_test helper: DRYs the repeated fail/count/print pattern ---
    int num_failed = 0;
    bool failed = false;

    auto check_test = [&](bool condition, const char * fmt, ...) __attribute__((format(printf, 3, 4))) {
        failed = !condition;
        num_failed += failed;
        if (failed || verbose) {
            va_list args;
            va_start(args, fmt);
            vprintf(fmt, args);
            va_end(args);
        }
    };

    // --- bitwise_mismatches helper for SoA format spec tests ---
    auto bitwise_mismatches = [](const float * a, const float * b, int n) -> int {
        int mismatches = 0;
        for (int i = 0; i < n; i++) {
            uint32_t ua, ub;
            memcpy(&ua, &a[i], 4);
            memcpy(&ub, &b[i], 4);
            if (ua != ub) mismatches++;
        }
        return mismatches;
    };

    // ========================================================================
    // Standard quantization tests (all GGML types)
    // ========================================================================
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 :
                type == GGML_TYPE_MXFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 :
                type == GGML_TYPE_MXFP6   ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 :
                type == GGML_TYPE_MXFP8   ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 : MAX_QUANTIZATION_TOTAL_ERROR;
            check_test(total_error < max_quantization_error,
                "%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[!(total_error < max_quantization_error)], total_error);

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            check_test(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR,
                "%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[!(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR)], reference_error);

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                                            type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                                          ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                                          : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0
                                          ? MAX_DOT_PRODUCT_ERROR_TERNARY
                                          : type == GGML_TYPE_NVFP4
                                          ? MAX_DOT_PRODUCT_ERROR_FP4
                                          : type == GGML_TYPE_MXFP4 || type == GGML_TYPE_MXFP6 || type == GGML_TYPE_MXFP8
                                          ? MAX_DOT_PRODUCT_ERROR_MXFP
                                          : MAX_DOT_PRODUCT_ERROR;
            check_test(vec_dot_error < max_allowed_error,
                "%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[!(vec_dot_error < max_allowed_error)], vec_dot_error);
        }
    }

    // ========================================================================
    // MXFP SoA roundtrip via dispatch
    // ========================================================================
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        const size_t buf_size = ggml_row_size(f.type, test_size);
        std::vector<uint8_t> tmp_q(buf_size);
        std::vector<float> tmp_out(test_size);

        ggml_mxfp_quantize_soa(f.type, test_data.data(), tmp_q.data(), test_size, false);
        ggml_mxfp_dequantize_soa(f.type, tmp_q.data(), tmp_out.data(), test_size);

        const float soa_error = array_rmse(test_data.data(), tmp_out.data(), test_size);
        check_test(soa_error < f.max_quant_error,
            "%5s SoA quantization error:          %s (%f)\n", f.name, RESULT_STR[!(soa_error < f.max_quant_error)], soa_error);
    }

    // ========================================================================
    // Hadamard tests
    // ========================================================================

    // Hadamard self-inverse: H(H(x)) == x
    {
        float original[32], transformed[32];
        for (int i = 0; i < 32; i++) {
            original[i] = 0.1f + 2.0f * cosf(i + 0.5f);
            transformed[i] = original[i];
        }
        ggml_mxfp_hadamard_32_inplace(transformed);
        ggml_mxfp_hadamard_32_inplace(transformed); // apply twice = identity

        float max_err = 0.0f;
        for (int i = 0; i < 32; i++) {
            float err = fabsf(transformed[i] - original[i]);
            if (err > max_err) max_err = err;
        }
        check_test(max_err < 1e-5f,
            "hadamard H(H(x))==x roundtrip:         %s (max_err=%.2e)\n", RESULT_STR[!(max_err < 1e-5f)], max_err);
    }

    // Hadamard known output: H([1,0,...,0]) = [1/sqrt(32), ...]
    {
        float unit[32] = {};
        unit[0] = 1.0f;
        ggml_mxfp_hadamard_32_inplace(unit);

        const float expected = MXFP_HADAMARD_32_NORM;
        float max_err = 0.0f;
        for (int i = 0; i < 32; i++) {
            float err = fabsf(unit[i] - expected);
            if (err > max_err) max_err = err;
        }
        check_test(max_err < 1e-7f,
            "hadamard unit vector:                  %s (max_err=%.2e, expected %.8f)\n",
            RESULT_STR[!(max_err < 1e-7f)], max_err, expected);
    }

    // Hadamard + quantize + dequant + Hadamard roundtrip (KV cache write/read path)
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        std::vector<float> original(test_size);
        std::vector<float> rotated(test_size);
        std::vector<float> recovered(test_size);
        generate_data(2.0, test_size, original.data());

        memcpy(rotated.data(), original.data(), test_size * sizeof(float));
        for (size_t b = 0; b < test_size / 32; b++) {
            ggml_mxfp_hadamard_32_inplace(&rotated[b * 32]);
        }

        const size_t buf_size = ggml_row_size(f.type, test_size);
        std::vector<uint8_t> qbuf(buf_size);
        ggml_mxfp_quantize_soa(f.type, rotated.data(), qbuf.data(), test_size, false);

        ggml_mxfp_dequantize_soa(f.type, qbuf.data(), recovered.data(), test_size);
        for (size_t b = 0; b < test_size / 32; b++) {
            ggml_mxfp_hadamard_32_inplace(&recovered[b * 32]);
        }

        float err = array_rmse(original.data(), recovered.data(), test_size);
        check_test(err < f.max_pipeline_error,
            "%5s Hadamard pipeline roundtrip:       %s (err=%.6f, max=%.6f)\n",
            f.name, RESULT_STR[!(err < f.max_pipeline_error)], err, f.max_pipeline_error);
    }

    // Fused Hadamard+SoA quantize must match separate Hadamard then SoA quantize
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        const size_t n = 128;
        std::vector<float> input(n);
        generate_data(3.0, n, input.data());

        const size_t buf_size = ggml_row_size(f.type, n);

        // Separate path: Hadamard then SoA quantize
        std::vector<float> rotated(n);
        memcpy(rotated.data(), input.data(), n * sizeof(float));
        for (size_t b = 0; b < n / 32; b++) {
            ggml_mxfp_hadamard_32_inplace(&rotated[b * 32]);
        }
        std::vector<uint8_t> sep_buf(buf_size);
        ggml_mxfp_quantize_soa(f.type, rotated.data(), sep_buf.data(), n, false);

        // Fused path
        std::vector<uint8_t> fused_buf(buf_size);
        ggml_mxfp_quantize_soa(f.type, input.data(), fused_buf.data(), n, true);

        bool match = (sep_buf == fused_buf);
        check_test(match, "%5s fused Hadamard+SoA quant:          %s\n", f.name, RESULT_STR[!match]);
    }

    // Hadamard roundtrip functions: output must match manual Hadamard + quant + dequant
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        const size_t n = 128;
        std::vector<float> input(n);
        generate_data(4.0, n, input.data());

        // Roundtrip function
        std::vector<float> rt_out(n);
        ggml_mxfp_hadamard_roundtrip(f.type, input.data(), rt_out.data(), n);

        // Manual: Hadamard -> quant -> dequant (same steps, no second Hadamard)
        const size_t buf_size = ggml_row_size(f.type, n);
        std::vector<float> manual_out(n);
        memcpy(manual_out.data(), input.data(), n * sizeof(float));
        for (size_t b = 0; b < n / 32; b++) {
            ggml_mxfp_hadamard_32_inplace(&manual_out[b * 32]);
        }
        std::vector<uint8_t> qbuf(buf_size);
        ggml_mxfp_quantize_soa(f.type, manual_out.data(), qbuf.data(), n, false);
        ggml_mxfp_dequantize_soa(f.type, qbuf.data(), manual_out.data(), n);

        float err = array_rmse(manual_out.data(), rt_out.data(), n);
        check_test(err == 0.0f,
            "%5s hadamard_roundtrip vs manual:       %s (rmse=%.6f)\n", f.name, RESULT_STR[err != 0.0f], err);
    }

    // ========================================================================
    // Element converter tests
    // ========================================================================

    // Converter vs canonical LUT values
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        int mismatches = 0;
        for (int j = 0; j < f.n_codes; j++) {
            const float converter_val = f.to_float((uint8_t)j);
            const float lut_val       = f.lut[j];

            if (isnan(converter_val) && isnan(lut_val)) continue;
            if (converter_val != lut_val) {
                if (mismatches == 0 || verbose) {
                    printf("  %s LUT mismatch at [%d]: converter=%.8g, lut=%.8g\n",
                           f.name, j, converter_val, lut_val);
                }
                mismatches++;
            }
        }
        check_test(mismatches == 0,
            "%5s converter vs LUT:                %s (%d/%d values match)\n",
            f.name, RESULT_STR[mismatches > 0], f.n_codes - mismatches, f.n_codes);
    }

    // Exhaustive round-trip: quantize(dequantize(i)) == i for all valid bit patterns
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        int rt_bad = 0;
        for (int j = 0; j < f.n_codes; j++) {
            if ((uint8_t)j == f.nan_bits && f.nan_bits != 0) continue;

            float val = f.to_float((uint8_t)j);
            if (isnan(val) || isinf(val)) continue;

            uint8_t back = f.to_quant(val);
            if (back != (uint8_t)j && !(val == 0.0f && f.to_float(back) == 0.0f)) {
                if (rt_bad == 0 || verbose) {
                    printf("  %s roundtrip: 0x%02X -> %.6g -> 0x%02X\n", f.name, j, val, back);
                }
                rt_bad++;
            }
        }
        check_test(rt_bad == 0,
            "%5s converter round-trip:             %s (%d/%d survived)\n",
            f.name, RESULT_STR[rt_bad > 0], f.n_codes - rt_bad, f.n_codes);
    }

    // Element converter edge cases (expected values validated against LUTs)
    {
        struct conv_check {
            const char * name;
            float        input;
            uint8_t      expected_bits;
            bool         is_saturation;
            const float * lut;
            float       (*to_float)(uint8_t);
            uint8_t     (*to_quant)(float);
        };

        const conv_check checks[] = {
            // FP4 E2M1 -[S(1)|E(2)|M(1)], bias=0
            { "fp4 zero",      0.0f,    0x00, false, kvalues_mxfp4_float, ggml_mxfp_fp4_e2m1_to_float, ggml_mxfp_float_to_fp4_e2m1 },
            { "fp4 sub 0.5",   0.5f,    0x01, false, kvalues_mxfp4_float, ggml_mxfp_fp4_e2m1_to_float, ggml_mxfp_float_to_fp4_e2m1 },
            { "fp4 norm 1.0",  1.0f,    0x02, false, kvalues_mxfp4_float, ggml_mxfp_fp4_e2m1_to_float, ggml_mxfp_float_to_fp4_e2m1 },
            { "fp4 max 6.0",   MXFP4_E2M1_MAX_FINITE, 0x07, false, kvalues_mxfp4_float, ggml_mxfp_fp4_e2m1_to_float, ggml_mxfp_float_to_fp4_e2m1 },
            { "fp4 neg -3.0", -3.0f,    0x0D, false, kvalues_mxfp4_float, ggml_mxfp_fp4_e2m1_to_float, ggml_mxfp_float_to_fp4_e2m1 },
            { "fp4 sat 100",  100.0f,   0x07, true,  kvalues_mxfp4_float, ggml_mxfp_fp4_e2m1_to_float, ggml_mxfp_float_to_fp4_e2m1 },

            // FP6 E2M3 -[S(1)|E(2)|M(3)], no NaN/Inf
            { "e2m3 zero",      0.0f,     0x00, false, kvalues_mxfp6_e2m3, ggml_mxfp_fp6_e2m3_to_float, ggml_mxfp_float_to_fp6_e2m3 },
            { "e2m3 sub",       0.125f,   0x01, false, kvalues_mxfp6_e2m3, ggml_mxfp_fp6_e2m3_to_float, ggml_mxfp_float_to_fp6_e2m3 },
            { "e2m3 max 7.5",   MXFP6_E2M3_MAX_FINITE, 0x1F, false, kvalues_mxfp6_e2m3, ggml_mxfp_fp6_e2m3_to_float, ggml_mxfp_float_to_fp6_e2m3 },
            { "e2m3 sat 100",   100.0f,   0x1F, true,  kvalues_mxfp6_e2m3, ggml_mxfp_fp6_e2m3_to_float, ggml_mxfp_float_to_fp6_e2m3 },
            { "e2m3 neg -2",   -2.0f,    0x30, false, kvalues_mxfp6_e2m3, ggml_mxfp_fp6_e2m3_to_float, ggml_mxfp_float_to_fp6_e2m3 },

            // FP8 E4M3 -[S(1)|E(4)|M(3)], bias=7
            { "e4m3 zero",      0.0f,     0x00, false, kvalues_mxfp8_e4m3, ggml_mxfp_fp8_e4m3_to_float, ggml_mxfp_float_to_fp8_e4m3 },
            { "e4m3 sub",       1.f/512,  0x01, false, kvalues_mxfp8_e4m3, ggml_mxfp_fp8_e4m3_to_float, ggml_mxfp_float_to_fp8_e4m3 },
            { "e4m3 max 448",   MXFP8_E4M3_MAX_FINITE, 0x7E, false, kvalues_mxfp8_e4m3, ggml_mxfp_fp8_e4m3_to_float, ggml_mxfp_float_to_fp8_e4m3 },
            { "e4m3 sat 500",   500.0f,   0x7E, true,  kvalues_mxfp8_e4m3, ggml_mxfp_fp8_e4m3_to_float, ggml_mxfp_float_to_fp8_e4m3 },
            { "e4m3 neg -1",   -1.0f,     0xB8, false, kvalues_mxfp8_e4m3, ggml_mxfp_fp8_e4m3_to_float, ggml_mxfp_float_to_fp8_e4m3 },
        };

        int conv_bad = 0;

        // validate expected_bits against LUTs
        for (const auto & c : checks) {
            if (c.lut && !c.is_saturation) {
                float lut_val = c.lut[c.expected_bits];
                if (c.input != lut_val && !(c.input == 0.0f && lut_val == 0.0f)) {
                    printf("  TEST BUG %s: expected_bits=0x%02X -> LUT=%.8g, but input=%.8g\n",
                           c.name, c.expected_bits, lut_val, c.input);
                    conv_bad++;
                }
            }
        }

        // Now test the quantize direction
        for (const auto & c : checks) {
            uint8_t got = c.to_quant(c.input);
            if (got != c.expected_bits) {
                if (conv_bad == 0 || verbose) {
                    printf("  %s: quantize(%.6g) = 0x%02X, expected 0x%02X\n",
                           c.name, c.input, got, c.expected_bits);
                }
                conv_bad++;
            }
        }

        // FP8 E4M3: 0x7F must dequantize to NaN
        {
            float nan_val = ggml_mxfp_fp8_e4m3_to_float(0x7F);
            if (!isnan(nan_val)) {
                if (conv_bad == 0 || verbose) {
                    printf("  e4m3 0x7F dequant: expected NaN, got %.6g\n", nan_val);
                }
                conv_bad++;
            }
        }

        check_test(conv_bad == 0,
            "  element converter edge cases:        %s (%d/%d passed)\n",
            RESULT_STR[conv_bad > 0],
            (int)(sizeof(checks)/sizeof(checks[0])) + 1 - conv_bad,
            (int)(sizeof(checks)/sizeof(checks[0])) + 1);
    }

    // ========================================================================
    // FP6 pack/unpack round-trip
    // ========================================================================
    {
        int pack_bad = 0;

        // Test all 64 possible 6-bit values in each of the 4 positions
        for (int pos = 0; pos < 4; pos++) {
            for (int val = 0; val < 64; val++) {
                uint8_t in[4] = {0, 0, 0, 0};
                in[pos] = (uint8_t)val;

                uint8_t packed[3], out[4];
                ggml_mxfp_pack_fp6x4(in, packed);
                ggml_mxfp_unpack_fp6x4(packed, out);

                if (out[pos] != (uint8_t)val) {
                    if (pack_bad == 0 || verbose) {
                        printf("  fp6 pack roundtrip: pos=%d val=0x%02X -> got 0x%02X\n",
                               pos, val, out[pos]);
                    }
                    pack_bad++;
                }
                // no crosstalk
                for (int k = 0; k < 4; k++) {
                    if (k != pos && out[k] != 0) {
                        if (pack_bad == 0 || verbose) {
                            printf("  fp6 pack crosstalk: pos=%d val=0x%02X leaked to pos=%d (0x%02X)\n",
                                   pos, val, k, out[k]);
                        }
                        pack_bad++;
                    }
                }
            }
        }

        // known-answer: [0x3F, 0x00, 0x3F, 0x00] -> {0x3F, 0xF0, 0x03}
        {
            uint8_t in[4] = {0x3F, 0x00, 0x3F, 0x00};
            uint8_t packed[3];
            ggml_mxfp_pack_fp6x4(in, packed);
            uint8_t expected[3] = {0x3F, 0xF0, 0x03};
            if (packed[0] != expected[0] || packed[1] != expected[1] || packed[2] != expected[2]) {
                if (pack_bad == 0 || verbose) {
                    printf("  fp6 known-answer: packed [%02X,%02X,%02X] expected [%02X,%02X,%02X]\n",
                           packed[0], packed[1], packed[2], expected[0], expected[1], expected[2]);
                }
                pack_bad++;
            }
        }

        check_test(pack_bad == 0, "  fp6 pack/unpack round-trip:           %s\n", RESULT_STR[pack_bad > 0]);
    }

    // ========================================================================
    // E8M0 tests
    // ========================================================================

    // E8M0 known-answer decode + HALF vs FULL (MXFP4 uses HALF, MXFP6/8 use FULL)
    {
        int e8m0_bad = 0;

        struct { uint8_t e; float expected; } e8m0_known[] = {
            {   0, 5.87747175e-39f }, // 2^(-127) (E8M0 special case)
            {   1, 1.17549435e-38f }, // 2^(-126) (min normal)
            { 126, 0.5f },     // 2^(126-127) = 2^(-1) = 0.5
            { 127, 1.0f },     // 2^(127-127) = 2^0 = 1.0
            { 128, 2.0f },     // 2^(128-127) = 2^1 = 2.0
            { 254, 1.70141183e+38f }, // 2^127 (max representable)
        };
        for (const auto & t : e8m0_known) {
            float got = ggml_e8m0_to_fp32(t.e);
            if (got != t.expected) {
                if (e8m0_bad == 0 || verbose) {
                    printf("  E8M0 decode e=%d: got %.8g, expected %.8g\n", t.e, got, t.expected);
                }
                e8m0_bad++;
            }
        }

        // HALF must be exactly half of FULL for all valid exponents
        for (int e = 0; e < 255; e++) {
            float full = ggml_e8m0_to_fp32((uint8_t)e);
            float half = ggml_e8m0_to_fp32_half((uint8_t)e);
            if (half != full * 0.5f) {
                if (e8m0_bad == 0 || verbose) {
                    printf("  E8M0 HALF!=FULL/2 at e=%d: half=%.8g, full/2=%.8g\n", e, half, full * 0.5f);
                }
                e8m0_bad++;
                break;
            }
        }

        check_test(e8m0_bad == 0, "  E8M0 known-answer + HALF/FULL:       %s\n", RESULT_STR[e8m0_bad > 0]);
    }

    // E8M0 rounding at sqrt(2) threshold
    {
        int round_bad = 0;

        {
            int e = ggml_mxfp_e8m0_base_estimate(1.0f, 0);
            if (e != 127) {
                printf("  E8M0 round: amax=1.0 -> e=%d, expected 127\n", e);
                round_bad++;
            }
        }
        {
            int e = ggml_mxfp_e8m0_base_estimate(2.0f, 0);
            if (e != 128) {
                printf("  E8M0 round: amax=2.0 -> e=%d, expected 128\n", e);
                round_bad++;
            }
        }
        {
            float below = 1.4142f;
            int e = ggml_mxfp_e8m0_base_estimate(below, 0);
            if (e != 127) {
                printf("  E8M0 round: amax=%.6f -> e=%d, expected 127 (no round)\n", below, e);
                round_bad++;
            }
        }
        {
            float at_sqrt2 = 1.41422f;
            int e = ggml_mxfp_e8m0_base_estimate(at_sqrt2, 0);
            if (e != 128) {
                printf("  E8M0 round: amax=%.6f -> e=%d, expected 128 (rounds up)\n", at_sqrt2, e);
                round_bad++;
            }
        }
        {
            int e_no_off = ggml_mxfp_e8m0_base_estimate(MXFP8_E4M3_MAX_FINITE, 0);
            int e_e4m3   = ggml_mxfp_e8m0_base_estimate(MXFP8_E4M3_MAX_FINITE, MXFP8_E4M3_EMAX_OFFSET);
            if (e_no_off - e_e4m3 != MXFP8_E4M3_EMAX_OFFSET) {
                printf("  E8M0 emax_offset: diff=%d, expected %d\n",
                       e_no_off - e_e4m3, MXFP8_E4M3_EMAX_OFFSET);
                round_bad++;
            }
        }

        check_test(round_bad == 0, "  E8M0 rounding boundary:              %s\n", RESULT_STR[round_bad > 0]);
    }

    // E8M0 scale computation: verify base exponent is reasonable for various amax values
    {
        const float test_amax[] = { 0.001f, 0.1f, 1.0f, MXFP4_E2M1_MAX_FINITE, 100.0f, MXFP8_E4M3_MAX_FINITE, 10000.0f };
        int bad = 0;
        for (float amax : test_amax) {
            int e_base = ggml_mxfp_e8m0_base_estimate(amax, 0);
            if (e_base < 1 || e_base > 254) {
                if (bad == 0 || verbose) {
                    printf("  E8M0 bad e_base=%d for amax=%.4f\n", e_base, amax);
                }
                bad++;
                continue;
            }
            float scale = ggml_e8m0_to_fp32((uint8_t)e_base);
            float ratio = amax / scale;
            if (ratio < 0.25f || ratio > 4.0f) {
                if (bad == 0 || verbose) {
                    printf("  E8M0 scale=%.6g for amax=%.4f, ratio=%.4f (expected ~1)\n",
                           scale, amax, ratio);
                }
                bad++;
            }
        }
        check_test(bad == 0,
            "  E8M0 scale sanity check:             %s (%d/%d passed)\n",
            RESULT_STR[bad > 0], (int)(sizeof(test_amax)/sizeof(test_amax[0])) - bad,
            (int)(sizeof(test_amax)/sizeof(test_amax[0])));
    }

    // EMAX_OFFSET produces valid E8M0 for each format's max finite value
    {
        int emax_bad = 0;
        for (int i = 0; i < n_mxfp_fmts; i++) {
            const auto & f = mxfp_fmts[i];
            int e_base = ggml_mxfp_e8m0_base_estimate(f.max_finite, f.emax_offset);
            if (e_base < 1 || e_base > 254) {
                if (emax_bad == 0 || verbose) {
                    printf("  %s emax_offset=%d: max_finite=%.1f gives e_base=%d (out of range)\n",
                           f.name, f.emax_offset, f.max_finite, e_base);
                }
                emax_bad++;
            }
        }
        check_test(emax_bad == 0, "  EMAX_OFFSET vs format max:           %s\n", RESULT_STR[emax_bad > 0]);
    }

    // EMAX_OFFSET no-clip: verify that no representable amax causes clipping
    {
        int clip_bad = 0;
        for (int i = 0; i < n_mxfp_fmts; i++) {
            const auto & f = mxfp_fmts[i];
            for (float amax = 0.01f; amax < f.max_finite * 4; amax *= 1.01f) {
                int e_base = ggml_mxfp_e8m0_base_estimate(amax, f.emax_offset);
                if (e_base < 0) e_base = 0;
                if (e_base > 254) e_base = 254;
                float scale = ggml_e8m0_to_fp32((uint8_t)e_base);
                float normalized_max = amax / scale;
                if (normalized_max > f.max_finite * 1.001f) {
                    if (clip_bad == 0 || verbose) {
                        printf("  %s clip: amax=%.4f scale=%.4g normalized=%.4f > max=%.1f\n",
                               f.name, amax, scale, normalized_max, f.max_finite);
                    }
                    clip_bad++;
                }
            }
        }
        check_test(clip_bad == 0, "  EMAX_OFFSET no-clip guarantee:        %s\n", RESULT_STR[clip_bad > 0]);
    }

    // ========================================================================
    // SoA layout and type system tests
    // ========================================================================

    // SoA layout: verify qs + e8m0 regions sum to row_size
    {
        bool section_failed = false;
        for (int i = 0; i < n_mxfp_fmts; i++) {
            const auto & f = mxfp_fmts[i];
            for (int nblocks : { 1, 4, 8, 32 }) {
                size_t qs_bytes   = (size_t)nblocks * f.qs_per_block;
                size_t e8m0_bytes = (size_t)nblocks;
                size_t total      = qs_bytes + e8m0_bytes;
                size_t row_size   = ggml_row_size(f.type, nblocks * 32);

                if (total != row_size) {
                    section_failed = true;
                    num_failed++;
                    if (verbose) {
                        printf("  %s SoA layout nblocks=%d: qs(%zu)+e8m0(%zu)=%zu != row_size(%zu)\n",
                               f.name, nblocks, qs_bytes, e8m0_bytes, total, row_size);
                    }
                }
            }
        }
        if (section_failed || verbose) {
            printf("  SoA layout size check:               %s\n", RESULT_STR[section_failed]);
        }
    }

    // block size consistency
    {
        bool ok = (QK_MXFP4 == 32 && QK_MXFP8 == 32 && QK_MXFP6 == 32);
        check_test(ok,
            "  MXFP block size == 32:               %s (QK4=%d, QK6=%d, QK8=%d)\n",
            RESULT_STR[!ok], QK_MXFP4, QK_MXFP6, QK_MXFP8);
    }

    // ggml_is_mxfp dispatch
    {
        bool check_ok = true;
        for (int i = 0; i < n_mxfp_fmts; i++) {
            if (!ggml_is_mxfp(mxfp_fmts[i].type)) { check_ok = false; }
        }
        if (ggml_is_mxfp(GGML_TYPE_F16)) { check_ok = false; }
        check_test(check_ok, "  ggml_is_mxfp:                   %s\n", RESULT_STR[!check_ok]);
    }

    // zero block produces E8M0=0
    {
        float zeros[32] = {};
        for (int i = 0; i < n_mxfp_fmts; i++) {
            const auto & f = mxfp_fmts[i];
            const size_t buf_size = ggml_row_size(f.type, 32);
            std::vector<uint8_t> buf(buf_size, 0xFF);

            ggml_mxfp_quantize_soa(f.type, zeros, buf.data(), 32, false);

            uint8_t e8m0 = buf[f.qs_per_block];
            check_test(e8m0 == 0,
                "%5s zero block E8M0:                   %s (e8m0=%d, expected 0)\n",
                f.name, RESULT_STR[e8m0 != 0], e8m0);
        }
    }

    // ========================================================================
    // SoA format spec: quantize, manually walk raw bytes, compare against reference dequant
    // Each format has genuinely different packing, so these are kept separate.
    // ========================================================================
    {
        const int nblocks = 2;
        const int nelems = nblocks * 32;
        float input[64];
        for (int i = 0; i < 64; i++) {
            input[i] = (i < 32) ? 0.1f * sinf(i + 0.5f) : 3.0f * cosf(i + 0.5f);
        }

        // MXFP4
        {
            const size_t buf_size = ggml_row_size(GGML_TYPE_MXFP4, nelems);
            std::vector<uint8_t> buf(buf_size);
            std::vector<float> ref_out(nelems);
            std::vector<float> manual_out(nelems);

            ggml_mxfp_quantize_soa(GGML_TYPE_MXFP4, input, buf.data(), nelems, false);
            ggml_mxfp_dequantize_soa(GGML_TYPE_MXFP4, buf.data(), ref_out.data(), nelems);

            const uint8_t * qs   = buf.data();
            const uint8_t * e8m0 = qs + nblocks * MXFP_QS_PER_BLOCK_E2M1;

            for (int b = 0; b < nblocks; b++) {
                const float d = ggml_e8m0_to_fp32(e8m0[b]);
                const uint8_t * block_qs = &qs[b * MXFP_QS_PER_BLOCK_E2M1];
                for (int j = 0; j < 16; j++) {
                    manual_out[b*32 + j]      = ggml_mxfp_fp4_e2m1_to_float(block_qs[j] & 0x0F) * d;
                    manual_out[b*32 + j + 16] = ggml_mxfp_fp4_e2m1_to_float(block_qs[j] >>   4) * d;
                }
            }

            int mm = bitwise_mismatches(ref_out.data(), manual_out.data(), nelems);
            check_test(mm == 0,
                "mxfp4 SoA format spec:                 %s (%d/%d match)\n",
                RESULT_STR[mm > 0], nelems - mm, nelems);
        }

        // MXFP6
        {
            const size_t buf_size = ggml_row_size(GGML_TYPE_MXFP6, nelems);
            std::vector<uint8_t> buf(buf_size);
            std::vector<float> ref_out(nelems);
            std::vector<float> manual_out(nelems);

            ggml_mxfp_quantize_soa(GGML_TYPE_MXFP6, input, buf.data(), nelems, false);
            ggml_mxfp_dequantize_soa(GGML_TYPE_MXFP6, buf.data(), ref_out.data(), nelems);

            const uint8_t * qs   = buf.data();
            const uint8_t * e8m0 = qs + nblocks * MXFP_QS_PER_BLOCK_E2M3;

            for (int b = 0; b < nblocks; b++) {
                const float d = ggml_e8m0_to_fp32(e8m0[b]);
                const uint8_t * block_qs = &qs[b * MXFP_QS_PER_BLOCK_E2M3];
                for (int j = 0; j < 32; j += 4) {
                    uint8_t vals[4];
                    ggml_mxfp_unpack_fp6x4(&block_qs[j * 3 / 4], vals);
                    for (int k = 0; k < 4; k++) {
                        manual_out[b*32 + j + k] = ggml_mxfp_fp6_e2m3_to_float(vals[k]) * d;
                    }
                }
            }

            int mm = bitwise_mismatches(ref_out.data(), manual_out.data(), nelems);
            check_test(mm == 0,
                "mxfp6 SoA format spec:                 %s (%d/%d match)\n",
                RESULT_STR[mm > 0], nelems - mm, nelems);
        }

        // MXFP8
        {
            const size_t buf_size = ggml_row_size(GGML_TYPE_MXFP8, nelems);
            std::vector<uint8_t> buf(buf_size);
            std::vector<float> ref_out(nelems);
            std::vector<float> manual_out(nelems);

            ggml_mxfp_quantize_soa(GGML_TYPE_MXFP8, input, buf.data(), nelems, false);
            ggml_mxfp_dequantize_soa(GGML_TYPE_MXFP8, buf.data(), ref_out.data(), nelems);

            const uint8_t * qs   = buf.data();
            const uint8_t * e8m0 = qs + nblocks * MXFP_QS_PER_BLOCK_E4M3;

            for (int b = 0; b < nblocks; b++) {
                const float d = ggml_e8m0_to_fp32(e8m0[b]);
                const uint8_t * block_qs = &qs[b * MXFP_QS_PER_BLOCK_E4M3];
                for (int j = 0; j < 32; j++) {
                    manual_out[b*32 + j] = ggml_mxfp_fp8_e4m3_to_float(block_qs[j]) * d;
                }
            }

            int mm = bitwise_mismatches(ref_out.data(), manual_out.data(), nelems);
            check_test(mm == 0,
                "mxfp8 SoA format spec:                 %s (%d/%d match)\n",
                RESULT_STR[mm > 0], nelems - mm, nelems);
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
