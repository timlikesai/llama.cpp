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
#include <stdarg.h>
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
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP6_E3M2 = 0.0020f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP8_E5M2 = 0.0012f;
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP6_E3M2 = 0.0015f;
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP8_E5M2 = 0.0012f;

constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_MXFP = 0.012f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};

// Generate canonical dequant LUT from the OCP MX spec formulas — NOT from our converters.
// This is the ground truth that any implementation (CPU, Metal, CUDA) must match.
// Ref: OCP MX Spec v1.0 Tables 4-5 (E2M1, E2M3), OCP FP8 Table 1 (E4M3).
static float spec_decode_mxfp(int exp_bits, int man_bits, int bias, uint8_t v) {
    const int total   = 1 + exp_bits + man_bits; // sign + exp + mantissa
    const int sign    = (v >> (total - 1)) & 1;
    const int exp     = (v >> man_bits) & ((1 << exp_bits) - 1);
    const int man     = v & ((1 << man_bits) - 1);
    const int max_exp = (1 << exp_bits) - 1;
    const int max_man = (1 << man_bits) - 1;

    if (exp == max_exp && man == max_man && exp_bits == 4) return NAN; // E4M3 NaN
    if (exp == max_exp && exp_bits == 5) { // E5M2 IEEE Inf/NaN
        if (man == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    float val;
    if (exp == 0) {
        val = (float)man * powf(2.0f, 1.0f - bias - man_bits); // subnormal
    } else {
        val = (1.0f + (float)man / (float)(1 << man_bits)) * powf(2.0f, (float)(exp - bias)); // normal
    }
    return sign ? -val : val;
}

static std::vector<float> generate_spec_lut(int exp_bits, int man_bits, int bias) {
    int n = 1 << (1 + exp_bits + man_bits);
    std::vector<float> lut(n);
    for (int i = 0; i < n; i++) lut[i] = spec_decode_mxfp(exp_bits, man_bits, bias, (uint8_t)i);
    return lut;
}

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

struct mxfp_test_format {
    ggml_type     type;
    uint8_t       nan_bits;          // bit pattern for NaN (0 = no NaN in format)
    float         max_quant_error;   // test threshold: SoA roundtrip RMSE
    float         max_pipeline_error;// test threshold: Hadamard pipeline RMSE
    int           exp_bits;          // for spec LUT generation (2 for FP4/FP6, 4 for FP8)
    int           man_bits;          // for spec LUT generation (1, 3, or 3)
    int           bias;              // for spec LUT generation (1, 1, or 7)

    // everything else derived from ggml type/format traits
    const char * name()         const { return ggml_get_type_traits(type)->type_name; }
    int          qs_per_block() const { return (int)ggml_get_type_traits(type)->type_size - 1; }
    int          n_vals()       const { return 1 << (qs_per_block() * 8 / 32); }
    const ggml_mxfp_format_traits * fmt() const { return ggml_mxfp_get_format_traits(type); }
};

static void hadamard_blocks(float * data, size_t n) {
    for (size_t b = 0; b < n / 32; b++) {
        ggml_mxfp_hadamard_32_inplace(&data[b * 32]);
    }
}

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static int bitwise_mismatches(const float * a, const float * b, int n) {
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        uint32_t ua, ub;
        memcpy(&ua, &a[i], 4);
        memcpy(&ub, &b[i], 4);
        if (ua != ub) mismatches++;
    }
    return mismatches;
}

static int soa_format_spec_check(ggml_type type, const float * input, int nelems) {
    const auto * fmt = ggml_mxfp_get_format_traits(type);
    const int nblocks = nelems / 32;
    const int qpb = fmt->qs_per_block;
    std::vector<uint8_t> buf(ggml_row_size(type, nelems));
    std::vector<float> ref_out(nelems), manual_out(nelems);

    ggml_mxfp_quantize_soa(type, input, buf.data(), nelems, false);
    ggml_mxfp_dequantize_soa(type, buf.data(), ref_out.data(), nelems);

    const uint8_t * qs = buf.data(), * e8m0 = qs + nblocks * qpb;
    for (int b = 0; b < nblocks; b++) {
        const float d = ggml_e8m0_to_fp32(e8m0[b]);
        const uint8_t * bqs = &qs[b * qpb];
        float * out = &manual_out[b * 32];
        switch (type) {
            case GGML_TYPE_MXFP4:
                for (int j = 0; j < 16; j++) {
                    out[j]      = fmt->to_float(bqs[j] & 0x0F) * d;
                    out[j + 16] = fmt->to_float(bqs[j] >>   4) * d;
                }
                break;
            case GGML_TYPE_MXFP6:
            case GGML_TYPE_MXFP6_E3M2:
                for (int j = 0; j < 32; j += 4) {
                    uint8_t v[4]; ggml_mxfp_unpack_fp6x4(&bqs[j * 3 / 4], v);
                    for (int k = 0; k < 4; k++) out[j + k] = fmt->to_float(v[k]) * d;
                }
                break;
            case GGML_TYPE_MXFP8:
            case GGML_TYPE_MXFP8_E5M2:
                for (int j = 0; j < 32; j++) out[j] = fmt->to_float(bqs[j]) * d;
                break;
            default: break;
        }
    }
    return bitwise_mismatches(ref_out.data(), manual_out.data(), nelems);
}

struct test_runner {
    int  num_failed = 0;
    bool verbose    = false;

    void check(bool condition, const char * fmt, ...) __attribute__((format(printf, 3, 4))) {
        num_failed += !condition;
        if (!condition || verbose) {
            va_list args;
            va_start(args, fmt);
            vprintf(fmt, args);
            va_end(args);
        }
    }
};

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

    test_runner T;
    T.verbose = verbose;

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
                type == GGML_TYPE_MXFP8   ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 :
                type == GGML_TYPE_MXFP6_E3M2 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP6_E3M2 :
                type == GGML_TYPE_MXFP8_E5M2 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP8_E5M2 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool ok = total_error < max_quantization_error;
            T.check(ok,
                "%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[!ok], total_error);

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            ok = reference_error < MAX_QUANTIZATION_REFERENCE_ERROR;
            T.check(ok,
                "%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[!ok], reference_error);

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                                            type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                                          ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                                          : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0
                                          ? MAX_DOT_PRODUCT_ERROR_TERNARY
                                          : type == GGML_TYPE_NVFP4
                                          ? MAX_DOT_PRODUCT_ERROR_FP4
                                          : type == GGML_TYPE_MXFP4 || type == GGML_TYPE_MXFP6 || type == GGML_TYPE_MXFP8 || type == GGML_TYPE_MXFP6_E3M2 || type == GGML_TYPE_MXFP8_E5M2
                                          ? MAX_DOT_PRODUCT_ERROR_MXFP
                                          : MAX_DOT_PRODUCT_ERROR;
            ok = vec_dot_error < max_allowed_error;
            T.check(ok,
                "%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[!ok], vec_dot_error);
        }
    }

    // Hadamard: self-inverse + known-answer
    {
        float data[32];
        for (int i = 0; i < 32; i++) data[i] = 0.1f + 2.0f * cosf(i + 0.5f);
        float orig[32]; memcpy(orig, data, sizeof(data));
        ggml_mxfp_hadamard_32_inplace(data);
        ggml_mxfp_hadamard_32_inplace(data);
        float err = max_abs_diff(data, orig, 32);
        T.check(err < 1e-5f, "  hadamard self-inverse:               %s (err=%.2e)\n", RESULT_STR[!(err < 1e-5f)], err);

        memset(data, 0, sizeof(data)); data[0] = 1.0f;
        ggml_mxfp_hadamard_32_inplace(data);
        float expect[32]; for (int i = 0; i < 32; i++) expect[i] = MXFP_HADAMARD_32_NORM;
        err = max_abs_diff(data, expect, 32);
        T.check(err < 1e-7f, "  hadamard unit vector:                %s (err=%.2e)\n", RESULT_STR[!(err < 1e-7f)], err);
    }

    const mxfp_test_format mxfp_fmts[] = {
        { GGML_TYPE_MXFP4,    0, MAX_QUANTIZATION_TOTAL_ERROR_MXFP4, MAX_MXFP_PIPELINE_ERROR_MXFP4, 2, 1, 1 },
        { GGML_TYPE_MXFP6,    0, MAX_QUANTIZATION_TOTAL_ERROR_MXFP6, MAX_MXFP_PIPELINE_ERROR_MXFP6, 2, 3, 1 },
        { GGML_TYPE_MXFP8, 0x7F, MAX_QUANTIZATION_TOTAL_ERROR_MXFP8, MAX_MXFP_PIPELINE_ERROR_MXFP8, 4, 3, 7 },
        { GGML_TYPE_MXFP6_E3M2, 0,    MAX_QUANTIZATION_TOTAL_ERROR_MXFP6_E3M2, MAX_MXFP_PIPELINE_ERROR_MXFP6_E3M2, 3, 2, 3 },
        { GGML_TYPE_MXFP8_E5M2, 0x7F, MAX_QUANTIZATION_TOTAL_ERROR_MXFP8_E5M2, MAX_MXFP_PIPELINE_ERROR_MXFP8_E5M2, 5, 2, 15 },
    };
    const int n_mxfp_fmts = sizeof(mxfp_fmts) / sizeof(mxfp_fmts[0]);

    // Per-format: Hadamard pipeline roundtrip (the end-to-end KV cache path)
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        std::vector<float> original(test_size);
        generate_data(2.0, test_size, original.data());
        std::vector<float> recovered(original.begin(), original.end());
        hadamard_blocks(recovered.data(), test_size);
        std::vector<uint8_t> qbuf(ggml_row_size(f.type, test_size));
        ggml_mxfp_quantize_soa(f.type, recovered.data(), qbuf.data(), test_size, false);
        ggml_mxfp_dequantize_soa(f.type, qbuf.data(), recovered.data(), test_size);
        hadamard_blocks(recovered.data(), test_size);
        float err = array_rmse(original.data(), recovered.data(), test_size);
        bool ok = err < f.max_pipeline_error;
        T.check(ok, "%5s Hadamard pipeline roundtrip:       %s (err=%.6f)\n", f.name(), RESULT_STR[!ok], err);
    }

    // Converter vs canonical LUT — ground truth for backend implementations
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        auto lut = generate_spec_lut(f.exp_bits, f.man_bits, f.bias);
        int bad = 0;
        for (int j = 0; j < f.n_vals(); j++) {
            float cv = f.fmt()->to_float((uint8_t)j), lv = lut[j];
            if (isnan(cv) && isnan(lv)) continue;
            if (cv != lv) { bad++; }
        }
        T.check(bad == 0, "%5s converter vs LUT:                %s (%d/%d match)\n",
            f.name(), RESULT_STR[bad > 0], f.n_vals() - bad, f.n_vals());
    }

    // Exhaustive element converter round-trip: quantize(dequantize(i)) == i for all codes
    for (int i = 0; i < n_mxfp_fmts; i++) {
        const auto & f = mxfp_fmts[i];
        int bad = 0;
        for (int j = 0; j < f.n_vals(); j++) {
            if ((uint8_t)j == f.nan_bits && f.nan_bits != 0) continue;
            float val = f.fmt()->to_float((uint8_t)j);
            if (isnan(val) || isinf(val)) continue;
            uint8_t back = f.fmt()->to_elem(val);
            if (back != (uint8_t)j && !(val == 0.0f && f.fmt()->to_float(back) == 0.0f)) { bad++; }
        }
        T.check(bad == 0, "%5s converter round-trip:             %s (%d/%d survived)\n",
            f.name(), RESULT_STR[bad > 0], f.n_vals() - bad, f.n_vals());
    }

    // Saturation + NaN edge cases (not covered by exhaustive roundtrip)
    {
        struct { ggml_type type; float input; uint8_t expected; } sat_cases[] = {
            { GGML_TYPE_MXFP4, 100.0f, 0x07 }, // saturate to max (6.0)
            { GGML_TYPE_MXFP6, 100.0f, 0x1F }, // saturate to max (7.5)
            { GGML_TYPE_MXFP8, 500.0f, 0x7E }, // saturate to max (448.0), not NaN (0x7F)
        };
        int bad = 0;
        for (const auto & c : sat_cases) {
            if (ggml_mxfp_get_format_traits(c.type)->to_elem(c.input) != c.expected) { bad++; }
        }
        if (!isnan(ggml_mxfp_fp8_e4m3_to_float(0x7F))) { bad++; } // 0x7F must be NaN
        T.check(bad == 0, "  saturation + NaN edge cases:         %s\n", RESULT_STR[bad > 0]);
    }

    // FP6 pack/unpack exhaustive round-trip + crosstalk
    {
        int bad = 0;
        for (int pos = 0; pos < 4; pos++) {
            for (int val = 0; val < 64; val++) {
                uint8_t in[4] = {0, 0, 0, 0};
                in[pos] = (uint8_t)val;
                uint8_t packed[3], out[4];
                ggml_mxfp_pack_fp6x4(in, packed);
                ggml_mxfp_unpack_fp6x4(packed, out);
                if (out[pos] != (uint8_t)val) { bad++; }
                for (int k = 0; k < 4; k++) {
                    if (k != pos && out[k] != 0) { bad++; }
                }
            }
        }
        T.check(bad == 0, "  fp6 pack/unpack round-trip:           %s\n", RESULT_STR[bad > 0]);
    }

    // E8M0: known-answer decode, HALF==FULL/2, round(log2) boundary, no-clip sweep
    {
        struct { uint8_t e; float decode_expected; float round_amax; int round_expected; } e8m0_cases[] = {
            { 0,   5.87747175e-39f,     1.0f, 127 }, // 2^(-127); exact power of 2
            { 127,            1.0f,     2.0f, 128 }, // 2^0;      exact power of 2
            { 128,            2.0f,  1.4142f, 127 }, // 2^1;      below sqrt(2) threshold
            { 254, 1.70141183e+38f, 1.41422f, 128 }, // 2^127;    above sqrt(2) threshold
        };
        int bad = 0;
        for (const auto & c : e8m0_cases) {
            if (ggml_e8m0_to_fp32(c.e) != c.decode_expected) { bad++; }
            if (ggml_mxfp_e8m0_base_estimate(c.round_amax, 0) != c.round_expected) { bad++; }
        }
        for (int e = 0; e < 255 && bad == 0; e++) { // HALF must equal FULL/2
            if (ggml_e8m0_to_fp32_half((uint8_t)e) != ggml_e8m0_to_fp32((uint8_t)e) * 0.5f) { bad++; }
        }
        for (int i = 0; i < n_mxfp_fmts; i++) { // no-clip: normalized max <= format max_finite
            const auto * fmt = mxfp_fmts[i].fmt();
            for (float amax = 0.01f; amax < fmt->max_finite * 4; amax *= 1.05f) {
                int e = ggml_mxfp_e8m0_base_estimate(amax, fmt->emax_offset);
                e = (e < 0) ? 0 : (e > 254 ? 254 : e);
                if (amax / ggml_e8m0_to_fp32((uint8_t)e) > fmt->max_finite * 1.001f) { bad++; }
            }
        }
        T.check(bad == 0, "  E8M0 decode + rounding + no-clip:    %s\n", RESULT_STR[bad > 0]);
    }

    // Per-format: ggml_is_mxfp, zero-block E8M0, SoA format spec byte walk
    {
        float zeros[32] = {};
        const int nelems = 64;
        float spec_input[64];
        for (int i = 0; i < 64; i++) {
            spec_input[i] = (i < 32) ? 0.1f * sinf(i + 0.5f) : 3.0f * cosf(i + 0.5f);
        }

        for (int i = 0; i < n_mxfp_fmts; i++) {
            const auto & f = mxfp_fmts[i];
            T.check(ggml_is_mxfp(f.type), "%5s ggml_is_mxfp:                      %s\n", f.name(), RESULT_STR[!ggml_is_mxfp(f.type)]);

            std::vector<uint8_t> buf(ggml_row_size(f.type, 32), 0xFF);
            ggml_mxfp_quantize_soa(f.type, zeros, buf.data(), 32, false);
            uint8_t e = buf[f.qs_per_block()];
            T.check(e == 0, "%5s zero block E8M0:                   %s\n", f.name(), RESULT_STR[e != 0]);

            int mm = soa_format_spec_check(f.type, spec_input, nelems);
            T.check(mm == 0, "%5s SoA format spec:                  %s\n", f.name(), RESULT_STR[mm > 0]);
        }
        T.check(!ggml_is_mxfp(GGML_TYPE_F16), "  ggml_is_mxfp(F16)==false:            %s\n", RESULT_STR[ggml_is_mxfp(GGML_TYPE_F16)]);
    }

    if (T.num_failed || verbose) {
        printf("%d tests failed\n", T.num_failed);
    }

    return T.num_failed > 0;
}
