// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

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
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 = 0.0070f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 = 0.0020f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_MXFP = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


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

    int num_failed = 0;
    bool failed = false;

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
                type == GGML_TYPE_NVFP4       ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 :
                type == GGML_TYPE_MXFP4_E2M1 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 :
                type == GGML_TYPE_MXFP6_E2M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 :
                type == GGML_TYPE_MXFP8_E4M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 : MAX_QUANTIZATION_TOTAL_ERROR;
            failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                                            type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                                          ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                                          : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0
                                          ? MAX_DOT_PRODUCT_ERROR_TERNARY
                                          : type == GGML_TYPE_NVFP4
                                          ? MAX_DOT_PRODUCT_ERROR_FP4
                                          : type == GGML_TYPE_MXFP4_E2M1 || type == GGML_TYPE_MXFP6_E2M3 || type == GGML_TYPE_MXFP8_E4M3
                                          ? MAX_DOT_PRODUCT_ERROR_MXFP
                                          : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    // MXFP SoA roundtrip: test from_float_soa → to_float_soa through the traits system
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (!qfns_cpu->from_float_soa || !qfns_cpu->to_float_soa) {
            continue;
        }

        const size_t buf_size = ggml_row_size(type, test_size);
        std::vector<uint8_t> tmp_q(buf_size);
        std::vector<float> tmp_out(test_size);

        qfns_cpu->from_float_soa(test_data.data(), tmp_q.data(), test_size);
        qfns_cpu->to_float_soa(tmp_q.data(), tmp_out.data(), test_size);

        const float soa_error = array_rmse(test_data.data(), tmp_out.data(), test_size);
        const float max_soa_error =
            type == GGML_TYPE_MXFP4_E2M1 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 :
            type == GGML_TYPE_MXFP6_E2M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 :
            type == GGML_TYPE_MXFP8_E4M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 : MAX_QUANTIZATION_TOTAL_ERROR;
        failed = !(soa_error < max_soa_error);
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s SoA quantization error:          %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], soa_error);
        }
    }

    // MXFP traits guards: ALL MXFP flash attention MUST use SoA layout.
    // Every MXFP type must have SoA traits for flash attention KV cache.
    // MXFP6/MXFP8 are KV-cache-only — they must NOT have AoS CPU dequant.
    // MXFP4 has AoS for model weights (upstream) but flash attention still uses SoA.
    {
        const ggml_type all_mxfp_types[] = { GGML_TYPE_MXFP4_E2M1, GGML_TYPE_MXFP8_E4M3, GGML_TYPE_MXFP6_E2M3 };
        for (ggml_type type : all_mxfp_types) {
            const auto * cpu = ggml_get_type_traits_cpu(type);

            // ALL MXFP types must have SoA paths (flash attention KV cache uses SoA exclusively)
            failed = !(cpu->from_float_soa && cpu->to_float_soa);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s SoA traits present:               %s\n", ggml_type_name(type), RESULT_STR[failed]);
            }
        }

        // MXFP6/MXFP8 are KV-cache-only — must NOT have AoS CPU dequant
        const ggml_type kv_only_types[] = { GGML_TYPE_MXFP8_E4M3, GGML_TYPE_MXFP6_E2M3 };
        for (ggml_type type : kv_only_types) {
            const auto * cpu = ggml_get_type_traits_cpu(type);
            failed = (cpu->to_float != nullptr);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s AoS CPU to_float absent:          %s\n", ggml_type_name(type), RESULT_STR[failed]);
            }
        }
    }

    // Hadamard orthogonality: H(H(x)) == x (self-inverse with 1/sqrt(32) normalization)
    {
        float original[32], transformed[32];
        // Use varied test data (not all zeros or constants)
        for (int i = 0; i < 32; i++) {
            original[i] = 0.1f + 2.0f * cosf(i + 0.5f);
            transformed[i] = original[i];
        }
        ggml_hadamard_32_inplace(transformed);
        ggml_hadamard_32_inplace(transformed); // apply twice = identity

        float max_err = 0.0f;
        for (int i = 0; i < 32; i++) {
            float err = fabsf(transformed[i] - original[i]);
            if (err > max_err) max_err = err;
        }
        // Should be exact up to floating-point rounding (~1e-6 for float)
        failed = !(max_err < 1e-5f);
        num_failed += failed;
        if (failed || verbose) {
            printf("hadamard H(H(x))==x roundtrip:         %s (max_err=%.2e)\n", RESULT_STR[failed], max_err);
        }
    }

    // SoA SIMD vs scalar reference cross-check
    {
        struct soa_cross_check {
            ggml_type type;
            void (*ref_dequant)(const void *, float *, int64_t);
        };

        const soa_cross_check checks[] = {
            { GGML_TYPE_MXFP4_E2M1, dequantize_row_mxfp4_soa },
            { GGML_TYPE_MXFP8_E4M3, dequantize_row_mxfp8_soa },
            { GGML_TYPE_MXFP6_E2M3, dequantize_row_mxfp6_soa },
        };

        for (const auto & c : checks) {
            const auto * cpu = ggml_get_type_traits_cpu(c.type);
            if (!cpu->from_float_soa || !cpu->to_float_soa) continue;

            const size_t buf_size = ggml_row_size(c.type, test_size);
            std::vector<uint8_t> tmp_q(buf_size);
            std::vector<float> out_ref(test_size);
            std::vector<float> out_simd(test_size);

            // Quantize with SoA
            cpu->from_float_soa(test_data.data(), tmp_q.data(), test_size);

            // Dequant with scalar reference
            c.ref_dequant(tmp_q.data(), out_ref.data(), test_size);

            // Dequant with CPU/SIMD path
            cpu->to_float_soa(tmp_q.data(), out_simd.data(), test_size);

            // Compare bitwise
            int mismatches = 0;
            for (size_t j = 0; j < test_size; j++) {
                uint32_t a, b;
                memcpy(&a, &out_ref[j], 4);
                memcpy(&b, &out_simd[j], 4);
                if (a != b) mismatches++;
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s SoA SIMD vs scalar ref:           %s (%zu/%zu match)\n",
                       ggml_type_name(c.type), RESULT_STR[failed],
                       test_size - mismatches, test_size);
            }
        }
    }

    // MXFP element converter validation against canonical LUT reference values.
    // Tests that IEEE-754 bit reconstruction in converters matches the OCP MX spec tables.
    {
        struct lut_test {
            const char * name;
            const float * lut;
            int           count;
            float       (*converter)(uint8_t);
        };

        const lut_test lut_tests[] = {
            { "fp8_e4m3", kvalues_mxfp8_e4m3, 256, fp8_e4m3_to_float },
            { "fp8_e5m2", kvalues_mxfp8_e5m2, 256, fp8_e5m2_to_float },
            { "fp6_e2m3", kvalues_mxfp6_e2m3,  64, fp6_e2m3_to_float },
            { "fp6_e3m2", kvalues_mxfp6_e3m2,  64, fp6_e3m2_to_float },
        };

        for (const auto & t : lut_tests) {
            int mismatches = 0;
            for (int i = 0; i < t.count; i++) {
                const float converter_val = t.converter((uint8_t)i);
                const float lut_val       = t.lut[i];

                // Both NaN → match. Otherwise must be bitwise identical.
                if (isnan(converter_val) && isnan(lut_val)) continue;
                if (converter_val != lut_val) {
                    if (mismatches == 0 || verbose) {
                        printf("  %s LUT mismatch at [%d]: converter=%.8g, lut=%.8g\n",
                               t.name, i, converter_val, lut_val);
                    }
                    mismatches++;
                }
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s converter vs LUT:                %s (%d/%d values match)\n",
                       t.name, RESULT_STR[failed], t.count - mismatches, t.count);
            }
        }

        // FP4 E2M1: converter is in ggml-common.h (static inline), LUT is kvalues_mxfp4_float
        {
            int mismatches = 0;
            for (int i = 0; i < 16; i++) {
                const float converter_val = ggml_mxfp_fp4_e2m1_to_float((uint8_t)i);
                const float lut_val       = kvalues_mxfp4_float[i];
                if (converter_val != lut_val) {
                    if (mismatches == 0 || verbose) {
                        printf("  fp4_e2m1 LUT mismatch at [%d]: converter=%.8g, lut=%.8g\n",
                               i, converter_val, lut_val);
                    }
                    mismatches++;
                }
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("fp4_e2m1 converter vs LUT:                %s (%d/16 values match)\n",
                       RESULT_STR[failed], 16 - mismatches);
            }
        }
    }

    // E8M0 scale computation: verify base exponent is reasonable for various amax values
    {
        const float test_amax[] = { 0.001f, 0.1f, 1.0f, 6.0f, 100.0f, 448.0f, 10000.0f };
        int bad = 0;
        for (float amax : test_amax) {
            // ggml_mxfp_e8m0_base_estimate returns unclamped e_base
            int e_base = ggml_mxfp_e8m0_base_estimate(amax, 0);
            if (e_base < 1 || e_base > 254) {
                if (bad == 0 || verbose) {
                    printf("  E8M0 bad e_base=%d for amax=%.4f\n", e_base, amax);
                }
                bad++;
                continue;
            }
            float scale = ggml_mxfp_e8m0_to_fp32((uint8_t)e_base);
            // Scale should be within 2x of amax (rough sanity check)
            float ratio = amax / scale;
            if (ratio < 0.25f || ratio > 4.0f) {
                if (bad == 0 || verbose) {
                    printf("  E8M0 scale=%.6g for amax=%.4f, ratio=%.4f (expected ~1)\n",
                           scale, amax, ratio);
                }
                bad++;
            }
        }
        failed = (bad > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  E8M0 scale sanity check:             %s (%d/%d passed)\n",
                   RESULT_STR[failed], (int)(sizeof(test_amax)/sizeof(test_amax[0])) - bad,
                   (int)(sizeof(test_amax)/sizeof(test_amax[0])));
        }
    }

    // SoA layout: verify offset macros produce correct byte positions
    {
        const struct { ggml_type type; int qs_per_block; } soa_types[] = {
            { GGML_TYPE_MXFP4_E2M1, MXFP4_SOA_QS_PER_BLOCK },
            { GGML_TYPE_MXFP8_E4M3, MXFP8_SOA_QS_PER_BLOCK },
            { GGML_TYPE_MXFP6_E2M3, MXFP6_SOA_QS_PER_BLOCK },
        };

        for (const auto & st : soa_types) {
            for (int nblocks : { 1, 4, 8, 32 }) {
                size_t expected_e8m0_off = (size_t)nblocks * st.qs_per_block;
                size_t actual_e8m0_off = MXFP_SOA_E8M0_OFFSET(nblocks, st.qs_per_block);
                size_t total = actual_e8m0_off + nblocks; // e8m0 region = 1 byte per block
                size_t row_size = ggml_row_size(st.type, nblocks * 32);

                bool offset_ok = (actual_e8m0_off == expected_e8m0_off);
                bool size_ok = (total == row_size);

                if (!offset_ok || !size_ok) {
                    failed = true;
                    num_failed++;
                    if (verbose) {
                        printf("  %s SoA layout nblocks=%d: e8m0_off=%zu (expected %zu), total=%zu (row_size=%zu)\n",
                               ggml_type_name(st.type), nblocks, actual_e8m0_off, expected_e8m0_off, total, row_size);
                    }
                }
            }
        }
        if (verbose) {
            printf("  SoA layout offset check:             %s\n", RESULT_STR[0]); // only prints failures above
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
