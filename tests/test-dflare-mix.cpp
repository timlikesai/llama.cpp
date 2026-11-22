// test-dflare-mix.cpp: check the DFlare tap mixing of Inkling DSpark drafts against a numeric oracle.
//
// The graph under test is llama_model_dflash::build_dflare_alpha, build_dflare_taps and build_dflare_mix, the
// per-stage mixing of the target features that docs/mxfp8/33-drafter-reference-semantics.md transcribes from the
// vLLM reference: softmax the tap scores, weight the taps by that stage's row, normalize with that stage's
// dflare_norm.
//
// The expected values below come from tests/test_dflare_mix.py, which runs the reference in numpy float64.
// Regenerate them with 'python3 tests/test_dflare_mix.py' and paste over the k_ref_ blocks. Both sides draw
// the inputs from the same LCG in the same order (fuse, norm, feats); the *_in checksums pin that agreement.

#include "models.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static const double k_ref_tiny_in = 10.7829105;
static const float k_ref_tiny_alpha[] = { 0.196628219f, 0.214352356f, 0.356656112f, 0.232363313f, 0.302527521f, 0.225657071f, 0.19242085f, 0.279394557f, 0.314587056f, 0.247474298f, 0.153726556f, 0.28421209f, 0.199171061f, 0.306061989f, 0.2543511f, 0.24041585f };
static const float k_ref_tiny_mix0[] = { 0.213810164f, -0.456779553f, 1.02656426f, -0.403666622f, 0.399498522f, -0.147004928f, 0.389215041f, 0.239200273f, -1.1282274f, -0.527689512f, -0.0747975364f, -0.311063817f };
static const float k_ref_tiny_mix1[] = { 0.628609965f, 1.4471976f, -1.39063262f, -0.0810070367f, 0.250017427f, -0.00931349536f, 0.882245394f, -0.215835467f, 0.870693637f, -0.798494089f, 0.204681859f, -0.0868385501f };
static const float k_ref_tiny_mix2[] = { 1.06168317f, -0.167469112f, 0.871752237f, 0.00713681858f, 0.0789153764f, 0.00900083795f, 1.3501742f, 0.0177372432f, -0.512749362f, 0.163177131f, 0.114762043f, -0.245134287f };
static const float k_ref_tiny_mix3[] = { 0.944183549f, -0.0957008409f, 0.195106473f, 0.269267438f, 0.459108189f, -0.0403046149f, 0.964282405f, 0.0521724395f, -0.216795817f, 0.352179401f, 0.0763493048f, -0.111416016f };

static const double k_ref_aligned_in = -12.8888729;
static const float k_ref_aligned_alpha[] = { 0.150563182f, 0.182592998f, 0.489804597f, 0.177039223f, 0.179637954f, 0.19273141f, 0.325397616f, 0.30223302f, 0.222701285f, 0.226109349f, 0.366456917f, 0.184732449f, 0.488384994f, 0.22219587f, 0.113066867f, 0.176352269f };
static const float k_ref_aligned_stat[][7] = {
    { -0.0585850303f, 0.39980981f, 1.75758268f, 0.125341972f, 0.000574376941f, -0.00238624682f, 0.00784164772f },
    { 0.0159832324f, 0.305412083f, 1.65198862f, 0.622019423f, 0.0509489083f, 0.00581417843f, -0.330020161f },
    { -0.0372143324f, 0.277791586f, 1.73304433f, -0.246808386f, -0.0169716513f, -0.0140570072f, 0.0862823428f },
    { 0.0553187516f, 0.331046879f, 1.69266818f, -0.195584634f, -0.110792039f, 0.0898016475f, 0.0240919274f },
};

static const double k_ref_inkling_in = -283.35075;
static const float k_ref_inkling_alpha[] = { 0.431841457f, 0.233959275f, 0.271977797f, 0.0622214717f, 0.294011814f, 0.208729599f, 0.164086131f, 0.333172456f, 0.0711697776f, 0.293867129f, 0.326542814f, 0.308420279f, 0.294421896f, 0.341829987f, 0.0892360791f, 0.274512037f };
static const float k_ref_inkling_stat[][7] = {
    { -0.0049040504f, 0.335357743f, 2.51311379f, 0.089721635f, -0.319499208f, -0.628238349f, -0.813709484f },
    { 0.00522118459f, 0.336560153f, 2.43678661f, 0.247175178f, 1.36501306f, -0.445650988f, 0.776983967f },
    { 0.00156271159f, 0.330818679f, 2.61730284f, 0.264814066f, 0.734263447f, 0.249557689f, 0.0385434747f },
    { -0.00741075468f, 0.334818247f, 2.78015149f, 0.303643542f, -0.255050131f, 0.208513011f, 0.305397563f },
};

// the draft mixes 4 target taps, one weight row per stage plus one for the markov head
struct dflare_case {
    const char    * name;
    int64_t         n_taps;
    int64_t         n_embd;
    int64_t         n_tokens;
    int64_t         n_mix;
    float           eps;
    double          in;                 // checksum of the inputs the oracle draws too
    const float   * alpha;              // {n_taps, n_mix} mixing weights
    const float   * mix[4];             // full {n_embd, n_tokens} outputs, NULL when only the stats are known
    const float   (* stat)[7];          // mean, meansq, max and four spot values per mix
};

static const dflare_case CASES[] = {
    { "tiny",    4,    6, 2, 4, 1e-6f, k_ref_tiny_in,    k_ref_tiny_alpha,
        { k_ref_tiny_mix0, k_ref_tiny_mix1, k_ref_tiny_mix2, k_ref_tiny_mix3 }, nullptr },
    { "aligned", 4,   64, 3, 4, 1e-6f, k_ref_aligned_in, k_ref_aligned_alpha,
        { nullptr, nullptr, nullptr, nullptr }, k_ref_aligned_stat },
    { "inkling", 4, 2304, 5, 4, 1e-6f, k_ref_inkling_in, k_ref_inkling_alpha,
        { nullptr, nullptr, nullptr, nullptr }, k_ref_inkling_stat },
};

// the LCG of test-col2im-1d.cpp: tests/test_dflare_mix.py draws the same sequence
static uint64_t g_rng = 0x12345678ULL;

static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

static void fill(ggml_tensor * t) {
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        ((float *) t->data)[i] = frand();
    }
}

static double checksum(const ggml_tensor * t) {
    double sum = 0.0;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        sum += ((const float *) t->data)[i];
    }
    return sum;
}

// mean, mean of squares, max and the values at four spread flat indices
static void stats_of(const ggml_tensor * t, float * out) {
    const int64_t n   = ggml_nelements(t);
    const float * src = (const float *) t->data;

    double mean = 0.0;
    double meansq = 0.0;
    float  max = src[0];

    for (int64_t i = 0; i < n; ++i) {
        mean   += src[i];
        meansq += (double) src[i] * src[i];
        max     = std::max(max, src[i]);
    }

    out[0] = (float) (mean / n);
    out[1] = (float) (meansq / n);
    out[2] = max;

    // the same double arithmetic as stats() in tests/test_dflare_mix.py
    static const double frac[4] = { 0.0, 1.0/3.0, 2.0/3.0, 1.0 };

    for (int i = 0; i < 4; ++i) {
        out[3 + i] = src[(int64_t) (frac[i] * (n - 1))];
    }
}

// compares n values against the oracle, keeps the largest error and prints the first mismatch
static bool compare(const float * got, const float * ref, int64_t n, const char * what, float & worst) {
    bool ok = true;

    for (int64_t i = 0; i < n; ++i) {
        const float err = std::abs(got[i] - ref[i]);
        worst = std::max(worst, err);

        if (err > 1e-5f) {
            if (ok) {
                printf("  %s: first mismatch at %lld, got %.9g ref %.9g err %.3g\n",
                        what, (long long) i, got[i], ref[i], err);
            }
            ok = false;
        }
    }

    return ok;
}

static bool run_case(const dflare_case & c) {
    struct ggml_init_params params = {
        /* .mem_size   = */ (size_t) 64 << 20,
        /* .mem_base   = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * fuse  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_taps, c.n_mix);
    ggml_tensor * norm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_embd, c.n_mix);
    ggml_tensor * feats = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_taps*c.n_embd, c.n_tokens);

    fill(fuse);
    fill(norm);
    fill(feats);

    bool ok = true;

    // the oracle draws the same inputs, so a mismatch here means the generator and the test drifted apart
    const double in = checksum(fuse) + checksum(norm) + checksum(feats);
    if (std::abs(in - c.in) > 1e-5) {
        printf("  inputs: got %.9g ref %.9g\n", in, c.in);
        ok = false;
    }

    ggml_tensor * alpha = llama_model_dflash::build_dflare_alpha(ctx, fuse);
    ggml_tensor * taps  = llama_model_dflash::build_dflare_taps(ctx, feats, c.n_embd);

    std::vector<ggml_tensor *> mix;
    for (int64_t i = 0; i < c.n_mix; ++i) {
        ggml_tensor * norm_i = ggml_view_1d(ctx, norm, c.n_embd, i*c.n_embd*ggml_element_size(norm));
        mix.push_back(llama_model_dflash::build_dflare_mix(ctx, taps, alpha, i, norm_i, c.eps));
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, alpha);
    for (ggml_tensor * t : mix) {
        ggml_build_forward_expand(gf, t);
    }
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, gf, 4) == 0);

    float worst = 0.0f;

    ok = compare((const float *) alpha->data, c.alpha, c.n_taps*c.n_mix, "alpha", worst) && ok;

    for (int64_t i = 0; i < c.n_mix; ++i) {
        if (c.mix[i]) {
            ok = compare((const float *) mix[i]->data, c.mix[i], ggml_nelements(mix[i]), "mix", worst) && ok;
        } else {
            float got[7];
            stats_of(mix[i], got);
            ok = compare(got, c.stat[i], 7, "stat", worst) && ok;
        }
    }

    printf("%-8s taps=%2lld embd=%4lld tokens=%2lld: max err %.3g %s\n", c.name,
            (long long) c.n_taps, (long long) c.n_embd, (long long) c.n_tokens, worst, ok ? "OK" : "FAIL");

    ggml_free(ctx);
    return ok;
}

// one-hot weights select one tap, so the mix has to be that tap normalized. this catches a transposed tap
// axis, which the oracle values also catch, but names the failure instead of only measuring it.
static bool run_one_hot(void) {
    const int64_t n_taps   = 4;
    const int64_t n_embd   = 5;
    const int64_t n_tokens = 3;
    const float   eps      = 1e-6f;

    struct ggml_init_params params = {
        /* .mem_size   = */ (size_t) 8 << 20,
        /* .mem_base   = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    ggml_tensor * alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_taps, n_taps);
    ggml_tensor * ones  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * feats = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_taps*n_embd, n_tokens);

    fill(feats);

    for (int64_t m = 0; m < n_taps; ++m) {
        for (int64_t t = 0; t < n_taps; ++t) {
            ((float *) alpha->data)[t + m*n_taps] = t == m ? 1.0f : 0.0f;
        }
    }

    for (int64_t i = 0; i < n_embd; ++i) {
        ((float *) ones->data)[i] = 1.0f;
    }

    std::vector<ggml_tensor *> mix;
    ggml_tensor * taps = llama_model_dflash::build_dflare_taps(ctx, feats, n_embd);

    for (int64_t m = 0; m < n_taps; ++m) {
        mix.push_back(llama_model_dflash::build_dflare_mix(ctx, taps, alpha, m, ones, eps));
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : mix) {
        ggml_build_forward_expand(gf, t);
    }
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, gf, 4) == 0);

    // tap m of token i is the block m*n_embd of row i, and mix m flattens as i*n_embd + e
    std::vector<float> ref;

    for (int64_t m = 0; m < n_taps; ++m) {
        for (int64_t i = 0; i < n_tokens; ++i) {
            const float * tap = (const float *) feats->data + i*n_taps*n_embd + m*n_embd;

            double sumsq = 0.0;
            for (int64_t e = 0; e < n_embd; ++e) {
                sumsq += (double) tap[e] * tap[e];
            }
            const float scale = 1.0f / sqrtf((float) (sumsq / n_embd) + eps);

            for (int64_t e = 0; e < n_embd; ++e) {
                ref.push_back(scale * tap[e]);
            }
        }
    }

    float worst = 0.0f;
    bool  ok    = true;

    for (int64_t m = 0; m < n_taps; ++m) {
        ok = compare((const float *) mix[m]->data, ref.data() + m*n_tokens*n_embd,
                    n_tokens*n_embd, "one hot", worst) && ok;
    }

    printf("one hot                                              : max err %.3g %s\n", worst, ok ? "OK" : "FAIL");

    ggml_free(ctx);
    return ok;
}

int main(void) {
    int fails = 0;

    for (const dflare_case & c : CASES) {
        if (!run_case(c)) {
            fails++;
        }
    }

    if (!run_one_hot()) {
        fails++;
    }

    printf(fails == 0 ? "all DFlare tap mixing checks passed\n" : "%d DFlare tap mixing checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
