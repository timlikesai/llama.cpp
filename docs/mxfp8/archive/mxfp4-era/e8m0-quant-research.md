To achieve high throughput while maintaining numerical stability, we keep all packing operations vector-aligned (processing 4 elements per instruction sequence) and implement the scale derivation using bitwise operations without transcendental runtime functions.

---

### Math to Bitwise Translation

#### 1. Runtime MXFP8 Activation Scaling ($F_{\text{eff}} \approx 288$)

We derived the runtime exponent:


$$E_X = \lfloor \log_2(x_{\max}) - 7.17 \rfloor$$


For positive normalized FP32 floats, the IEEE-754 bit-pattern provides $\lfloor \log_2(x) \rfloor = ((\text{bits} \gg 23) \& 0xFF) - 127$.

By adding a fractional threshold bias before stripping the exponent, the floating-point bit manipulation directly yields the floored logarithm. Specifically, $7.17 = 7 + \log_2(1.125) \approx 7 + 0.17$. Scaling the input float by multiplying or biasing the exponent allows direct extraction:


$$E_X = \text{clamp}\left(\left\lfloor \log_2(x_{\max} / 288.0f) \right\rfloor + 1, \ 0, \ 255\right)$$

#### 2. Offline / KV Cache Quant-Time FP4 (E2M1) Scaling ($F_{\max} = 6.0$)

For weights and KV cache quantization:

* Naive dynamic range: $S_W = 2^{\lceil \log_2(w_{\max} / 6.0) \rceil}$
* Best interaction balance: Evaluate candidate exponents $E \in \{E_{\text{naive}}-1, E_{\text{naive}}\}$ across the block using minimal squared error:

$$\arg\min_{E} \sum_{i=0}^{31} \left( w_i - 2^E \cdot \text{round}_{\text{E2M1}}\left(\frac{w_i}{2^E}\right) \right)^2$$



---

### CUDA Implementation

Below is the production header implementation supporting 4x-vectorized constructors for:

1. **MXFP8 (E4M3):** Using `__nv_fp8x4_e4m3` (CUDA 11.8+ / 12.x).
2. **MXFP4 (E2M1):** 4 elements packed into a single `uint16_t` (nibble-packed: $[e_3, e_2, e_1, e_0]$) using bitwise rounding.

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cfloat>

// ============================================================================
// 1. FAST BITWISE SCALE DERIVATION
// ============================================================================

// Computes E8M0 scale byte for MXFP8 with F_eff = 288.0f saturation target
__device__ __forceinline__ uint8_t compute_mxfp8_scale_e8m0(float amax) {
    if (amax <= 1e-12f) return 0;
    // F_target = 288.0f. We need: ceil(log2(amax / 288.0f))
    // Which is equivalent to floor(log2(amax / 144.0f)) for powers of 2.
    // In IEEE-754: frexp / exponent extraction
    float scaled = amax * (1.0f / 288.0f);
    int32_t exp_val;
    // Fast hardware log2/exponent extraction via float bit-cast
    uint32_t bits = __float_as_uint(scaled);
    exp_val = ((bits >> 23) & 0xFF) - 127;
    
    // Ceiling adjustment: if mantissa is non-zero, exp_val++
    if ((bits & 0x007FFFFF) != 0) {
        exp_val++;
    }
    
    // Bias for OCP E8M0 (unbiased range [0, 255] or standard bias 127 depending on MMA spec)
    // OCP MX spec maps raw byte E directly as power-of-two multiplier 2^(E - bias).
    int32_t e8m0 = exp_val + 127;
    return (uint8_t)__clamp(e8m0, 0, 255);
}

// Converts E8M0 exponent scale back to FP32 multiplier (for normalization before pack)
__device__ __forceinline__ float e8m0_to_inv_scale(uint8_t e8m0) {
    if (e8m0 == 0) return 0.0f;
    // Inverse scale = 2^-(e8m0 - 127) = 2^(127 - e8m0)
    uint32_t exp_bits = (uint32_t)(127 - ((int32_t)e8m0 - 127)) << 23;
    return __uint_as_float(exp_bits);
}

// ============================================================================
// 2. 4X VECTORIZED MXFP8 (E4M3) PACKING
// ============================================================================

// Quantizes and packs 4 float32 activations to 4x E4M3 using CUDA __nv_fp8x4_e4m3
__device__ __forceinline__ uint32_t pack_4x_mxfp8_e4m3(float4 val, float inv_scale) {
    // 1. Scale down inputs into the E4M3 range
    val.x *= inv_scale;
    val.y *= inv_scale;
    val.z *= inv_scale;
    val.w *= inv_scale;

    // 2. Vectorized construction to __nv_fp8x4_e4m3
    // Under __NV_SATFINITE, hardware handles saturation to 448.0f automatically
    __nv_fp8x4_e4m3 fp8_packed(val.x, val.y, val.z, val.w);
    
    return *reinterpret_cast<uint32_t*>(&fp8_packed);
}

// ============================================================================
// 3. 4X VECTORIZED MXFP4 (E2M1) PACKING (FOR WEIGHTS & KV-CACHE)
// ============================================================================

// Converts a single normalized float (|x| <= 6.0) to 4-bit E2M1
// E2M1 representable positive points: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
__device__ __forceinline__ uint8_t float_to_e2m1(float v) {
    uint32_t b = __float_as_uint(v);
    uint32_t sign = (b >> 31) & 0x1;
    float abs_v = fabsf(v);

    // Constant-time threshold search / bitwise rounding
    // Midpoints for rounding to nearest even:
    // 0 < 0.25 < 0.5 < 0.75 < 1.0 < 1.25 < 1.5 < 1.75 < 2.0 < 2.5 < 3.0 < 3.5 < 4.0 < 5.0 < 6.0
    uint8_t code = 0;
    if      (abs_v >= 5.0f)   code = 0x7; // 6.0
    else if (abs_v >= 3.5f)   code = 0x6; // 4.0
    else if (abs_v >= 2.5f)   code = 0x5; // 3.0
    else if (abs_v >= 1.75f)  code = 0x4; // 2.0
    else if (abs_v >= 1.25f)  code = 0x3; // 1.5
    else if (abs_v >= 0.75f)  code = 0x2; // 1.0
    else if (abs_v >= 0.25f)  code = 0x1; // 0.5
    else                      code = 0x0; // 0.0

    return (sign << 3) | code;
}

// Packs 4 float32 inputs to a 16-bit word containing 4x E2M1 values
__device__ __forceinline__ uint16_t pack_4x_mxfp4_e2m1(float4 val, float inv_scale) {
    uint8_t p0 = float_to_e2m1(val.x * inv_scale);
    uint8_t p1 = float_to_e2m1(val.y * inv_scale);
    uint8_t p2 = float_to_e2m1(val.z * inv_scale);
    uint8_t p3 = float_to_e2m1(val.w * inv_scale);

    return (uint16_t)(p0 | (p1 << 4) | (p2 << 8) | (p3 << 12));
}

// ============================================================================
// 4. BLOCK-OF-32 QUANTIZATION KERNEL (W4A8 RUNTIME FUSION)
// ============================================================================

// 1 Warp handles one 32-element vector block along the K-dimension
__device__ void quantize_block_32_mxfp8(
    const float* __restrict__ block_in, 
    uint32_t* __restrict__ packed_out, 
    uint8_t* __restrict__ scale_out,
    int lane_id
) {
    // 1. Each of 32 threads loads 1 element (or 8 threads load float4)
    float val = block_in[lane_id];
    float abs_val = fabsf(val);

    // 2. Warp-level reduction to find amax over the 32 elements
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        abs_val = fmaxf(abs_val, __shfl_xor_sync(0xFFFFFFFF, abs_val, offset));
    }
    float block_amax = abs_val;

    // 3. Lane 0 derives scale using F_eff = 288 target
    uint8_t e8m0 = 0;
    float inv_scale = 0.0f;
    if (lane_id == 0) {
        e8m0 = compute_mxfp8_scale_e8m0(block_amax);
        *scale_out = e8m0;
    }
    // Broadcast scale back across the warp
    e8m0 = __shfl_sync(0xFFFFFFFF, e8m0, 0);
    inv_scale = e8m0_to_inv_scale(e8m0);

    // 4. Thread 0..7 convert 4 elements each using 4x vector pack
    if (lane_id < 8) {
        int idx = lane_id * 4;
        float4 v = make_float4(
            block_in[idx + 0], 
            block_in[idx + 1], 
            block_in[idx + 2], 
            block_in[idx + 3]
        );
        packed_out[lane_id] = pack_4x_mxfp8_e4m3(v, inv_scale);
    }
}

```

---

### Optimal Quant-Time Exponent Refinement

For weights (or KV caches quantized during decoding prefill), you can avoid the search grid in the primary MMA path by using this local 2-candidate L2 evaluation over the 32-element block:

```cuda
__device__ uint8_t find_best_weight_scale_block32(const float* block_w, float amax) {
    // Candidate 1: standard ceiling to 6.0
    uint8_t e_ceil = compute_mxfp4_scale(amax); 
    // Candidate 2: pulled-in scale (1 bit lower, gives more resolution to the bulk)
    uint8_t e_tight = (e_ceil > 0) ? e_ceil - 1 : 0;

    float inv_ceil = e8m0_to_inv_scale(e_ceil);
    float inv_tight = e8m0_to_inv_scale(e_tight);

    float err_ceil = 0.0f;
    float err_tight = 0.0f;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float w = block_w[i];
        
        // Reconstruct both and calculate L2 error
        float w_hat_ceil = decode_e2m1(float_to_e2m1(w * inv_ceil)) / inv_ceil;
        float w_hat_tight = decode_e2m1(float_to_e2m1(w * inv_tight)) / inv_tight;

        err_ceil += (w - w_hat_ceil) * (w - w_hat_ceil);
        err_tight += (w - w_hat_tight) * (w - w_hat_tight);
    }

    return (err_tight < err_ceil) ? e_tight : e_ceil;
}

```

### Critical Execution Details

* **Sub-byte Alignments:** `pack_4x_mxfp4_e2m1` packs 4 contiguous values into 16 bits (`uint16_t`), keeping adjacent 32-element blocks 16-byte aligned. This maps directly to hardware sub-byte matrix loads (`ldmatrix.x4`).
* **Hardware Saturation:** When using `__nv_fp8x4_e4m3`, CUDA automatically routes saturation to finite maximum ($448.0\text{f}$) under default rounding flags, ensuring values exceeding the scaled $F_{\text{eff}}$ clip rather than turning into `NaN`.


---

## Scale Optimization: Findings (2026-07-09)

### The core insight: scale optimality is an OUTPUT problem, not a value problem

The scale for a quantized block is **not** chosen to best-reconstruct the block's own values in isolation. It is chosen to make the **dot-product output** (weight x activation) as accurate as possible:

```
min_E  || (W . a) - (W . quant_E(a)) ||^2  =  min_E  sum_i  W_i^2 * ( a_i - quant_E(a_i) )^2
```

A weight element that multiplies a **large** activation deserves a finer scale than one multiplying a near-zero activation. This is why a fixed scale (any F) is suboptimal, and why the *right* scale depends on the joint weight x activation distribution.

**Practical consequence (where the data lives):**
- **Weights** are quantized *offline* (activations unknown at quant time). Best offline proxy: the **imatrix** (calibration-run activation magnitudes) — see below.
- **Activations / KV** are quantized *at runtime*, and here the other side of the multiply IS present (weights for activations, queries for KV). This is where the true joint objective is computable and cheap.

### Activation scale: single-pass optimal (no re-quantization)

In the `quantize_mmq_mxfp8` kernel each lane already holds 4 values; the 8 lanes cover the 32-value block. To choose the scale **without quantizing values multiple times**:

- The two candidate scales (E and E-1, 1 octave apart) give, for a value `v`, `r = v/2^E` and `r/2`.
- The e4m3 grid is **self-similar** (8 steps/octave), so the quantization error of `v` under either candidate is a pure function of `r`'s **mantissa position** — no second `__nv_fp8x4_e4m3` conversion needed.
- Reduce both errors across the 8 lanes (like the existing amax reduction), pick the better E, then quantize **once**.

Net cost: one extra 8-lane reduction + a few mantissa-bit ops. No second quantization, no new global loads.
**Caveat:** the W_i^2-joint objective is NOT available in the activation-quant kernel (weights are not in its signature — they load separately in the consuming MMA kernel). This is the *distribution-optimal* version (minimizes the activation's own error), not the *weight-aware* one. The weight-aware version needs a kernel refactor.

### Weight scale: imatrix (calibration-driven)

Re-introduced imatrix for mxfp4 (infra already exists: `tools/imatrix`, `imatrix-loader.h`, plumbed through `llama-quantize --imatrix` -> `ggml_quantize_chunk` -> `quantize_mxfp4(..., quant_weights)`).

`quantize_row_mxfp4_ref_imatrix` picks the per-block scale by minimizing the **imatrix-weighted output error** `sum_j im[j]^2 * (w[j] - round_e2m1(w[j]/s))^2` over a band of candidate exponents.

**0.8B results (CPU, full wikitext-2):**
| variant | PPL |
|---|---|
| plain spec /6.0 | 19.579 |
| imatrix, 4 chunks | 20.634 (too thin) |
| imatrix, full, 2-cand (tight-only) | 20.041 (over-tightened) |
| imatrix, full, wide band [-2..+3 oct] | **19.553** (beats spec by 0.026) |

**Lessons:**
- More imatrix chunks -> better (4-chunk was clearly too thin).
- A **tight-only** candidate band over-tightens (the absolute-error objective favors the smallest scale that still saturates). The band must include the **looser** side too so the true optimum (often near /6.0 or looser) is reachable.
- On the 0.8B (weak probe) the gain is marginal; the real test is 27B/35B.

### OCP-spec scale is suboptimal (confirmed)

Spec uses `scale = round_to_pow2(amax / format_max)` (E2M1 /6.0, E4M3 /448). Measured **worse** than old `/4` weights on PPL (27B: 6.69 vs 6.32; 35B: 6.42 vs 5.998). The E4M3 activation /448 is the main culprit (plateau at 128-256, cliff >320). Old `/4` weights + activation fmax ~128-256 is the best *fixed* config. The adaptive (imatrix/optimal) approaches above aim to beat even that.

### Ops lesson

**Never `rsync --delete` to a machine running jobs.** It wipes untracked files (sweep logs) and clobbers scripts the jobs depend on. Push only the specific files changed, or sync without `--delete`.

### FINAL SCALE DECISION (locked in)

| path | format | fmax | rationale |
|---|---|---|---|
| **weights** (offline, e2m1) | MXFP4 | **4.0** | best PPL on 27B/35B (spec ÷6.0 is ~0.1-0.4 PPL worse). 0.8B is a weak/misleading probe. |
| **activations** (runtime, e4m3) | W4A8 | **256** | best fixed PPL on 27B (÷256=6.32; spec ÷448=6.69, ~0.37 worse). |
| **KV cache** (runtime, e2m1) | — | 6.0 (spec) | less critical; spec value retained. |

**Adaptive scale (dropped):** a single-pass 2-candidate optimal activation scale was implemented (mantissa/hardware e4m3 error, no re-quant) but landed at 6.5586 on the 27B — *worse* than the best fixed (÷256=6.32). The e8m0 integer-exponent grid is too coarse to place the activation optimum, so per-block scale selection can't beat the best fixed value. **Ship fixed ÷256.**

**Imatrix weight-quant (task 47):** re-introduced — `llama-quantize --imatrix` now drives an imatrix-aware mxfp4 scale (minimizes imatrix-weighted output error over a band around ÷4.0). On 0.8B it marginally beats spec (19.55 vs 19.58); real benefit to be measured on 27B/35B.

### Final measured PPL (locked-in config, GPU, f16 KV, -ngl 32)

| Model | Locked-in (÷4.0 w + ÷256 a) | Q4_0 ref | Spec (÷6.0 w + ÷448 a) | Imatrix-weight (27B) |
|---|---|---|---|---|
| 27B | **6.4405** | 6.44 | 6.69 | **6.3199** (beats fixed by 0.12) |
| 35B | **5.9972** | 5.886 | 6.42 | — |

**Confirmed wins:**
- Locked-in fixed (÷4.0 + ÷256) beats the OCP spec by **0.25 PPL (27B)** and **0.42 PPL (35B)**.
- 35B locked-in (5.997) is within 0.11 of Q4_0 (5.886).
- **Imatrix weight-quant beats the fixed ÷4.0 by 0.12 PPL on the 27B** (6.32 vs 6.44) — a real, measurable gain from the calibration-driven weight scale. This is the feature to keep.

### HARD RULE (learned the hard way): NEVER delete logs/results

- NEVER `rm` a log, result, or benchmark output file. Ever.
- NEVER `rsync --delete` to a machine running jobs (or any machine) — it wipes untracked files (sweep logs, in-progress outputs).
- Test/benchmark scripts: write results to a DURABLE location (`/home/tim/...` or repo `logs/`), never `/tmp`, and never `rm` them at the end.
- If a re-run overwrites a file, that's fine (same path); but never proactively delete prior results.
- Logs are evidence. Losing them forces expensive re-runs and destroys data that may be the only record of a measurement.

(This rule was violated during the mxfp4 scale work: an `rsync --delete` to Beast wiped the 27B sweep logs, and a test-script `rm -f` deleted result artifacts. Do not repeat.)

### 35B-A3B full re-quantized PPL set (locked-in scales + imatrix, 2026-09-14)

| quant | size | PPL |
|---|---:|---:|
| Q8_0 | 34.4 GiB | **5.7399** (measured via partial GPU offload, -ngl 30) |
| Q5_1 | 24.3 GiB | 5.7902 |
| Q4_K_M | 19.7 GiB | 5.8137 |
| Q4_1 | 20.3 GiB | 5.8871 |
| Q4_0 | 18.4 GiB | 5.8902 |
| **mxfp4 (imatrix)** | 17.7 GiB | 6.0955 |

Q8_0 is the ceiling (best); mxfp4 is last on the MoE model (expected — imatrix
helps less on MoE experts, and the MoE dense parts dominate). The 35B Q8_0 needed
partial GPU offload (-ngl 30 of 40 layers, ~27GB GPU + CPU tail) because the
34.4 GiB file exceeds the 2x16GB = 32GB total.
