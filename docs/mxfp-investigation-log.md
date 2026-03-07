# MXFP Flash Attention — Investigation Log

Tracking issues, root causes, and fixes discovered during development.

## Issue 1: CUDA tg128 Regression (8-14% slower)

**Status**: FIXED (commit 6832beadc)

**Symptom**: After commit 70bace301 (Vulkan MXFP flash attention), all MXFP types showed tg128
regression on CUDA:
- mxfp8_e4m3/mxfp4: 180.8 → 166.7 t/s (-7.8%)
- mxfp6_e2m3/mxfp4: 179.7 → 157.2 t/s (-12.5%)

**Root Cause**: Commit 70bace301 replaced ALL CUDA hardware intrinsics (`__nv_fp8_e4m3`,
`__nv_cvt_float_to_fp8`, `__nv_cvt_float_to_fp6`, etc.) with portable IEEE bit manipulation
functions (`mxfp_detail::*`) for HIP/MUSA compatibility. This affected:
1. `mse_error()` — called 96 times per 32-element block during E8M0 scale search
2. `quantize_elem()` — called during new KV cache entry quantization
3. `write_qs()` — called during set_rows for each new token's K/V
4. `dequant_elem()` — called during VEC kernel K/V dequantization

The MSE search inner loop was the main bottleneck — 96 software IEEE conversions per block
vs 96 single-cycle HW intrinsic calls.

**Fix**: Added `#if CUDART_VERSION >= 12050` (FP8) and `#if CUDART_VERSION >= 12080` (FP6)
guards to all trait functions, preserving portable fallback for HIP/MUSA.

**Bisect Path**: d0ef6448 (179.9) → 311958b13 (180.7) → d2b1bd174 (180.8) → 78f3a3a9b (180.9)
→ dab6b8216 (180.9) → 4719423e5 (180.9) → bc388001d (180.8) → **70bace301 (166.7)**

**Lesson**: When making code portable across backends, ALWAYS use conditional compilation for
the fast path. Never assume that "portable" code is "good enough" — software FP8/FP6 conversion
is 8-14% slower than HW intrinsics in hot paths.


## Issue 2: CUDA pp512 Regression (6000 → 1900 t/s)

**Status**: FIXED (commit 6832beadc)

**Symptom**: When MXFP dispatch was changed to use VEC kernel unconditionally (all batch sizes),
prompt processing throughput dropped 3x.

**Root Cause**: The VEC kernel dequantizes K/V for every query token individually. With batch=512
(prompt processing), that's 512x redundant dequant work. The MMA kernel dequantizes K/V tiles
into shared memory once, then all query tokens in the batch reuse them.

**Fix**: Dispatch MMA for batch > 2, VEC for batch ≤ 2 on Blackwell. Non-Blackwell falls back
to VEC (the only available path).


## Issue 3: GLM-4.7-Flash 20x Throughput Drop

**Status**: FIXED (two bugs)

**Symptom**: GLM-4.7-Flash pp512 dropped from ~5,500 to ~305 t/s.

**Root Cause**: TWO independent bugs combined:

**Bug 1**: `mxfp_layout_ok` in `fattn.cu` had `nb[1] % 16 == 0` check that rejected MLA tensors
(D_K=576 MXFP4: 18×17=306 bytes, 306%16≠0). This disabled MXFP flash attention for MLA models.
**Fix**: Removed 16B alignment requirement — only check block-type-size alignment.

**Bug 2**: `supports_op` in `ggml-cuda.cu` rejected MXFP4 MUL_MAT for batch > MMVQ_MAX_BATCH_SIZE.
Since the MODEL WEIGHTS are MXFP4 (MXFP4_MOE GGUF), this caused the scheduler to keep all weight
tensors on CPU (15.5 GiB CPU_Mapped, 191 graph splits). Even f16 KV cache was affected because
the model weights couldn't be placed on GPU.
**Fix**: Removed MMVQ batch size restriction for MXFP4 MUL_MAT.

**Verified**: After both fixes, f16 pp512 recovered from 302→5,588 t/s, tg128 from 61→136.5 t/s.
MXFP flash attention now works for MLA: mxfp4 pp512=5,414, mxfp8_e4m3 pp512=4,945.

**Lesson**: When `supports_op` returns false, the scheduler relocates tensors to CPU — this
affects ALL operations using those tensors, not just the rejected op. A seemingly harmless
batch-size restriction on MUL_MAT can cause the entire model to be CPU-mapped.


## Issue 4: E5M2 Quality Degradation on MLA Models

**Status**: EXPLAINED — Expected Behavior

**Symptom**: mxfp8_e5m2 GPU PPL = 15.05 on GLM-4.7-Flash (MLA), vs 11.70 on Qwen3-Coder
(non-MLA). CPU PPL = 11.55 (also bad, but less extreme).

**Root Cause**: **MLA skips Hadamard rotation** because V shares K buffer (V_is_K_view).
Rotating K would corrupt V. Without Hadamard, E5M2's 2 mantissa bits can't handle outlier
values in attention heads. PPL degrades progressively over longer sequences:
- Chunk 1: PPL 6.72 (OK)
- Chunk 13: PPL 16.6 (terrible)
- Final: PPL 15.05

E4M3 (3 mantissa bits) handles this much better without Hadamard: GPU PPL 10.58.
On non-MLA models where Hadamard IS applied, E5M2 works fine (PPL 11.70).

**Recommendation**: For MLA models, use E4M3 not E5M2 for K type.

**Future Direction**: Develop MLA-compatible rotation. Options:
1. Apply rotation in the absorbed projection space before K/V merging
2. Use a separate rotation for Q that matches the un-rotated K
3. Accept that MLA models need higher-precision MX types (E4M3 or E2M3)


## Issue 5: Intel iGPU mxfp4 Flash Attention Crash

**Status**: ROOT CAUSE IDENTIFIED

**Symptom**: Intel ARL integrated GPU crashes with `ErrorDeviceLost` when running mxfp4 flash
attention via Vulkan. f16/q8_0/q4_0 flash attention all work fine.

**Device Info**:
```
Intel(R) Graphics (ARL) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 32
shared memory: 65536 | int dot: 0 | matrix cores: none
```

**Root Cause**: The MXFP_Q_PREPROCESS section in `flash_attn.comp` uses subgroup operations
unconditionally:

- `flash_attn.comp:109-110`: `gl_SubgroupInvocationID`, `gl_SubgroupID`
- `flash_attn.comp:134,142-146`: `subgroupShuffleXor()` for Hadamard butterfly
- `flash_attn.comp:177-180`: `subgroupShuffle()` for vec4 assembly
- `flash_attn.comp:272-278`: `subgroupAll()`, `gl_NumSubgroups`

The `supports_op` check at `ggml-vulkan.cpp:15214-15217` gates on:
```c
if (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
    return false;
}
```

Intel iGPU likely reports `subgroup_shuffle=true` and `subgroup_vote=true` (it supports
Vulkan subgroups in general), but crashes when executing the specific MXFP shader — possibly
due to:
1. Subgroup size mismatch (expects 32 for Hadamard, Intel may use different size)
2. Intel driver bug with specific subgroup op combinations
3. Shader timeout from extremely slow execution on the low-power iGPU

**Fix options**:
1. Add explicit Intel iGPU rejection for MXFP flash attention in `supports_op`
2. Add a subgroup size check (require subgroupSize >= 32 for MXFP)
3. Create a shared-memory fallback for Hadamard that doesn't use subgroup shuffles


## Issue 6: Vulkan E3M2 Quantizer Threshold Bug

**Status**: FIXED (commit 3e8dceab5)

**Symptom**: mxfp6_e3m2 PPL = 3334 on Vulkan (should be ~12).

**Root Cause**: E3M2 subnormal threshold in `copy_to_quant.comp` was `f32_exp < 0` (copied
from E2M3 code). E3M2 has bias=3, so subnormal range is `f32_exp <= -3` (i.e. `f32_exp < -2`).
E2M3 has bias=1, so its threshold `f32_exp < 0` is correct for E2M3 but wrong for E3M2.

**Fix**: Changed threshold from `f32_exp < 0` to `f32_exp < -2` in the E3M2 quantizer.

**Lesson**: When adapting quantization code between MX format variants, ALWAYS recalculate
the bias-dependent thresholds. Don't copy-paste from one format to another.


## Issue 7: Vulkan MXFP Flash Attention — Matched Types Only

**Status**: KNOWN LIMITATION

**Symptom**: Mixed K/V types (e.g., K=mxfp8_e4m3, V=mxfp4) fall back to non-flash attention
path, which is ~20x slower.

**Root Cause**: `ggml-vulkan.cpp` line ~15105: `src[1]->type != src[2]->type → return false`
for FLASH_ATTN_EXT with MXFP types. The Vulkan shaders only support matched K/V types.

**Impact**: On CUDA, mixed types are the recommended config (e.g., mxfp6_e2m3 K + mxfp4 V)
for best quality-memory tradeoff. On Vulkan, users must use matched types (mxfp4/mxfp4) or
accept the non-flash fallback penalty.

**Future Fix**: Add V-type specialization constants to Vulkan MXFP flash attention shaders,
or generate multiple shader variants. Medium complexity.


## Issue 8: Vulkan MXFP tg128 Performance Gap (21%)

**Status**: INVESTIGATED — Architectural Limitation

**Symptom**: All MXFP types show ~21% tg128 regression vs f16 on Vulkan (148 vs 189 t/s).
On CUDA, the same gap is only 2.5% (184 vs 189 t/s).

**Root Cause**: Vulkan has no hardware MXFP support. CUDA Blackwell has `mma.mxf4` and
FP8/FP6 hardware intrinsics that convert in silicon. Vulkan must do all dequant in software
via SPIR-V bit manipulation (~10 instructions per 4 elements vs 1 for f16 vec4 load).

**Evidence**: ALL MXFP types cluster at identical ~148 t/s regardless of dequant complexity:
- FP4 (simplest), FP6 E2M3, FP6 E3M2, FP8 E4M3, FP8 E5M2 — all ~148 t/s
- This proves the bottleneck is common block-access overhead, not per-element conversion

**Optimizations Attempted** (commit 17455822a):
1. LUT → arithmetic dequant (MXFP4): +0.8% (noise)
2. Branchless dequant (all types): 0%
3. Disable SHMEM_STAGING at Br=1: 0%

**Bug Fixed**: E4M3 NaN check was incorrect for MX format. MX E4M3 has no NaN — exp=15
mant=7 = ±480, not NaN. The old code returned NaN for valid max-range values.

**pp512 gap is only ~5%** (5,300 vs 5,500 t/s) because prompt processing amortizes dequant
across many query rows per tile (Br >> 1).

**Lesson**: When Vulkan lacks hardware support for a quantization format, software dequant
overhead per KV position is irreducible. The gap is proportional to the ratio of dequant
instructions to total work per position. At batch=1 (tg), dequant dominates. At batch>>1
(pp), dot product work dominates and dequant is amortized.
