# Metal Mixed K/V Flash Attention Optimization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the ~35% throughput regression on mixed K/V type flash attention (mxfp8+mxfp4, mxfp6+mxfp4, q8_0+q4_0) by adding native template instantiations instead of runtime function-constant dispatch.

**Architecture:** The Metal flash attention kernel template already has separate K and V type parameters, but all instantiations use K=V (matched). Mixed K/V falls back to `char*` pointer arithmetic + `dequant_v_dispatch()` switch — causing 35% tg regression and 55% pp regression. Fix: add native cross-type template instantiations for the 3 most important mixed combos, so the compiler gets typed pointer access and full optimization. Keep the function-constant dispatch as fallback for any combo without a native kernel.

**Tech Stack:** Metal Shading Language (MSL), Objective-C++, C++

---

## Background

### Current Performance (M2 Air, Qwen3-4B Q8_0, 16 chunks)

| Config | pp512 | tg128 | vs matched |
|--------|-------|-------|------------|
| q8_0+q8_0 (matched) | 340 | 19.1 | baseline |
| q8_0+q4_0 (mixed) | 154 | 12.9 | **55% pp / 32% tg slower** |
| mxfp4+mxfp4 (matched) | 333 | 17.4 | baseline |
| mxfp8_e4m3+mxfp4 (mixed) | 231 | 11.5 | **31% pp / 34% tg slower** |

### Root Cause

When K!=V, the V dequant path at `ggml-metal.metal:6431` (non-vec) and `:7174` (vec) uses:
1. `char*` byte pointer arithmetic instead of typed block pointers
2. `dequant_v_4x4_dispatch()` / `dequant_v_t4_dispatch()` — switch-based dispatch
3. Manual byte offset computation: `v_row + (i/nl_v_eff)*v_bs`

When K==V, the template-native `deq_v()` / `deq_v_t4()` function pointers give the compiler typed access, aligned loads, and inlined dequant.

### Target Mixed Combos (3 primary)

1. **K=mxfp8_e4m3 + V=mxfp4** — best quality/memory ratio for MXFP
2. **K=mxfp6_e2m3 + V=mxfp4** — best memory savings with excellent quality
3. **K=q8_0 + V=q4_0** — mainstream quantized, benefits many users

### Files to Modify

- `ggml/src/ggml-metal/ggml-metal.metal` — kernel template instantiations
- `ggml/src/ggml-metal/ggml-metal-device.cpp` — pipeline lookup (base name + function constants)

### Key Dimensions

Non-vec head sizes (13): 32, 40, 48, 64, 72, 80, 96, 112, 128, 192, 192×128, 256, 576×512
Vec head sizes (8): 32, 64, 96, 128, 192, 192×128, 256, 576×512
NE values per vec head size: dk32→4, dk64→2, dk96→4, dk128→1, dk192→2, dk192×128→2, dk256→1, dk576×512→2

---

## Task 1: Add Non-Vec Mixed Kernel Instantiations

**Files:**
- Modify: `ggml/src/ggml-metal/ggml-metal.metal` (after line ~6819, before `#undef FA_TYPES`)

**Step 1: Add K=mxfp8_e4m3 + V=mxfp4 non-vec instantiations**

Insert after the mxfp6_e3m2 matched block (line 6819), before `#undef FA_TYPES`:

```metal
// Mixed K/V: K=mxfp8_e4m3, V=mxfp4 (native template — no dispatch overhead)
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 32,  32>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 40,  40>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 48,  48>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 72,  72>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 80,  80>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 96,  96>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 112, 112>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 192, 192>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_mxfp8_e4m3_v_mxfp4_e2m1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp8, 2, dequantize_mxfp8_e4m3, block_mxfp4, 2, dequantize_mxfp4, 576, 512>;
```

**Step 2: Add K=mxfp6_e2m3 + V=mxfp4 non-vec instantiations**

```metal
// Mixed K/V: K=mxfp6_e2m3, V=mxfp4
template [[host_name("kernel_flash_attn_ext_mxfp6_e2m3_v_mxfp4_e2m1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp6, 2, dequantize_mxfp6_e2m3, block_mxfp4, 2, dequantize_mxfp4, 32,  32>;
// ... (same 13 head sizes as above, substituting block_mxfp6/dequantize_mxfp6_e2m3 for K)
template [[host_name("kernel_flash_attn_ext_mxfp6_e2m3_v_mxfp4_e2m1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_mxfp6, 2, dequantize_mxfp6_e2m3, block_mxfp4, 2, dequantize_mxfp4, 576, 512>;
```

**Step 3: Add K=q8_0 + V=q4_0 non-vec instantiations**

```metal
// Mixed K/V: K=q8_0, V=q4_0
template [[host_name("kernel_flash_attn_ext_q8_0_v_q4_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q4_0, 2, dequantize_q4_0, 32,  32>;
// ... (same 13 head sizes, K=block_q8_0/dequantize_q8_0, V=block_q4_0/dequantize_q4_0)
template [[host_name("kernel_flash_attn_ext_q8_0_v_q4_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q4_0, 2, dequantize_q4_0, 576, 512>;
```

**Step 4: Build to verify shader compilation**

```bash
cd /Users/tim/code/llama.cpp && cmake --build build -j 2>&1 | tail -20
```
Expected: Clean compilation (no Metal shader errors).

**Step 5: Commit**

```bash
git add ggml/src/ggml-metal/ggml-metal.metal
git commit -m "Metal: add native mixed K/V flash attention non-vec kernel instantiations

Add template instantiations for mxfp8_e4m3+mxfp4, mxfp6_e2m3+mxfp4,
and q8_0+q4_0 mixed K/V combinations. These give the compiler typed
pointer access and full optimization, avoiding the char* dispatch path."
```

---

## Task 2: Add Vec Mixed Kernel Instantiations

**Files:**
- Modify: `ggml/src/ggml-metal/ggml-metal.metal` (after the mxfp6_e3m2 vec block, before `#undef FA_TYPES`)

**Step 1: Add K=mxfp8_e4m3 + V=mxfp4 vec instantiations**

Note: vec kernels use `nl=8` for all quantized types, and `_t4` dequant functions. NE values vary by head size.

Insert after the mxfp6_e3m2 vec block:

```metal
// Mixed K/V vec: K=mxfp8_e4m3, V=mxfp4
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_mxfp8_e4m3_v_mxfp4_e2m1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp8, 8, dequantize_mxfp8_e4m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 576, 512, 2>;
```

**Step 2: Add K=mxfp6_e2m3 + V=mxfp4 vec instantiations**

```metal
// Mixed K/V vec: K=mxfp6_e2m3, V=mxfp4
template [[host_name("kernel_flash_attn_ext_vec_mxfp6_e2m3_v_mxfp4_e2m1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp6, 8, dequantize_mxfp6_e2m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 32, 32, 4>;
// ... (same 8 head sizes, substituting block_mxfp6/dequantize_mxfp6_e2m3_t4 for K)
template [[host_name("kernel_flash_attn_ext_vec_mxfp6_e2m3_v_mxfp4_e2m1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_mxfp6, 8, dequantize_mxfp6_e2m3_t4, block_mxfp4, 8, dequantize_mxfp4_t4, 576, 512, 2>;
```

**Step 3: Add K=q8_0 + V=q4_0 vec instantiations**

```metal
// Mixed K/V vec: K=q8_0, V=q4_0
template [[host_name("kernel_flash_attn_ext_vec_q8_0_v_q4_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q4_0, 8, dequantize_q4_0_t4, 32, 32, 4>;
// ... (same 8 head sizes)
template [[host_name("kernel_flash_attn_ext_vec_q8_0_v_q4_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q4_0, 8, dequantize_q4_0_t4, 576, 512, 2>;
```

**Step 4: Build to verify**

```bash
cd /Users/tim/code/llama.cpp && cmake --build build -j 2>&1 | tail -20
```

**Step 5: Commit**

```bash
git add ggml/src/ggml-metal/ggml-metal.metal
git commit -m "Metal: add native mixed K/V flash attention vec kernel instantiations"
```

---

## Task 3: Update Pipeline Lookup for Native Mixed Kernels

**Files:**
- Modify: `ggml/src/ggml-metal/ggml-metal-device.cpp:1262-1355` (non-vec pipeline)
- Modify: `ggml/src/ggml-metal/ggml-metal-device.cpp:1357-1446` (vec pipeline)

**Step 1: Define the native mixed combo check**

In both `ggml_metal_library_get_pipeline_flash_attn_ext()` and `_vec()`, after computing `mxfp_v_type`, check if we have a native kernel for this K/V combo. If so, use a different base name and set `mxfp_v_type = 0`.

For `ggml_metal_library_get_pipeline_flash_attn_ext()` (non-vec), modify lines ~1282-1315:

```cpp
    // Check if we have a native mixed K/V kernel (avoids dispatch overhead)
    bool native_mixed = false;
    if (op->src[1]->type != op->src[2]->type) {
        const auto kt = op->src[1]->type;
        const auto vt = op->src[2]->type;
        native_mixed =
            (kt == GGML_TYPE_MXFP8_E4M3 && vt == GGML_TYPE_MXFP4_E2M1) ||
            (kt == GGML_TYPE_MXFP6_E2M3 && vt == GGML_TYPE_MXFP4_E2M1) ||
            (kt == GGML_TYPE_Q8_0       && vt == GGML_TYPE_Q4_0);
    }

    if (native_mixed) {
        snprintf(base, 256, "kernel_%s_%s_v_%s_dk%d_dv%d",
                "flash_attn_ext",
                ggml_type_name(op->src[1]->type),
                ggml_type_name(op->src[2]->type),
                dk, dv);
        mxfp_v_type = 0; // native kernel handles V type directly
    } else {
        snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
                "flash_attn_ext",
                ggml_type_name(op->src[1]->type),
                dk, dv);
    }
```

**Step 2: Apply same change to vec pipeline lookup**

Same pattern for `ggml_metal_library_get_pipeline_flash_attn_ext_vec()` at lines ~1378-1408, using `"flash_attn_ext_vec"` in the snprintf.

**Step 3: Build and verify**

```bash
cd /Users/tim/code/llama.cpp && cmake --build build -j 2>&1 | tail -20
```

**Step 4: Commit**

```bash
git add ggml/src/ggml-metal/ggml-metal-device.cpp
git commit -m "Metal: route mixed K/V flash attention to native kernels

When K/V combo has a native template instantiation (mxfp8+mxfp4,
mxfp6_e2m3+mxfp4, q8_0+q4_0), use it directly instead of the
function-constant dispatch path. This gives typed pointer access
and eliminates the ~35% mixed-type throughput regression."
```

---

## Task 4: Run Flash Attention Tests

**Step 1: Run test-backend-ops flash attention tests**

```bash
cd /Users/tim/code/llama.cpp && ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -b Metal0 2>&1 | tail -40
```

Expected: All 9120 flash attention tests pass (includes all MXFP mixed K/V combos).

**Step 2: If any failures, check the native mixed kernels specifically**

Look for failures involving the 3 native mixed combos:
- `mxfp8_e4m3` K + `mxfp4_e2m1` V
- `mxfp6_e2m3` K + `mxfp4_e2m1` V
- `q8_0` K + `q4_0` V

Common issues:
- Wrong `ns10`/`ns20` stride values (K and V strides differ for mixed types)
- Mismatched `nl` values (should be 2 for non-vec, 8 for vec for all quantized types)
- Pipeline name typo (host_name must exactly match the lookup snprintf)

**Step 3: If tests pass, commit any fixes and run full test suite**

```bash
cd /Users/tim/code/llama.cpp && ./build/bin/test-backend-ops -b Metal0 2>&1 | tail -20
```

---

## Task 5: Benchmark Before/After on M2 Air

**Step 1: Run kv-bench with the same model and configs**

```bash
cd /Users/tim/code/llama.cpp && bash kv-bench-local.sh /Users/tim/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf 16
```

**Step 2: Compare mixed-type results against baseline**

Key metrics to compare (baseline from pre-optimization run):

| Config | Baseline pp512 | Baseline tg128 | Target pp512 | Target tg128 |
|--------|---------------|----------------|-------------|-------------|
| q8_0+q4_0 | 154 | 12.9 | ~300+ | ~18+ |
| mxfp8_e4m3+mxfp4 | 231 | 11.5 | ~310+ | ~16+ |
| mxfp6_e2m3+mxfp4 | 193 | 11.1 | ~290+ | ~15+ |

Matched types should be UNCHANGED (regression check):

| Config | Expected pp512 | Expected tg128 |
|--------|---------------|----------------|
| f16 | ~357 | ~19.6 |
| q8_0+q8_0 | ~340 | ~19.1 |
| mxfp4+mxfp4 | ~333 | ~17.4 |

**Step 3: Save benchmark results**

Results are auto-saved to `test-results/kv-bench-*` by the script.

**Step 4: Commit benchmark results and push**

```bash
git add test-results/
git commit -m "Metal: benchmark results for native mixed K/V flash attention

M2 Air, Qwen3-4B Q8_0, 16 chunks. Compare mixed K/V performance
before and after native template instantiations."
git push
```

---

## Task 6 (Follow-up): Add Secondary Mixed Combos

After confirming the primary 3 combos show improvement, add:
4. K=mxfp8_e5m2 + V=mxfp4
5. K=mxfp6_e3m2 + V=mxfp4

Same pattern as Tasks 1-3 but with the e5m2/e3m2 types. Lower priority since these are less commonly used.

---

## Task 7 (Follow-up): Cross-Machine Benchmark Matrix

Run `kv-bench-local.sh` on each machine and fill in:

| Machine | GPU | BW | f16 tg | mxfp4 tg | mx8+4 tg | mx6+4 tg | q8+q4 tg |
|---------|-----|----|--------|----------|----------|----------|----------|
| M2 Air 24GB | 8c | 100 | | | | | |
| M2 Max Studio 32GB | 30c | 400 | | | | | |
| M2 Ultra Studio 64GB | 60c | 800 | | | | | |
| M4 Max MBP 64GB | 40c | 546 | | | | | |
| M4 Max MBP 128GB | 40c | 546 | | | | | |

Use: `Qwen3-4B-Instruct-2507-Q8_0.gguf` for consistency, 16 chunks.
For larger machines, also test with a 32B+ model to hit bandwidth limits.

---

## Notes

- The function-constant dispatch path is **kept as fallback** for any K/V combo without a native kernel (e.g., mxfp8_e5m2+mxfp6_e2m3). No code is removed.
- PSOs are compiled on-demand, so unused kernel instantiations don't affect startup time. They only increase the metallib binary size slightly.
- The `mxfp_type` function constant (Q preprocessing) is independent of V type and continues to work correctly — it's set based on K type only.
- The `ns10`/`ns20` stride constants are computed from `src[1]->nb[1]` and `src[2]->nb[1]` respectively, so they automatically reflect the different K and V block sizes.
