# Vulkan MXFP Flash Attention — Performance Analysis & Optimization Notes

## Current State (2026-03-13)

**Branch:** `mxfp4-flash-attention-v3`
**Benchmark:** Qwen3-Coder-30B-A3B, D=128, 16-chunk, RTX 5070 Ti

| Config | PPL | pp512 | tg128 | % f16 tg |
|--------|-----|-------|-------|----------|
| f16 | 11.19 | 5,410 | 157.9 | 100% |
| q8_0+q4_0 | 11.14 | 5,237 | 181.0 | 114.6% |
| mxfp4 | 11.84 | 4,384 | 119.6 | 75.7% |
| mxfp8_e4m3 | 11.16 | 4,010 | 110.4 | 69.9% |

**Gap to close:** MXFP tg128 is ~70-76% of f16. Target: 85-90%.

## Architecture Overview

### Three Vulkan FA Shader Paths

| Path | Shader | When Used | MXFP Approach |
|------|--------|-----------|---------------|
| Scalar | `flash_attn.comp` | N=1 (token gen) | Direct register dequant, subgroup shuffles |
| CM1 | `flash_attn_cm1.comp` | Mixed K/V types | Cooperative matrix 16x16, shmem staging |
| CM2 | `flash_attn_cm2.comp` | N>1 (prompt), matched types | Cooperative matrix 2, shmem staging |

**Key dispatch rule** (`ggml-vulkan.cpp:2996`): For N=1 (token generation), scalar
is ALWAYS used, even on cm2-capable hardware. This means **cm2 optimizations only
affect prompt processing (pp512), NOT token generation (tg128)**.

### Why F16 is Fast (0 barriers)

The non-MXFP path uses `coopMatLoadTensorNV` with a decode function callback. The
driver handles all memory staging internally — zero explicit barriers in the iteration
loop. MXFP can't use this because SoA layout has qs and e8m0 in separate memory
regions, which the decode function's single-pointer model can't efficiently address.

### Why MXFP is Slow (shmem staging)

MXFP requires explicit dequant to shared memory, then `coopMatLoadTensorNV` from shmem:
```
Global → registers (dequant) → shmem → coopmat registers
vs f16:
Global → coopmat registers (driver-managed, single step)
```

This double memory traffic + barrier overhead is the fundamental cost.

## Barrier Analysis (CM2 Path)

### Current: 2 barriers per iteration (optimized from 4)

Separate k_shmem/v_shmem enables parallel K+V dequant:
```
K dequant → k_shmem ┐
V dequant → v_shmem ┘ (parallel, different memory)
barrier ← all dequant writes visible
coopMatLoad K from k_shmem
coopMatLoad V from v_shmem
barrier ← loads complete before next iter overwrites shmem
KQ MMA → softmax → VKQ MMA (all register compute, no shmem)
```

**Result:** Reducing from 4→2 barriers had NO measurable performance impact.
Confirms barriers are not the bottleneck.

## Memory Access Analysis (CM2 Dequant Loop)

### Bank Conflicts: ZERO
- HSK_pad=128, float16_t=2B, row=256B = 64 bank-width units
- Thread mapping: tid 0-31 → row 0, elem_off 0,4,8,...,124
- Each thread hits a unique bank. No conflicts.

### Global Memory Coalescing: PERFECT
- First warp reads uint32_t[0..31] = 1 full 128B cache line
- Each warp in each iteration reads one contiguous cache line
- No wasted bandwidth.

### E8M0 Scale Redundancy: 8x OVERLOAD
- Within a 32-element MXFP block, 8 threads (4 elements each) share ONE E8M0 scale
- Each thread loads its own copy from global memory = 8 redundant reads
- CUDA VEC loads scale ONCE per 32-element block (0.03 loads/elem vs 0.25 loads/elem)
- **Optimization opportunity: ~4-9% dequant time savings via subgroup broadcast**

## Scalar Path Analysis (Token Generation)

The scalar path (`flash_attn.comp`) is used for tg128. Key differences from CM2:

| Metric | Scalar | CM2 |
|--------|--------|-----|
| Q preprocessing | Registers + subgroup shuffles | Shmem staging (3 barriers) |
| K/V dequant | Direct per-element, no shmem | Shmem staging (2 barriers) |
| Dot product | vec4 dot (single instruction) | MMA (hardware tensor core) |
| V accumulation | Register-only | Coopmat from shmem |

### E8M0 Scale Problem in Scalar Path
- Scale loaded PER 4-element dequant call (0.25 loads/elem)
- CUDA VEC loads ONCE per 32-element block (0.03 loads/elem)
- **8x more scale reads than CUDA** — this is the biggest optimization target

### Comparison: CUDA VEC vs Vulkan Scalar

| Metric | CUDA VEC | Vulkan Scalar |
|--------|----------|---------------|
| K reads/elem | 0.31 | 0.37-0.50 |
| E8M0 loads/elem | 0.03 | 0.25 |
| Thread utilization | 100% warp | 70-80% workgroup |
| V accumulation | Register-only, 1 dequant/tile | Repeated dequants per K block |
| HW dequant | fp8x4→float4 (1 instruction) | uint32 + shift (2 steps) |

## CUDA Reference Patterns

### CUDA MMA (Prompt Processing)
- **Double-buffer:** K[N+1] prefetched via cp.async WHILE computing KQ on K[N]
- **cp.async:** Global→shmem overlapped with compute (no Vulkan equivalent)
- **V inline dequant:** V dequanted directly to half2 in shmem (not staged)
- **2 barriers per iteration** (pipeline sync + VKQ completion)

### CUDA VEC (Token Generation)
- **On-demand dequant:** K/V loaded per-element from global, no shmem
- **Scale amortization:** E8M0 loaded once per 32-element block
- **HW intrinsics:** fp8x4→float4 in 1 instruction (vs software dequant)
- **Zero barriers:** Register-only accumulation

## Metal Reference Patterns

### Metal VEC Kernel
- **Register-only V dequant:** V dequanted during accumulation, not staged
- **Simdgroup shuffles:** For reductions, no shmem needed
- **3 barriers per iteration** (Q load, Q*K^T, softmax)
- **Inline dequant constants:** LUTs in L1 cache

### What Makes Metal Fast
1. SoA aligned uint32_t loads (same as us)
2. Register-only V dequant (avoids shmem write-after-read hazard)
3. Simdgroup shuffle for reductions (no memory)
4. Template specialization (compiler unrolls everything)

## Optimization Opportunities

### Priority 1: E8M0 Scale Amortization (Scalar Path — affects tg128)

Load E8M0 scale once per 32-element block, broadcast to all threads processing
elements within that block. Options:
- `subgroupBroadcast(scale, lane_that_loaded_it)`
- Cache scale in register across the 8 consecutive dequant calls within same block

**Expected gain:** 4-9% of dequant time → 1-3% of tg128

### Priority 2: Scalar Path K/V Loop Optimization

The scalar path's K/V dequant calls `dequantize4()` which recomputes SoA offsets
(bwr, row_byte, etc.) on every call. These offsets are predictable within a row:
- `bwr` is constant for all blocks in the same row
- `row_byte` is constant for all blocks in the same row
- Only `blk` and `eib` change within a row

**Approach:** Pre-compute row offsets once, then iterate blocks with simple arithmetic.

### Priority 3: Vectorized Scale+Dequant

Instead of:
```glsl
float d = e8m0_to_fp32(scale_byte);
float val = fp8_e4m3_to_float(qs_byte) * d;
```

Combine scale decode + element dequant into a single operation that avoids the
intermediate float multiply.

### Priority 4: Subgroup-Level K Dot Product (Scalar Path)

Instead of each thread computing a partial dot product independently, use subgroup
shuffles to share K elements across threads for higher arithmetic density.

## No-Go Approaches (Confirmed)

1. **SoA decode functions for coopMatLoadTensorNV:** Driver can't coalesce e8m0
   reads from separate memory region. Confirmed by Vulkan spec research.
2. **cp.async equivalent:** No Vulkan/GLSL equivalent exists.
3. **Reducing barriers below 2:** Already at minimum for shmem staging pattern.
4. **Shared memory bank conflict optimization:** Already zero conflicts.
5. **Global memory coalescing:** Already perfect.

## Vulkan-Specific Notes

- `barrier()` provides both execution AND memory visibility for shared variables
- `coopMatLoadTensorNV` with workgroup scope is an implicit sync point
- Subgroup shuffles are measurably faster than shmem on RTX 5070 (Blackwell)
- `setTensorLayoutStrideNV` alignment hints (`& ~7`) enable faster load paths
- `gl_CooperativeMatrixClampModeConstantNV` avoids per-element bounds checking
