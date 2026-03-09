# MXFP KV Cache SoA Optimization Plan

**Goal:** EXCEED f16 in both pp and tg for MXFP KV cache on Metal. MXFP4 transfers 73% less KV data than f16 — with efficient aligned loads and fast dequant, we should be FASTER, not slower. Current gap: pp ~96%, tg ~90% of f16. Target: >100% of f16 in both.

**Testing:** ONLY use `kv-bench-local.sh` (Metal) or `kv-bench.sh` (Docker/CUDA). NEVER run llama-bench directly. NEVER pipe or redirect command output.

**Machine:** M4 Max, 128GB RAM. Linux machine with 2x 5070 Ti for CUDA.

**Default bench model:** Qwen3-Coder-30B MXFP4_MOE at `/Users/tim/.lmstudio/models/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/`

**Branch:** `mxfp4-flash-attention-v3` (fork: `git@github.com:timlikesai/llama.cpp.git`)

---

## RULES FOR AI AGENTS

1. **Read this file before doing anything.** This is the plan. Follow it.
2. **Never abandon a plan step.** If something produces wrong results, debug it. Check the Notes section. Check the Consistency Checklist. The bugs are probably already documented here.
3. **Never revert to AoS.** SoA is the goal. AoS is the fallback that already works at 91%. We need 95%+. SoA is how we get there.
4. **Update this file** as you complete work. Check off items. Add findings. Record benchmark results.
5. **If you're stuck**, re-read the Pipeline Map and Notes sections below. The answer is probably there.

---

## Current State (2026-03-08)

### What the code does now (SoA working):

**set_rows: WRITES SoA** — all MXFP set_rows kernels write SoA format per full multi-head row:
- Layout: `[qs_block0|qs_block1|...|qs_blockN][e8m0_0|e8m0_1|...|e8m0_N]`
- MXFP4: qs at `ind * 16`, e8m0 at `nk0 * 16 + ind`
- MXFP8: qs at `ind * 32`, e8m0 at `nk0 * 32 + ind`
- MXFP6: qs at `ind * 24`, e8m0 at `nk0 * 24 + ind`

**FA kernels: READ SoA with full-row addressing** — uses `dequant_mxfp_soa_4x4()`/`dequant_mxfp_soa_t4()`
with full-row base pointer, absolute block index (head_block_offset + bidx), and NS10/NS20 as total nblocks.

**Q preprocessing: ENABLED in 4x4 kernel, DISABLED in VEC kernel** (VEC costs ~2 t/s).

### Benchmark Results (SoA, 2026-03-08, all perplexity correct):

| Config | PPL | pp512 | tg128 | % of f16 | Gap to target |
|--------|-----|-------|-------|----------|---------------|
| f16 baseline | 11.17 | 1,507 | 107.4 | 100% | — |
| mxfp8_e4m3+mxfp4 | 11.21 | 1,435 | 95.8 | 89.2% | +5.8% needed |
| mxfp8_e5m2+mxfp4 | 11.97 | 1,438 | 94.9 | 88.3% | +6.7% needed |
| mxfp6_e2m3+mxfp4 | 11.56 | 1,440 | 95.5 | 89.0% | +6.0% needed |
| mxfp6_e3m2+mxfp4 | 12.01 | 1,431 | 95.5 | 89.0% | +6.0% needed |
| mxfp4 matched | 11.77 | 1,447 | 96.6 | 90.0% | +5.0% needed |

### Key bug that was fixed:

The SoA corruption (PPL=30M) was caused by a **head offset mismatch**: set_rows writes SoA across
the full multi-head row (e.g., nk0=16 blocks for 4 heads × 4 blocks/head). E8M0 values are at
offset `nk0 * qs_per_block` (byte 256 for MXFP4). But the FA kernel used per-head addressing with
`nblocks = DK/32 = 4`, expecting e8m0 at `4*16+bidx = 64+bidx`. Additionally, the AoS-based head
stride (nb12 = 68 bytes) didn't match the SoA qs stride (64 bytes per head), so even qs offsets
were wrong for heads > 0.

**Fix:** FA dequant now uses full-row base pointer (subtracting `k_head_byte_off`), absolute block
indices (`mxfp_head_k_boff + bidx_within_head`), and `NS10`/`NS20` as total nblocks per row.

---

## STEP 0: Fix SoA bugs and restore SoA [IN PROGRESS]

### 0a. Fix P0: E8M0 clamp in Q preprocessing [ALREADY DONE]
- [x] `mxfp4_compute_e8m0()` line 884: already `clamp(e_base, 0, 254)` — was never 255
- [x] `mxfp_compute_e8m0_mse()` line 1271: already `clamp(e_base, 0, 254)` — was never 255
- [x] MSE search range line 883/1270: already `min(..., 254)` — was never 255
- The PLAN's original analysis was based on incorrect assumptions about the code state

### 0b. Fix P1: Scale application timing [TESTED — REVERTED]
- [x] Tested applying `*= args.scale` before Q preprocessing in 4x4 kernel
- [x] Result: PPL went from 11.77 to 11.86 (WORSE, not better)
- [x] Root cause: Metal dequantizes Q back to float (unlike CUDA block-scaled MMA), so
      applying scale≈0.088 before MXFP4 quantization makes values smaller, losing precision
- [x] **Decision: Keep scale applied AFTER QK MMA (the current approach). This is correct for
      Metal's dequant+standard MMA path. CUDA's approach is for block-scaled MMA only.**
- [x] VEC Q preprocessing: tested re-enabling, costs ~2 t/s. Left disabled for now.

### 0c. Restore SoA set_rows write [DONE]
- [x] MXFP4: qs at `ind * 16`, e8m0 at `nk0 * 16 + ind`
- [x] MXFP8: qs at `ind * 32`, e8m0 at `nk0 * 32 + ind`
- [x] MXFP6: qs at `ind * 24`, e8m0 at `nk0 * 24 + ind`
- [x] Fixed `dequantize_mxfp4_soa` 4x4 function (was reading AoS offsets as debug workaround)

### 0d. Activate SoA FA dequant [DONE]
- [x] 4x4 kernel K: routes through `dequant_mxfp_soa_4x4()` when `FC_flash_attn_ext_mxfp_type > 0`
- [x] 4x4 kernel V: routes through `dequant_mxfp_soa_4x4()` for MXFP V types (1-5),
      keeps AoS dispatch for q4_0(6)/q8_0(7), keeps template for non-MXFP matched
- [x] VEC kernel K: routes through `dequant_mxfp_soa_t4()` when `FC_flash_attn_ext_vec_mxfp_type > 0`
- [x] VEC kernel V: same pattern as 4x4 but with t4 variants

### 0e. Test SoA end-to-end [DONE]
- [x] f16 baseline: PPL 11.17 ✓
- [x] mxfp4 SoA: PPL 11.77 ✓ (Δ+0.60 from f16)
- [x] All MXFP configs produce correct PPL ✓
- [x] All MXFP configs run at ~89-90% of f16 throughput

### 0f. Debug SoA PPL corruption [DONE]

**Root cause found:** Head offset mismatch between set_rows (full-row SoA) and FA (per-head addressing).

CUDA uses `mxfp_soa_head_offsets()` to compute qs_off and e_off from the full row base.
Metal's FA kernel pre-adjusts k/v by `ikv2*nb12` (AoS head stride), but SoA has different
per-head byte offsets (qs stride = 64 per head, not 68).

**Fix:** Save `k_head_byte_off`/`v_head_byte_off` and `mxfp_head_k_boff`/`mxfp_head_v_boff` at
kernel start. In SoA dequant calls, use `row_ptr - head_byte_off` as full-row base,
`head_boff + bidx` as absolute block index, and `NS10`/`NS20` as total nblocks.
Applied to all 8 SoA dequant call sites (4x4: 2K+2V, VEC: 1K+1V) plus pad buffer paths.

---

## STEP 1: Metal SoA set_rows + FA dequant for all MXFP types [PARTIALLY DONE — see STEP 0]

### 1a. Shared SoA infrastructure in ggml-common.h [DONE]
- `MXFP4_SOA_QS_PER_BLOCK = 16`, `MXFP8_SOA_QS_PER_BLOCK = 32`, `MXFP6_SOA_QS_PER_BLOCK = 24`

### 1b. CUDA mxfp-traits.cuh updated to use shared constants [DONE]

### 1c. Metal SIMD set_rows kernels for all MXFP types [DONE — but writing AoS, need SoA]
- SIMD-parallel kernels exist and work correctly (32 threads per 32-element block)
- Currently write AoS format — need to change to SoA (STEP 0c)

### 1d. Metal SoA aligned-load FA dequant for all MXFP types [DONE — but dead code]
- `dequant_mxfp_soa_4x4()` and `dequant_mxfp_soa_t4()` exist at lines 824/843
- Currently never called — need to activate (STEP 0d)

### 1e. Metal ggml-metal-ops.cpp dispatch [DONE]
- `is_mxfp_simd` covers all 5 MXFP types, routes to SIMD kernels

### Lessons learned (cross-backend)
- **Metal cannot export kernels templated on `static inline` function pointers.** Compiles fine, crashes at runtime with null pipeline. Use int template params + switch instead.
- **SoA aligned loads are the single biggest win.** This pattern is universal — every backend benefits from contiguous aligned data.
- **MXFP4 constant LUT (16 entries) is register-speed on Apple GPU.** No benefit from arithmetic decode.

---

## STEP 2: Clean up AoS code paths [DEFERRED — until SoA is working]

Previously marked DONE but then SoA was reverted, undoing the cleanup. Redo after STEP 0 succeeds.

### 2a. AoS dequant function definitions must remain
- Referenced by FA template instantiations as template params
- Even when SoA path handles dequant at runtime, the definitions are needed for compilation

### 2b. Dead code to delete after SoA works
- [ ] Old serial set_rows code (if any remains)
- [ ] AoS-specific FA dequant call sites (replaced by SoA dispatch)
- [ ] MXFP V dispatch cases in `dequant_v_4x4_dispatch` / `dequant_v_t4_dispatch` (SoA handles these)

---

## STEP 3: Metal FA performance — close the gap to >95% [TODO — after STEP 0]

### Analysis: why MXFP FA is slower than f16 FA

The FA VEC kernel (token generation path) has three code paths for K dequant:
1. **Path A — native template match** (q8_0, q4_0, f16): Direct `dot(float4, float4)`, zero dequant overhead.
2. **Path B — MXFP SoA dispatch**: `dequant_mxfp_soa_t4()` switch on mxfp_type, then per-type dequant.
3. **Path C — generic dequant**: Block-based dequant for other types.

### 3a. Research: profile FA decode [TODO]
- [ ] Use Metal System Trace (Instruments) to get per-kernel GPU time for tg path
- [ ] Compare FA kernel time: f16 vs mxfp4 vs mxfp8 vs mxfp6
- [ ] Determine: is the gap ALU-bound (dequant cost) or memory-bound (SoA access pattern)?
- [ ] Check occupancy: does MXFP dequant use more registers, reducing occupancy?

### 3b. Arithmetic MXFP8 decode [TESTED — LUT WINS on Apple GPU]
LUT = 94.7 t/s, Arithmetic = 90.2 t/s. Keep LUT on Metal.

### 3c. MXFP6 decode improvements [TODO]
- [ ] Check if t4 MXFP6 dequant can use a single uint32_t load instead of 3 byte loads

### 3d. E8M0 scale hoisting [ANALYZED — NO OPPORTUNITY]
Each inner loop iteration hits a different block. No redundant loads.

### 3e. Half-precision K dot product [TESTED — NO EFFECT on Apple M4]
Apple M4 has unified ALU — same throughput for half and float.

### 3f. Wider V loads for matched configs [TODO]
- [ ] Check if V inner loop can process 8 or 16 elements per iteration instead of 4

### 3g. NE parameter tuning [DONE — NE=4 wins]
Applied NE=4 to all MXFP dk128 VEC instantiations. +1.3% to +2.5% across configs.

### 3h. Reduce function constant dispatch overhead [TODO]
- [ ] Compare generated assembly for Path A vs Path B
- [ ] Consider native MXFP template specializations that inline SoA dequant

---

## STEP 4: Cross-backend pattern sharing [TODO]

### 4a. Document Metal-specific vs portable patterns
### 4b. Update CUDA mxfp-traits.cuh with Metal wins [TODO]
### 4c. Benchmark CUDA on Linux 2x 5070 Ti [TODO]

---

## STEP 5: Vulkan SoA [TODO]

### 5a. Research current Vulkan MXFP implementation
### 5b. Vulkan set_rows shaders → SoA layout
### 5c. Vulkan FA shaders → SoA dequant
### 5d. Test on Linux 2x 5070 Ti

---

## STEP 6: CPU SoA [TODO]

### 6a. Research current CPU MXFP implementation
### 6b. CPU set_rows → SoA layout
### 6c. CPU FA dequant → SoA

---

## Cross-Backend Patterns Reference

### SoA Memory Layout (all backends, defined in ggml-common.h)
```
Per FULL row (all heads packed): [qs_block0|qs_block1|...|qs_blockN][e8m0_0|e8m0_1|...|e8m0_N]

qs region:  nblocks_total * QS_PER_BLOCK bytes, contiguous and aligned
e8m0 region: nblocks_total bytes, starts at nblocks_total * QS_PER_BLOCK
Total row size = nblocks_total * (QS_PER_BLOCK + 1) = same as AoS

MXFP4: QS_PER_BLOCK = 16 bytes (32 4-bit values, packed)
MXFP8: QS_PER_BLOCK = 32 bytes (32 8-bit values)
MXFP6: QS_PER_BLOCK = 24 bytes (32 6-bit values, packed)

IMPORTANT: SoA spans ALL heads in the row, not per-head. The ggml tensor strides
(nb[1]) use AoS block size for per-head views. FA kernels must use full-row base
pointer + absolute block indices, not per-head pointers + per-head nblocks.
Metal: NS10 = nb11/nb10 = total blocks per row. CUDA: stride_blocks = nb_row/block_size.
```

### Aligned Load Pattern (all backends)
```
MXFP4: uint32_t at bidx*16 + offset → always 4-byte aligned
MXFP8: uint32_t at bidx*32 + offset → always 4-byte aligned
MXFP6: uint32_t at bidx*24 + il*12  → always 4-byte aligned (4x4 path)
        3 bytes at bidx*24 + il*3   → byte-aligned only (t4 path)
```

### Integer Subtype IDs (all backends)
```
0 = not MXFP
1 = MXFP4 (E2M1)
2 = MXFP8 E4M3
3 = MXFP8 E5M2
4 = MXFP6 E2M3
5 = MXFP6 E3M2
```

### SIMD Set_Rows Pattern (all backends)
```
1. Load 32 floats (1 per thread/lane)
2. Optional Hadamard butterfly via shuffle_xor
3. Compute amax via subgroup/simd max
4. Compute E8M0: round(log2(amax)) - emax_offset + 127
5. Quantize each element to target format
6. Write qs to SoA qs region, e8m0 to e8m0 region
7. For 6-bit types: cooperative packing (4 threads → 3 bytes via shuffle)
```

### Backend-Specific Equivalents
```
                    Metal              CUDA                Vulkan
shuffle_xor:        simd_shuffle_xor   __shfl_xor_sync     subgroupShuffleXor
shuffle:            simd_shuffle       __shfl_sync         subgroupShuffle
max reduce:         simd_max           warp reduce         subgroupMax
sum reduce:         simd_sum           warp reduce         subgroupAdd
bit cast:           as_type<>()        __float_as_uint     floatBitsToUint
cached load:        device ptr         __ldg()             buffer load
specialization:     function_constant  template param      spec_constant
int8 dot product:   N/A                dp4a/__dp4a         N/A (extension)
```

---

## Cross-Backend MXFP Flash Attention Pipeline Map

**This is the definitive reference for implementation consistency.**

The fundamental insight: MXFP flash attention is a **memory pipeline** problem. The FP math
is small and fast — a few multiplies and LUT lookups. Performance comes from **byte-aligned,
pipelined memory transfers** enabled by the SoA layout. Every backend must move the same data
through the same mathematical steps. Hardware intrinsics create minor deviations; the structure
and math must be identical.

### Architecture: Why SoA Unlocks Performance

```
Traditional AoS (array of structs):  [e8m0_0|qs0_0..qs0_15][e8m0_1|qs1_0..qs1_15]...
  → 17-byte stride (MXFP4), 25-byte (MXFP6), 33-byte (MXFP8)
  → Non-power-of-2 → can't align memory transfers → serialized loads

SoA (struct of arrays):              [qs0_0..qs0_15|qs1_0..qs1_15|...][e8m0_0|e8m0_1|...]
  → qs region: nblocks × 16B (MXFP4) / 24B (MXFP6) / 32B (MXFP8)
  → All power-of-2 aligned: uint32_t loads always land on 4B boundaries
  → GPU memory controllers can pipeline multiple aligned loads in parallel
  → E8M0 region: contiguous bytes, loaded separately (1 byte per block)
```

**This alignment is what makes everything else work.** Without it, loads serialize,
occupancy drops, and dequant-heavy types (MXFP6, MXFP8) fall to 60-70% of f16.
With it, we get 90%+ because the GPU's memory pipeline is fully utilized.

### Pipeline Overview (both backends)

```
┌──────────────┐     ┌───────────────┐     ┌───────────────┐     ┌──────────┐
│  set_rows     │────>│  Q preprocess │────>│  KQ attention │────>│  VKQ     │
│  (KV store)   │     │  (per query)  │     │  (per KV pos) │     │  accum   │
└──────────────┘     └───────────────┘     └───────────────┘     └──────────┘
 float → SoA MXFP     float → MXFP → float   SoA K → dequant      SoA V → dequant
 Hadamard + E8M0      Hadamard + E8M0         × preprocessed Q     × softmax(QK)
 quantize + pack      quantize + dequant      softmax               accumulate
```

### Stage 1: set_rows — Float to SoA MXFP (KV Cache Write)

**Purpose:** Store incoming KV vectors in quantized SoA format for later FA consumption.
**Math must match exactly between backends.** Any difference here propagates through all attention.

```
Step   │ Operation              │ CUDA (reference)              │ Metal                         │ Match?
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
1.1    │ Load 32 floats         │ 1 thread loads 32 values      │ 32 threads × 1 value (SIMD)   │ ✓ same data
       │                        │ sequential from src0          │ via simd_shuffle              │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
1.2    │ Hadamard-32 rotation   │ hadamard_32_inplace()         │ hadamard_32_simd()            │ ✓ same math
       │ (if K type)            │ in-register butterfly         │ simd_shuffle_xor butterfly    │   1/√32 norm
       │                        │ 5 stages, ×(1/√32)           │ 5 stages, ×0.17677669...      │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
1.3    │ Compute amax           │ sequential max over 32 vals   │ simd_max(abs(val))            │ ✓ same result
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
1.4    │ E8M0 scale (MSE)       │ round(log2(amax)) via IEEE    │ round(log2(amax)) via IEEE    │ ✓ same algo
       │                        │ bit extract, ±R MSE search    │ bit extract, ±R MSE search    │
       │                        │ clamp to [0, 255]             │ clamp to [0, 255]             │ ⚠ SEE NOTE 1
       │                        │ mxfp_traits::mse_error()      │ mxfp4_compute_e8m0() etc      │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
1.5    │ Quantize elements      │ traits::write_qs()            │ best_index_mxfp4_metal() etc  │ ✓ type-specific
       │                        │ MXFP4: float_to_fp4_e2m1      │ MXFP4: decision boundaries    │
       │                        │ MXFP8: float_to_fp8 intrinsic │ MXFP8: float_to_fp8_e4m3_rn  │
       │                        │ MXFP6: float_to_fp6 intrinsic │ MXFP6: float_to_fp6_e2m3_rn  │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
1.6    │ Write SoA layout       │ qs → row_base + bidx*qs_size  │ qs → row_base + bidx*qs_size  │ ✓ same layout
       │                        │ e8m0 → row_base + total_qs    │ e8m0 → row_base + total_qs    │
       │                        │        + bidx                 │        + bidx                 │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
1.7    │ MXFP6 bit packing      │ pack_fp6x4(): 4→3 bytes       │ simd cooperative: 4 threads   │ ✓ same bits
       │ (6-bit types only)     │ sequential                    │ → 3 bytes via simd_shuffle    │
```

### Stage 2: Q Preprocessing — Simulate KV Quantization Loss on Queries

**Purpose:** The Q vectors must "see" the same quantization loss as K. Since K was Hadamard-rotated
and quantized, Q must undergo the same transform so that dot(Q,K) is computed in the quantized domain.

**THIS IS WHERE CUDA AND METAL FUNDAMENTALLY DIVERGE.**

```
Step   │ Operation              │ CUDA (reference)              │ Metal (current)               │ Match?
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
2.1    │ Load Q from global     │ float2 loads from Q_f2        │ Q loaded to threadgroup sq[]  │ ✓
       │                        │ ×scale applied immediately    │ scale applied later (in QK)   │ ⚠ SEE NOTE 2
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
2.2    │ Hadamard-32            │ hadamard_32_inplace()         │ hadamard_32_simd()            │ ✓ same math
       │                        │ (if apply_hadamard)           │ (if DK == DV)                 │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
2.3    │ E8M0 for Q block       │ MSE search, same as set_rows  │ MSE search via                │ ✓ same algo
       │                        │ clamp to [0, 255]             │ mxfp_compute_e8m0_mse()       │ ⚠ SEE NOTE 1
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
2.4    │ Quantize Q elements    │ Same quantize functions as    │ Same quant functions as        │ ✓
       │                        │ set_rows (traits::quantize)   │ set_rows (float_to_fp*_rn)    │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
       │                        │                               │                               │
       │ *** CRITICAL FORK ***  │                               │                               │
       │                        │                               │                               │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
2.5a   │ CUDA: Q stays quantized│ Q stored as packed MXFP in    │                               │
       │                        │ shared memory tile_Q_qs[]     │                               │
       │                        │ E8M0 stored in tile_Q_sc[]    │                               │
       │                        │ Used directly in block-scaled │                               │
       │                        │ MMA (mma_block_scaled)        │                               │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
2.5b   │ Metal: Q dequantized   │                               │ Dequant back to float:        │
       │ back to float/half     │                               │ mxfp4_roundtrip(val, scale)   │
       │                        │                               │ → copysign(qval*scale, val)   │
       │                        │                               │ Written back to sq[] as q_t   │ ⚠ SEE NOTE 3
       │                        │                               │ Used in standard simdgroup MMA│
```

### Stage 3: KQ Attention — Query × Key Dot Product

```
Step   │ Operation              │ CUDA (reference)              │ Metal (current)               │ Match?
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
3.1    │ Load K from SoA        │ cp.async 16B chunks           │ device ptr + aligned loads    │ ✓ same data
       │ (qs region)            │ → shared memory tile_K_qs     │ uint32_t loads in dequant     │
       │                        │ FP6: raw 24B → expand to 32B  │ FP6: uint32_t + mask          │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
3.2    │ Load K E8M0 scales     │ Byte loads → tile_K_sc        │ Loaded inside dequant func    │ ✓ same data
       │                        │ FP4: paired (2 per uint32)    │ e8m0_to_fp32() per block      │
       │                        │ FP6/8: individual             │                               │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
       │                        │                               │                               │
       │ *** CRITICAL FORK ***  │                               │                               │
       │                        │                               │                               │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
3.3a   │ CUDA: block-scaled MMA │ K in quantized form in smem   │                               │
       │                        │ Q in quantized form in smem   │                               │
       │                        │ mma_block_scaled(D,K,Q,Ksc,Qsc)                               │
       │                        │ HW applies E8M0 scales during │                               │
       │                        │ the MMA itself (fused)        │                               │
       │                        │ Result: float accumulators    │                               │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
3.3b   │ Metal: dequant + MMA   │                               │ K dequantized to float/half:  │
       │                        │                               │ dequant_mxfp_soa_4x4() →     │
       │                        │                               │   LUT[raw] × e8m0_to_fp32(e) │
       │                        │                               │ → written to threadgroup sk[] │
       │                        │                               │ Q already dequantized in sq[] │
       │                        │                               │ Standard simdgroup_multiply_  │
       │                        │                               │ accumulate(mqk, mq, mk, mqk) │
       │                        │                               │ Result: float accumulators    │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
3.4    │ Math equivalence       │ Q_quant × K_quant × Qscale    │ Q_dequant × K_dequant         │ MUST MATCH
       │                        │ × Kscale (block-scaled MMA)   │ = (Q_quant×Qscale) ×          │
       │                        │                               │   (K_quant×Kscale)            │
       │                        │ Both compute:                 │ Both compute:                 │
       │                        │ sum(dequant(qi) × dequant(ki))│ sum(dequant(qi) × dequant(ki))│
```

### Stage 4: Softmax

```
Step   │ Operation              │ CUDA (reference)              │ Metal (current)               │ Match?
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
4.1    │ Apply attention scale  │ Pre-applied to Q in step 2.1  │ ×args.scale after QK store    │ ✓ same result
       │                        │ (×scale before Hadamard)      │ (after MMA output)            │ ⚠ SEE NOTE 2
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
4.2    │ Logit softcap (opt)    │ softcap × tanh(x)             │ softcap × tanh(x)             │ ✓
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
4.3    │ Add mask               │ += slope × mask[j,i]          │ += slope × mask[j,i]          │ ✓
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
4.4    │ Online softmax         │ Track running max + rowsum    │ Track running max + rowsum    │ ✓
       │                        │ fast_expf_mxfp() cubic poly   │ exp() (Metal built-in)        │
       │                        │ Rescale VKQ accumulators      │ Rescale VKQ accumulators      │
```

### Stage 5: VKQ Accumulation — Softmax(QK) × V

```
Step   │ Operation              │ CUDA (reference)              │ Metal (current)               │ Match?
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
5.1    │ Load V from SoA        │ Dequant V → half2 in smem     │ Dequant V on-the-fly          │ ✓ same math
       │                        │ load_V_f16/mxfp8/mxfp6()     │ dequant_mxfp_soa_4x4/t4()    │
       │                        │ MXFP4: LUT × e8m0 scale      │ MXFP4: LUT × e8m0 scale      │
       │                        │ MXFP8: dequant × scale        │ MXFP8: LUT × scale           │
       │                        │ MXFP6: unpack 6b × scale      │ MXFP6: unpack 6b × scale     │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
5.2    │ VKQ MMA                │ half2 V tile × softmax(QK)    │ float/half V × softmax(QK)    │ ✓
       │                        │ via MMA instruction           │ via simdgroup MMA             │
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
5.3    │ Accumulate             │ VKQ_C[i] += (rescaled)        │ so4[j][i] += (rescaled)       │ ✓
───────┼────────────────────────┼───────────────────────────────┼───────────────────────────────┼────────
5.4    │ Final normalize        │ VKQ / rowsum                  │ VKQ / rowsum                  │ ✓
```

### Notes: Known Inconsistencies and Bug Suspects

**NOTE 1 — E8M0 clamp range:**
- CUDA set_rows: `clamp(e_base, 0, 255)` — allows e=255 (Inf scale)
- CUDA Q quantize: `clamp(e_base, 0, 255)` — allows e=255
- Metal set_rows: `clamp(e_int, 0, 254)` — caps at 254 (no Inf)
- Metal Q preprocess mxfp4_compute_e8m0: `clamp(e_base, 0, 255)` — **allows e=255**!
- E8M0=255 → scale=Inf → `0 × Inf = NaN` in round-trip. **This is the probable NaN bug.**
- **FIX:** Metal mxfp4_compute_e8m0 (and all mxfp_compute_e8m0_mse variants) must clamp
  to [0, 254] to match the Metal set_rows clamp, OR both must match CUDA's [0, 255] with
  safe handling of Inf scale.
- **ALSO:** The MSE search range `e_lo/e_hi` must be clamped consistently. CUDA clamps
  `min(255, ...)` in the search but `max(0, min(255, ...))` for best_e. Metal Q preprocess
  uses `min(255, ...)` for search and `clamp(0, 255)` for best_e.

**NOTE 2 — Scale application timing:**
- CUDA applies `×scale` to Q values before Hadamard and quantization (step 2.1)
- Metal applies `×args.scale` to QK dot product result after MMA (step 4.1)
- **Mathematically equivalent** IF the scale factor doesn't affect quantization. But it DOES:
  - CUDA: Q is scaled, then Hadamard'd, then E8M0 is computed on scaled values
  - Metal: Q is NOT scaled, Hadamard'd, E8M0 computed on unscaled values
  - Different amax → different E8M0 → different quantization → different results
- **This is a correctness difference for perplexity** even if throughput is unaffected.
- **FIX:** Metal must apply `×scale` before Q preprocessing, not after QK.

**NOTE 3 — Round-trip NaN risk (MXFP4):**
```
mxfp4_roundtrip(val, scale):
  inv_scale = scale > 0 ? 1/scale : 0    // if scale=Inf → inv_scale=0
  normalized = abs(val) * inv_scale        // 0 or finite
  qval = decision_boundary(normalized)     // 0 for small normalized
  return copysign(qval * scale, val)       // 0 * Inf = NaN!
```
When E8M0=255, scale=Inf. If qval=0 (input is small relative to Inf), `0×Inf = NaN`.
This NaN propagates through all subsequent attention computation for that Q block.

**NOTE 4 — MXFP8/6 Q preprocessing round-trip:**
Same pattern as MXFP4. The generic `mxfp_roundtrip<quant, dequant>()` does:
```
float q = dequant(quant(val * inv_scale)) * scale
```
If scale=Inf and quant produces 0 → `0 × Inf = NaN`.

### Consistency Checklist (items to fix)

- [ ] **P0 (NaN bug):** Clamp E8M0 to [0, 254] in ALL Metal Q preprocessing paths
  - `mxfp4_compute_e8m0()` line 871: change `clamp(e_base, 0, 255)` → `clamp(e_base, 0, 254)`
  - `mxfp_compute_e8m0_mse()`: same clamp change
  - This matches Metal set_rows which already clamps to 254
  - Alternatively, add `0×Inf` guard in round-trip: `if (!isfinite(scale)) return 0.0f;`
- [ ] **P0 (NaN bug):** Also clamp MSE search range to 254: `e_hi = min(..., 254)`
- [ ] **P1 (scale timing):** Apply attention scale to Q before preprocessing, not after QK
  - In Metal FA 4x4 kernel: multiply `sq[j*DK + ...] *= args.scale` before mxfp_preprocess_q_dispatch
  - In Metal FA VEC kernel: same
  - Remove the `×args.scale` from the softmax step that currently applies it
  - This makes Metal match CUDA's approach (scale baked into Q before quantization)
- [ ] **P2 (CUDA consistency):** Verify CUDA also produces valid PPL (it should — it's the reference)
- [ ] **P3 (E8M0 range):** Decide on unified clamp policy across ALL backends:
  - Option A: clamp to [0, 254] everywhere (safe, no Inf)
  - Option B: clamp to [0, 255] everywhere + guard against Inf in round-trip
  - CUDA currently allows 255 in Q quantize — verify it doesn't hit NaN there too
- [ ] **P4 (MXFP6/8 variants):** Verify all 5 MXFP types use consistent E8M0 clamp
- [ ] **P5 (cross-backend test):** Add perplexity regression test to CI for all KV types

### Memory Transfer Summary (the real performance story)

```
                    Bytes per       Aligned    Transfers per    Pipeline
                    32-element blk  width      head (DK=128)    depth
MXFP4 qs:          16 B           uint32_t    4 blocks × 4     16 loads
MXFP4 e8m0:        1 B            byte        4 blocks         4 loads
MXFP8 qs:          32 B           uint32_t    4 blocks × 8     32 loads
MXFP8 e8m0:        1 B            byte        4 blocks         4 loads
MXFP6 qs:          24 B           uint32_t*   4 blocks × 6     24 loads
MXFP6 e8m0:        1 B            byte        4 blocks         4 loads
f16 (baseline):    256 B          half4       16 loads          16 loads

* MXFP6: 24B not perfectly aligned, use uint32_t+mask or 3-byte loads

Total memory per head per KV position:
  MXFP4: 68 B (73% reduction vs f16)
  MXFP8: 132 B (48% reduction vs f16)
  MXFP6: 100 B (61% reduction vs f16)
  f16:   256 B (baseline)
```

The key to performance: **fewer bytes to transfer × aligned loads = GPU memory pipeline
can issue more loads in parallel, overlapping computation with transfer.** NE=4 multiplies
this by processing 4 KV positions simultaneously — 4× the memory loads in flight.

### VEC Kernel Pipeline (Token Generation — the critical path)

```
Token generation = 1 query × many KV positions.
Each iteration processes NE=4 KV positions:

For each KV position group (4 positions):
  ┌─ Memory: Load 4× K qs (aligned uint32_t)     ← PIPELINED
  ├─ Memory: Load 4× K e8m0 (byte)               ← PIPELINED
  ├─ Compute: Dequant K = LUT[qs] × e8m0_to_f32  ← SMALL
  ├─ Compute: dot(Q_preprocessed, K_dequant)      ← SMALL
  ├─ Memory: Load 4× V qs (aligned)               ← PIPELINED
  ├─ Memory: Load 4× V e8m0 (byte)               ← PIPELINED
  ├─ Compute: Dequant V = LUT[qs] × e8m0_to_f32  ← SMALL
  └─ Compute: VKQ += softmax(QK) × V_dequant     ← SMALL

Memory dominates. Aligned SoA loads are what make this fast.
```
