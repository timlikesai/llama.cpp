# MXFP4 Flash Attention Optimization Roadmap

Branch: `mxfp4-flash-attention-v3`
Date: 2026-02-28

## Current Performance Baseline

| Metric | Value |
|--------|-------|
| Model | Qwen3-Coder-30B-A3B (MXFP4_MOE GGUF) |
| GPU | 2x RTX 5070 Ti (sm_120, consumer Blackwell) |
| PPL (59-chunk) | 9.8622 (+0.13 vs F16) |
| KV memory | 63.75 MiB (6x reduction vs F16 384 MiB) |
| pp512 | ~5,916 t/s |
| tg128 | ~150 t/s (no spec), ~247 t/s (n-gram spec) |

## Instruction Selection (Already Optimal)

The kernel uses `mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64` which is
the correct instruction for MXFP4 data (32-element E8M0-scaled blocks). Key decisions:

- **scale_vec::2X** (32 elements/scale) — matches MXFP4 block size. 4X would require
  NVFP4 format (16-element blocks, E4M3 scales), incompatible with GGUF MXFP4 weights.
- **kind::mxf4** — both Q and K/V are MXFP4. mxf4nvf4 would require NVFP4 on one side.
- **E8M0 scales** — required by MX spec and hardware. No smaller scale format exists.
- **dp4a + LUT** in VEC kernel — no scalar FP4 dot product instruction on Blackwell.
- **Scalar butterfly Hadamard** — correct for 32 values in registers; HadaCore tensor
  core Hadamard only helps for large batched transforms in shared memory.

## Phase 1: Low-Effort Optimizations

### 1a. cp.async for K Tile Loads
**Impact: Medium | Effort: Low**

K quantized data (FP4 nibbles) goes directly to shared memory without dequantization,
making it a perfect candidate for `cp.async.cg.shared.global`. Currently all K loads are
synchronous (every thread participates in `reinterpret_cast` loads from global memory).

Benefits:
- Frees warp threads from load participation during K tile fetch
- Hardware-managed async pipeline overlaps loads with prior computation
- The kernel already includes `cp-async.cuh` but does not use it for MXFP4

Does NOT apply to V loads (V requires FP4→F16 dequantization before shared memory).

### 1b. Reduce MSE E8M0 Search Range (±2 → ±1)
**Impact: Low | Effort: Low**

The KV cache quantization tests 5 E8M0 candidates (±2 around amax estimate). With
Hadamard pre-rotation equalizing block values, the amax estimate is already reliable.
Reducing to 3 candidates (±1) halves the search cost with likely negligible PPL impact.

### 1c. Approximate exp() in Softmax
**Impact: Low-Medium | Effort: Low**

Replace `expf()` calls in softmax (which serialize on the SFU) with a polynomial
approximation of 2^x. Flash Attention 4 uses a cubic polynomial for this.

## Phase 2: Medium-Effort Optimizations

### 2a. V Double-Buffering
**Impact: Medium | Effort: Medium**

V is loaded synchronously after `__syncthreads()` — it blocks the pipeline while K MMA
could overlap. Double-buffering V (like K is already double-buffered) would overlap V
dequantization with the next iteration's K loads.

Requires additional shared memory for a second V buffer (~17 KB for D=128, nbatch_fa=64).
Current total smem is ~41 KB; adding a second V buffer brings it to ~58 KB, still under
Blackwell's 128 KB limit.

### 2b. TMA for K/V/Mask Loads
**Impact: Medium-High | Effort: Medium**

Tensor Memory Accelerator provides hardware-accelerated bulk transfers with automatic
address generation and OOB handling. Would replace the entire `flash_attn_ext_mxfp4_load_K`
function with a tensor map descriptor + `cp.async.bulk`.

Benefits:
- Eliminates all thread participation in address calculation and OOB checks
- Hardware handles multi-dimensional tensor addressing
- Available on sm_90+ (works on our sm_120)

## Not Viable for sm_120

### tcgen05.mma / TMEM Migration
**Impact: Very High | Effort: Very High | sm_100 only (datacenter Blackwell)**

5th-gen tensor core instructions with dedicated tensor memory (TMEM) for accumulators.
This is what Flash Attention 4 uses with full warp specialization. NOT available on
consumer Blackwell (sm_120 / RTX 5070 Ti) — requires sm_100a (B200, GB200).

### Cluster-Level Cooperation / DSMEM
**Impact: Medium | Effort: Very High | sm_100 only for full feature set**

Thread block clusters with distributed shared memory. While basic cluster launch works
on sm_120, the full DSMEM features needed for flash attention cooperation require sm_100a.
