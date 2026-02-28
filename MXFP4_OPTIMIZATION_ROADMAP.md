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
| pp512 | ~7,200 t/s (was ~5,916 before fast_expf) |
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

## Phase 1: Low-Effort Optimizations (COMPLETED)

### 1a. cp.async for K Tile Loads — NOT VIABLE
**Impact: N/A | Effort: Low | Status: Investigated, not viable**

MXFP4 SoA compact layout has row stride = 24 blocks × 17 bytes = 408 bytes. Since
408 % 16 = 8, odd-numbered KV rows are only 8-byte aligned. Both `cp_async_cg_16`
and `int4` vectorized loads require 16-byte alignment. Would require padding
`block_mxfp4` to 32 bytes (doubling memory), defeating the purpose.

TMA (Phase 2b) has the same alignment constraint and is also not viable.

### 1b. Reduce MSE E8M0 Search Range (±2 → ±1) — DONE ✓
**Commit: 18598686 | PPL impact: negligible**

Reduced from 5 candidates to 3. Hadamard pre-rotation makes the amax estimate reliable.

### 1c. Approximate exp() in Softmax — DONE ✓
**Commit: db7ee2b6 | pp512: 5,916 → 7,200 t/s (+21.7%) | PPL: +0.021**

Cubic polynomial approximation of 2^x (from Flash Attention 4) with IEEE 754 bit
manipulation. Added underflow guard for `xi < -126` to prevent garbage floats.

## Phase 2: Medium-Effort Optimizations

### 2a. V Double-Buffering
**Impact: Medium | Effort: Medium**

V is loaded synchronously after `__syncthreads()` — it blocks the pipeline while K MMA
could overlap. Double-buffering V (like K is already double-buffered) would overlap V
dequantization with the next iteration's K loads.

Requires additional shared memory for a second V buffer (~17 KB for D=128, nbatch_fa=64).
Current total smem is ~41 KB; adding a second V buffer brings it to ~58 KB, still under
Blackwell's 128 KB limit.

### 2b. TMA for K/V/Mask Loads — NOT VIABLE
**Impact: N/A | Effort: Medium | Status: Same alignment issue as cp.async**

TMA's `cp.async.bulk` also requires 16-byte aligned addresses. The 17-byte
`block_mxfp4` stride makes row bases non-16-byte-aligned for odd KV positions.
Same fundamental constraint as Phase 1a.

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
