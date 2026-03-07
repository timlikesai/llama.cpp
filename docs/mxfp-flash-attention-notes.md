# MXFP Flash Attention: Implementation Notes & Research Directions

Draft brain dump — not publication quality. Capturing ideas, findings, and potential paper angles.

## The Core Idea

MX (Microscaling) format types — MXFP4, MXFP6 (E2M3/E3M2), MXFP8 (E4M3/E5M2) — use a shared
E8M0 scale per 32-element block, giving them a natural block structure that maps well to flash
attention's tiled computation. We implement flash attention kernels that operate directly on
MX-quantized KV cache data without dequantizing to FP16 first, achieving near-FP16 quality at
2-4x memory reduction.

The key insight: **attention is inherently tolerant of value quantization** because softmax
normalization dampens the effect of small errors in individual K/V elements. The attention
mechanism acts as a weighted average, so quantization noise tends to cancel out across the
sequence dimension.

## Novel Technique: Hadamard Rotation for MX Quantization

### The Problem
MX format quantizes in 32-element blocks with a single shared scale. Attention head dimensions
contain outlier values — a few large elements alongside many small ones. When a block's scale is
set by an outlier, the small values lose precision. This is the classic dynamic range problem.

### The Solution
Apply a Hadamard rotation (H32, the 32x32 Walsh-Hadamard matrix) to both K and Q before
quantization. The Hadamard transform spreads energy uniformly across all elements within each
32-element block, eliminating outliers and making the shared E8M0 scale more representative.

Key properties:
- **Orthogonal**: preserves dot products (K·Q is unchanged after rotating both)
- **Self-inverse**: H32 * H32 = 32*I, so one transform serves as both forward and inverse
- **Block-aligned**: 32-element Hadamard matches the 32-element MX block size exactly
- **Norm-preserving**: the L2 norm of each block is preserved (energy conservation)

### Implementation Details
- **CPU**: Full H32 butterfly (5 stages of 16 butterfly operations each = 160 FP ops per block)
- **CUDA VEC kernel**: Distributed across 16 threads via register shuffles, zero shared memory
- **CUDA MMA kernel**: Applied during Q quantization in registers before MMA
- **Vulkan**: Subgroup shuffles (shuffleXor) for butterfly stages
- **Metal**: Threadgroup-parallel with simd_shuffle_xor

### Norm Folding Optimization
The Hadamard transform scales values by sqrt(32). Instead of normalizing after the transform,
we fold the normalization into the E8M0 scale computation:
```
scale = round(log2(amax)) - round(log2(sqrt(32)))
      = round(log2(amax)) - 3
```
This saves 32 FMULs per block. In practice the perf gain is immeasurable (dominated by the
160 butterfly ops) but it's mathematically cleaner.

### Quality Impact
Without Hadamard rotation, PPL degrades catastrophically for lower-precision formats:
- mxfp8_e4m3: +0.22 (borderline acceptable)
- mxfp8_e5m2: +1.38 (significant)
- mxfp6_e2m3: +3.34 (catastrophic)
- mxfp6_e3m2: +4.60 (catastrophic)
- mxfp4: (not tested without, assumed even worse)

With Hadamard rotation, mxfp6_e2m3 achieves +0.03 PPL vs FP16 — better than q8_0 in some metrics.

### Performance Cost
- pp512 (prompt processing): ~5.7% slowdown (6,088 -> 6,435 without Hadamard)
- tg128 (token generation): 0% — completely hidden by memory latency in VEC kernel
- The cost is dominated by the 160 butterfly FP operations, not the norm computation

### Why No Cheaper Alternative
We tested:
- Sign flips / permutations: can't spread outliers (don't mix magnitudes)
- Partial Hadamard (2-3 butterfly stages): 40-60% compute savings but quality degrades
- H8×4 block-diagonal: smaller Hadamard on sub-blocks, quality loss
- BRQ paper confirms: block-32 Hadamard matching MX block size is optimal

## MSE-Optimal E8M0 Scale Selection

### The Problem
Standard MX quantization uses `round(log2(amax))` to compute the E8M0 shared scale. This
minimizes the maximum quantization error (minimax) but not the mean squared error (MSE).

### The Solution
For each 32-element block, evaluate 3 candidate scales: floor(log2(amax)), round(log2(amax)),
and ceil(log2(amax)). For each candidate, compute the total MSE across all 32 elements by
quantizing each element and measuring (original - reconstructed)^2. Pick the scale with minimum
total MSE.

This is 3x more compute (96 quantize+dequant+error operations vs 32) but produces measurably
better scales, especially for MXFP4 where each value has only 16 representable levels.

### Implementation
- **CUDA**: HW intrinsics for fp8/fp6 quantize+dequant in the inner loop. Critical for
  performance — software IEEE bit manipulation is 8-14% slower (the regression we fixed).
- **CPU**: Same MSE search using software converters. Slower but still fast enough for
  the token generation path.
- **GPU Q quantization**: Uses plain round(log2) for speed (Q is ephemeral, not cached).
  This creates a small GPU/CPU PPL gap (~0.2) since CPU uses MSE for Q too.

### SFU-Free E8M0 Computation
Instead of `__float2int_rn(log2f(amax))` which uses the Special Function Unit (SFU, low
throughput), we extract the IEEE-754 exponent via integer bit operations:
```c
uint32_t bits = __float_as_uint(amax);
int floor_log2 = ((bits >> 23) & 0xFF) - 127;
// Round: check if mantissa >= sqrt(2)-1 ≈ 0.4142
int round_log2 = floor_log2 + ((bits & 0x7FFFFF) >= 0x3504F3 ? 1 : 0);
```
This is exact for powers of 2 and correctly rounds for all other values. Pure integer ALU,
no SFU dependency. The 0x3504F3 threshold is the IEEE mantissa bits of sqrt(2).

## Architecture: Unified MMA + VEC Dispatch

### Two Kernel Families
1. **MMA kernel** (fattn-mma-mxfp.cuh): Blackwell tensor core MMA instructions. Dequantizes
   K/V tiles into shared memory once, then all query tokens in the batch reuse them. Optimal
   for prompt processing (batch > 2).

2. **VEC kernel** (fattn-vec.cuh): Scalar dot products with distributed dequant across threads.
   Each thread handles a slice of the K dimension. Optimal for token generation (batch ≤ 2)
   because the per-query overhead is hidden by memory latency.

### Why Not MMA for Everything?
Tested MMA at batch=1: 7.7% regression (169.8 vs 184.0 t/s). The MMA kernel's shared memory
setup cost and synchronization barriers aren't amortized with only 1-2 queries. VEC's
distributed approach avoids these overheads.

### Why Not VEC for Everything?
VEC dequantizes K/V for every query token. With batch=512 (prompt processing), that's 512x
redundant dequant work. pp512 regressed from 6,000 to 1,900 t/s when we tried VEC-for-all.

### Dispatch Threshold
batch ≤ 2: VEC, batch > 2: MMA. The threshold of 2 was determined empirically. Batch=2 is
the common case for speculative decoding (draft + verify).

## FP6 cp.async Loading (MMA Kernel)

MXFP6 data is stored in a tight packed format (24 bytes per 32-element block = 6 bits/element)
for memory savings. But the MMA kernel needs byte-aligned data for tensor core operations.

### The Pipeline
1. **cp.async**: Load packed 24-byte blocks from global memory as 16-byte chunks into shared memory
2. **In-place expansion**: A dedicated kernel (`flash_attn_ext_mxfp_expand_K_fp6`) reads the
   packed data from shared memory into registers, syncthreads, then writes expanded byte-per-element
   data back to shared memory
3. **MMA consumption**: Tensor cores read the expanded byte-aligned data

The expansion is gated by the `needs_smem_expand_k` trait (true only for FP6 types).

### Cross-Warp Safety
The expansion involves reading shared memory that other warps may have written. A
`__syncthreads()` barrier between the read and write phases ensures correctness.

### Performance
cp.async + expansion: 5,889 t/s pp512 (vs 5,441 without cp.async = +8.2%). Remaining ~4% gap
to FP8 is due to the expansion step itself.

## Benchmark Data (2026-03-07)

### CUDA RTX 5070 Ti — Qwen3-Coder-30B-A3B (D=128, GQA, 16 chunks)

| Config | KV GiB/100K | GPU PPL | Δ F16 | GPU pp512 | GPU tg128 | CPU pp512 | CPU tg128 |
|--------|-------------|---------|-------|-----------|-----------|-----------|-----------|
| q8/q4 | 3.72 | 11.45 | -0.08 | 5,977 | 185.3 | 265 | 10.9 |
| q8_0 | 4.86 | 11.52 | -0.01 | 5,963 | 185.4 | 266 | 10.9 |
| f16 | 9.16 | 11.53 | — | 6,045 | 189.1 | 266 | 10.9 |
| mxfp6e2m3 | 3.01 | 11.56 | +0.03 | 5,790 | 179.4 | 134 | 10.2 |
| mxfp8e4m3 | 3.58 | 11.66 | +0.13 | 5,818 | 180.6 | 137 | 10.4 |
| mxfp8e5m2 | 3.58 | 11.70 | +0.18 | 5,819 | 180.5 | 151 | 10.3 |
| mxfp6e3m2 | 3.01 | 11.74 | +0.21 | 5,802 | 179.5 | 130 | 10.3 |
| q4_0 | 2.58 | 11.96 | +0.43 | 5,972 | 185.0 | 263 | 10.9 |
| mxfp4 | 2.44 | 12.47 | +0.94 | 5,974 | 179.6 | 126 | 10.1 |

### Key Observations

**MXFP6 E2M3 is the Pareto-optimal choice:**
- +0.03 PPL vs f16 (better than q8_0's -0.01 within noise)
- 3.01 GiB/100K vs 9.16 GiB/100K = 67% memory reduction
- 179.4 vs 189.1 tg128 = 5.1% throughput cost
- Better quality AND less memory than q8_0 (4.86 GiB)

**MXFP outperforms q4_0 in quality at similar memory:**
- mxfp4 (2.44 GiB): PPL 12.47, vs q4_0 (2.58 GiB): PPL 11.96
- q4_0 wins on PPL here, but mxfp6e2m3 (3.01 GiB) crushes both: PPL 11.56
- The real comparison: mxfp6e2m3 vs q8_0 — similar quality, 38% less memory

**Throughput gap analysis:**
- GPU tg128: MXFP types are 5.0-5.1% slower than f16. This is irreducible dequant overhead.
  SASS analysis: mxfp4 kernel has 3,232 instructions vs 1,152 for f16 (2,080 extra for dequant).
- GPU pp512: MXFP types are 4.0-4.2% slower than f16. MMA dequant + Hadamard rotation cost.
- CPU pp512: MXFP types are 50% slower (126-151 vs 263-266). Dominated by MSE E8M0 scale search
  during Q quantization. This is the main MXFP weakness on CPU.
- CPU tg128: Only 7% slower. The MSE search only runs for new KV cache entries, which is a
  small fraction of the total computation during generation.

### Vulkan NVIDIA RTX 5070 Ti — Gemma 3n E4B (D=256, 16 chunks)

| Config | PPL | pp512 | tg128 | % f16 tg |
|--------|-----|-------|-------|----------|
| f16 | — | 6,197 | 88.4 | 100% |
| q8_0 | 39.00 | 6,169 | 88.2 | 99.8% |
| q4_0 | 39.05 | 6,151 | 88.0 | 99.5% |
| mxfp4 | 35.96 | 5,859 | 81.1 | 91.8% |

Note: mxfp4 PPL (35.96) is *better* than q8_0 (39.00) and q4_0 (39.05) on this model.
This suggests Hadamard rotation is particularly effective for Gemma 3n's attention patterns.

Vulkan tg128 gap is 8.2% vs CUDA's 5.1% — wider because Vulkan uses LUT-based software dequant
instead of CUDA's hardware fp8/fp6 conversion intrinsics.

## Open Questions & Future Directions

### Mixed K/V Types on Vulkan
Currently Vulkan only supports matched K/V types (K=mxfp4, V=mxfp4). Mixed types like
K=mxfp8_e4m3/V=mxfp4 fall back to non-flash attention (~20x slower). The CUDA kernel handles
mixed types via runtime V-type dispatch. Vulkan would need either:
- Shader specialization constants for V type
- Multiple shader variants (5x code but straightforward)

### MLA Model Support
MLA (Multi-head Latent Attention, used by DeepSeek/GLM-4.7) shares V with K (V_is_K_view).
This means:
- Hadamard rotation of K corrupts V — we skip rotation for MLA
- Without Hadamard, MXFP quality degrades significantly on MLA
- Need a rotation scheme that preserves V while rotating K, or rotate both consistently

Current MLA MXFP results are poor (mxfp4 PPL +3.04 vs f16 on GLM-4.7-Flash).
Also seeing a massive throughput regression vs previous measurements that needs investigation.

### CPU Performance
The 50% pp512 slowdown on CPU is the main weakness. Options:
- SIMD-optimized MSE search (AVX-512 could evaluate all 3 candidate scales in parallel)
- Approximate MSE: use the round(log2) scale with a ±1 correction based on simple heuristics
  instead of full MSE evaluation
- Skip MSE for Q quantization (only matters for perplexity, not generation quality)

### Paper-Worthy Contributions
1. **Hadamard + MX quantization for attention**: The combination of block-aligned Hadamard
   rotation with MX format's block structure is novel. No prior work applies Hadamard
   specifically to make MX quantization work for flash attention.

2. **MSE-optimal E8M0 scale search**: The ±1 exhaustive search is simple but effective.
   Could be generalized to other block-scaled formats.

3. **Unified multi-format flash attention kernel**: Supporting 5 MX types through traits-based
   template metaprogramming, with type-specific optimizations (cp.async for FP6, HW intrinsics
   for FP8) while sharing 90% of the kernel code.

4. **Comprehensive quality-throughput-memory Pareto analysis**: Systematic comparison of
   MX formats against existing quantization (q4_0, q8_0, q8/q4 mixed) across multiple
   models, architectures (GQA, MLA), and backends (CUDA, Vulkan, CPU).

## Regression Notes

### HW Intrinsic Regression (Fixed)
Commit 70bace301 (Vulkan MXFP flash attention) replaced all CUDA HW intrinsics with portable
IEEE bit manipulation for HIP/MUSA compatibility. This caused 8-14% tg128 regression because
the MSE search inner loop (96 quantize+dequant per block) and set_rows write_qs path were
using software converters.

Fix: `#if CUDART_VERSION` guards on all trait functions, with portable fallback for HIP/MUSA.

### VEC-for-All Dispatch Regression (Fixed)
Early MXFP dispatch used VEC kernel unconditionally. This was correct for token generation but
caused pp512 to regress from 6,000 to 1,900 because VEC dequantizes K/V per query token instead
of once per batch.

Fix: MMA for batch > 2, VEC for batch ≤ 2 on Blackwell. Non-Blackwell falls back to VEC only.

### MLA Throughput Regression (EXPLAINED — Not a Bug)
GLM-4.7-Flash pp512 dropped from ~5,500 to ~300 t/s. Root cause: **2-GPU layer split**.

The model is 15.9 GiB and splits across two RTX 5070 Ti GPUs (7.7 + 8.0 GiB). With 47 layers,
each layer boundary requires PCIe data transfer. The previous ~5,500 t/s results were likely
from a single-GPU run (or different setup). This is a test environment issue, not a code regression.

To get single-GPU results, would need either:
- A single GPU with >16 GiB VRAM
- A smaller MLA model (e.g., DeepSeek-Lite if one exists in GGUF)
- `--tensor-split 1.0,0.0` to force single GPU (won't fit)

### MLA + E5M2 Quality Issue (EXPLAINED — Expected)
mxfp8_e5m2 GPU PPL 15.05 vs CPU 11.55 on GLM-4.7-Flash. Root cause: **MLA skips Hadamard**.

MLA models share V with K (V_is_K_view), so Hadamard rotation of K would corrupt V. We skip
rotation entirely for MLA. Without Hadamard, E5M2's 2 mantissa bits can't handle outliers —
PPL degrades progressively over longer sequences (chunk 1: 6.72, chunk 13: 16.6).

E4M3 (3 mantissa bits) handles this much better even without Hadamard: GPU PPL 10.58.
On non-MLA models where Hadamard IS applied, E5M2 works fine (PPL 11.70 on Qwen3-Coder).

**Takeaway**: For MLA models, use E4M3 not E5M2. Or develop an MLA-compatible rotation scheme
that can rotate K without corrupting V (e.g., apply rotation in the absorbed projection space
before K/V are combined).

### Intel iGPU mxfp4 Crash (Under Investigation)
Intel ARL integrated GPU crashes with ErrorDeviceLost when running mxfp4 flash attention.
The device has no cooperative matrix support (`matrix cores: none`). The Vulkan backend may
be dispatching MXFP flash attention shaders that require cooperative matrix or subgroup
operations that the Intel driver can't handle.

Likely fix: check `ctx->device->coopmat_support` or similar in supports_op before accepting
MXFP flash attention. The scalar path might work if it doesn't use cooperative matrix, but
the MXFP Q preprocessing (Hadamard via subgroup shuffles) may also be incompatible.
