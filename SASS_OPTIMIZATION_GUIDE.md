# SASS-Driven CUDA Kernel Optimization Guide

A practical guide for using SASS (Shader ASSembly) analysis to iteratively optimize CUDA kernels in llama.cpp. Developed during MXFP flash attention VEC kernel optimization; applicable to any CUDA kernel in the project.

## Overview

SASS is the native GPU instruction set — the actual machine code the GPU executes. Examining SASS reveals exactly what the compiler produced, exposing:

- Unnecessary instructions the compiler couldn't optimize away
- Suboptimal memory access patterns (scalar loads where vector loads are possible)
- Missed opportunities for hardware intrinsics
- Register pressure and spills
- Whether your source-level "optimization" actually changed anything

The core workflow is: **edit code, build, dump SASS, compare, benchmark, repeat**.

## Prerequisites

- Docker build environment (CUDA 13.1 devel image for `cuobjdump`)
- The docker build for llama.cpp: `cd ~/code/services && docker compose build llama-cpp-ultra`
- The three scripts in the repo root: `sass-diff.sh`, `vec-iterate.sh`, `kv-bench.sh`

## The Iteration Cycle

### 1. Set a baseline

Before making any changes, capture the current state:

```bash
# Full iteration: build, bench all configs, extract SASS, save as baseline
./vec-iterate.sh --set-baseline
```

This does three things:
- Builds the docker image
- Benchmarks pp512 + tg128 for mxfp6, mxfp4, mxfp8, and f16
- Extracts SASS for the target kernel family and saves it as the baseline snapshot

### 2. Make a code change

Edit the kernel source. For the VEC kernel, the key files are:

| File | What it contains |
|------|-----------------|
| `ggml/src/ggml-cuda/fattn-vec.cuh` | VEC kernel main loop, Q quantization, thread config |
| `ggml/src/ggml-cuda/fattn-common.cuh` | K dot product, V dequantization, dispatch wrappers |
| `ggml/src/ggml-cuda/fattn-mma-mxfp.cuh` | MMA kernel (prompt processing path) |
| `ggml/src/ggml-cuda/mxfp-traits.cuh` | KV cache quantization (set_rows write path) |
| `ggml/src/ggml-cuda/set-rows.cu` | KV cache write kernel dispatch |

### 3. Build, bench, and compare

```bash
# Full iteration: build, bench, SASS, compare to baseline
./vec-iterate.sh

# Faster: skip SASS (just bench)
./vec-iterate.sh --bench-only

# Faster: skip bench (just SASS)
./vec-iterate.sh --sass-only

# Single config only
./vec-iterate.sh --config mxfp4
```

### 4. Analyze the results

Look at three things:
1. **Benchmark numbers**: Did pp512/tg128 change?
2. **Instruction count delta**: More or fewer instructions?
3. **Instruction category changes**: Which instruction types changed?

### 5. Track progress

```bash
# Show history of all iterations
./vec-iterate.sh --history
```

## Using `sass-diff.sh` Directly

For more control over SASS analysis, use `sass-diff.sh`:

```bash
# Save baseline/current snapshots independently
./sass-diff.sh --baseline
./sass-diff.sh --current

# Compare baseline vs current
./sass-diff.sh --compare

# Detailed instruction stats for a snapshot
./sass-diff.sh --stats baseline
./sass-diff.sh --stats current

# Target different kernel families
./sass-diff.sh --kernel "flash_attn_ext_mma" --baseline
./sass-diff.sh --kernel "k_set_rows" --baseline
./sass-diff.sh --kernel "mul_mat_vec" --baseline

# List all kernel function names in the .so
./sass-diff.sh --list-kernels
```

### Default kernel filter

By default, `sass-diff.sh` filters to `flash_attn_ext_vecILi128ELi1E` — the D=128, ncols=1 VEC kernels, which are the hot path for token generation (tg128). Use `--kernel` to target other kernels.

### Useful kernel filter patterns

| Pattern | What it matches |
|---------|----------------|
| `flash_attn_ext_vecILi128ELi1E` | VEC kernels, D=128, ncols=1 (tg128 hot path) |
| `flash_attn_ext_vec` | All VEC kernel variants |
| `flash_attn_ext_mma` | All MMA kernel variants (prompt processing) |
| `k_set_rows_mxfp_soa` | MXFP KV cache write kernels |
| `quantize_f32_mxfp` | MXFP quantization kernels |
| `mul_mat_vec` | Matrix-vector multiply kernels |
| `mul_mat_q` | Matrix-matrix quantized multiply kernels |

## Reading SASS Output

### Instruction format

Each SASS instruction looks like:

```
        /*0070*/                   LDG.E.128.SYS R16, [R2.64+UR14] ;
```

- `/*0070*/` — byte offset within the kernel (hex)
- `LDG` — instruction opcode (Global Load)
- `.E.128.SYS` — modifiers (Extended addressing, 128-bit load, System coherence)
- `R16` — destination register
- `[R2.64+UR14]` — source address (64-bit register + uniform register offset)

### Key instruction categories

| Instruction | Meaning | What to watch for |
|-------------|---------|-------------------|
| **LDG** | Global memory load | Count and width (.32 vs .64 vs .128). Fewer, wider loads = better |
| **STG** | Global memory store | Should be minimal in read-heavy kernels |
| **LDS** | Shared memory load | Fast but limited bandwidth |
| **STS** | Shared memory store | Often paired with barriers |
| **FFMA** | FP32 fused multiply-add | Core compute — more = more arithmetic work |
| **HFMA2** | FP16 fused multiply-add | Half-precision compute (2 ops per instruction) |
| **HMMA** | Tensor core MMA | Matrix multiply-accumulate (MMA kernel only) |
| **IMAD** | Integer multiply-add | Includes dp4a (int8 dot product) |
| **PRMT** | Byte permute | Used in LUT-based dequantization. High count = LUT overhead |
| **LOP3** | 3-input logic op | Bit manipulation (nibble extract, masking) |
| **SHF** | Funnel shift | Bit shifting (part of dequant paths) |
| **MUFU** | Special function unit | Transcendentals: exp2, log2, rcp, rsqrt. Expensive |
| **I2F** | Integer to float | Type conversion overhead |
| **F2FP** | Float to float packed | Format conversion (e.g., FP32 to FP16) |
| **MOV** | Register move | Register pressure indicator — excessive MOVs = spilling |
| **SHFL** | Warp shuffle | Cross-lane communication (reductions, Hadamard) |
| **WARPSYNC** | Warp synchronization | Barrier cost |
| **S2R** | Special register read | Reading threadIdx, blockIdx, etc. |

### Decoding mangled kernel names

The C++ mangled names encode template arguments. The pattern for flash_attn_ext_vec is:

```
_Z18flash_attn_ext_vecILi128ELi1EL9ggml_type39ELS0_39ELb0EE...
                       ^^^^^  ^^^              ^^      ^^  ^^
                       D=128  ncols=1     type_K=39  type_V  has_residual
```

ggml_type enum values:

| Value | Type |
|-------|------|
| 1 | GGML_TYPE_F16 |
| 2 | GGML_TYPE_F32 |
| 8 | GGML_TYPE_Q8_0 |
| 39 | GGML_TYPE_MXFP4 (E2M1) |
| 40 | GGML_TYPE_MXFP8 (E4M3) |
| 41 | GGML_TYPE_MXFP6_E2M3 |
| 42 | GGML_TYPE_MXFP6_E3M2 |
| 43 | GGML_TYPE_MXFP8_E5M2 |

So `ggml_type39ELS0_39` = K=MXFP4, V=MXFP4. And `ggml_type43ELS0_39` = K=MXFP8_E5M2, V=MXFP4.

## Benchmarking with `kv-bench.sh`

```bash
# Full suite: all configs, 16 chunks perplexity
./kv-bench.sh

# Quick throughput check (no perplexity)
./kv-bench.sh --skip-perplexity --config mxfp4

# Quick perplexity (2 chunks = fast, 16 = accurate)
./kv-bench.sh --skip-bench --chunks 2

# Different model
./kv-bench.sh --model glm-4.7-flash --config f16

# Skip docker build (use existing image)
./kv-bench.sh --skip-build --skip-perplexity --config mxfp4
```

**Important**: Always `docker compose stop llama-cpp-ultra` before benchmarking to free GPU memory. The scripts handle this automatically when they run the build step, but with `--skip-build` you must do it manually.

## Optimization Methodology

### Step 1: Profile before optimizing

Before diving into SASS, understand where time is spent:

- **tg128** (token generation): Dominated by MLP weight loads (~90%). The VEC kernel is a small fraction. Small instruction count reductions translate to small throughput gains.
- **pp512** (prompt processing): MMA kernel + KV cache writes. More compute-bound, so instruction reductions matter more.

### Step 2: Identify the hot kernel

Use `--kernel` to filter to the kernel you care about. Look at instruction counts to find the most expensive variant:

```bash
./sass-diff.sh --stats current
```

The per-kernel listing shows instruction counts sorted by size. Focus on the largest kernels.

### Step 3: Compare to the simplest baseline

The f16/f16 VEC kernel (type 1/1) is the minimum-instruction-count baseline. Compare your quantized kernel's instruction categories against f16 to understand where extra instructions come from.

Example comparison (VEC kernel, D=128, ncols=1):

| Kernel | Instructions | Key difference |
|--------|-------------|----------------|
| f16/f16 | 1,152 | Simple load + HFMA2 |
| q8_0/q8_0 | 1,984 | +832 for dequant |
| mxfp4/mxfp4 | 3,232 | +2,080 for LUT + dp4a + E8M0 |
| mxfp8/mxfp4 | 3,128 | +1,976 for K fp8 dequant + V fp4 dequant |

### Step 4: Look for red flags in instruction mix

Things that suggest optimization opportunities:

- **High MUFU count**: SFU calls (log2, exp2, rcp). Can often be replaced with integer bit tricks. Example: we replaced `log2f(amax)` with IEEE-754 bit extraction, saving ~2.6% tg128.
- **High PRMT count**: LUT-based dequant overhead. Consider hardware conversion intrinsics.
- **Excessive MOV**: Register spilling. Try reducing live variables or shared memory usage.
- **Scalar LDG where vector possible**: Single-byte loads where 4/8/16-byte loads would work.
- **SHFL without corresponding compute**: Overhead from warp shuffles (Hadamard, reductions).

### Step 5: Make surgical changes and verify

The most important rule: **change one thing at a time**. After each change:

1. Build and check SASS — did the instruction count actually change?
2. If SASS is unchanged, the compiler already handled it. Move on.
3. If SASS changed, benchmark to see if it matters at runtime.
4. A change that reduces instructions but doesn't improve benchmarks is still useful knowledge — it tells you the kernel is not compute-bound at that point.

### Step 6: Know when to stop

Diminishing returns are real. Indicators that you've reached the optimization floor:

- **Constant gap across context lengths**: If the overhead doesn't scale with context, it's per-invocation fixed cost that can only be reduced by reducing instructions.
- **Instruction count close to f16 baseline**: You can't do less work than the data format requires.
- **Benchmark noise exceeds expected gain**: If changes produce <0.5% differences, you're in the noise.
- **Compute vs memory bound**: If the kernel is memory-bound (waiting on global loads), reducing instructions in the compute path won't help.

## Case Study: VEC Kernel MXFP4 Optimization

### Starting point: 177.6 t/s tg128 (f16 baseline: 188.8)

### Optimization 1: nthreads 32 to 16 (+1.4%)
- **Hypothesis**: With 16 threads per K dot product, each warp processes 2 K/V rows in parallel instead of 1, halving loop iterations.
- **SASS change**: Fewer loop iterations, but more work per thread.
- **Result**: 177.6 to ~180 t/s. The warp-level parallelism improvement outweighed per-thread overhead.

### Optimization 2: MSE E8M0 removal (+2.6%)
- **Hypothesis**: The KV write path tested 3 candidate E8M0 scales via full quantize/dequant/MSE loop (~480 FLOPs per block). Direct bit extraction should be much cheaper.
- **SASS change**: Eliminated MUFU (log2f), removed the MSE search loop entirely.
- **Result**: ~180 to 184.0 t/s. The KV write path was the actual bottleneck, not the VEC kernel.

### Investigation: Hadamard isolation (0% tg128 impact)
- **Hypothesis**: Walsh-Hadamard rotation on K (during write) and Q (during VEC) adds overhead.
- **Method**: Temporarily disabled K Hadamard in `llama-kv-cache.cpp` and Q Hadamard in `fattn-vec.cuh`. Built and benchmarked.
- **Result**: tg128 unchanged (184.0). Hadamard costs zero at batch=1 because only 32 blocks are processed. But pp512 improved 5.7% without K Hadamard — useful knowledge for the MMA path.

### Investigation: Context-length scaling
- **Method**: Benchmarked tg128 at 512, 4K, and 32K context.
- **Result**: Gap is constant at ~2.5% regardless of context. The overhead is per-token fixed cost (0.14ms = ~2.9us per layer), not per-KV-row.

### Final state: 184.0 t/s (2.5% gap to f16 = irreducible dequant overhead)

The 2,080 extra SASS instructions for MXFP4 vs f16 are the fundamental cost of interpreting quantized data. This cannot be eliminated — it's the irreducible minimum for the data format.

## Applying This to Other Kernels

### MMA kernel (prompt processing)

```bash
./sass-diff.sh --kernel "flash_attn_ext_mma" --baseline
# ... make changes ...
./sass-diff.sh --kernel "flash_attn_ext_mma" --current
./sass-diff.sh --kernel "flash_attn_ext_mma" --compare
```

Key files: `fattn-mma-mxfp.cuh`, `mma.cuh`

### MMVQ kernel (matrix-vector quantized multiply)

```bash
./sass-diff.sh --kernel "mul_mat_vec_mxfp" --baseline
```

Key files: `mmvq-mxfp-soa.cuh`

### MMQ kernel (matrix-matrix quantized multiply)

```bash
./sass-diff.sh --kernel "mul_mat_q.*mxfp" --baseline
```

Key files: `mmq.cuh`

### Set-rows / quantization kernel

```bash
./sass-diff.sh --kernel "k_set_rows" --baseline
```

Key files: `set-rows.cu`, `mxfp-traits.cuh`

## Tips and Pitfalls

1. **The compiler is smart.** Many source-level "optimizations" (reordering operations, manual strength reduction, explicit register hints) produce identical SASS. Always check SASS before benchmarking — if it didn't change, the benchmark won't change either.

2. **`__ldg` rarely helps on modern GPUs.** The compiler and hardware handle L1/L2 caching well. We found `__ldg` hints produced identical SASS on sm_120 in most cases.

3. **Don't optimize in the noise.** Benchmark variance on this system is ~0.5% for tg128. Changes under 1% need multiple runs to confirm.

4. **Memory-bound vs compute-bound matters.** The VEC kernel at short context is largely memory-bound (waiting for KV row loads from VRAM). Extra compute instructions hide behind memory latency. This is why 2.8x more instructions only costs 2.5% throughput.

5. **KV write path matters for tg128.** The set_rows kernel runs every token to write the new KV entry. At batch=1, this is a tiny kernel (32 threads, 32 blocks) but every instruction counts because there's no memory latency to hide behind.

6. **Build inside Docker, never locally.** The host CUDA version (12.4) cannot target sm_120a. Always build via `docker compose build llama-cpp-ultra`. The SASS extraction also runs cuobjdump inside a Docker container.

7. **Stop the service before benchmarking.** `docker compose stop llama-cpp-ultra` frees GPU memory. Running benchmarks while the service is loaded will give wrong numbers.

8. **Temporary experiment changes must be reverted.** When testing hypotheses (like disabling Hadamard), make the change, build, bench, then immediately revert. Never commit experimental correctness-breaking changes.
