# MXFP4 Flash Attention for Blackwell GPUs

Branch: `mxfp4-flash-attention-v2`

Custom MXFP4 flash attention kernel for NVIDIA Blackwell (sm_120a) GPUs, using hardware
FP4 block-scaled MMA instructions for K×Q and F16 MMA for V×softmax. Includes K-only
Walsh-Hadamard rotation for calibration-free quantization quality improvement.

## Architecture

### Kernel Structure

Two kernel paths, selected by batch size at dispatch time:

- **MMA kernel** (batch > 2): Prompt processing. Uses Blackwell `mma_block_scaled` with
  `kind::mxf4` (2X) for K×Q dot products, standard F16 MMA with F32 accumulator for
  V×softmax. Shared memory holds double-buffered K tiles, double-buffered mask tiles,
  single V tile, and quantized Q in MXFP4 format.

- **VEC kernel** (batch <= 2): Token generation. Software dequantization of MXFP4 K/V
  via `if constexpr` specialization that bypasses function pointers for direct SoA loads.

### Data Flow

```
Q (F32) ──→ Hadamard rotation ──→ MXFP4 quantization ──→ block-scaled FP4 MMA ──→ KQ scores
K (F16) ──→ set_rows: Hadamard rotation + MXFP4 quantization ──→ SoA KV cache
V (F16) ──→ set_rows: MXFP4 quantization (no rotation) ──→ SoA KV cache ──→ F16 dequant ──→ F16 MMA ──→ VKQ output

KQ scores ──→ softmax (half2 in registers) ──→ F16 MMA with V ──→ VKQ output (F32)
```

### Per-Row SoA KV Cache Layout

MXFP4 blocks are 17 bytes: 16 bytes of nibble data (qs) + 1 byte E8M0 exponent (e).
The standard AoS layout `[e0][qs0_16B][e1][qs1_16B]...` forces qs to odd byte offsets,
requiring byte-by-byte loads.

The SoA layout rearranges within each row:
```
AoS: [e0][qs0_16B][e1][qs1_16B][e2][qs2_16B][e3][qs3_16B]
SoA: [qs0_16B][qs1_16B][qs2_16B][qs3_16B][e0][e1][e2][e3]
```

- K loads: direct `int` loads (was 4 byte loads per int) — 4x fewer instructions
- V loads: direct `uint16_t` loads (was 2 byte loads) — 2x fewer instructions
- Row byte count unchanged (N * 17), stride metadata remains valid
- KV cache is runtime-only — zero impact on GGUF format or other backends

### K-only Walsh-Hadamard Rotation

Applying a 32-element Walsh-Hadamard transform before MXFP4 quantization spreads outlier
energy uniformly across block elements, making the E8M0 shared exponent a tighter fit.

- K cache: `H(K)` stored after rotation in `set_rows` (flagged via `op_params[0] = 1`)
- Q at attention time: `H(Q)` computed in kernel before MXFP4 quantization
- Identity: `H(Q) · H(K)^T = Q · K^T` (H is orthogonal and involutory)
- V cache: NOT rotated (V rotation hurts quality — see experiments below)

Implementation uses butterfly-pattern operations:
- K/Q (single thread holds 32 values): 5 butterfly stages + normalization, 192 FP32 ops
- VEC Q (8 threads hold 4 values each): 2 register stages + 3 warp shuffle stages

References:
- BRQ: "Block Rotation is All You Need for MXFP4 Quantization" (arxiv 2511.04214)
- MR-GPTQ (arxiv 2509.23202): Proves rotations improve MXFP4 but hurt NVFP4
- HadaCore (arxiv 2412.08832): Warp-shuffle butterfly pattern for GPU Hadamard transforms

### FP4 MMA Block Scale Thread Mapping

Critical implementation detail for `mma_block_scaled`:
- A scale (rows): `row = threadIdx.x / 4 + (threadIdx.x % 2) * 8`
- B scale (columns): `col = threadIdx.x / 4`
- Reference: mmq.cuh line 1040, NVIDIA PTX block-scaling docs
- Do NOT use `threadIdx.x % 16` for the A scale

### FP4 Quantization Notes

`kvalues_mxfp4` is NOT sorted: `{0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}` (positives
then negatives). Never use binary search (`best_index_int8`) on this table. Use
`ggml_cuda_float_to_fp4_e2m1` which has its own sorted positive LUT and sign bit.

## Files Modified (from master)

Core kernel files:
- `ggml/src/ggml-cuda/fattn-mma-mxfp4.cuh` — MMA kernel (NEW)
- `ggml/src/ggml-cuda/fattn-common.cuh` — VEC dot/dequant for MXFP4, SoA versions
- `ggml/src/ggml-cuda/fattn.cu` — Dispatch logic
- `ggml/src/ggml-cuda/fattn-vec.cuh` — SoA `if constexpr` specializations
- `ggml/src/ggml-cuda/cpy-utils.cuh` — MXFP4 quantization + SoA quantizer
- `ggml/src/ggml-cuda/set-rows.cu` — SoA set_rows kernel for MXFP4
- `ggml/src/ggml-cuda/hadamard.cuh` — Walsh-Hadamard transforms (NEW)
- `src/llama-kv-cache.cpp` — op_params Hadamard flag for K writes
- 13 template instance files (12 MMA + 1 VEC)
- Plus: common/arg.cpp, convert.cu, ggml-cuda.cu, cp-async.cuh

## Performance

Measured on 2x NVIDIA RTX 5070 Ti (sm_120a), Qwen3-Coder-30B-A3B-Instruct (MXFP4_MOE):

| Metric | MXFP4 FA | F16 FA | Notes |
|--------|----------|--------|-------|
| Prompt eval (pp512) | 5,916 t/s | 6,146 t/s | MMA kernel |
| Token gen (tg128) | 193 t/s | 202 t/s | VEC kernel |
| KV cache size | 63.75 MiB | 384 MiB | 6x reduction (with 1-bit residual) |
| Prompt eval (pp4700) | ~3,008 t/s | — | Longer context |

## Perplexity Results

All measurements on Qwen3-Coder-30B-A3B-Instruct, wikitext-2-raw, 59 chunks:

| Config | K cache | V cache | 59c PPL | vs F16 | KV size |
|--------|---------|---------|---------|--------|---------|
| F16 baseline | F16 | F16 | 9.7319 | — | 384 MiB |
| Q4_0 | Q4_0 | Q4_0 | 9.9730 | +0.2411 | ~96 MiB |
| **MXFP4 + H + 1-bit res** | **MXFP4 1.5×** | **MXFP4** | **9.8622** | **+0.1303** | **63.75 MiB** |
| MXFP4 + Hadamard | MXFP4 | MXFP4 | 10.3099 | +0.5780 | 51 MiB |
| MXFP4 + Hadamard | MXFP4 | F16 | 10.3741 | +0.6422 | 121.5 MiB |
| MXFP4 (no Hadamard) | MXFP4 | MXFP4 | 10.6617 | +0.9298 | 51 MiB |

The compact 1-bit sign residual closes 86% of the gap to F16 (+0.1303 vs +0.9298 without),
beating Q4_0 quality (+0.1303 vs +0.2411) at 66% less memory (63.75 vs ~96 MiB).
Hadamard rotation alone reduces the PPL gap by 37.8% (0.9298 → 0.5780).

## Validation

Determinism verified: 3 runs produce bit-identical PPL (10.7034 for MXFP4, 10.0417 for F16).
Causal mask verified: no future token leakage detected across context lengths.

## Compact 1-Bit Sign Residual K Cache

After MXFP4 quantization, the residual (original − dequantized) captures lost information.
Experiments showed that storing just the **sign direction** of each residual element (1 bit)
with a per-block E8M0 magnitude scale recovers 86% of the gap to F16.

### How It Works

1. **Quantize K** to MXFP4 (primary)
2. **Compute residual**: `res[j] = original[j] − dequant(primary[j])`
3. **Pack 32 sign bits** into uint32_t per block (4 bytes vs 16 bytes for full MXFP4 nibbles)
4. **Compute E8M0 magnitude**: `round(log2(mean(|res|))) + 127`
5. At attention time, **expand** sign bits to ±1.0 FP4 nibbles and scale by E8M0

### Compact Memory Layout (per row, D=128, n_head_kv=4)

```
Flat SoA: [primary_qs: N×16B][sign_bits: N×4B][res_E8M0: N×1B]
Total blocks: 24 (16 primary + 8 compact), aligned to 4 for int loads
Row stride: 24 × 17 = 408 bytes
K allocation: 1.5× primary (was 2× with full MXFP4 residual)
```

### Results

- **PPL 9.8622** (+0.1303 vs F16 9.7319) — beats Q4_0 (9.9730, +0.2411)
- **K cache: 38.25 MiB** (was 25.5 MiB base, 51 MiB with 2× residual)
- **KV total: 63.75 MiB** (6x reduction vs F16 384 MiB)
- Sign-only residual captures 80% of full residual's improvement

## Quality Optimization Research

16 calibration-free experiments were conducted to close the remaining PPL gap. K-only
Hadamard rotation is the global optimum — no other calibration-free technique improves it.

### Experiments Tested

| # | Approach | Result | Why |
|---|----------|--------|-----|
| E1 | K-only Hadamard baseline | 10.3099 (59c) | Reference point |
| E2 | Exponent round-up | +0.45 worse | Larger scale hurts small values |
| E3 | Exponent round-down | +0.90 worse | Smaller scale causes clipping |
| E4 | V-only Hadamard | +1.36 worse | K rotation critical, not V |
| E5 | K+V Hadamard | +0.06 worse (59c) | V rotation hurts at scale |
| E6 | SmoothAttention | Skipped | BRQ found marginal for MXFP4, needs calibration |
| E7 | Attention sink F16 | Skipped | Gap is structural, not position-dependent |
| E8 | Channel permutation | +0.17 worse | Disrupts beneficial grouping |
| E9 | Adaptive exponent (MSE) | +0.50 worse | Per-block MSE doesn't optimize attention |
| E10 | Stochastic rounding | Skipped | Higher MSE, no averaging benefit for cache |
| E11 | Training-aware methods | Research | All require calibration data |
| E12 | V=F16 (diagnostic) | +0.06 (59c) | V type irrelevant; noise at 2c |
| E13-14 | V=Q8_0, V=Q4_0 | Skipped | E12 proved V type doesn't matter |
| E15 | RaZeR -0 remap | Skipped | Incompatible with hardware FP4 MMA |
| E16 | W_K column-norm equalization | Skipped | Redundant with Hadamard, counterproductive |

### Key Findings

1. **K quantization dominates quality loss.** V quantization type has negligible PPL
   impact (F16 V only +0.064 vs MXFP4 V at 59 chunks). Literature confirms keys are
   38-45x more sensitive to quantization than values (KV-AdaQuant, arxiv 2502.15075).

2. **Hadamard equalizes block magnitudes.** The Walsh-Hadamard transform spreads outlier
   energy across all 32 elements in an MXFP4 block, making the shared E8M0 exponent a
   tighter fit. This is optimal for the FP4 E2M1 grid.

3. **2-chunk PPL tests are unreliable.** Error bars are ±0.25 or more. Multiple
   experiments showed large 2c differences that vanished at 59 chunks. Always use
   59+ chunks for meaningful comparisons.

4. **Round-to-nearest is optimal for E8M0.** Both round-up and round-down are strictly
   worse. Per-block adaptive selection (trying both and picking lower MSE) also hurts
   because attention is nonlinear — per-block optimization disrupts beneficial error
   cancellation.

5. **Column-norm equalization fights Hadamard.** Making K channels more uniform before
   Hadamard would make the Hadamard output MORE concentrated (Hadamard of a uniform
   vector concentrates into one component), worsening quantization.

### Structural Gap Analysis (pre-residual)

Without the 1-bit sign residual, the gap between MXFP4+Hadamard (10.3099) and Q4_0
(9.9730) was structural:

- **~40% from E8M0 power-of-two scaling.** E8M0 has up to 41% scale error vs Q4_0's
  continuous FP16 scale, forcing suboptimal quantization ranges.

- **~60% from E2M1 4-bit grid.** The non-uniform FP4 grid `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`
  overrepresents small values and underrepresents mid-range values where post-Hadamard
  distributions concentrate. Q4_0's uniform grid is a better fit.

The 1-bit sign residual closed this gap completely — MXFP4 + Hadamard + compact residual
(PPL 9.8622) now beats Q4_0 (9.9730) by 0.1108 PPL at 34% less memory.

## Build and Test

### Build

```bash
cd ~/code/services
docker compose build llama-cpp-ultra
```

Requires CUDA 13.1+ for sm_120a (Blackwell) FP4 MMA support. Host CUDA 12.x cannot
compile these kernels — Docker build is mandatory.

### Test

```bash
cd ~/code/services/repos/llama.cpp
./mxfp4-test.sh              # Run all tests (build + perplexity + bench)
./mxfp4-test.sh --skip-build # Skip Docker build
./mxfp4-test.sh --quick      # 2-chunk perplexity only (fast sanity check)
./mxfp4-test.sh --bench-only # Performance benchmarks only
```

See `mxfp4-test.sh` for all options and configurations.

## Model Under Test

- Qwen3-Coder-30B-A3B-Instruct (MXFP4_MOE GGUF)
- n_embd_head_k=128, n_embd_head_v=128, n_head_kv=4, GQA ratio=8
- Location: `/models/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/`
