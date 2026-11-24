# 44 — Are the quantization paths intrinsic-backed? (audit, 2026-11-23)

Question posed: *"we may be using scalar quantization of the activations to mxfp8 when intrinsics are
available"* — and more generally, is every conversion/quantization using the hardware path, given the
weights and these cards were designed for each other.

**Short answer: yes on every path that matters. Nothing to fix, and two of the suspicions are now
settled with code citations instead of hope.**

## 1. NVFP4 layers (20 of 24) — fp4 activations on the ×4 MMA: CONFIRMED

`ggml/src/ggml-cuda/mma.cuh:1304`:

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

Both operands are `e2m1`, K=64, `scale_vec::4X`. The gating is real, not aspirational:

- `mmq.cu:137` — `can_use_q4 = (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc)`
- `mmq.cu:86-91` — NVFP4 dispatches `mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>` when allowed, else `GGML_PREC_Q8`
- `mmq.cu:255` — `y_is_fp4 = use_native_fp4 && src0->type != GGML_TYPE_MXFP8`
- `mmq.cu:287,322,362` — the NVFP4 `use_native_fp4` branch passes `src1_scale` through, i.e. the
  activation side carries its own `ue4m3` block scales.

So the checkpoint's `required_activation_quantization: nvfp4` for those layers is what the kernel
actually does. There is **no** `to_nvfp4`-named function (grep: 0 hits) — the activation quantization is
selected through the `prec_src1` template parameter and the shared `quantize_*_mmq` machinery, which is
why name-based greps mislead here.

## 2. MXFP8 layers (4 of 24) — fp8 activations on `mxf8f6f4`: CONFIRMED, and at the hardware ceiling

`mma.cuh:1290`:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

`e4m3.e4m3` = **both operands are fp8** (activations are mxfp8, not f16, not dequantized). PTX defines no
K=64 block-scaled form for 8-bit operands, so `1X`/K=32 is the widest available — the ×4 throughput of
the FP4 path is not reachable for MXFP8 by software choice. `mmq-config-blackwell.cuh:36` records the
geometry consequence: same tile shape as the FP4 types, but the VRAM tile is `MMQ_ITER_K` rather than
`MMQ_ITER_K_FP4` because it is 1 byte per value.

Measured contribution: **+4.6% prefill** — and the ceiling is low because MXFP8 is only ~3.9 GiB of the
~14.6 GiB of expert weights (~27%), with the other 20 layers already on the native FP4 MMA.

## 3. Decode (single token) — MXFP8 vector path uses the hardware fp8 convert: CONFIRMED

`mmvf.cu:350-373` (my Phase 3 path) reads pairs and calls `dequantize_mxfp8_pair`, and
`dequantize.cuh:124-133`:

```c
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) \
    && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_fp8x2_e4m3 x2;
    x2.__x = (uint16_t) q[0] | ((uint16_t) q[1] << 8);
    const float2 v = ggml_cuda_cast<float2>(x2);      // -> __nv_cvt_fp8x2_to_halfraw2
```

`__CUDA_ARCH__` is 1200 on these cards, so the hardware branch is the one that compiles (`convert.cuh:53-65`
also has the `fp8x2 -> bf162 / float2` variants). The dot product itself is `float2` + `ggml_cuda_mad`.

**One inherent limit, worth understanding rather than "fixing":** a payload pair is assembled from two
1-byte loads, because an MXFP8 block is 33 bytes (1 scale + 32 values), so `qs` is odd-aligned on every
other block — a natural 16-bit load would be misaligned and CUDA requires natural alignment. A warp still
touches 64 contiguous payload bytes, so it coalesces. The same fact is why
`mmvf.cu:512` asserts `stride_row % 2 == 0 || std::is_same_v<T, block_mxfp8>`.

## 4. What was checked and found NOT worth optimizing

| candidate | finding |
|---|---|
| The MXFP8 **activation quantizer** (`quantize.cu`, scalar-per-thread vs NVFP4's `float8` loads + two-level reduce) | It does not appear in the kernel profile at all (see `37-prefill-kernel-profile.md`). Restructuring it would be work for a measurement of zero. |
| `.scale` sidecar application for NVFP4 | Now handled by upstream's generic pass (`19bc41dbb`); the scale is folded into the MMA epilogue (`mmq.cuh:479-506` loads weight scales once per tile, `y_scale_used` for the fused-MUL case) — no extra elementwise pass over the output. |
| Relative-bias gather dtype | Moot: the tensor no longer exists. `#32808` (adopted in `84d9bc0dc`) makes the FA kernel gather `table[indices[k,q], h, q]` itself, in f32, with no `n_kv`-wide intermediate at all. My earlier f16-gather work (`e1f2296fa`) was superseded by deleting the gather. |

## 5. Still open, and where the real time is going

The intrinsics question is settled; the remaining perf losses are **memory movement**, not arithmetic:

1. **Drafter host-copy tap delivery** (`#3` here): `speculative.cpp:1150` requires `n_embd_state == 0`
   for `use_dev_features`, and Inkling's tap channel sets `n_embd_state > 0` — so we are on host copies,
   moving 4 taps x `n_ubatch` x 2304 f32 (**~75 MB per prefill ubatch**) synchronously to RAM, measured
   at about **-15% prefill**. Proposed fix: give the state-row device branch (`llama-batch.h:71`,
   `include/llama.h:1220`) a source other than `h_nextn` — the target's layer-input device buffer — and
   delete the precondition. Design detail and the reason `h_nextn` is not reusable (DSpark's confidence
   output already occupies `t_h_nextn`) in `36-drafter-step3a-readout.md`.
2. **Scalar copy / f32 gather kernels** (`#4`): last profile had `cpy_scalar<float,float>` at 9.5% over
   2,960 calls and `k_get_rows_float_vec<float>` at 11.1% over 400 calls. Both predate the indexed-bias
   merge, so they need re-attribution before anyone optimizes them — see `#2`.
3. **Launch-bound MoE decode** (`#6`): ~384 tiny expert GEMVs per token, using 29-40 GB/s of ~900 GB/s
   available. This is why the `mmvq` route was neutral (`20-…`): the cost is launches and occupancy, not
   ALU form. The lever is fewer/bigger kernels, not faster ones.
