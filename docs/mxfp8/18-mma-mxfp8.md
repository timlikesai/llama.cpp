# Phase 4b (CUDA Tier C) - native `mma.kind::mxf8f6f4` W8A8 for `GGML_TYPE_MXFP8`

Brief: `docs/mxfp8/briefs/R-mma-mxfp8.md`. Task: extend master's existing MMQ block-scaled
MMA infrastructure to MXFP8 and measure prefill/decode.

## Status: DONE — committed as c39ab0a00 plus tolerance follow-up 6c1424737. The lane timed out with the code written and building; the parent ran the gates and landed it.

## STEP 0 answer: YES - master's FP4/Q8_1 tile geometry holds MXFP8 unchanged

No new SRAM layout is needed. The arithmetic, all from the current tree:

| quantity | FP4 today (`mmq.cuh`) | MXFP8 |
|---|---|---|
| mma shape | `kind::mxf4(.mxf4nvf4).block_scale.m16n8k64` (`mma.cuh:1288,1295`) | `kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32` |
| `tile_A` | `tile<16, 8, int>` = 16 rows x 8 int32 = 512 B | **same** (512 B = 16 x 32 e4m3) |
| `tile_B` | `tile<8, 8, int>`, 2 regs/lane | **same** (256 B = 8 x 32 e4m3) |
| asm operands | 4 x A, 2 x B, 4 x D, a_scale, b_scale | **byte-for-byte the same operand skeleton** (`ref/mma-kind-mxf8f6f4-all3.inc:33-45`) |
| x tile row | 64 payload int32 = 256 B = 512 e2m1 (`MMQ_ITER_K_FP4`) | 64 payload int32 = 256 B = **256 e4m3** = `MMQ_ITER_K` |
| scale area/row | 8 int32 = 16 blocks x 1 e8m0 byte, 2 packed per word (`block_fp4_mmq` comment `mmq.cuh:53`) | 8 int32 = 8 blocks; **1 word per block, scale in byte 0** - the same 8 words, nothing packed |
| y block | `block_fp4_mmq` = 4 x `uint32` + 128 B payload = 144 B (`mmq.cuh:55-58`) | `block_fp8_mmq` = 4 x `uint32` + 128 B payload = 144 B - **identical size and stride** |
| `sram_stride` | `LAYOUT_FP4` = 2*32 + 8 + 4 = 76 and asserted == `LAYOUT_Q8_1` (`mmq.cuh:164`) | 76, unchanged |
| `MMQ_TILE_Y_K` | 36 int32 = 4 scale words + 32 payload | 36 int32 = 4 scale words + 32 payload |

So MXFP8 needs **no new loader arithmetic, no new layout, no new stride** - the FP4 native path's
pointer math (`x_sc = x_qs + 2*MMQ_TILE_NE_K`, `y_sc = y`, `y_qs = y + 4`,
`scaleA[n][frag] = x_sc[row*sram_stride + k0/tile_A::J]`, `mmq-vec-dot.cuh:1186-1215`) already
indexes exactly one scale word per fragment once the frag covers 32 8-bit values instead of 64
nibbles. `tidx_A = g + (t%2)*8` (`mmq-vec-dot.cuh:1198`) is the PTX 16-row scale-lane map and is
the same for `1X` and `2X` (`ref/mmq-vec-dot-mxfp-mma.inc:31`).

Two things that are NOT free:
1. The e8m0 scale must be read with `ggml_cuda_e8m0_to_fp32(e)` - **no `*0.5f`** (the half in
   `vec_dot_mxfp4_q8_1` / `mmq-load-tiles.cuh:1677` is because `kvalues_mxfp4` are doubled).
   In the MMA route the scale byte goes to the tensor core raw, so this only matters for the
   non-MMA/dequant routes and for the precision-floor harness (gate 2).
2. Activations must be quantized to e4m3 + e8m0. There is no such quantizer on master; the FP4
   one (`quantize.cu:337 quantize_mmq_mxfp4`) is the template. `quantize.cu/.cuh` is **outside my
   brief fence** - see Corrections.

## Port surface — decided per site (as implemented)

Every MXFP4/NVFP4 touchpoint, widened or not, per the diff:

| site | change |
|---|---|
| `common.cuh` | new `ggml_cuda_type_traits<GGML_TYPE_MXFP8>` (qk/qr/qi/bs) |
| `mmq.cuh` | MXFP8 in the layout/type switches; **no new `sram_layout`, no new stride** — reuses `LAYOUT_FP4` |
| `mma.cuh` | `mma.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32` with `e4m3.e4m3` operands, alongside the existing fp4 wrappers |
| `mmq-config-blackwell.cuh` | MXFP8 config entries mirroring MXFP4 geometry |
| `mmq-load-tiles.cuh` | tile loader for the e4m3 payload (32 values/byte each, so no nibble handling at all) |
| `mmq-vec-dot.cuh` | vec_dot for the block-scaled MMA route |
| `mmq.cu` | dispatch case, `prec_src1` pinned to the native format, and `mmq_supported = blackwell_mma_available(cc)` |
| `quantize.cu` / `.cuh` | new e4m3 + ue8m0 activation quantizer (see Corrections #1 — outside the fence, and necessary) |
| `generate_cu_files.py` | MXFP8 added to the W4A4 list plus a new `TYPES_MMQ_NATIVE_SRC1_ONLY` set, so the instance declares only the block-scaled case; generated `mmq-instance-mxfp8.cu` (compiled into the build without a reconfigure) |

## Gates — final results (parent, after the lane's 45-minute budget expired)

| Gate | Result |
|---|---|
| `test-backend-ops -b CUDA0 -p mxfp8` | **237/237** |
| `test-backend-ops -b CUDA1 -p mxfp8` | **237/237** |
| `test-backend-ops -b CPU -p mxfp8` | **306/306**, unregressed |
| `test-quantize-fns` | exit 0 |
| `MUL_MAT_ID` / fusion at MoE shapes | inside the 237 — the before-fail/after-pass set was exactly the `type_b=f32` MUL_MAT and MUL_MAT_ID cases |
| precision floor | observed max NMSE **7.4e-4**: ~10x inside master's own `F8_E4M3` + `NATIVE_FP8` allowance (5e-3), 27x inside the fp4-activation allowance (2e-2) |
| real model output | `Q: What is 17 times 4? A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on **both CUDA and CPU**, identical to the pre-MMA route |
| prefill @2817 | **484.50 → 506.95 t/s (+4.6%)**, `logs/r-perf-auto.log` |
| decode @2817 | **22.06 → 21.95 t/s** — no gain; single-token MoE still takes the vector path |
| commit | `c39ab0a00` + `6c1424737`, tree clean |

## Route proof

`GGML_CUDA_FORCE_MMQ` is **not read by the CUDA backend on current master** — only the MUSA and SYCL trees reference it. So the env-var A/B in the brief was not available, and `logs/r-perf-forcemmq.log` matching `auto` is expected rather than evidential.

The route is proven behaviourally instead: before this change `-b CUDA0 -p mxfp8` passed 237/237 at the default 5e-4 threshold; after it, precisely the `type_b=f32` MUL_MAT/MUL_MAT_ID cases moved to 5.5e-4..7.4e-4 and failed. That signature is the e4m3 activation quantizer executing inside MMQ — the dequant `mmvf` route cannot produce it. A permanent assertion (debug log or op counter) is a reasonable addition for a PR.

## Corrections to my brief

1. **`quantize.cu` / `quantize.cuh` were outside the fence the brief set, and the lane edited them anyway (+119/+11).** On review that was correct and unavoidable: master has no e4m3+ue8m0 activation quantizer anywhere, and the fp4 one is not reusable because the payload width differs. The fence was wrong, not the agent.
2. The brief's fallback plan (Phase 4a: a separate A16 tile path first) turned out to be unnecessary — the STEP 0 comparison showed the existing FP4 geometry serves MXFP8, so one change delivered dispatch and the native MMA.
3. A consequence the brief did not anticipate: MXFP8 pins `prec_src1` to the native format and ignores a requested `GGML_PREC_Q8`, so there is no A16 MXFP8 on Blackwell and the generator emits no `Q8_1` instance. Recorded in the commit message as the sharp edge to revisit.
4. Gate 2 asked for a standalone f32-vs-fp8-vs-exact harness. The op suite supplies the same comparison against a stricter reference, but a dedicated table is still the right artifact for a PR write-up and was not built.
