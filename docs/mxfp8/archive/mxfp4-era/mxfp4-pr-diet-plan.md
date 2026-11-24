# MXFP4 PR Diet Plan

Branch: `mxfp4-quant-kv` vs upstream merge-base `95887577a`
Original footprint: **36 files, +1019/-252 = 1271 lines changed**
Final footprint: **36 files, +818/-162 = 980 lines (-23%)** — all tests green
Goal: same feature set (dense mxfp4 layers, W4A8, mxfp4 KV cache, UOS scales for KV + W4A4) at ~50%.

## STATUS: three release-blocking bugs found and fixed (validated 2026-09-25)

The branch was NOT "mostly working" — test-backend-ops fails in all three mxfp4 matmul
paths. Root causes, all fixed in the working tree and validated (auto/q4/q8 prec all pass
MUL_MAT + MUL_MAT_ID; CPY + SET_ROWS + FLASH_ATTN_EXT pass):

1. **W4A8 vec_dot read the wrong config** (`ggml_cuda_mmq_vec_dot_mxfp4_mxfp8_mma`):
   `get_sram_stride`/`get_K_vram` were queried without `prec_src1`, defaulting to Q8 ->
   ampere config (stride 72, K_vram 256) instead of the Blackwell FP8 config (148, 512).
   Every MMQ shape (n>8) produced garbage; n<=8 passed only because MMVQ masks it.
   Fix: pass GGML_PREC_MXFP8 (2 lines).

2. **W4A4 used the wrong MMA instruction** (mma.cuh): the PR replaced upstream's
   `kind::mxf4.scale_vec::2X.m16n8k64` (packed nibbles) with `kind::mxf8f6f4.scale_vec::1X.
   m16n8k32` (8-bit containers) for the e2m1 x e2m1 case, but `vec_dot_fp4_fp4_mma` still
   feeds packed-nibble k64 data. Fix: restored the upstream instruction for Q4; the
   mxf8f6f4 k32 variant is only needed (and only used) for e4m3 activations.

3. **W4A4 activation kernel rewrite was broken in 4 ways** (quantize_mmq_mxfp4):
   non-atomic `|=` RMW on d4 scale words into uninitialized workspace memory; one byte
   per 4 values with OR-collided nibbles; wrong byte index (ignored the warp's group
   base, so warp 1 clobbered warp 0 and bytes 64..127 were never written); launcher
   vals_per_block mismatch. Fix per R1 below: reverted to the upstream kernel + the
   1-line UOS scale change. Also reverted the launcher restructure, which had hoisted
   `GGML_ASSERT(ne00 % QK_NVFP4 == 0)` to fire for MXFP4 (crash for k=96 MoE shapes).

4. **The Q8_1 path for mxfp4 on Blackwell was deleted, breaking upstream's own W4A8 tests**:
   the PR replaced the `mul_mat_q_case<MXFP4, GGML_PREC_Q8>` instantiation with the MXFP8
   one, and the dispatch keyed on src0 type rather than prec — so an explicit Q8 request
   (upstream's test_mul_mat_w4a8/mul_mat_id_w4a8 cases) silently ran the e4m3 path at
   ~5.5e-4 NMSE vs the 5e-4 tolerance, while upstream's true Q8_1 path passes. Fixed by
   restoring the Q8 instantiation + dispatch (quantize_mmq_q8_1, ampere config, generic
   Q8_1 MMA funcs) and mapping Q4->Q4, Q8/F32->Q8_1, default->MXFP8 in get_prec_src1.
   The util-funcs switch returns mxfp8 funcs only for MXFP8 prec. The test tolerance
   distinguishes W4A8 (1e-3) from env-forced W4A4 (2e-2).

All R-items are now implemented (R6 was verified already-minimal: mxfp4 must stay in
FA_TYPES because the GGML_CUDA_FA_<K>_<V> defines are consumed as C++ identifiers).
Validated: full test-backend-ops suite on CUDA0, the prec matrix (auto/q4/q8) on
MUL_MAT+MUL_MAT_ID, CPY+SET_ROWS+FLASH_ATTN_EXT, test-quant-type-selection 11/11,
and a clean sm_89 compile of the mmq-instance-mxfp4 TU (R8 guards removed).

## Where the 1271 lines live

| Bucket | Files | Lines | Verdict |
|---|---|---|---|
| W4A4 activation kernel rewrite | quantize.cu (mxfp4 kernel) | ~185 | **flotsam** — feature content is 1 line |
| W4A8 activation kernel | quantize.cu (mxfp8 kernel + launchers) | ~200 | feature, but a clone of the mxfp4 kernel |
| W4A8 MMA core | mmq-vec-dot, mmq-load-tiles, mma.cuh, mmq-config, mmq.cuh | ~290 | feature core |
| KV cache fattn | fattn-common, fattn-vec, fattn.cu, instances, cmake | ~180 | feature core |
| KV cache plumbing | cpy, cpy-utils, set-rows, convert, ggml-cuda.cu, arg.cpp | ~100 | feature core |
| Dense mxfp4 weights | llama-quant, llama.h, gguf-py, model-loader, quantize.cpp, ggml-quants.c | ~150 | mostly feature |
| Shared helpers | ggml-impl.h, common.cuh | ~65 | consolidation (good) |
| Tests | test-backend-ops | ~80 | partially flotsam |
| Misc glue | mmq.cu, dequantize.cuh, cmake, generator, bench | ~110 | mixed |

## Recommendations, ranked by savings

### R1 — DONE: reverted the `quantize_mmq_mxfp4` rewrite; W4A4-UOS is now a one-line patch

The rewritten kernel was not just churn — it was broken (see STATUS bug 3). Reverted to the
upstream kernel with a single scale change:

```cpp
- const uint8_t e = compute_e8m0_scale(amax);
+ // UOS (arxiv 2607.24377) for the W4A4 activation path
+ const uint8_t e = ggml_e8m0_scale(amax, GGML_MXFP4_FMAX_UOS, true);
```

`compute_e8m0_scale` stays deleted (ggml_e8m0_scale replaces it). The float4 vectorization
can be revisited as a follow-up perf PR, written correctly this time.

Perf risk is negligible: activation quantization is O(T*K) vs the GEMM's O(T*K*N); even a 2x
slower quantize kernel is noise. Keep the float4 vectorization as a follow-up perf PR where it
can be benchmarked on its own. This also keeps the MoE scatter path byte-identical to upstream
except the scale, which is what reviewers will look at first.

### R2 — DONE: one `quantize_mmq_mxfp<scatter, prec>` kernel for both activation formats

DONE: reverted the launcher restructure in `quantize_mmq_fp4_cuda` /
`quantize_scatter_mmq_fp4_cuda` back to upstream's per-branch structure. This also fixed
the crash from the hoisted `GGML_ASSERT(ne00 % QK_NVFP4 == 0)` firing for MXFP4 (mxfp4 only
guarantees ne00 % 32; killed MoE shapes with k=96).

DONE: folded `quantize_mmq_mxfp8` into a single `quantize_mmq_mxfp<scatter, prec>` on the
upstream kernel geometry (lane-per-value, b-loop over 2 scale groups, full-warp amax
reduce). The store index expression `quad*16 + b*8 + group_id` turned out identical for
both formats — only the element type (char2 vs uint32) and the scale store differ, so the
if constexpr sections are small. Validated with the full prec matrix.

### R3 — Collapse the duplicated Blackwell config table (~30 saved)

mmq-config-blackwell.cuh duplicates all 16 MXFP4 `CASE`s with only the SRAM layout different:

```cpp
constexpr ggml_cuda_mmq_sram_layout L =
    prec_src1 == GGML_PREC_MXFP8 ? GGML_CUDA_MMQ_SRAM_LAYOUT_FP8 : GGML_CUDA_MMQ_SRAM_LAYOUT_FP4;
// one 16-CASE list using L
```

### R4 — Unify the three CPU mxfp4 quantizers in ggml-quants.c (~55 saved)

`quantize_row_mxfp4_ref`, `quantize_row_mxfp4_kv`, and `quantize_row_mxfp4_impl` are the same
function with three scale policies (RNE-4.0 / UOS-7.25 / imatrix-searched). One static impl
parameterized by `(fmax, round_up, quant_weights)` + three thin wrappers (-40).
`best_index_mxfp4` also duplicates `ggml_float_to_fp4_e2m1_index` (same grid, 2x convention) —
make one call the other (-12).

### R5 — Trim test flotsam (~37 saved)

Drop `init_tensor_mxfp4_ties` and the `initialize_tensors` override in `test_cpy` (-34). The
cpy nmse tolerance already absorbs single-code tie differences, so the custom init gates
nothing the standard uniform init wouldn't. Keep the 8 fattn cases, the set_rows/cpy tolerance
entries, and the 2e-2 -> 1e-3 tightening (a nice W4A8 consequence).

### R6 — cmake: small simplification only (~4 saved, verified constraint)

mxfp4 must stay in `FA_TYPES`: the bottom loop emits `GGML_CUDA_FA_<K>_<V>` compile
definitions for every pair, and `FATTN_VEC_CASE` in fattn.cu consumes them as C++
identifiers (undefined = hard error). So the `+1` and the 3-line `continue()` filter in the
"all" branch are already minimal. The only trimmable piece is the 4-line mxfp4-pairing
`FATAL_ERROR` in the user-combos branch — dropping it degrades the error for a bad
`GGML_CUDA_FA_QUANTS=mxfp4-q8_0` to a missing-.cu-file at build time. Marginal; keep or
drop based on taste.
### R7 — generate_cu_files.py: derive the MXA8 source (~10 saved)

Replace the 12-line `SOURCE_MMQ_MXA8` block with a one-liner when writing the mxfp4 instance:
`SOURCE_MMQ.replace("DECL_MMQ_CASE(", "DECL_MMQ_CASE_MXA8(")`. Keep `TYPES_KV_EXTRA`.

### R8 — Verify, then likely drop, the non-Blackwell guards (~17 saved, needs build check)

The `NO_DEVICE_CODE` in mma.cuh and the `#if defined(BLACKWELL_MMA_AVAILABLE)` wrappers around
both vec-dot bodies: upstream compiled fine without them (the asm is already inside `#if`; the
else branch was a plain no-op). Verify with a non-Blackwell-only build
(`CMAKE_CUDA_ARCHITECTURES=89`) and drop if it passes.

### R9 — llama-quant.cpp: merge with the MXFP4_MOE precedent (~5 saved)

- OUTPUT branch: `if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE || ftype == LLAMA_FTYPE_MOSTLY_MXFP4)` -> Q8_0.
- Fold TOKEN_EMBD into the catch-all: `new_type = category == tensor_category::TOKEN_EMBD ? GGML_TYPE_Q8_0 : GGML_TYPE_MXFP4;` with a single comment.

### R10 — DONE: explicit prec requests are now honored for MXFP4

Was: `get_prec_src1` returned `prec == Q4 ? Q4 : MXFP8` before the validity assert, so an
explicit F32/Q8 request was silently downgraded to MXFP8 (upstream maps those to Q8_1),
and a Q8 request would have mismatched the ampere config against the mxfp8 util funcs.

Fix (in tree): Q4 -> Q4, F32/Q8 -> Q8_1 (ampere config + generic Q8_1 MMA funcs via the
third switch), else MXFP8. The util-funcs switch returns the mxfp8 funcs only for
`prec_src1 == GGML_PREC_MXFP8` and breaks otherwise. Validated: GGML_CUDA_MMQ_PREC=q8
passes the full mxfp4 suite. The test tolerance distinguishes the paths
(max_nmse_err_mxfp4_blackwell: 1e-3 default, 2e-2 for env-forced q4).

## Do NOT cut — irreducible feature core (~600 lines)

- **fattn mxfp4** (~190): `dequantize_V_mxfp4` (the shfl pairing halves V-cache read bytes in
  the attention inner loop — a naive per-thread nibble read would save ~15 lines at real perf
  cost, don't), `fattn_vec_mxfp4_map` + `vec_dot_fattn_vec_KQ_mxfp4`, the Q_reg interleave,
  the K/V-type-mismatch guard. The f16-Q KQ dot (BF16 pattern) is the right accuracy choice
  for a 4-bit K format; the q4_0-style dp4a alternative would save ~25 lines but quantizes Q
  to q8_1 — not recommended.
- **W4A8 MMA core** (~150): `ggml_cuda_mmq_vec_dot_mxfp4_mxfp8_mma`, `load_tiles_mxfp4_mxfp8`,
  the e4m3 `mma_block_scaled_fp4` variant, `block_fp8_mmq` + `SRAM_LAYOUT_FP8`. The mxf8f6f4
  m16n8k32 semantics (byte-per-value e2m1 weights, e4m3 activations) share too little with the
  fp4_fp4 path to unify without constexpr soup.
- **KV plumbing** (~90): cpy both directions, set-rows, convert.cu dequant registrations,
  supports_op, arg.cpp, llama-bench. All follow existing per-type patterns exactly; this is
  the table-stakes cost of a new KV cache type.
- **Dense mxfp4 registration** (~10): llama.h, gguf-py, model-loader, quantize.cpp,
  test-quant-type-selection.
- **Shared host/device helpers in ggml-impl.h** (~35): `ggml_e8m0_scale` +
  `ggml_float_to_fp4_e2m1_index` are what make the CPU and CUDA quantizers agree bit-for-bit
  on ties. This *is* the consolidation, not flotsam (it already deleted
  `ggml_cuda_float_to_fp4_e2m1` and `compute_e8m0_scale`).
- **`mul_mat_q_process_tile` loop generalization**: already net-negative lines vs upstream.

## One behavior change to surface explicitly in the PR description

R1's one-liner changes gpt-oss/MoE mxfp4 activation numerics vs upstream (RNE-fmax-4.0 ->
UOS-7.25), because the kernel is shared between the MoE scatter path and the dense W4A4 path.
The current PR already does this implicitly via the rewrite; the diet just makes it visible.
Justify with the perplexity data from mxfp4-scale.md in the PR description, or gate UOS to
dense-only behind a template param (~+5 lines) if MoE numerics must stay untouched.

## Final budget

| Item | Was | Final |
|---|---|---|
| quantize.cu | 356 | 175 (R1 revert + R2 unified kernel + nvfp4 byte-identical) |
| mmq-config-blackwell.cuh | 56 | 39 (R3) |
| ggml-quants.c | 94 | 71 (R4) |
| test-backend-ops.cpp | 79 | 48 (R5, + tolerance helper) |
| common.cmake | 8 | 8 (R6: verified already minimal) |
| generate_cu_files.py | 21 | 10 (R7) |
| mma.cuh + vec-dot guards | 29 | ~11 (R8, sm_89-verified; mma.cuh now purely additive) |
| llama-quant.cpp | 10 | 6 (R9) |
| mmq.cu dispatch | 37 | 45 (R10 + Q8_1 path restoration) |
| common.cuh | 28 | 12 (nvfp4 helper restored; mxfp4 helpers purely additive) |
| feature core (fattn, W4A8 MMA, KV plumbing, helpers) | ~553 | ~555 |
| **Total** | **1271** | **980 (-23%)** |

Cleanup round (post-R-items): nvfp4 is now byte-identical to upstream everywhere
(compute_e8m0_scale + quantize_mmq_nvfp4 + common.cuh helper restored; the mxfp4
utils are purely additive). mma.cuh adds the W4A8 instruction as a nested else-if
without touching the upstream mxf4/mxf4nvf4 blocks. The unified activation kernel
is DRYed into quantize_mmq_mxfp_pack4/quantize_mmq_mxfp_store helpers + a slim
kernel that reads as upstream's mxfp4 kernel plus a scale line. The fattn-vec
macro typo fix was reverted (out of scope), and the W4A4 comment now correctly
says mxf4 (W4A4 uses kind::mxf4; only W4A8 uses kind::mxf8f6f4).

The remaining ~600 lines are the irreducible feature core: fattn mxfp4 KV kernels (~190),
W4A8 MMA (vec_dot + load_tiles + mma variant + block/config, ~210), KV plumbing
(cpy/set-rows/convert/supports_op/arg.cpp, ~90), shared host/device helpers (~40),
dense-mxfp4 registration (~20), tests (~50). Cutting further means cutting features:
the V-dequant shfl pairing, the f16-Q KQ path accuracy, or the W4A4-dense path.

The more meaningful review metric: files with heavy rework of existing upstream code
(quantize.cu, mmq-config, ggml-quants.c, tests, llama-quant, generator) dropped from
~700 to ~390 churned lines, and the mxfp4 activation kernel is upstream's own plus one
scale line.

## Suggested final commit shape

1. `ggml: shared e8m0/e2m1 quantization helpers (host/device)` — ggml-impl.h, common.cuh,
   ggml-quants.c unification, cpu trait switch.
2. `cuda: mxfp4 KV cache (fattn vec, cpy, set-rows, UOS)` — fattn + plumbing + tests.
3. `cuda: W4A8 (mxf8f6f4) path for mxfp4 dense layers` — GGML_PREC_MXFP8, mmq, templated
   activation kernel, configs.
4. `llama: dense MXFP4 quantization (ftype, imatrix scale search)` — llama-quant, gguf-py,
   tools.

## Housekeeping

The repo root has untracked research files (e8m0.md, mxfp4-scale*.md, logs/, plot scripts,
ppl-vs-size.png...). Keep them out of the PR; move durable research into `docs/` or a separate
branch, and add `logs/` to .gitignore locally if needed.
