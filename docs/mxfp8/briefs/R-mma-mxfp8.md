# Brief R — Phase 4b: native `mma.kind::mxf8f6f4` W8A8 for MXFP8, **lifted from stale branches but written to current master conventions**

## STEP 0 — extend what master already has; DO NOT port the branch's separate FP8 layout (parent's measured comparison)

Master's MMQ already carries block-scaled FP types by **reusing int-quant tile geometry**, so the likely correct change is small. Evidence, all verified in the current tree:

| anchor | what it shows |
|---|---|
| `mmq.cuh:131` | `GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, // MXFP4 and NVFP4 on Blackwell.` — **one layout serves two types** already |
| `mmq.cuh:164` | `static_assert(ggml_cuda_mmq_get_sram_stride(LAYOUT_FP4) == ggml_cuda_mmq_get_sram_stride(LAYOUT_Q8_1), "Wrong tile size for MXFP4")` — the FP4 layout is **deliberately stride-identical to Q8_1** |
| `mmq.cuh:408-409` | `ggml_cuda_mmq_get_tile_x_sizes_from_type`: `case GGML_TYPE_MXFP4: return MMQ_DP4A_TXS_Q8_1;` and `GGML_TYPE_NVFP4: return MMQ_DP4A_TXS_Q8_0_16;` — block-scaled FP rides **existing** int tile sizing |
| `mmq-vec-dot.cuh:1172` | *“Shared MMA kernel for MXFP4 and NVFP4 on Blackwell.”* — a shared kernel to extend with a third type, not three separate ones |
| `mmq-load-tiles.cuh:1626-1698` | the MXFP4 loader: `bxi->e` via `ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f` (line 1677), `QI_MXFP4`-indexed payload, `threads_per_row = iter_k / QK_MXFP4` (line 1698) |
| `mmq.cuh:257, 291, 963` | `prec_src1 != GGML_PREC_Q4 && (NVFP4 || MXFP4)` special-cases — the pattern to widen to MXFP8 |
| `mmq.cuh:727-748, 897, 1715-1717` | the `VDR_MXFP4_Q8_1_MMQ` vec-dot-route enum + dispatch, `DECL_MMQ_CASE`, `DECL_MMQ_CASE_W4A4` |

**Why MXFP8 should fit this shape rather than need a new layout:** `block_mxfp4` is `{uint8_t e; uint8_t qs[16];}` (32 values/block) and `block_mxfp8` is `{uint8_t e; uint8_t qs[32];}` (32 values/block) — **same scale encoding (`e8m0`), same values-per-block, only the payload is 8-bit instead of 4-bit-packed nibbles**. And a 32-byte e4m3 payload is exactly `Q8_1`-shaped, which is what the existing layout is sized for.

**So the comparison you must actually perform before writing anything** (and record the answer in the doc):
1. Read `load_tile_x_q` for MXFP4 and ask: with the payload reinterpreted as e4m3 bytes instead of nibble pairs, and the scale read as `ggml_cuda_e8m0_to_fp32(e)` with **no `*0.5f`**, does the existing FP4/Q8_1 tile layout hold MXFP8 unchanged? If yes: MXFP8 may need only a `case` in the config table, `tile_x_sizes`, the vec-dot route enum, and the MMA wrapper — **no new loader at all**.
2. Enumerate every current `GGML_TYPE_MXFP4` / `NVFP4` touchpoint (`grep -n "MXFP4\|NVFP4" ggml/src/ggml-cuda/*.cuh ggml/src/ggml-cuda/*.cu`) and decide per site: widen the existing condition, add a case, or not applicable. That list **is** the port surface — the stale branch's separate `SRAM_LAYOUT_FP8`, `block_fp8_mmq`, `CASE_MXFP` and hand-written loader are the thing we are explicitly NOT copying (their `CASE_MXFP` even puts `J` in a different argument position than master's `CASE`).
3. Only if step 1 fails should a new SRAM layout appear — and then justify it in the doc with the stride/`tile_x_sizes` arithmetic that forced it.

Use the branch refs for the **facts inside them** (instruction encoding, scale byte-packing rules, `tidx_A` lane mapping, the `e4m3.e4m3` dtype field, the arch guard + cuBLAS fallback policy, and the FP8 activation-quantizer rounding fix from `0a409c8d8`) — those are the parts that are expensive to rediscover. Everything structural comes from master.

## What prior art gives us (already extracted — read these, don't re-derive)

Reference dumps in `docs/mxfp8/ref/` (full unfiltered diffs also in `logs/priorart-*.diff`):

| file | source | what it is |
|---|---|---|
| `mma-kind-mxf8f6f4-all3.inc` | `mxfp4-soa-model-weights:ggml/src/ggml-cuda/mma.cuh` ~1040-1145 | **the working instruction set**: `e4m3.e4m3`, `e5m2.e5m2`, `e2m3.e2m3`, `e3m2`, plus `kind::mxf4` and `kind::mxf4nvf4` — all with the same operand skeleton |
| `mma-soa-mxfp8-block.inc` | same branch, ~1020-1150 | surrounding wrapper style |
| `mmq-config-blackwell-mxfp68.cuh` | `mxfp68-quant` | `CASE_MXFP` — "mxfp4, mxfp6 and mxfp8 share the same tile geometry" |
| `load_tile_x_q-mxfp8.inc` | `mxfp68-quant:mmq-load-tiles.cuh` ~1600-1720 | **the tile loader for `block_mxfp8`** (vectorized `QK_MXFP8/4` word loads) + the mxfp6 bitstream loader |
| `mmq-vec-dot-mxfp-mma.inc` | `mxfp68-quant:mmq-vec-dot.cuh` ~1235-1400 | the block-scaled vec_dot: "mxfp weights (e2m1/e2m3/e4m3) x e4m3 activations (m16n8k32, scale_vec::1X ue8m0)" |
| `mmq-cuh-mxfp8-blocks.inc` | `mxfp68-quant:mmq.cuh` 55-100, 690-715 | `block_fp8_mmq` (`static_assert(sizeof == sizeof(block_q8_1_mmq))`) + type classification sites |
| `mmq-cu-dispatch.inc` | `mxfp68-quant:mmq.cu` 60-95, 130-160, 285-330 | dispatch cases, the y-size note, and the arch guard + fallback |

**Facts worth keeping, from `9a02817b2`** ("pack the FP8 scale operands in a format the mma instruction expects") — the single most error-prone part, quoted in `docs/mxfp8/02-prior-art-branches.md`:
- *"the mma only reads byte 0 of the scale operand, so shift each frag's scale byte down"* → `scaleA[n][frag] = sa >> (8*frag);` (a `uint4`/`.x .y .z .w` load is the **wrong** form and was the bug).
- SFA lane index `tidx_A = g + (t % 2) * 8` with `g = tid/4`, `t = tid%4`; *"scale_vec::1X with byte-id 0 takes scale_A for rows g and g+8 from the quad lanes with t%4 == 0 and 1"*.
- `FP8_SCALES_PER_WORD = QK_FP8_MMQ / QK_MXFP4`; `static_assert(nfrags == FP8_SCALES_PER_WORD, "one scale word covers all fragments and one y block")`.
- Activation block: 128 e4m3 values, `d4[0]` packs 4 ue8m0 scales (byte f covers values `32f..32f+31`), `d4[1..3]` pad to `MMQ_TILE_Y_K`.
- Their `mmq.cu` comment: *"mxfp8 activation blocks (e4m3 codes + ue8m0 scale) have the same size and value count as q8_1"*.
- Their arch guard: *"mxfp6/mxfp8 MMQ needs the Blackwell block-scaled mma; other arches use the cuBLAS dequant fallback"* → `if ((type == MXFP6 || type == MXFP8) && !blackwell_mma_available(cc)) return false;` — matches DR-4.
- Target instruction: `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {D0..D3},{A0..A3},{B0,B1},{D0..D3}, a_scale, {0,0}, b_scale, {0,0};`
- Good news: **MXFP8 needs no nibble shuffle** (the `lo_mask 0x0F0F0F0F` / `hi_mask 0x3C3C3C3C` bits-[5:2] dance is FP4-only — an e4m3 weight byte is already a byte operand).

## Scope note: this brief supersedes `P-mmq-mxfp8.md`

Lane `P` (an A16-style tile path as a stepping stone to the MMA) timed out having written no code, and STEP 0 above makes the two-step plan likely unnecessary: if MXFP8 slots into the existing FP4/Q8_1 geometry, one change delivers both `mmq` dispatch **and** the native MMA. Treat 4a as a fallback, not a first step.

## Reconciliation table — the branch is ~6 weeks stale, DO NOT copy-paste

| concern | current master (verify yourself) | stale branch | required action |
|---|---|---|---|
| wire format | `block_mxfp8 { uint8_t e; uint8_t qs[32]; }`, `QK_MXFP8 32`, 33 B | **identical** | none — the loaders operate on our bytes as-is |
| SRAM layout enum | `GGML_CUDA_MMQ_SRAM_LAYOUT_{Q8_0,Q8_1,Q2_K,Q3_K,Q6_K,FP4,NVFP4}` — **no FP8 member** | references `..._FP8` | **add** the enumerator + its `ggml_cuda_mmq_get_sram_stride` case, sized for one scale word per `FP8_SCALES_PER_WORD*QK_MXFP4` |
| config macro | `CASE(type, nthreads, occupancy, I, J, layout, K_vram, stream_k, fallback)` — per-type lines, `J` in position 5 | `CASE_MXFP(J_, nthreads_, occupancy_, I_, layout, K_vram_, stream_k_, fallback_)` — **`J` is position 1** | rewrite as master-style `CASE(GGML_TYPE_MXFP8, ...)` lines; a literal copy would silently mis-set `J` and produce wrong tiles or garbage perf |
| config selection fn | `ggml_cuda_mmq_get_config` special-cases NVFP4/MXFP4 on `prec_src1` (Blackwell vs ampere fallback) | predates `prec_src1` | integrate MXFP8 into the **current** `prec_src1` logic; keep the ampere/non-Blackwell route falling back to Tier A (`mmvf`), never to a wrong tile path |
| y-side block | master has `block_q8_1_mmq`, `block_fp4_mmq`, `QK8_1_MMQ`, `QK_FP4_MMQ` | `block_fp8_mmq`, `QK_FP8_MMQ` | re-add following master's naming/static_assert style |
| MMQ plumbing switch | `mmq.cuh` has per-type `case` lists (type_can_use_* / sram layout / scale index helpers) | same idea, fewer types | add MXFP8 to **every** switch the branch touches — grep for each branch hit and find the master equivalent. Miss one and you get a compile error at best, silent tile corruption at worst |
| template instances | `template-instances/mmq-instance-*.cu` + `generate_cu_files.py` | same mechanism | regenerate; check `CMakeLists.txt` for whether GLOB needs a reconfigure — if it does, **ask the parent before reconfiguring** |
| mma.cuh wrapper | master already has `kind::mxf4nvf4 … .ue4m3` | adds several `kind::mxf8f6f4` forms | mirror master's existing wrapper structure/`constexpr` dispatch style |
| activation quantizer | `to_fp8_mmq` (f32 → e4m3 + 4×ue8m0) | same idea | re-implement in the current quantize path, **including the `+0.5f` rounding** the mxfp68 branch fixed for e4m3 (`0a409c8d8` "round e4m3 correctly") |

## Correctness is already proven — do not disturb it

`~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` (18,974,614,688 B, bit-exact NVFP4+MXFP8 experts) generates `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` **identically on CPU and CUDA**, `The capital of France is → Paris.`, at prefill 484.50 / decode 22.06 t/s (2817 tokens) vs the lossy reference's 492.32 / 23.62. Suite: CPU 306/306, CUDA0 237/237 for `-p mxfp8`. Today's root cause of wrong output was upstream `690b87d88` (loop slots aliased before the `.scale` pass), cherry-picked as `489106434`.

## Gates

1. `test-backend-ops -b CUDA0 -p mxfp8` and `-b CUDA1 -p mxfp8` fully green, **and** `-b CPU -p mxfp8` still 306/306. Include `MUL_MAT_ID` / `MUL_MAT_FUSION(use_id=1)` at the real expert shapes (`n_ff 1152/2304`, 128 experts, top-8).
2. A **synthetic precision floor** measurement, because the whole justification for W8A8 is that e4m3 activations are near-lossless: mul_mat with f32 activations (Tier A route) vs FP8 activations (this route) vs a dequantized exact reference — report max relative error and cosine similarity, and state explicitly whether the W8A8 error is below IQ4_NL's quantization error (the standard DR-4 sets for accepting it).
3. Real model, unchanged correctness: the `A: 68 …` string must still appear with the mmq/MMA route selected **and** with `GGML_CUDA_FORCE_MMQ=1`, plus CPU unchanged. Prove which path executed (env var, or a temporary instrumented print behind a flag that you then remove) — do not report a speedup you didn't route-verify.
4. Perf: prefill + decode at 2817 tokens vs baselines — current Tier A `mmvf` route = **484.50 / 22.06 t/s**, lossy IQ4_NL = 492.32 / 23.62. **Never pass an exact `-ngl`** (it disables memory fitting and OOMs this 2-card setup); don't use `llama-bench` for this model.
5. A negative result is a success: if native MMA is slower than `mmvf` for Inkling's shapes, say so with numbers and leave the heuristic unchanged. Do not force the win.
6. Commit only if green, in llama.cpp style split (`ggml-cuda : … MMA` then `CUDA : …`), `docs/mxfp8/**` and `scripts/**` untracked.

## Rules

- **No cherry-picks from these branches, no rebase, never touch `master`.** Adapt hunks by hand to current conventions and cite the source line in a comment (`// lifted from mxfp68-quant:mmq-load-tiles.cuh:1667, adapted to the CASE macro signature in 4a0bd27a0`) so a reviewer can follow.
- **Builds:** another agent may be compiling in `build-cuda`. Always `cmake --build build-cuda -j 22` (NEVER reconfigure/configure), tee to `logs/r-mmq-build-<n>.log`, read the **complete** output. On a lock/partial-object conflict, wait and retry — don't `rm` build artifacts.
- `nvcc` here is CUDA 13.2 targeting `sm_120a` (already configured); `kind::mxf8f6f4` needs `BLACKWELL_MMA_AVAILABLE` — confirm the guard macro and the arch gate, and make non-Blackwell builds compile (dead-code the asm behind the same guard master uses).
- Fence: you own `ggml/src/ggml-cuda/{mma.cuh,mmq.cu,mmq.cuh,mmq-config-*.cuh,mmq-load-tiles.cuh,mmq-vec-dot.cuh,template-instances/**}` and `docs/mxfp8/17-*`→ use `docs/mxfp8/18-mma-mxfp8.md`. Do **not** touch `conversion/**`, `scripts/inkling_native/**`, `src/**`, `ggml-cuda.cu`, `mmvf.cu`, `mmvq.cu`, `convert.cu`, `dequantize.cuh`, `getrows.cu` — those hold the already-proven behavior and another agent is working in `conversion/` on vision.
- Check `git status` before you start editing and never revert work that isn't yours. No docker/systemctl. Both GPUs free. **Never delete a file or log.**
- Write `docs/mxfp8/18-mma-mxfp8.md` within your first 4 tool calls and update as you measure — five tasks this session timed out having produced nothing, which is the failure mode to avoid above all. 45-minute budget; finalize at 38; leave the build green.
