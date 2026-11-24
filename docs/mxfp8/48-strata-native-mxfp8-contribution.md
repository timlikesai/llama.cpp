# 48. Strata: what it already has for MXFP8, and the contribution that is actually missing

Investigated 2026-11-23 in `/home/tim/code/services/repos/Strata` (remote
`https://github.com/Niko1221/Strata`, HEAD `73d4f4196`, MIT). Question posed: *their MXFP8 layer code
probably does not use the mixed MMA - can we contribute the work from this branch so the engine can run
the native weights directly, like our GGUF does on llama.cpp?*

**Short answer: the premise is wrong in their favour, and there is still a real, narrow contribution.**
Strata already emits the identical PTX instruction we wrote, already quantizes activations to MXFP8,
already ships a *lossless* MXFP8 weight codec, already loads `thinkingmachines/Inkling-2-Mini` +
`mtp.safetensors` **with zero requantization**, and already has a bit-exact test for it. What it cannot
do is read **GGUF type 44** - so our validated lossless GGUF is unusable there, and the path that would
have to handle it silently downgrades it.

Repo shape for orientation: `app/` (Tauri desktop), `terminal/` (`strata` CLI, `strata_cli/`,
`strata_control/`), `engine/` (the C++/CUDA/HIP/SYCL/Metal engine + `serve/` Python API). Language mix:
1562 `.py`, 1184 `.cpp`, 559 `.hpp`, 319 `.cu`, 132 `.cuh`. It has its own GGUF reader - it does **not**
link ggml - with three vendored copies of `ggml-common.h` under `engine/third_party/`.

## 1. What they already have (each line verified by reading it)

| capability | where | evidence |
|---|---|---|
| **Mixed W8A8 block-scaled MMA, MXFP8 x MXFP8** | `engine/src/prefill/moe_fp8_mxf_tc.cu:116`, `engine/src/inkling2/cuda/ink_moe_fp4_sm120.cu:83` | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0` - **byte-for-byte the instruction in our `mma.cuh`** |
| MXFP8 **activation** quantizer | `moe_fp8_mxf_tc.cu:411` `quant_act_mxfp8_kernel`; `ink_moe_cuda.cuh:79-81` `xq_mx8 / xsc_mx8 / hq_mx8` | activations really go to `e4m3 + ue8m0`, not f16/q8 |
| Lossless **MXFP8 weight codec** | `engine/include/strata/quant/sk_codecs.hpp:25` `MX8 = 16, // Lossless MXFP8: F8_E4M3 weights + UE8M0/32w block scales (.strata v1.1)`; decode contract `dequant_mx8_row(weights_e4m3, scales_ue8m0, ncols)` at `:108-115` | same block geometry as our `block_mxfp8`: 32 values + one `2^(e-127)` scale |
| **Native checkpoint loader, zero requantization** | `engine/src/inkling2/ink_safetensors_loader.h:1-3` *"Direct Official HF safetensors loader for thinkingmachines/Inkling-2-Mini + mtp.safetensors - Zero-requantization adapter (NVFP4 -> SF4 code_id=15, MXFP8 -> MX8 code_id=16, BF16 trunk + DSpark)"*; `:419` *"Index all routed experts ... with ZERO requantization!"*; `:476` casts `SkCode::MX8` | they ingest the exact checkpoint we converted, including the drafter |
| The MXFP8 layers identified as **4 of 48** | `ink_moe_fp4_sm120.cu:10-11` *"MX8 (MXFP8, L20..23 = 4/48 visits)"* | independently matches our file: the 8 type-44 tensors are `blk.20..23.ffn_{gate_up,down}_exps.weight` |
| Same for **NVFP4**, `mxf4nvf4 ... scale_vec::4X.m16n8k64.e2m1.e2m1` | `ink_moe_fp4_sm120.cu:8-9` | matches our `mma.cuh:1304`; their comment *"Zero bit-reordering needed!"* mirrors our own finding that the rows are already 32 B-aligned |
| **cuBLASLt MXFP8 (VEC32_UE8M0) dense prefill** | `engine/src/prefill/gemm_mxfp8.cuh:1` + `sf_swizzle_off()` at `:33` | with fused `dequant_q8_0_to_mxfp8` / `dequant_q6_k_to_mxfp8` producers |
| MX8 on **HIP and Metal** too | `engine/src/inkling2/hip/ink_hip_codecs.hip.h:27,1184` (`CODE_MX8 = 16`, *"Layers 20..23"*), `metal/ink_metal_common.h:43` (`FMT_MX8`) | so a GGUF-type gap, if fixed, should be fixed for all three backends |
| Bit-exact **test suite** for MX8 | `engine/tests/quant/strata_v1_1_inkling2_test.cpp:118,192` *"All 5 mixed codecs (SF4/MX8/S3_K/S2_K/S1.5) + Batch4 + Loop/SITU/AttnRes/SConv4/DSpark match bit-exact!"*, plus `ink_kernel_test.cu:105 test_sm120_mx8()` | the harness a PR would extend already exists |
| Product-level support | `docs/MODELS.md:260-285` - dedicated `strata-inkling` engine, `.strata` builds **SQ-10 / SQ-15 / SQ-20** (*SQ-20 = 19.3 GB, "near-lossless native SF4+MX8"*), Unsloth Dynamic GGUFs, built-in **DSpark** drafting, `Inkling-2-Mini-Vision.strata` (0.64 GB), `strata launch --family inkling-2-mini --model SQ-15` | SQ-20's 19.3 GB is essentially our 19.61 GB file |

So on the MMA question there is nothing to teach them. Our `15649f8e8`/`c39ab0a00` MXFP8 tile path is
the same design already realized there, in a dedicated Inkling kernel with more shape specialization.

## 2. The actual gap: GGUF cannot carry MXFP8 into Strata

`engine/include/strata/quant/types.hpp` is their type table, and its ids **are** the GGUF ids by design
(`:3` *"The ids are the GGUF file format's (ggml_type), so a GGUF's type field and a pack's type field
are valid here unchanged. An id this table does not know is answered with nullptr"*). The enum is:

```cpp
IQ1_M = 29, BF16 = 30, MXFP4 = 39, NVFP4 = 40, Q1_0 = 41, Q2_0 = 42,   // :16
```

- **`MXFP8 = 44` is absent**, and so is any `block_mxfp8` in the whole tree (`grep block_mxfp8` -> 0 hits
  outside the three vendored `ggml-common.h`, which also lack it).
- Consequence, followed through the code: `ggml_to_quant_type()` (`ink_gguf_loader.h:44`) is a bare cast
  -> `type_info(44)` returns `nullptr` -> `gguf_row_bytes()` returns 0 (`:69-71`) ->
  **`"GGUF: unsupported tensor type ..."`** for experts (`:304`) or dense (`:463`). So
  `Inkling-2-Mini-FULL.gguf` is refused at load. That is fail-safe, which is good engineering on their
  side - but it means our artifact does not run there.
- Their lossless GGUF passthrough is `SkCode::GGUF_RAW`, whose documented set is
  `Q8_0/Q4_K/Q5_K/Q6_K/IQ3_S/IQ4_NL/IQ4_XS/MXFP4` (`sk_codecs.hpp:26`; Metal `ink_metal_common.h:44`
  adds BF16/F16/F32/Q4_0/Q3_K/Q2_K/IQ3_XXS) - **MXFP8 is in neither list**, and the supported-format
  summary they show users (`engine/src/inkling2/ep_dual_gpu.h:19`) says
  `Community .gguf (NVFP4/Q8_0/Q4_K/Q6_K/IQ3_S/IQ4_NL/IQ4_XS/MXFP4)` - again no MXFP8.

### 2b. The secondary finding, and the more valuable one to report

`engine/src/inkling2/ink_gguf_transcode.h:84-114`:

```cpp
// Choose target S_K codec preserving source GGUF precision without blowing VRAM budget.
inline SkCode choose_sk_code_for_ggml(uint32_t ggml_type) noexcept {
    ...
    switch (ggml_type) {
        case 0: case 1: case 8: case 9: case 15: case 30: return SkCode::S8_K;   // >= 8 bpw
        case 6: case 7:  case 13: case 14:               return SkCode::S6_K;   // 5..6 bpw
        // <= 4 bpw (Q2_K ... IQ4_*, MXFP4) -> S4_K
        default:                                          return SkCode::S4_K;
    }
}
```

39 (MXFP4, 4.25 bpw), 40 (NVFP4, 4.5 bpw) and a future 44 (MXFP8, **8.25 bpw**) all fall into
`default -> S4_K`, a 4-bit trellis - i.e. the function that promises to *preserve source GGUF precision*
halves it for MXFP8, silently, on the `GGUF_RAW -> S_K` streaming path (used by the Metal runner,
`ink_metal_common.h:574,719`, and the VRAM-saving route). This is worth telling them even if they never
take the type-44 patch, because it is a quiet quality regression class rather than a crash.

We can attach real evidence that these layers are unforgiving: in our lossless validation, the same
Inkling MXFP8 layers loaded with a **wrong scale convention** measured **PPL 5,260,013.9** vs 5.5757
correct, and KLD 13.787 (`docs/mxfp8/20-lossless-validation.md`). A 2x bit-depth cut is less violent than
that, but it is the same layers with no headroom narrative.

## 3. What we bring that they do not have

Not kernels - **format semantics, a validated artifact, and a test oracle**:

1. **`block_mxfp8` as it actually exists in the wild.** Our `ggml/src/ggml-cuda/convert.cuh` /
   `ggml-common.h` work defines it: 32 x `e4m3` payload + 1 x `ue8m0` scale per block = 33 B,
   `QK_MXFP8 = 32`, `vec_dot_type = MXFP8` (so the activation format is *itself*, which is why their
   `TypeInfo.act` choice for row 44 needs thought - MXFP4/NVFP4 both say `kQ80` there).
   The `types.cpp` row that mirrors their format is literally:
   ```cpp
   {Type::MXFP8, "mxfp8", QK_MXFP8, sizeof(block_mxfp8), /* act */ Type::MXFP8, true, false},
   ```
2. **A file to test with.** They have no GGUF containing type 44 - we do, and it is byte-verified
   against the checkpoint: `Inkling-2-Mini-FULL.gguf`, 798 tensors, type histogram
   `{F32: 541, BF16: 209, NVFP4: 40, MXFP8: 8}`, the 8 being `blk.20..23.ffn_{gate_up,down}_exps.weight`.
   Plus `conversion/inkling2.py`, which regenerates it from the HF directory.
3. **A fact that shrinks their PR.** I checked whether a type-44 GGUF needs the per-expert scale plumbing
   that NVFP4 needs: **it does not.** All 40 `.scale` tensors in our file are F32 and belong to the 40
   NVFP4 tensors; none shares a prefix with a type-44 tensor, because MXFP8 carries its `ue8m0` scale
   inside the block. So their `load_scale(..., /*required=*/ gu_type == 40, ...)` at
   `ink_gguf_loader.h:849-866` is already correct for 44, and `lm.dn_scale` likewise. That removes the
   single most error-prone part of the change (double-applying a scale), which is exactly the failure we
   measured at PPL 5.26e6.
4. **The repack shape is already proven on their side.** MX8 stores payload and scales as **separate
   planes** (`dequant_mx8_row(weights, scales, ncols)`), while GGUF **interleaves** them per 33-byte
   block. Their `is_nvfp4_to_sf4` branch (`ink_gguf_loader.h:869-884`) does the identical job for NVFP4:
   detect the native type, compute a split-plane body layout, copy bytes. An `is_mxfp8_to_mx8` sibling is
   a *de-interleave, no arithmetic*, hence lossless by construction - and their alignment note holds
   (`K = 2304` and `inter = 1152` are both multiples of 32).
5. **The trap list.** `test-backend-ops`-style coverage is not enough for a new ggml type: our
   `QI_MXFP8 = 2` vs `8` bug (`b16720b83`) silently dropped 3/4 of every row with no assert; and
   `ggml_get_rows` hard-coding an F32 return (`ggml.c`, `// TODO: implement non F32 return`) made one of
   my own "optimizations" measure a literal 0.0% change. Both are cheap to mention so nobody re-derives
   them in their tree.

## 4. Proposed contribution, in the order I would send it

| # | change | files | size | risk |
|---|---|---|---|---|
| **PR-A** | Teach the type table GGUF id 44 and route type-44 experts to the existing **MX8** path losslessly | `quant/types.hpp:16`, `quant/types.cpp` (row), vendored `ggml-common.h` (`block_mxfp8`, `QK_MXFP8`), `artifact/gguf_reader.hpp:140 block_geometry()`, `ink_gguf_loader.h:869+` (`is_mxfp8_to_mx8` + body layout), `SkCode::GGUF_RAW`/`FMT_*` doc lists, `ep_dual_gpu.h:19`, `docs/MODELS.md` | ~120-180 lines + test | medium: the layout math must match both the CUDA and Metal/HIP consumers; a wrong stride is silent |
| **PR-B** | Stop the silent 4-bit downgrade for native block-scaled types: explicit `case 39/40/44`, and prefer *refuse* over requantize for unknown ids | `ink_gguf_transcode.h:86` | ~15 lines | low, but changes behaviour for currently-"working" GGUFs - worth an issue first |
| **Issue** | Report 2b with the PPL-sensitivity evidence; ask which they prefer (lossless repack vs hard error) before writing code | - | - | none |
| **Offer** | A byte-verified type-44 GGUF + `conversion/inkling2.py` as the test fixture, so their bit-exact suite can assert *GGUF-MXFP8 == safetensors-MX8 == checkpoint bytes* | new test in `strata_v1_1_inkling2_test.cpp` or `ink_gguf_transcode_test.cpp` | small | none |

The strongest framing for them is not "we ported fp8 MMA" - it is **"your MX8 kernels and bit-exact tests
have no GGUF data source; here is one, plus a class of silent requantization to close."**

## 5. Honest context for our own work

- Tim's original goal - run the unadulterated weights on this hardware, fast - is, per this reading,
  **already reachable in Strata today via the HF checkpoint** (SQ-20, or the direct safetensors loader),
  with their own DSpark drafting and vision. Our GGUF contribution buys *interop with the llama.cpp
  ecosystem*, not a capability Strata lacks.
- What our llama.cpp branch uniquely provides: a **single self-contained GGUF** (trunk + native experts +
  embedded 54-tensor drafter, 19.61 GB, byte-exact) plus the deployment we measured - and the code fixes
  that made it run fast: MXFP8 type + CUDA tile path, alias-aware layer placement, the indexed-bias FA
  adoption, the fitter zero-span guard, padded-vocab draft logits, `can_reuse` state-presence.
- Deciding question, therefore: **which engine is actually faster on these two cards?** TODO #26 - build
  Strata, load the local checkpoint, and compare against `1,036 / 52.1` t/s at 262,144 (and
  `1,091 / 40.9` at 8K). That needs GPU exclusivity (their container competes with ours) and an engine
  build, so it waits for a go-ahead.

## 6. Not yet verified (do not assert these downstream)

- Whether `choose_sk_code_for_ggml()` is reachable for expert bodies on the **CUDA** path at all, or only
  on Metal/hybrid streaming - I read the Metal call sites, not a CUDA one. The S4_K claim is about the
  transcode path, which is where that function lives.
- Whether MX8's on-disk scale plane is linear or cuBLASLt-swizzled for the MoE kernel
  (`sf_swizzle_off()` at `gemm_mxfp8.cuh:33` is the *dense* cuBLASLt path; the MoE kernel uses
  `s_grid`/`decode_half16_mxfp8`). PR-A must read the MX8 producer in `ink_safetensors_loader.h:419-480`
  and match it exactly; guessing here is the silent-corruption class of bug.
- Whether their Inkling GGUF support has been exercised on a file containing MXFP4 (39), which would
  already show the 2b behaviour in practice.
- No build of Strata has been attempted, so nothing here is a claim about their runtime behaviour -
  every statement above is from reading their source at the cited line.
