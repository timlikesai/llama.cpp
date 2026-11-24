# Brief D — Blackwell sm_120a MXFP8 MMA facts + ecosystem references

**Deliverable:** write `/home/tim/code/services/repos/llama.cpp/docs/mxfp8/05-hw-and-ecosystem.md`.
**Do not edit source.** Use official docs (NVIDIA PTX ISA, CUDA C++ Programming Guide, CUTLASS/vLLM/SGLang/TE/TensorRT-Model-Optimizer repos) and cite URLs + doc section/table names for every number. Print a 15-line summary to stdout.

## Setup we must design for

2× GeForce RTX 5060 Ti 16 GB, **compute capability 12.0 (sm_120)**, driver 615.71.09, **CUDA 13.2 (nvcc 13.2.86)**. llama.cpp master already compiles Blackwell FP4 tensor-core paths and its `ggml/src/ggml-cuda/CMakeLists.txt` (≈:63-95) appends `120a-real` and rewrites plain `12X` → `12Xa` because "the Blackwell FP4 tensor core instructions are not forwards compatible".
Existing master asm: `ggml/src/ggml-cuda/mma.cuh:1288` `mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0`, `:1295` `...kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`.

Goal: MXFP8 **weights** (e4m3 payload, ue8m0 scale per 1×32 along K) with **FP8 activations** (the checkpoint's `required_activation_quantization` for the mxfp8 group is literally mxfp8_e4m3 / block 32 / ue8m0) — i.e. W8A8 block-scaled tensor-core matmul, plus a decode-friendly path for MoE (`top-8 of 128` experts ⇒ small M).

## Questions

1. **PTX syntax/semantics** of `mma.sync.aligned.kind::mxf8f6f4.block_scale.*`: legal shapes (m16n8k32?), which `.scale_vec::` values (1X/2X/4X) are legal on `sm_120a` (vs `sm_100a`/`sm_121a`), the full legal A/B dtype cross-product for `kind::mxf8f6f4` (e4m3×e4m3, e4m3×e2m1, e5m2×…, and whether **A4×W8** and **A8×W4** mixed operands are permitted — our prior branch used the mixed-type property for W4A8), accumulator types, scale-type (ue8m0 only?), and `satfinite` requirements. Quote the PTX ISA tables.
2. **SFA/SFB register layout for m16n8k32** — how many 32-bit scale registers per warp, and which thread holds which scale byte (the "scale factor A matrix"/"scale factor B matrix" fragment layout + the idesc/format for `kind::mxf8f6f4`). This is the make-or-break detail; quote the ISA section/figures exactly and include the mapping formula (row ↔ lane).
3. **Does sm_120 (consumer Blackwell) have `tcgen05` / TMEM?** (expected: no — it's sm_100a/sm_101a; confirm, since a tcgen05 plan would be a dead end here.) Confirm `wgmma` is sm_90a-only. Confirm `mma.sync` with `.e4m3/.e5m2` (non-block-scaled, FP8 MMA) is available on sm_89/sm_120 — that gives us a simpler fallback where the e8m0 scale is applied in software during accumulation.
4. **ue8m0 semantics**: bias (127), value 255 = NaN, exponent range, and how a ue8m0 block scale interacts with a per-tensor fp32 global scale (NVFP4's two-level recipe: `amax`/`outer_scale_denominator`). Confirm whether *training-time* scale-selection rules (e.g. `fouroversix_mae`) have any inference-time consequence (expected: none — the bytes are already chosen).
5. **Fast conversions / memory layout**: `cvt.rn.f16x2.e4m3x2` / `__nv_cvt_fp8x2_to_halfraw2` throughput on sm_120; `__nv_fp8_storage_t` satfinite conversions; and the **33-bytes-per-32-elements** problem: not power-of-two → consequences for smem swizzling, 128-bit `lds` patterns, and `vec_dot` alignment. Compare to how llama.cpp handles MXFP4 (17 B/block) and NVFP4 (36 B/block) in `mmq-load-tiles.cuh` — read that file locally and report the actual constraint/technique (this is local code reading, not web).
6. **Ecosystem reference implementations to read as layout documentation**:
   - **Local vLLM checkout** at `/home/tim/code/services/repos/vllm` — grep for `mxfp8`, `mxf_8`, `e8m0`, `scaled_mma`, `sm120`, `block_scaled` and report file paths + which intrinsics/PTX they use (this is a concrete, high-value reference for sm_120 MXFP8 GEMM).
   - CUTLASS blockscaled MMA examples (which archs, how they build the SFA/SFB layout).
   - NVIDIA TensorRT Model Optimizer (producer of this checkpoint's `hf_quant_config.json` / `quant_method: modelopt|inkling2`) — its MXFP8/NVFP4 export layout conventions.
   - **llama.cpp upstream**: search GitHub PRs/issues for MXFP8 *weights* (types, `GGML_TYPE_` numbering disputes), and for Inkling-2-Mini follow-up PRs (nvfp4/mxfp8 native conversion, unsloth GGUF notes). Report PR numbers + state + chosen names/layouts — we want to converge with upstream, not diverge. Also check whether master's `docs/` or the MXFP4/NVFP4 PR descriptions mention an MXFP8 plan.
7. **Practical toolchain**: minimum toolkit for `kind::mxf8f6f4` on sm_120a; whether nvcc needs explicit `-gencode arch=compute_120a,code=sm_120a`; known ptxas pitfalls for block-scaled MMA on sm_120; whether CUDA 13.x deprecates anything we'd use.

## End with

`## IMPLICATIONS FOR OUR DESIGN` — ≤10 bullets: which MMA kind/scale_vec we must use, what SFA layout code we must write, whether an e4m3×e4m3 non-scaled MMA + software e8m0 fallback is viable, whether FP8 activations are required (vs FP16 activations with dequant), and anything that would block us on sm_120 specifically.
