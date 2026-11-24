# Brief A — Index prior art for MXFP8 / FP8 block-scaled matmul revival

**Deliverable:** write `/home/tim/code/services/repos/llama.cpp/docs/mxfp8/02-prior-art-branches.md`.
**Do not edit any source file.** Read-only research + one markdown output.
Cite `file:line` for every claim. Mark anything unverified as `UNVERIFIED`.
Target length 200–400 dense lines. Print a 15-line summary to stdout when done.

## Goal of the parent project (context)

We are adding **`GGML_TYPE_MXFP8`** (32 elements per block = `{uint8_t e /*ue8m0*/; uint8_t qs[32] /*e4m3*/}` = 33 B/block, 8.25 bpw — the direct analogue of `block_mxfp4` at `ggml/src/ggml-common.h:215-219`) to llama.cpp master, so that the *unadulterated* `thinkingmachines/Inkling-2-Mini` checkpoint can be repacked into GGUF **bit-exactly** and run on 2× RTX 5060 Ti (sm_120, CUDA 13.2, native nvfp4 + mxfp8 tensor cores).

Repo: `/home/tim/code/services/repos/llama.cpp`, branch **`mxfp8-weights`** off master `29b52d398` ("model : Inkling 2 Mini (#32694)").

## Verified starting facts (do not re-derive)

- Branch `mxfp4-w4a8` (tip `9a02817b2`) = 3 commits over merge-base `e30ad80fd` (2026-11-05, 445 commits behind master):
  - `6764dd2b8` ggml-cuda: fp8 activations for MXFP4 on Blackwell
  - `5bbceb6df` ggml-cuda: pack the FP8 scale operands
  - `9a02817b2` ggml-cuda: name the block-scaled mma after its operands
  Files touched (vs base): `ggml.h` (+GGML_PREC_FP8 = 35), `mma.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `mmq.cu`, `mmq.cuh`, `quantize.cu`, `quantize.cuh`, `template-instances/generate_cu_files.py`, `mmq-instance-mxfp4.cu`, `ggml-vulkan.cpp`, `tests/test-backend-ops.cpp`.
- Related: `w4a8-beast` / `w4a8-reviewable` both at `e5bb13fbe` ("ggml-cuda : mxfp4 W4A8 path via the mxf8f6f4 MMA on Blackwell"); worktrees `repos/wt-w4a8` (e5bb13fbe), `repos/wt-w4a4` (e467e5b5e).
- Patch archives: `/home/tim/code/services/repos/mxfp4-w4a8-full.diff` (54 KB), `mxfp4-w4a8-reviewable.patch`, `mxfp4-w4a8-fixes-tightening.patch`. Dir `repos/mxfp4-pr-assets/`. Untracked `scripts/w4a-harness/` in the repo.
- Other stale MXFP branches (KV-cache/flash-attn oriented, useful for E8M0 layout lore): `mxfp-types-and-quants`, `mxfp-fa-cuda`, `mxfp-fa-scalar`, `mxfp4-soa-model-weights`, `metal-mixed-mxfp-kv`, `mxfp68-quant`, `mxfp4-flash-attention*`.
- Untracked design docs at repo root: `mxfp8-scale.md`, `e8m0.md`, `e8m0-quant-research.md`, `mxfp4-scale.md`, `mxfp4-imatrix-accuracy.md`, `mxfp4-pr-diet-plan.md`, `mxfp4-scale-ab-plan.md`. Plus `/home/tim/code/services/docs/` (e.g. `kv-cache-investigation.md`).

## Required content

1. **Per-branch/commit map** with exact hashes: for the block-scaled MMA work, quote the full PTX asm strings used (e.g. `mma.sync.aligned.kind::mxf8f6f4...`), operand register counts, how the **ue8m0 scale factors are packed into SFA/SFB registers** (which thread holds which byte), K-tile size, and the arch guard macro(s) that enable it (`__CUDA_ARCH__`, `GGML_CUDA_FFMA`, FP4-specific feature macros).
2. **FP8 activation quantizer** added on `mxfp4-w4a8` (`ggml/src/ggml-cuda/quantize.cu` / `.cuh`): kernel names, input dtype, block size, scale format (e8m0 vs f32), output layout (separate scale tensor vs interleaved), how it's dispatched from the mul_mat path, and any new `GGML_PREC_*` plumbing (`GGML_PREC_FP8 = 35`, plus how `prec_src1` selects the native path in `mmq.cu`).
3. **mmq plumbing diff**: what changed in `mmq.cu`, `mmq.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh` — new template params, `sram_stride`/k-tile handling for 8-bit operands, vec-dot for fp8x2, and how MXFP4 weight tiles are loaded on the block-scaled path vs the classic dp4a path.
4. **Tests**: what `tests/test-backend-ops.cpp` gained (case names, shapes, tolerances).
5. **REUSE INVENTORY table** — one row per hunk: `reuse-verbatim` / `reuse-after-rebase` / `rewrite`, with reason. Base the rebase judgement on a real comparison with current master:
   `git diff --stat e30ad80fd master -- ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh ggml/src/ggml-cuda/mmq-load-tiles.cuh ggml/src/ggml-cuda/mmq-vec-dot.cuh ggml/src/ggml-cuda/quantize.cu ggml/src/ggml-cuda/mma.cuh`
   and read the current versions of those files (note master now has `mmq-config-blackwell.cuh`, `mmq-instance-nvfp4.cu`, `native_fp4_available`, `GGML_TYPE_F8_E4M3 = 43`).
6. **Lore from the untracked .md docs** — quote the parts about ue8m0 scales (incl. byte 255 = NaN), `mxf8f6f4` operand formats, satfinite conversions, four-over-six scale selection, and any measured perf numbers/PPL results.
7. **A short "what is NOT in any branch" list** — e.g. is there *any* prior MXFP8-as-a-*weight*-type work anywhere on this machine (vs MXFP4/6/8 as KV-cache types)? Check `git log --all --oneline --grep=-i mxfp8`, `git grep -l MXFP8 $(git rev-list --all --max-count=... )` — use judgment; a targeted sweep of branch tips (`git grep -l "MXFP8" <branch>`) is enough.
