# Brief E — Layout simulator + bit-exactness harness for the Inkling-2-Mini lossless repack

**Deliverable (code + doc):**
- `/home/tim/code/services/repos/llama.cpp/scripts/inkling_native/layout.py` — reference model of the byte layouts.
- `/home/tim/code/services/repos/llama.cpp/scripts/inkling_native/check_layout.py` — runnable checker (exit 0 = all invariants hold).
- `/home/tim/code/services/repos/llama.cpp/docs/mxfp8/06-lossless-repack-design.md` — the layout contract + measured size budget.
**Do NOT touch any existing source file** (no edits to `conversion/`, `ggml/`, `src/`). Only create files under those three paths. Print a 15-line summary to stdout.

## Why this exists

We want a GGUF of `thinkingmachines/Inkling-2-Mini` whose expert weights are **the original bytes, merely repacked** (no dequant/requant). Before touching C++ we must pin down the byte-level contract and *prove* round-trip equality on the real checkpoint. Everything here is pure numpy/torch on local data — no build needed.

## Checkpoint (local, read-only)

`~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/`
- `model-00001-of-00001.safetensors` (19,526,019,168 B; 1244 tensors). Read the safetensors header (8-byte LE length + JSON) and use `numpy.fromfile`/mmap with computed offsets — **never** load the whole file into RAM. 64 GB RAM; keep peak RSS < 6 GB.
- Expert tensors (E=128 experts, all `[128, rows, cols]`):
  - **NVFP4** layers 0–19: `model.llm.layers.{L}.mlp.experts.w13_weight` U8 `(128,2304,1152)` + `w13_weight_scale` F8_E4M3 `(128,2304,144)` + `w13_weight_amax` F32 `(128,)` + `w13_weight_outer_scale_denominator` F32 `(1,)`; `w2_weight` U8 `(128,2304,576)` + `w2_weight_scale` `(128,2304,72)`. Block 1×16 along K; scale is e4m3; dequant = `e2m1[nibble] * e4m3[scale] * (amax[e]/denom)`.
  - **MXFP8** layers 20–23: `w13_weight` F8_E4M3 `(128,2304,2304)` + `w13_weight_scale` U8 `(128,2304,72)`; `w2_weight` F8_E4M3 `(128,2304,1152)` + `w2_weight_scale` U8 `(128,2304,36)`. Block 1×32 along K; scale byte is **ue8m0** (value = 2^(b-127), b=255 is NaN ⇒ assert absent).
- Existing GGUFs for cross-checking (already on disk): `/home/tim/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` (contains 40 native **NVFP4** expert tensors for blk.0–19 + F32 `[128]` `.scale` sidecars; blk.20–23 experts are **Q8_0**), and `/home/tim/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf`. Read with the repo's `gguf-py` (`sys.path.insert(0,'gguf-py')`). GGUF type enum: F32=0, Q8_0=8, BF16=30, IQ4_NL=20, MXFP4=39, NVFP4=40, F8_E4M3=43.
- llama.cpp master block layouts (`ggml/src/ggml-common.h`): `block_mxfp4 {uint8_t e; uint8_t qs[16];}` (17 B / 32 elems, QK=32); `block_nvfp4 {uint8_t d[4]; uint8_t qs[32];}` (36 B / 64 elems, QK=64, sub-block 16). Repacker to mirror: `conversion/base.py:1240-1265` (`repack_mxfp4_blocks`, note its nibble de-interleave/re-interleave: HF packs element j in low nibble and j+16 in high nibble, ggml packs 2i/2i+1) and `conversion/base.py:1310-1335` (`_nvfp4_pack`, note it **strips the sign bit** of the e4m3 scale to make ue4m3, and packs `[4 scales][32 qs]`).
- Row permutation: `conversion/inkling2.py:170-176` slices the gate/up halves with an interleaved `row_slice` (stride 2, offset by `half`). A bit-exact repack must apply the same permutation to the scale rows. Confirm against the real unsloth GGUF if feasible, else document as UNVERIFIED.

## Tasks

1. **`layout.py`**: pure-python defs of `pack_mxfp8_blocks(weight_e4m3[np.uint8 view], scale_u8) -> np.uint8` producing ggml `block_mxfp8 = {e, qs[32]}` (33 B) row-major over K, and the matching `unpack/ dequant` (use an e4m3 LUT — copy the table approach from `conversion/inkling2.py:18-26`), plus `pack_nvfp4_blocks` matching master's `_nvfp4_pack` exactly (including the `& 0x7F` sign-strip and the super-block grouping), plus the nibble-order fixup for HF→ggml 4-bit packing (mirror `repack_mxfp4_blocks`'s lo/hi interleave logic).
2. **`check_layout.py`** invariants, all on *real slices* of the checkpoint (a few experts/rows only, for speed — use the safetensors header for offsets):
   - a. Every MXFP8 scale byte `< 255`; every e4m3 payload byte decodes finite (no NaN/±448 saturation surprises) — print min/max scale exponent per layer.
   - b. `dequant(unpack(pack(x))) == dequant_hf_reference(x)` **exactly** for sampled rows of both w13 and w2 in layers 20 and 23 (compare against `conversion/inkling2.py`'s `_dequant_expert_chunk` semantics — import it if importable, else re-implement and cross-check).
   - c. Same for NVFP4 layers 0 and 19, including the `(amax/denom)` global factor (report whether the factor is a power of two per expert — if yes, a sign-strip/renormalize trick is exact; if no, we must keep the sidecar `.scale` tensor like master's NVFP4 path does).
   - d. **Cross-check against the shipped unsloth GGUF**: for `blk.0.ffn_gate_up_exps.weight`, compare its NVFP4 bytes to what `layout.py` produces from the HF tensors (byte-identical? which fields differ — payload? scales? row order?) and compare `blk.0.ffn_gate_up_exps.scale` to `amax/denom`. This tells us whether the *existing* public file is already a valid lossless NVFP4 repack (huge shortcut if so) or an approximation. Report the exact diff statistics.
   - e. **Size budget table**: bytes per type for a fully lossless repack (NVFP4 experts incl. e4m3 scales + F32 `.scale` sidecars; MXFP8 experts incl. e8m0 scales; all other tensors kept BF16/F32 as in the checkpoint; separately what Q8_0/BF16 alternatives cost) and a second column at what fits in 2×16 GB with fp8 KV at 32k/64k/128k/262k ctx (use: 24 layers, 8 kv heads, head_dim 128, `sliding_window_size 1024` on 20 of 24 layers — note the SWA layers cap KV growth; show the arithmetic).
   - f. `--quick` flag (default) that samples; `--full` that sweeps all expert tensors (must still stream, no full-file load).
3. **`06-lossless-repack-design.md`**: the byte-level contract (diagram each block layout), the row-permutation rule, the decision on how the NVFP4 per-expert global factor is stored (sidecar `.scale` — matching master's convention), the measured size budget, and a "what must change in `conversion/inkling2.py`" list with function names + line refs. Include the **exact** list of GGUF tensor names + types + shapes for the target file.

## Constraints

- Never write outside those 3 paths. Never modify `/home/tim/models/**` or the safetensors (open read-only).
- No `rm` of any file. Log output to `scripts/inkling_native/check_layout.log` (tee, don't delete).
- A background CUDA build is running in this repo — do not run cmake/make.
- If a numpy/torch detail is ambiguous, resolve it by *reading master's code* and by comparing to the unsloth GGUF, and say which you used.
