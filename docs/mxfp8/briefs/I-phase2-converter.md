# Brief I — Phase 2: make the Inkling-2 conversion lossless (MXFP8 + NVFP4 pass-through)

**Deliverables**
1. Edit `conversion/inkling2.py` **in place** (no V2 files) so an explicit opt-in produces a GGUF whose routed-expert bytes are the checkpoint's own, merely repacked.
2. `scripts/inkling_native/verify_native_gguf.py` — standalone verifier: for every expert tensor in a produced GGUF, decode it and compare to an independent HF reference; assert **0 differing values** (±0.0 sign differences allowed, reported separately).
3. Append a `## Phase 2 result` section to `docs/mxfp8/06-lossless-repack-design.md` (do not rewrite it) with the produced file's tensor-type inventory and byte totals.

**Read first:** `docs/mxfp8/00-status.md` (findings + corrected facts), `docs/mxfp8/01-roadmap.md` (DR-1…DR-6), `docs/mxfp8/briefs/00-HOW-TO-WORK.md`, and `docs/mxfp8/04-inkling2-master.md` if it exists (row-permutation + who-can-emit-NVFP4 answers).

**Established facts (do not re-derive)**
- `GGML_TYPE_MXFP8 = 44` exists on this branch: `block_mxfp8 = {uint8_t e /*ue8m0*/; uint8_t qs[32] /*e4m3*/}`, 33 B / 32 elems; gguf-py has `MXFP8 = 44`, `QUANT_SIZE_MAP[MXFP8] = (32, 1+32)`, `LlamaFileType.MOSTLY_MXFP8 = 43`. CPU paths work (306/306 op tests); **CUDA reports not supported — that is expected and fine**.
- Checkpoint: experts `layers.0–19` = NVFP4 (U8 packed payload, `*_weight_scale` F8_E4M3 1×16, `*_weight_amax` F32[E], `*_weight_outer_scale_denominator` F32[1] **= 2688.0 = 6×448**), `layers.20–23` = MXFP8 (F8_E4M3 payload `(E,rows,cols)`, `*_weight_scale` U8 ue8m0 `(E,rows,cols/32)`, **no amax/denominator**). E=128.
- **`amax/denom` is a power of two for 0 of 128 experts** → the NVFP4 global factor must be written as the per-expert F32 sidecar `.scale` that master's runtime already loads and multiplies (see `src/llama-model.cpp:1917-1990`; the public unsloth file ships exactly these). **Do not fold it into the e4m3 block scales.**
- Every NVFP4 block-scale byte in this checkpoint has its sign bit clear, so master's e4m3→ue4m3 `& 0x7F` strip (`conversion/base.py:1324`) is exactly lossless here.
- Master's current Inkling path **dequantizes both expert groups to F32** (`conversion/inkling2.py:176`) and the converter's `filter_tensors`/`modify_tensors` renames `w13_weight`→`gate_up_proj` etc. The gate/up row interleave lives at `conversion/inkling2.py:170-176` — **any repack must apply the identical permutation to payload rows AND their scale rows**, or values silently mismatch their scales. This is the single highest-risk detail; prove it with the verifier on ≥2 experts per tensor.
- Reference repack conventions to imitate: `conversion/base.py:repack_mxfp4_blocks` (nibble de/re-interleave HF→ggml order for 4-bit) and `_nvfp4_pack` (`[4 ue4m3 scales][32 qs]` super-blocks). **MXFP8 has no nibble issue** — 1 byte per element, so the payload bytes are copied verbatim into the 33-byte block; the only transform is interleaving the scale byte at the head of each 32-element group.
- Bit-exact MoE pass-through precedents in-tree: `conversion/deepseek.py:749-751`, `conversion/deepseek41.py:223`, `conversion/gpt_oss.py:58` (all `add_tensor(..., raw_dtype=...)`).
- Target arch name for output is **`inkling2_loop`** with `block_count = 24` (the ggml-org file is the current-naming reference; the unsloth file uses the retired `inkling2`/`block_count=48` scheme).

**Constraints**
- Conversion of the full model needs ~20.4 GB of output plus streaming headroom: 61 GB RAM, 412 GB free disk. Keep tensors **lazy** (`LazyNumpyTensor`/`LazyChunkedTensor`) — never materialize all experts. Add a `--only-layers`-style escape hatch or a tiny synthetic path so the verifier can run on 2 layers without a 20 GB write.
- **Never delete a log or artifact.** Tee conversion output to `logs/phase2-convert.log`. Never write to `/tmp`.
- **Do not touch docker, GPUs, or services.** CPU-only (`--outtype` path, `llama-gguf`/`llama-print` or python to inspect).
- One writer per file: stay inside `conversion/inkling2.py`, `scripts/inkling_native/`, `docs/mxfp8/06-*.md`, `logs/`. Do not edit `conversion/base.py` unless strictly unavoidable — if you must, keep it additive and say why in the doc (another agent may be in `ggml/src/ggml-cuda/`).
- **Write files early and often**: the stub verifier + a first working MXFP8-only export before you touch NVFP4, then layer the rest. A previous run of a sibling brief died at the 30-minute timeout having written nothing.
- Opt-in only: default converter behaviour must be byte-for-byte unchanged for everyone else. Name the flag `--native-experts` (or follow the repo's existing flag style if one already fits).

**Acceptance**
1. `verify_native_gguf.py` exits 0 on a 2-layer sample: MXFP8 layers 20/21 → **0 differing values**, NVFP4 layers 0/19 → 0 differing values after applying the `.scale` sidecar.
2. A produced full file's inventory is exactly: 8 MXFP8 expert tensors, 40 NVFP4 expert tensors, 48 F32 `[128]` `.scale` sidecars, rest BF16/F32 as in the checkpoint; total ≈ 20.4 GB (report the measured number and the arithmetic).
3. `llama-print`/`llama-cli -m <file>` **on CPU** loads it and produces tokens (slow is fine) — proving the GGUF is structurally valid end-to-end. If CPU inference of a 20 GB model is impractical in your budget, load a 2-layer sample instead and say so.
