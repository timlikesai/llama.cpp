# Brief J — Native NVFP4 expert export for Inkling-2 (finish the ~18 GB lossless file)

**Deliverables (own exactly these files; nothing else):**
1. `conversion/inkling2.py` — extend the existing `--native-experts` path (already emits MXFP8) so the 20 **nvfp4_nearest** layers are also repacked verbatim instead of dequantized to BF16.
2. `scripts/inkling_native/verify_nvfp4_gguf.py` — new verifier (do **not** edit `verify_mxfp8_gguf.py`, `layout.py`, or `check_layout.py`; import and reuse them).
3. `docs/mxfp8/11-phase2b-nvfp4.md` — progress notes, written EARLY and updated as you go.

**Why this is the last mile.** A conversion with only MXFP8 native measured **44.95 GiB**: `BF16 218 tensors / 40.90 GiB`, `MXFP8 8 / 3.92 GiB`, `F32 478 / 0.14 GiB`. The 40 dequantized NVFP4 expert tensors are ~36.5 GiB of BF16 where native NVFP4 needs ~11.5 GiB. Making them native lands the file at **≈18 GB** (independently computed body size: 18,014.3 MiB), which is the whole point of the project.

**Established facts — do not re-derive, do not re-verify:**
- `GGML_TYPE_NVFP4 = 40` already exists upstream; `block_nvfp4 = {uint8_t d[4]; uint8_t qs[32];}` = 36 bytes per **64** elements (`QK_NVFP4=64`, sub-block 16). gguf-py already has `NVFP4: (64, 4+32)`. **No ggml/C++ change is needed.**
- Master's reference packer is `conversion/base.py:_nvfp4_pack` (:1309-1332): unpack ModelOpt nibbles, **strip the e4m3 scale sign bit (`& 0x7F`) to ue4m3**, then interleave `[4 scale bytes][32 payload bytes]` super-blocks. For THIS checkpoint the sign strip is exactly lossless — every scale byte already has its sign bit clear (verified).
- **Proof this works:** the public `/home/tim/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` contains `blk.0–19.ffn_gate_up_exps.weight` / `ffn_down_exps.weight` with `tensor_type = 40 (NVFP4)` that are **byte-identical** (2,985,984 / 1,492,992 bytes; 0 differing scale bytes, 0 differing payload bytes) to what `scripts/inkling_native/layout.py:pack_nvfp4_blocks()` builds from the HF tensors. Its `blk.N.ffn_gate_up_exps.scale` / `ffn_down_exps.scale` are F32 `[128]` and bitwise `== amax/outer_scale_denominator` for 128/128 experts. Match that layout exactly; you can diff against that file as ground truth (it is 17.63 GB — read single tensors via `GGUFReader`, never the whole file).
- The global factor **must** be a sidecar: `outer_scale_denominator` = 2688.0 (= 6×448) and per-expert `amax` ∈ [0.10, 0.64]; `amax/denom` is **never** a power of two (0/128 experts, 128 distinct factors). Do not fold it into the block scales.
- Sidecar tensor names must be what the runtime already loads: `blk.N.ffn_gate_up_exps.scale` / `blk.N.ffn_down_exps.scale`, F32 shape `[n_expert]` (see `src/llama-model.cpp:1917-1990`; `.input_scale` is `TENSOR_NOT_REQUIRED` and this checkpoint has none — omit it).
- Row order: reuse `_expert_row_slices(experts, rows, w13)` — payload rows and scale rows must be selected by the same slice (gate/up is interleaved; `down` is identity). MXFP8 export already does this and passed byte verification.
- Checkpoint names: `model.llm.layers.{0..19}.mlp.experts.{w13_weight,w2_weight}` (U8 packed), `..._weight_scale` (F8_E4M3, `(E,rows,cols/16)`), `..._weight_amax` (F32 `(E,)`), `..._weight_outer_scale_denominator` (F32 `(1,)`).
- **Watch the nibble order.** MXFP8 needed no reordering (1 byte/element). NVFP4 does: HF packs element j in the low nibble and j+16 in the high nibble of each byte, ggml packs 2i/2i+1. `layout.py:nibbles_to_elements` / `elements_to_ggml_nibbles` encode the verified transform, and `base.py:1240-1265` (`repack_mxfp4_blocks`) documents the same convention. Get this wrong and the file will look right by size but be numerically garbage — so verify per-expert, not just per-layer.

**Acceptance gates (report each with the command and its output):**
1. `verify_nvfp4_gguf.py` on a fresh conversion: for every one of the 40 NVFP4 tensors, decode `{d[4], qs[32]}` blocks and compare against `layout.hf_dequant`-style reference × `.scale` sidecar → **0 differing values** on sampled experts/rows (±0.0 sign differences reported separately).
2. Cross-check `blk.0` (and one later layer) byte-for-byte against unsloth's file → byte-identical.
3. Full conversion to `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf`, tee to `logs/phase2b-convert.log`; report the final `ls -la` size (expect ≈18 GB) and the type inventory (expect **40 NVFP4 + 8 MXFP8 + 48 F32 `.scale`**, rest BF16/F32).
4. `build-cuda/bin/llama-print` or `llama-cli --stdout` **on CPU** confirms the file loads and reports tensor types (GPU is off-limits).

**Hard rules:** write `docs/mxfp8/11-phase2b-nvfp4.md` within your first 5 tool calls and keep updating it; **25-minute budget** — at 20 minutes stop and finalize; never delete any file or log; no docker / GPU / service commands; keep tensors lazy (61 GB RAM, do not materialise all experts — the dequant path streams in 128-row chunks for a reason); default converter behaviour (no `--native-experts`) must stay byte-for-byte unchanged.
