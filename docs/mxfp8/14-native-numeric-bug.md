# 14 — Debug: bit-exact NATIVE Inkling file is numerically WORSE than lossy IQ4_NL

Status: **IN PROGRESS** — localizing the CPU/CUDA divergence (tool call ~12 for the first measurement).

## 0. Summary / TL;DR

*To be finalized with the measurements below.*

Working conclusion at the point of writing: the file's bytes are not the fault (the value-level
harness `scripts/inkling_native/verify_nvfp4_values.py` reproduces the checkpoint to 0.0 relative
error through ggml's own decode convention on both NVFP4 and MXFP8), so the divergence is in a
runtime path, not in the converter. The device matrix (§3.1) decides which tensor group and which
backend.

## 0b. Summary — parent update, 01:35

**The file is not the bug and neither is MXFP8.** Three-way value checks put the file's expert decode at exactly the checkpoint (`0.0` relative error), and the §3.1 device matrix shows native-**NVFP4** experts produce wrong text on **both** CPU and CUDA while the lossy IQ4_NL file is correct on both. Moving MXFP8 to CPU leaves output byte-identical to all-GPU, so the type this project added is not implicated. Resolution path is in §4.


See `logs/m2-device-matrix.log` (4 prompts x 7 device configurations, same binary, same seed,
temp 0). Baselines from the earlier runs:

| file | backend | output |
|---|---|---|
| NATIVE | CPU `-ngl 0` | `Q: What is 17 times 4? A: 17 times 4. 4. 5. 5. 6. 7. 18.` |
| NATIVE | CUDA auto | `Q: What is 17 times 4? A: 4.0 17.0 17.0 4.0 17.0 17.0` |
| IQ4_NL | CUDA auto | `Q: What is 17 times 4? A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` |

## 2. Code reading: the decode contract, per backend

Facts collected by reading the code (all verified against the source, line numbers from HEAD
`295d7807f`):

- `block_nvfp4` is `{ uint8_t d[4]; uint8_t qs[32]; }` for 64 values, one UE4M3 sub-scale per
  16 values (`ggml/src/ggml-common.h:226-231`). `block_mxfp8` is `{ uint8_t e; uint8_t qs[32]; }`
  (`:233-238`).
- **Both** host decoders return `ue4m3 * 0.5` to cancel the doubled `kvalues_mxfp4` table:
  CPU `ggml_ue4m3_to_fp32` (`ggml/src/ggml-impl.h:521-535`) and CUDA
  `ggml_cuda_ue4m3_to_fp32` (`ggml/src/ggml-cuda/common.cuh:908-...`, `... / 2` on every branch).
  So H2 (an asymmetric 0.5) is **not** the divergence at the decoder level.
- CPU NVFP4 paths: `dequantize_row_nvfp4` (`ggml/src/ggml-quants.c:646`) and the dot
  `ggml_vec_dot_nvfp4_q8_0_generic` (`ggml/src/ggml-cpu/quants.c:339`), which uses a **Q8_0**
  activation and reads the sub-scale with the same `s*(qk_sub/2)+j` nibble interleave as the
  dequantize. No `vec_dot_nvfp4_*` exists in `ggml/src/ggml-cpu/llamafile/` or `vec.cpp`: NVFP4
  is served by the generic traits only.
- CUDA NVFP4 paths: MMVF/dequantize (`convert.cu:384-420`), MMVQ
  `vec_dot_nvfp4_q8_1` (`vecdotq.cuh:334-361`, a **Q8_1** activation), MMQ
  `ggml_cuda_mmq_load_tiles_nvfp4` (`mmq-load-tiles.cuh:1726-1780`), and the Blackwell native
  FP4 MMA in `mma.cuh:1295` (`kind::mxf4nvf4.block_scale.scale_vec::4X ... .ue4m3`).
- `llm_build_base::build_lora_mm_id` (`src/llama-graph.cpp:1884-1932`) is the only place the
  per-expert F32 `.scale` sidecar is applied: reshape to `[1, n_expert, 1]`, repeat over tokens,
  `get_rows` with the selected ids, then one `MUL` on the fused `gate_up` result. It is
  backend-independent graph code.
- Immediately above it: `if (w->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(res, GGML_PREC_BF16);`
  (`src/llama-graph.cpp:1902-1904`). That is a **CUDA-only** accumulation-precision hint for the
  native-FP4 tensor-core path; the CPU ignores it. Prime suspect for a one-sided numeric change.

## 3. Discriminating experiments

### 3.1 Device matrix (which tensor group, which backend)

Same NATIVE file, one tensor group moved to the other backend with `-ot`/`-cmoe`:

| tag | config | verdict |
|---|---|---|
| native-cpu | `-ngl 0` | (see log) |
| native-gpu | default auto-fit | |
| native-gpu-moe-on-cpu | all `ffn_*_exps` on CPU, rest CUDA | |
| native-gpu-mxfp8-on-cpu | `blk.20-23.ffn_*exps` on CPU | |
| native-gpu-nvfp4-on-cpu | `blk.0-19.ffn_*exps` on CPU | |
| iq4nl-cpu / iq4nl-gpu | lossy reference | |

Script: `scripts/inkling_native/probe_device_matrix.sh`, log: `logs/m2-device-matrix.log`.

### 3.2 Op-level numeric probe (which function)

TODO - MUL_MAT_ID at the real Inkling expert shapes, CPU vs CUDA vs a float reference computed
from the file's own bytes.

## 4. Root cause — LOCALIZED (parent, from §3.1 device matrix)

| tag | output | verdict |
|---|---|---|
| native-cpu | `17 times 4. 4. 5. 5. 6. 7. 18.` | wrong |
| native-gpu | `4.0 17.0 17.0 4.0 17.0 17.0` | wrong |
| native-gpu-moe-on-cpu | `17 times 18, 19, 20, 21, 22, 23, 24,` | wrong |
| **native-gpu-mxfp8-on-cpu** | `4.0 17.0 17.0 4.0 17.0 17.0` — **identical to native-gpu** | MXFP8 placement irrelevant |
| **native-gpu-nvfp4-on-cpu** | `17 times 18, 19, 20, 21, …` | NVFP4 wrong on CPU **too** |
| iq4nl-cpu / iq4nl-gpu | `68 B: 64 C: 72 …` | correct |

**Fault = native-NVFP4 expert execution, on both backends.** MXFP8 (what this project added), the file bytes (§6), the non-expert tensors (bit-exact BF16, max abs diff 0) and the CUDA Tier A path (237/237) are all exonerated. `GGML_PREC_BF16` cannot be the whole story — CPU ignores that hint and is still wrong.

**Open contradiction:** the harness proves `decode(bytes) x sidecar == checkpoint` at 0.0 error, yet the runtime is wrong on both backends. Both can be true only if the harness shares an assumption with the *converter* that the runtime decode does not — a **convention** difference in `block_nvfp4` (which 16 elements `d[s]` covers, or the nibble arrangement inside those 8 bytes), not a value/ordering bug. The public unsloth file uses the same encoding and has never been numerically validated either.

**Two experiments decide ownership, in order:**
1. Round-trip one real expert through the **actual C decoder** (a 1-tensor GGUF + `to_float`, or 10 lines of ggml C) instead of a python mirror. If ggml's `dequantize_row_nvfp4` disagrees with the python model, the repack convention is ours to fix (and unsloth's too).
2. Run any upstream-supported native-NVFP4 checkpoint (Nemotron/Qwen NVFP4). If native NVFP4 is fine *outside* Inkling, the culprit is Inkling's fused gate_up + `.scale` + SituGLU combination in upstream graph code.

## 5. Fix (or upstream defect)

TODO — pending §4 experiment 1, which decides whether the repack convention is ours or the MoE scale/activation ordering is upstream.

## 6. Corrections to my brief

- The `.scale` sidecar is **not** ignored by either backend: `build_lora_mm_id` applies it as a
  `get_rows` + `MUL` on the fused result, and `verify_nvfp4_values.py` (log
  `logs/m-valuecheck-fixed.log`) shows `decode(bytes) x sidecar == checkpoint` at 0.0 relative
  error for `blk.0`/`blk.17` NVFP4 and `blk.20`/`blk.23` MXFP8. The `no-0.5` control in the same
  run is exactly 2x wrong, which proves the harness is sensitive to a factor of two.
- H2 (an asymmetric 0.5 between CPU and CUDA NVFP4) is refuted at the decoder level: both
  `ggml_ue4m3_to_fp32` and `ggml_cuda_ue4m3_to_fp32` divide by 2.
- H3 (sidecar applied before the activation being wrong) is refuted by the math: the checkpoint
  dequantizes *both* halves with the same per-expert factor (`conversion/inkling2.py:51`), so
  scaling the fused `[gate; up]` result before the split is equivalent.

## 7. Open questions

- Does `GGML_PREC_BF16` on the NVFP4 `MUL_MAT_ID` output make CUDA take Blackwell native FP4
  (W4A4) math, and is that enabled for a *loaded* NVFP4 tensor or only for on-the-fly-quantized
  ones?
