# 15 — Debug + fix: Inkling-specific native-expert execution fault

Status: **IN PROGRESS** — brief `briefs/O-nvfp4-block-convention.md`: settle whether ggml's C
NVFP4 decoder reads our repack the way the converter writes it, using ggml's *compiled* decoder
(not a python port) on the real file bytes.

Related: `14-native-numeric-bug.md` (device matrix that produced this scope), brief
`briefs/N-inkling-native-expert-fix.md`.

## 0. Control evidence (measured by parent, not re-derived here)

| model | native types | backend | result |
|---|---|---|---|
| `esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/*-MID-HIGH.gguf` | 448 NVFP4, 168 `.scale` **unfused** | CUDA | **coherent** (`logs/nvfp4-control.log`) |
| `ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf` | none | CUDA + CPU | **correct** `A: 68 B: 64 C: 72 …` |
| `timlikesai/Inkling-2-Mini-NATIVE.gguf` (ours) | 40 NVFP4 **fused** `ffn_gate_up_exps` + 40 `.scale`, 8 MXFP8 | CUDA | wrong `4.0 17.0 17.0 …` |
| same | same | CPU `-ngl 0` | wrong **differently** `17 times 4. 4. 5. 5. 6. 7. 18.` |

Device matrix (`logs/m2-device-matrix.log`): MXFP8 on CPU == all-GPU output byte-identical ->
MXFP8 not implicated. NVFP4 experts on CPU still wrong. CPU and CUDA disagree with each other on
identical bytes.

So: native NVFP4 works in general. The fault is specific to Inkling's expert path.

## 1. Inkling-specific features, and the hypothesis attached to each

| # | feature | hypothesis | probe |
|---|---|---|---|
| H-A | MoE **cache / slots** path (`moe_cache->get_experts()`, `GGML_OP_MUL_MAT_ID` + `slots`) | the cache repacks expert rows into a contiguous buffer with a type-assumption that is wrong for a native (16-block-subbyte / 33-byte MXFP8) tensor, and the bug is backend-dependent because the copy/run kernels differ | toggle the cache off (`common/arg.cpp`), or `--no-moe-fuse`-style flag; compare output |
| H-B | **Fused** `ffn_gate_up_exps` + single per-expert `.scale` -> `layer.ffn_up_exps_s` | the `.scale` sidecar is applied at a place that is only correct for the unfused layout (e.g. after the gate/up split, or broadcast over the wrong half of the fused row) | run the NATIVE file with gate/up **unfused** or with the scale folded into the weights; compare |
| H-C | **Clamped SituGLU** | scale applied after the clamp (or the clamp fed the wrong pre-activation) | graph read + numeric probe |

Working preference on entry: **H-A**, because it is the only hypothesis that naturally explains
CPU and CUDA being wrong *differently* on byte-identical weights.

## 1b. The decode contract, stated side by side (code reading, HEAD `295d7807f`)

Conventions written down so the experiment has something to falsify. All readers are of the
64-element super-block `{ uint8_t d[4]; uint8_t qs[32]; }` (`ggml-common.h:226-231`).

| reader | sub-block -> scale byte | byte j of a sub-block -> elements | scale value |
|---|---|---|---|
| C `dequantize_row_nvfp4` (`ggml-quants.c:646`) | `d[s]` covers elements `16s..16s+15` | `qs[s*8+j]`: low nibble -> elem `16s+j`, high -> elem `16s+8+j` | `ggml_ue4m3_to_fp32` = e4m3 * 0.5, `kvalues_fp4` = 2 x E2M1 |
| C `ggml_vec_dot_nvfp4_q8_0_generic` (`ggml-cpu/quants.c:339`) | same, `q8_block = s/2`, `q8_off = (s%2)*16` | same byte, lo -> `q8_off+j`, hi -> `q8_off+8+j` | same |
| C x86 SIMD `ggml_vec_dot_nvfp4_q8_0` (`arch/x86/quants.c:1086`) | `d[0..1]` with `y[2ib]`, `d[2..3]` with `y[2ib+1]` | `unpacklo/hi_epi64` of the (lo,hi) nibble LUTs restores element order `16s..16s+15` | `GGML_CPU_UE4M3_TO_FP32` = `ggml_table_f32_ue4m3` built from `ggml_ue4m3_to_fp32` (`ggml-cpu.c:3919`) |
| CUDA MMVQ `vec_dot_nvfp4_q8_1` (`vecdotq.cuh:334`) | `is = iqs0>>1` -> `d[is]`; `bq8_1 + (is>>1)`, `i8 = (is&1)*4` — valid because `QK8_1` is **32** here (`ggml-common.h:269`), 2 sub-blocks per q8_1 block | `v0.x/v0.y` (lo/hi nibbles of sub-block bytes 0..3) dot `qs[i8+0]/qs[i8+2]` = elements 0..3 / 8..11 | `ggml_cuda_ue4m3_to_fp32` = e4m3 / 2 |
| converter `_pack_nvfp4_chunk` (`conversion/inkling2.py:83`) | `raw[..:4] = scales[4b+s]` for elements `64b+16s..` | `raw[4+s*8+j] = codes[j] | codes[8+j] << 4` | checkpoint e4m3 bits verbatim (sign bit asserted clear) |
| upstream reference `_nvfp4_pack` (`conversion/base.py:1312`) | `d_ue.reshape(out, n_super, 4)` | `vals[:,:,:8] | (vals[:,:,8:] << 4)` | `& 0x7F` sign strip |

All six agree on the two things the brief listed as suspects: the nibble pairing (`j`, `j+8`) and
which 16 elements `d[s]` covers. The checkpoint's own semantics (`_dequant_expert_chunk:45-47`)
is the *other* pairing (`2i`, `2i+1`), which both packers explicitly convert out of. So on paper
the repack convention is not the fault — which is exactly why this needs the compiled decoder and
not a seventh python port.

## 1c. Decisive experiment: ggml's own C decode of the file bytes

`scripts/inkling_native/probe_c_decode.py`: loads `build-cuda/bin/libggml-base.so` with ctypes and
calls the exported `dequantize_row_nvfp4` (symbol present — `logs/nm-ggml-base-nvfp4.txt`) on the
real bytes of `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf`, multiplies by the `.scale` sidecar
read from the same file, and compares elementwise against the checkpoint weights recomputed from
`~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*`.

_TBD._

## 2. Measurements

_TBD as they land._

## 3. Root cause

_TBD._

## 4. Fix

_TBD._

## 5. Gate

| check | result |
|---|---|
| NATIVE CUDA auto-fit `17 times 4` -> `68` | pending |
| NATIVE agrees with IQ4_NL on 4 prompts | pending |
| CPU vs CUDA agree | pending |
| `test-backend-ops -b CUDA0 -p mxfp8` 237/237 | pending |
| `test-backend-ops -b CPU -p mxfp8` 306/306 | pending |
| prefill + decode t/s vs G0 (589.72 / 21.94 / 32.00) | pending |
