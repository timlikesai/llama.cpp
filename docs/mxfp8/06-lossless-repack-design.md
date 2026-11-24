# 06 — Lossless expert repack: byte contract and measured evidence

Deliverable of brief E. Companion code: `scripts/inkling_native/layout.py` (layout
reference model) and `scripts/inkling_native/check_layout.py` (runnable harness).
Last harness run: `scripts/inkling_native/check_layout.log` — **26/26 PASS, 0 failures,
0 warnings** (`--quick`: experts 0/64/127, layers 0, 19, 20, 23, 128-row streamed windows).

Measurements are from the local checkpoint
`~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/model-00001-of-00001.safetensors`
(19,526,019,168 B, 1244 tensors) and
`/home/tim/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf`.
Repo claims carry `file:line` against `mxfp8-weights` = master `29b52d398`.

## 1. Verdict

* **NVFP4 (layers 0-19): the shipped public GGUF is already a bit-exact lossless repack.**
  `blk.0.ffn_gate_up_exps.weight` (NVFP4, dims `(2304, 2304, 128)`) is **byte-identical** to
  what `layout.pack_nvfp4_blocks()` produces from the HF tensors — 2,985,984/2,985,984 bytes
  for expert 0, with **0** differing scale bytes and **0** differing payload bytes.
  `blk.0.ffn_down_exps.weight` likewise 1,492,992/1,492,992. No new ggml type is needed for NVFP4.
* **MXFP8 (layers 20-23) needs only `GGML_TYPE_MXFP8` = 33 B per 32 elements**; the repack is a
  pure bit move and is **bitwise** identical to master's own dequant reference (check b: 0
  differing values out of 47,775,744 checked).
* **Both repacks are byte-count neutral against the checkpoint** (NVFP4 10,935.0 MiB, MXFP8
  4,009.5 MiB in both formats) — 36 B/64 elems == 8 payload + 1 scale byte per 16; 33 B/32 == 1
  payload + 1/32 scale byte.
* **The NVFP4 per-expert global factor is not a power of two** (0/128 experts, 128 distinct
  factors per tensor), so it cannot be folded into the e4m3 sub-block scales. It must stay an
  F32 `.scale` sidecar — which is exactly what the shipped file does: its `.scale` values are
  **bitwise equal to `amax / outer_scale_denominator`** for 128/128 experts.
* `outer_scale_denominator` is **2688.0 = 6 x 448** (the NVFP4 recipe constant) for every tensor
  measured; `w13_weight_amax` for blk.0 spans [0.105285123, 0.331052512].

## 2. Block layouts (the contract)

```
block_mxfp4   17 B / 32 elems   ggml/src/ggml-common.h:214-219   (QK_MXFP4  = 32)
  byte  0        : e   ue8m0 scale, decoded as 0.5 * 2^(e-127)
  bytes 1..16    : qs  byte j = element j (low nibble) | element j+16 (high nibble)

block_nvfp4   36 B / 64 elems   ggml/src/ggml-common.h:221-227   (QK_NVFP4  = 64, sub 16)
  bytes  0..3    : d[4]  ue4m3 sub-block scales (e4m3 bits with the sign stripped), 1 per 16 elems
  bytes  4..35   : qs    32 packed code bytes; inside sub-block s, byte j =
                        element s*16 + j (low nibble) | element s*16 + 8 + j (high nibble)

block_mxfp8   33 B / 32 elems   PROPOSED, the analogue of block_mxfp4
  byte  0        : e   ue8m0 scale byte, copied verbatim from the checkpoint
  bytes 1..32    : qs  32 e4m3 payload bytes, element order, copied verbatim
```

Decode rules the harness proved (all against `conversion/inkling2.py:33-65`):

| format | ggml decode | equals checkpoint dequant |
|---|---|---|
| MXFP8 | `e4m3(qs[j]) * 2^(e-127)` | **bitwise** (check b) |
| NVFP4 | `kvalues_mxfp4[code] * ue4m3_half(d[s])` = `(2*E2M1) * (0.5*e4m3)` | numerically (check c) |

**MXFP8 must not reuse `GGML_E8M0_TO_FP32_HALF`** (`ggml/src/ggml-impl.h:496-514`). The half
scale exists to cancel the doubled `kvalues_mxfp4` table; an e4m3 payload is not doubled, so the
HALF decode is exactly 2x wrong. The harness asserts this negative case ("shortcut would be wrong
here = True" in every `b.*` line). Use `ggml_e8m0_to_fp32` (`ggml-impl.h:469-492`).

### Nibble order: the trap that cost an hour

The checkpoint (compressed-tensors / ModelOpt) and ggml use **different** 4-bit orders:

* checkpoint: byte `i` holds element `2i` (low nibble), element `2i+1` (high nibble) — proven by
  `conversion/inkling2.py:42-43` (`_E2M1[packed & 15] -> values[:, 0::2]`).
* ggml: within a **block** (32 elems for mxfp4, 16 for a nvfp4 sub-block), byte `j` holds
  element `j` (low) and element `j + block/2` (high) — `conversion/base.py:1262` and `:1325`.

The re-interleave is therefore **per block, never per row**: `layout.elements_to_ggml_nibbles(v,
group)` takes the group size explicitly. Applying it to a whole row silently mixes elements from
different scale groups and produces ~93 % wrong values while still looking self-consistent.

## 3. Row order for fused gate/up (was UNVERIFIED in the brief; now confirmed)

`conversion/inkling2.py:170-176` walks `half in range(2)` with
`row_slice = slice(2*start + half, 2*stop + half, 2)`, and the non-packed path at
`conversion/inkling2.py:255-258` does
`reshape(E, rows//2, 2, K).transpose(1, 2)`. Both say: the checkpoint interleaves gate/up
(row `2j` = gate `j`, row `2j+1` = up `j`), the GGUF stores them **blocked** [gate; up].

GGUF row `r` (0-based within an expert, `half = rows/2`) takes checkpoint row

```
hf_row(r) = 2 * (r % half) + (r // half)         # layout.w13_hf_row_of_gguf_row()
```

so GGUF rows `[0, half)` = checkpoint even rows, `[half, rows)` = checkpoint odd rows.
Scale rows are permuted by the **same** index vector (scales are 1 x 16 / 1 x 32 along K only, so
`scale.shape[1] == weight.shape[1]` and one row permutation covers both).

Measured on `blk.0` (2,985,984 NVFP4 bytes per expert, all 2304 rows, expert 0):

| row order | gate_up bytes equal | down bytes equal |
|---|---|---|
| identity | 51,520 (1.7 %) | **1,492,992 (100 %)** |
| `hf_row(r)` (gate/up-permuted) | **2,985,984 (100 %)** | n/a |
| alt-perm (`r//2 + (r%2)*half`) | 51,274 (1.7 %) | n/a |

`ffn_down_exps` uses identity, as expected.

## 4. NVFP4 global factor: keep the `.scale` sidecar

Master's NVFP4 convention: `ModelBase._repack_nvfp4` (`conversion/base.py:1334-1343`) writes the
packed tensor plus a sidecar `.scale` (`_write_scale_tensor` `conversion/base.py:335-339`,
`_write_scales_tensor` `:341-345` — omitted when the factor is trivially 1.0).

For Inkling the factor is per expert: `factor[e] = float32(w*_weight_amax[e]) /
float32(w*_weight_outer_scale_denominator)` (`conversion/inkling2.py:46-53`). Measured:

| tensor | amax range | denom | factor range | power-of-two experts |
|---|---|---|---|---|
| L0.w13 | 0.105285123 .. 0.331052512 | 2688.0 | 3.91685717e-05 .. 0.000123159421 | 0/128 |
| L0.w2 | 0.152233139 .. 0.444106638 | 2688.0 | 5.66343515e-05 .. 0.00016521824 | 0/128 |
| L19.w13 | 0.101429179 .. 0.400198668 | 2688.0 | 3.77340693e-05 .. 0.000148883439 | 0/128 |
| L19.w2 | 0.152584225 .. 0.643172026 | 2688.0 | 5.67649658e-05 .. 0.000239275309 | 0/128 |

Decision: **emit `blk.{0..19}.ffn_{gate_up,down}_exps.scale` as F32 `[128]` holding
`amax/2688.0`**, computed in float32 exactly as `conversion/inkling2.py:53` does
(`np.float32(a) / np.float32(d)`, one multiply at the end). This is byte-identical to what the
shipped unsloth file stores (128/128 experts, max rel diff 0.0). Cost: 80 tensors x 512 B = 40 KiB.
Do **not** try the sign-strip/renormalise trick to absorb the factor into `d[]`: with a
non-power-of-two factor that would requantise the e4m3 scales (lossy).

Sign-strip is safe for a different reason: the checkpoint never sets the sign bit on an NVFP4
scale. Check a measured **0** sign-bit bytes and **0** NaN (`0x7F`) bytes in 34,560 rows x
{144, 72} sub-blocks (layers 0 and 19), byte range 88..126 -> exponent field 11..15, i.e. e4m3
values 2^-4 .. 1.75. So `scale & 0x7F` (`conversion/base.py:1324`) is a no-op on real data.

MXFP8 layers need **no** sidecar at all: no `amax`/`denominator` companions exist
(`conversion/inkling2.py:138-139` asserts their absence), so the repack is fully self-contained.

## 5. Measured invariants

| check | result | evidence |
|---|---|---|
| a. ue8m0 scales never 255; e4m3 payloads finite | PASS | L20/L23 w13+w2: scale bytes 113..118 = exponents -14..-9, byte-255 count 0, NaN scale bytes 0, NaN payload bytes 0 over 6912 rows x {2304, 1152} |
| a. NVFP4 e4m3 scales: no NaN, no sign bit | PASS | 8 report lines, sign-bit bytes 0, NaN bytes 0 (see §4) |
| b. MXFP8 `dequant(unpack(pack(x))) == checkpoint dequant` | PASS bitwise | 0 differing of 47,775,744 values (layers 20, 23; w13, w2; experts 0/64/127); byte placement exact |
| c. NVFP4 round-trip incl. `amax/denom` | PASS numerically | 0 differing of 47,775,744; **`amax/denom` is never a power of two** |
| c. NVFP4 round-trip bitwise | not exact, by design | 547,754 / 272,868 / 543,558 / 273,435 values differ only in the **sign of zero** — `kvalues_mxfp4` is `int8` and cannot carry `-0.0`, while `E2M1[8] = -0.0`. Stored bytes are identical; only ggml's decoder prints `+0.0`. Numerically irrelevant. |
| d. unsloth GGUF NVFP4 bytes | PASS | §1, §2, §3 above |
| e. size budget | measured | §6 |
| f. `--quick` sampling / `--full` sweep | `--quick` implemented and run; **`--full` not run** (budget) | see §10 |

## 6. Size budget (MiB, text backbone only; vision and mtp skipped as master does at `conversion/inkling2.py:188`)

Checkpoint split, from the safetensors header:

| group | MiB |
|---|---|
| NVFP4 payloads (20 layers x 2 tensors) | 9720.0 |
| NVFP4 e4m3 scales | 1215.0 |
| NVFP4 `amax` + `outer_scale_denominator` F32 | 0.02 (21,888 B) |
| MXFP8 e4m3 payloads (4 layers x 2 tensors) | 3888.0 |
| MXFP8 ue8m0 scales | 121.5 |
| other `model.llm.*` tensors (BF16/F32) | 3069.8 |
| non-text tensors in the shard (vision) | 607.0 |

| GGUF body | MiB | note |
|---|---|---|
| **lossless repack** (NVFP4 36/64 + MXFP8 33/32 + other as-is + 40 KiB sidecars) | **18014.3** | 17.59 GiB; shipped files are 17.63 GB |
| NVFP4 experts as `block_nvfp4` | 10935.0 | == checkpoint bytes (size-neutral) |
| MXFP8 experts as `block_mxfp8` | 4009.5 | == checkpoint bytes (size-neutral) |
| MXFP8 experts as Q8_0 (34/32) | 4131.0 | +121.5 MiB and **lossy**: a 32-elem block spans up to 2^18 in magnitude, which int8 x f16 cannot hold exactly. What the shipped UD file does for blk.20-23. |
| MXFP8 experts as flat F8_E4M3 (1 B/elem) | 3888.0 | -121.5 MiB, lossy (drops the ue8m0 scales) |
| MXFP8 experts as BF16 | 7776.0 | +3766.5, lossy |
| MXFP8 experts requantised to MXFP4 (17/32) | 2065.5 | -1944.0, lossy |
| NVFP4 experts requantised to Q8_0 | 20655.0 | +9720 |
| all experts BF16 | 46656.0 | does not fit |

KV headroom. 24 layers, `local_layer_ids` = 20 entries -> 20 SWA layers capped at
`sliding_window_size` 1024, 4 full layers (5, 11, 17, 23); `num_key_value_heads` 8,
`head_dim`/`swa_head_dim` 128 (`config.json` text_config). Per token per layer
= 2 (k,v) x 8 x 128 x bytes = fp8 2048 B, q8_0 2176 B, f16 4096 B.

```
KV_bytes(N) = (20 * min(N, 1024) + 4 * N) * 2048      # fp8, SWA capped
  N= 32768: (20480 +  131072) * 2048 =  296.0 MiB   (uncapped 1536.0)
  N= 65536: (20480 +  262144) * 2048 =  552.0 MiB   (uncapped 3072.0)
  N=131072: (20480 +  524288) * 2048 = 1064.0 MiB   (uncapped 6144.0)
  N=262144: (20480 + 1048576) * 2048 = 2088.0 MiB   (uncapped 12288.0)
```

Against 2 x 16384 MiB = 32768 MiB of VRAM:

| ctx | fp8 KV (SWA capped) | weights + fp8 KV | slack of 32768 MiB | weights + f16 KV, no SWA cap | verdict |
|---|---|---|---|---|---|
| 32k | 296.0 | 18310.3 | 14457.7 | 21086.3 | FITS both |
| 64k | 552.0 | 18566.3 | 14201.7 | 24158.3 | FITS both |
| 128k | 1064.0 | 19078.3 | 13689.7 | 30302.3 | FITS both (2465.7 left) |
| 262k | 2088.0 | 20102.3 | 12665.7 | 42590.3 | uncapped **OVER by 9822.3** |

The SWA cap is what makes 262k fit: with `--swa-full` (or a build that keeps all 24 layers
dense) f16 KV at 262k alone is 24576 MiB and the model no longer fits on 2 x 16 GB.
These are weight+KV sums only — compute graphs, CUDA context and the MoE scratch are on top.

## 7. Target GGUF tensor list (names, types, shapes)

Dims are in GGUF order (fastest-varying first), i.e. `(K, rows, experts)`; GGUF tensor-info
dimension fields are **uint64** (`gguf-py/gguf/gguf_reader.py:288`), offsets uint64, type uint32.
Types per `gguf-py/gguf/constants.py:7305-7342`: `F32=0`, `Q8_0=8`, `IQ4_NL=20`, `BF16=30`,
`MXFP4=39`, `NVFP4=40`, `F8_E4M3=43`, **`MXFP8=44` (already reserved in the Python enum)**.

| tensor name | type | dims | bytes | count |
|---|---|---|---|---|
| `blk.{0..19}.ffn_gate_up_exps.weight` | NVFP4 (40) | (2304, 2304, 128) | 382,205,952 ea | 20 |
| `blk.{0..19}.ffn_gate_up_exps.scale` | F32 (0) | (128,) | 512 ea | 20 |
| `blk.{0..19}.ffn_down_exps.weight` | NVFP4 (40) | (1152, 2304, 128) | 191,102,976 ea | 20 |
| `blk.{0..19}.ffn_down_exps.scale` | F32 (0) | (128,) | 512 ea | 20 |
| `blk.{20..23}.ffn_gate_up_exps.weight` | MXFP8 (44) | (2304, 2304, 128) | 700,710,912 ea (2376 B/row) | 4 |
| `blk.{20..23}.ffn_down_exps.weight` | MXFP8 (44) | (1152, 2304, 128) | 350,355,456 ea (1188 B/row) | 4 |
| everything else | as master emits today | - | 3069.8 MiB total | - |

The 40 NVFP4 weight + 40 `.scale` pairs above are exactly what the shipped unsloth file already
contains (verified for blk.0; the byte-identity of the other 19 layers is §10 item 1).
No `.scale` for the MXFP8 layers.

## 8. What must change

ggml side (briefs B/G own this; listed for completeness):

1. `ggml/src/ggml-common.h` near `:214` — add `#define QK_MXFP8 32` and
   `typedef struct { uint8_t e; uint8_t qs[QK_MXFP8]; } block_mxfp8;` + the `static_assert`
   (33 B), mirroring `block_mxfp4` at `:214-219`.
2. `ggml/src/ggml-quants.c` next to `dequantize_row_mxfp4` (`:572-590`) — `dequantize_row_mxfp8`
   using `ggml_e8m0_to_fp32` (`ggml/src/ggml-impl.h:469-492`), **not** `..._half` (§2).
3. Reject `e == 255` on load (ue8m0 NaN), as `validate_e_e8m0` does for mxfp4
   (`ggml/src/ggml-quants.c:5461`, used at `:5489`).
4. `ggml_type_size` / `GGML_QUANT_SIZES` row `MXFP8 -> {33, 32}`, `ggml_is_quantized`,
   `ggml_compute_rows_mv`/`from_float` cases, `test_quantize_support_fn` in the C++ tests.

converter side, `conversion/inkling2.py` (this is brief E's "what must change" list):

5. `Inkling2Model.dequant_model` (`:116-183`) currently dequantises every expert tensor to F32 via
   `gguf.LazyChunkedTensor` (`:176`) and deletes the companions (`:178-179`), which forces
   requantisation downstream. Add a native passthrough branch taken when the output may keep the
   source format: skip the F32 materialisation for `scheme in ("nvfp4_nearest", "mxfp8_nearest")`
   and hand the packed bytes through instead. `_dequant_expert_chunk` (`:33-65`) stays as the
   fallback (and is the reference this harness mirrors).
6. The shape/validation block (`:145-160`) is reusable as-is — it already proves `block ==
   (1, 16)` for NVFP4 and `(1, 32)` for MXFP8 from the stored shapes and warns when
   `hf_quant_config.json` disagrees (it does for MXFP8: `weight_block_shape: [32, 32]`).
   The native path must trust the stored shapes, like master does.
7. `Inkling2Model.modify_tensors` (`:250-274`): keep the name mapping (`.w13_weight` ->
   `.gate_up_proj` -> `FFN_GATE_UP_EXPS`, `.w2_weight` -> `.down_proj` -> `FFN_DOWN_EXPS`), but
   yield a uint8 byte tensor of shape `(experts, rows, bytes_per_row)` instead of the lazy F32
   tensor, and apply the §3 permutation to weight and scale rows for `w13` only.
8. New `_mxfp8_expert_tensor()` next to `_mxfp4_expert_tensor` (`conversion/base.py:1267-1295`)
   and a `pack_mxfp8_blocks()` next to `repack_mxfp4_blocks` (`:1230-1265`), plus
   `_nvfp4_pack` reuse (`:1309-1332`) for the NVFP4 path — all lazy per expert, because
   `gguf_writer` holds every added tensor until the final write (the comment at
   `conversion/base.py:1271-1274` is exactly this warning).
9. Write the packed bytes with `gguf_writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.MXFP8)`
   following `_repack_nvfp4` (`conversion/base.py:1334-1343`), and the per-expert factor with
   `_write_scales_tensor` (`conversion/base.py:341-345`) as `.scale` (F32 `[128]`).
10. `Inkling2Model.tensor_force_quant` (`:245-248`) must not touch the expert tensors (it already
    only forces F32 for conv/clip/res-query/rel-bias); confirm no `-ot`/`--tensor-type` override
    requantises them, and that `.scale` sidecars are not treated as weights.

## 9. Corrections to brief E

* Brief: "HF packs element j in low nibble and j+16 in high nibble" (describing the **source**).
  That is the **destination** rule. The source is element `2i` low / `2i+1` high
  (`conversion/inkling2.py:42-43`). The re-interleave must be per 32-elem (mxfp4) or per 16-elem
  (nvfp4 sub-block) group.
* Brief: "AMAX/denom ... if yes, a sign-strip/renormalize trick is exact". Answer: not a power of
  two for any expert, and the sign bit is never set anyway, so the two questions are independent
  — the sign strip is exact regardless (§4).
* Brief: `GGMLQuantizationType` list is right, and additionally `MXFP8 = 44` already exists in
  `gguf-py/gguf/constants.py:7342` (so the id is picked; the C++ enum/backend work is not).
* Brief: "read with the repo's `gguf-py`" — **gguf-py cannot be imported in this venv**:
  `gguf-py/gguf/__init__.py:9` -> `metadata.py:5` -> `import yaml`, and pyyaml is not installed in
  `.venv`. `check_layout.py` therefore carries its own minimal GGUF header parser (auto-detects
  the uint64 dimension field; `gguf-py/gguf/gguf_reader.py:288` confirms uint64 is canonical).
  Header of this file is ~11.9 MB (200,058 tokens + 424,413 merges), so a naive few-MB header
  window silently desynchronises.
* Brief: peak RSS. The harness mmaps the checkpoint and reads only the windows it needs; the GGUF
  cross-check maps lazily and touches one expert per tensor. Nothing approaches the 6 GB ceiling.
* Not a correction but a surprise worth flagging: `kvalues_mxfp4` (int8) makes ggml print `+0.0`
  where the checkpoint holds `-0.0`. Bytes are preserved; only the dequantised sign of zero is not.

## 10. Next steps

1. Re-run check (d) across **all 20 NVFP4 layers** and all 128 experts (today: blk.0, expert 0,
   all 2304 rows) to certify "the shipped NVFP4 experts are lossless" as a blanket statement, and
   the same for the `.scale` sidecars of every layer.
2. Run `--full` (all experts, all quantised layers, streamed; estimated a few minutes and
   ~14 GB of reads) and attach its log; `--quick` is what §5 reports.
3. Add the same harness in `convert_hf_to_gguf.py`-adjacent form as a repo test once MXFP8 lands
   (the C++ side should assert `dequantize_row_mxfp8` agrees with `layout.dequant_mxfp8_blocks`).
4. Decide the UX for the native path: a `--preserve-expert-format`-style flag vs. auto-detect on
   `nvfp4_nearest`/`mxfp8_nearest`, and whether the vision tower / mtp stay skipped
   (`conversion/inkling2.py:188`) for this build.
5. Verify the claim (from the brief, not re-measured here) that the unsloth file ships
   blk.20-23 experts as Q8_0 — the harness could assert it in one line by reading the tensor types
   it already parses.
6. MXFP4 requantisation of MXFP8 (-1944 MiB) is the only option that buys real space; if it is
   pursued, brief E's harness is the place to measure its error against the bit-exact baseline.

## 11. Running the harness

```sh
cd /home/tim/code/services/repos/llama.cpp
.venv/bin/python scripts/inkling_native/check_layout.py --quick | tee scripts/inkling_native/check_layout.log
# --full streams every expert of every quantised layer; --no-gguf skips the unsloth cross-check
```

`.venv/bin/python` (numpy 2.5.3) is required; the `python3` on PATH has no numpy.
Exit status is 0 only if every strict invariant passed. Checks (d) and (e) are advisory
(warnings) because they describe a third-party file and a budget, not an invariant of this repo.
