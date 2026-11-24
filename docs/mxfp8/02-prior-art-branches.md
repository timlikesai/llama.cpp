# 02 — Prior art: Blackwell block-scaled MMA from `mxfp4-w4a8` / `w4a8-beast`

Extracted **inline by the parent** after this brief timed out in four separate delegated runs. Sources are
verbatim `git show` output, saved unfiltered so nothing is lost to a pipe:

| artifact | content |
|---|---|
| `logs/priorart-mma-asm.diff` | full diff of `5bbceb6df` "CUDA : Blackwell FP8 mma for FP4 weights" (445 lines) |
| `logs/priorart-scale-pack.diff` | full diff of `9a02817b2` "CUDA : pack the FP8 scale operands in a format the mma instruction expects" (272 lines) |
| `logs/priorart-mma-cuh.inc` | complete `mma.cuh` at `5bbceb6df` |
| branch commits | `6764dd2b8` (native Blackwell tensor core support, 2026-02-26) → `5bbceb6df` → `9a02817b2`; `e5bb13fbe` on `w4a8-beast` (2026-03-09) |

## The instruction (mma.cuh:1287-1308 at `5bbceb6df`)

Three variants, selected at compile time by `type` / `prec_src1`. **All share one operand skeleton** —
4 A registers, 2 B registers, f32 accumulator, and the two scale operands as **register indices** with
`{0, 0}` filling the second id slot of each:

```cpp
// mxfp4 weights x FP8 (e4m3) activations   <- the mixed-type W4A8 path, our direct template
asm volatile(
  "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
  "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
  "%10, {0, 0}, %11, {0, 0};"
  : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
  : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]),
    "r"(a_scale), "r"(b_scale));

// mxfp4 x mxfp4   (kind::mxf4, scale_vec::2X, m16n8k64, e2m1.e2m1, ue8m0)   — same skeleton
// nvfp4 x nvfp4   (kind::mxf4nvf4, scale_vec::4X, m16n8k64, e2m1.e2m1, ue4m3) — same skeleton
```

**Read the dtype field as `<A>.<B>`**: `e2m1.e4m3` means A = the FP4 weight, B = the FP8 activation.
For **MXFP8 weights × FP8 activations** (Phase 4b, our case) the required form is the same
`kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 … .ue8m0` with **`e4m3.e4m3`** — `mxf8f6f4` is
exactly the mixed 8/6/4-bit family, so W8A8 is a dtype-field change, not a new instruction. `k32`
and `scale_vec::1X` already match MXFP8's one-ue8m0-per-32-K granularity, which is why this
prior art transfers so cleanly (UNVERIFIED: that A=e4m3 is legal at m16n8k32 for our operand
counts — confirm against the PTX ISA before writing the kernel).

## Scale-operand packing (`9a02817b2`, the commit that made it work)

The subtle part, quoted from the diff comments:

- **"the mma only reads byte 0 of the scale operand, so shift each frag's scale byte down"** — scales
  for the 4 fragments live as 4 bytes inside one `uint32_t`, and each fragment's operand is produced with
  `scaleA[n][frag] = sa >> (8*frag);` (and `scaleB[frag] = sb >> (8*frag);`).
  Earlier code wrongly loaded a `uint4` and used `.x/.y/.z/.w` directly.
- `static_assert(nfrags == FP8_SCALES_PER_WORD, "one scale word covers all fragments and one y block");`
  with `static constexpr int FP8_SCALES_PER_WORD = QK_FP8_MMQ / QK_MXFP4;` (`mmq.cuh:31`).
- SFA lane index: `const int g = threadIdx.x / 4; const int t = threadIdx.x % 4;`
  then **"scale_vec::1X with byte-id 0 takes scale_A for rows g and g + 8 from the quad lanes with
  t%4 == 0 and 1"** → `const int tidx_A = g + (t % 2) * 8;`
- Shared-memory stride for the FP8 layout, from `ggml_cuda_mmq_get_sram_stride`:
  `MMQ_ITER_K/8 + MMQ_ITER_K/(FP8_SCALES_PER_WORD*QK_MXFP4) + 2` — "256 mxfp4 values in 32 words +
  one scale word per 128 values + 2 words of padding" (the *old* form `MMQ_ITER_K/8 + MMQ_ITER_K/QK_MXFP4 + 4`
  allocated a stride per-fragment and is what the packing commit fixed).

## FP8 activation block (the quantizer's output format)

```cpp
// activation block for the mxfp block-scaled mma (Blackwell): 128 e4m3 values,
// d4[0] packs 4 ue8m0 scales (byte f covers values 32f..32f+31), d4[1..3] pad to MMQ_TILE_Y_K
struct block_fp8_mmq {
    uint32_t d4[4];
    int8_t   qs[QK_FP8_MMQ];
};
```
`QK_FP8_MMQ = QK8_1_MMQ = 4*QK8_1 = 128`. So activations are 128 e4m3 values with **four** ue8m0
sub-scales packed little-endian into `d4[0]`, one per 32 values — structurally the *same* shape as
`block_mxfp8 { uint8_t e; uint8_t qs[32]; }`, just batched 4-at-a-time for the MMA's fragment width.

## FP4 operand bit placement (why the payload needs a shift)

> "an mxfp4 byte holds value j in the low nibble and j + 16 in the high one, so the low nibbles of an
> int are a0 (A cols 4t..4t+3) and the high ones a2; **kind::mxf8f6f4 wants e2m1 in bits [5:2]**"

Implemented with `lo_mask = 0x0F0F0F0F`, `hi_mask = 0x3C3C3C3C`. **MXFP8 needs no equivalent shuffle** —
an e4m3 weight byte is already a full byte operand — which makes our case *simpler* than the prior art
here, at the cost of needing the `e4m3.e4m3` dtype form above.

## src1 precision selection (`mmq.cu:ggml_cuda_mmq_get_prec_src1`)

Documented in that commit as: *"A set prec is the lowest rank the caller allows (see ggml.h), so a type
with no path at that rank uses the next better one; off Blackwell src1 is always Q8_1."*
`GGML_PREC_FP8` already exists as an enum value (`ggml/include/ggml.h:1035`, added by `d7b4c998d`).

## Reuse verdict for MXFP8 (Phase 4b)

| piece | verdict | why |
|---|---|---|
| `mma.sync…kind::mxf8f6f4…` skeleton + `{0,0}` scale-id operands | **reuse near-verbatim** | identical shape; change the dtype field to `e4m3.e4m3` |
| scale byte-shifting (`sa >> 8*frag`), `FP8_SCALES_PER_WORD`, `tidx_A` | **reuse verbatim** | MXFP8 weights have exactly one ue8m0 per 32 values, same granularity as the FP8 activations in that path |
| `block_fp8_mmq` + the e4m3 activation quantizer | **reuse, needs rebase** | branch is 6 weeks stale; check against current `mmq.cuh`'s `block_fp4_mmq`/`q8_1_mmq` layout and the new `mmq-config-blackwell.cuh` |
| `lo_mask`/`hi_mask` nibble→bits[5:2] shift | **not needed** | e4m3 weights are byte operands |
| tile loader for FP4 payload (`mmq-load-tiles.cuh`) | **rewrite** | our row is 33 B/32 values (`{e, qs[32]}`), a different SRAM layout than both FP4 and the branch's FP8-activation block |
| `prec_src1` plumbing | **reuse verbatim** | unchanged semantics, already supports FP8 |

**Ordering note:** Phase 4a (plain `mmq` A16 tile path for MXFP8, lane `P`) is independent of all of this
and is what should land first — it de-risks dispatch and gives a measurement baseline. 4b then swaps the
dot for the native MMA above.
