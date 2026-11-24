# 21 — Why `Inkling-DEQ-Q80.gguf` emits non-finite logits

Brief: `docs/mxfp8/briefs/V-dequant-nan.md`
Status: **root cause narrowed to the runtime path, converter exonerated; KL baseline NOT yet
restored** (needs the one experiment listed in §6, or a build). Over budget (35 min), so this
documents measurements, not speculation — every claim below has a log or a script.
Constraints honoured: **no builds**, no `llama-quantize` runs (same reason), no edits outside
`conversion/inkling2.py` + `scripts/inkling_native/` + `docs/mxfp8/21-*` + `logs/`.
`conversion/inkling2.py` was **not** touched, so the native path is untouched by construction.
Nothing deleted. `Inkling-DEQ-Q80.gguf` and both `u-kld-*.bin` files are intact.

## Headline

**The dequantized reference file's data is fine.** Three of the four hypotheses in the brief are
refuted by measurement, and a fourth (doc 20 §2.1's "prime suspect": an MXFP8 -> Q8_0 block
scale/overflow bug) is refuted outright. The nan comes from the **runtime path**, and it is
**not device-specific**.

Separately: `Inkling-2-Mini-NATIVE.gguf` is **not healthy on CPU** — it also returns nan PPL at
`-ngl 0`. Doc 20 only ever measured NATIVE under auto-fit (CUDA), so "NATIVE is healthy" is a
CUDA-only claim and needs re-checking.

## 1. Step 1 — offline value scan: hypothesis (a) REFUTED

`scripts/inkling_native/scan_gguf_finite.py` (new) walks every tensor's raw blocks off the
memmap in bounded chunks, in the exact format `ggml/src/ggml-common.h` defines:
`block_q8_0 = {half d; int8 qs[32]}` (34 B / 32 el), IQ4_NL 18 B / 32 el, F32, BF16. It counts
non-finite **block scales**, non-finite **values**, and reports the largest finite magnitude.
It asserts each tensor's byte count is a whole number of blocks (the brief's shape warning).

```
./scripts/inkling_native/scan_gguf_finite.py --gguf ~/models/timlikesai/Inkling-DEQ-Q80.gguf
tensors  : 704   types: {F32: 478, Q8_0: 226}
scanned 704 tensors, 27.81 GB of tensor data
RESULT: no non-finite block scales or values found          -> logs/v21-scan-deq-q80.log

./scripts/inkling_native/scan_gguf_finite.py --only exps
scanned 48 tensors, 25.99 GB of tensor data
RESULT: no non-finite block scales or values found
largest finite magnitudes seen: blk.17.ffn_down_exps.weight Q8_0 max|x|=0.852177
                                                              -> logs/v21-scan-deq-experts.log
```

Zero inf/nan anywhere in 27.81 GB, so:
- no fp16 scale overflow: the experts' largest block value bound is **0.85** (`127 * max|d|`),
  four orders of magnitude below the 65504 x 127 ceiling where a half scale would saturate;
- no `e4m3`/`ue8m0` byte 254/255 leaking through: the converter already refuses those
  (`conversion/inkling2.py:59` dequant path and `:112` native pack path refuse UE8M0 byte 255;
  `:66-67` refuses any non-finite dequantized weight, and the run that produced this file
  completed). Byte 254 (= 2^127) is not refused but would have shown up as a non-finite scale
  in the scan above, and did not.

## 2. Step 2 — cross-check against the source of truth: the numbers are RIGHT

`scripts/inkling_native/check_deq_vs_iq4nl.py` (new) samples experts {0, 64, 127} x 4 rows of
all 48 shared `ffn_*_exps.weight` tensors in both files, dequantizes each with `gguf.quants`,
and compares. Chosen because the healthy public build and DEQ-Q80 differ in *only* the expert
storage type (brief V), so it is the same bytes from two independent conversion paths:

```
expert tensors in DEQ=48 in IQ4_NL=48, shared=48     -> logs/v21-check-deq-vs-iq4nl.log
  blk.0.ffn_down_exps.weight      max|d|=0.01096 rel=9.19%  rms=0.00212  max|x|=0.119
  ... (all 40 ex-NVFP4 layers: rms 0.0015-0.0022, max rel <= 13.9%)
  blk.20.ffn_down_exps.weight     max|d|=0      rel=0.0000% rms=0        max|x|=0.109
  blk.20..23 (all 8 tensors)      max|d|=0      rel=0.0000%
worst relative disagreement: 13.8862%  (4-bit scale, expected)
```

Two facts worth freezing:
1. **The 8 ex-MXFP8 tensors (blk.20-23) are bit-identical between DEQ-Q80 and the public file** —
   the public build stores those same experts as Q8_0 too, same shapes, same bytes.
2. The 40 ex-NVFP4 tensors differ from the public build by exactly 4-bit rounding
   (`rms ~ 0.002` against `max|x| ~ 0.1`). No sign of a row/column permutation, a wrong
   `amax/denominator` fold, or an interleaving slip — those would read as ~100 %, not ~10 %.

=> The converter's dequantized values are correct. **No converter fix is warranted**, and
doc 20 §2.1's "prime suspect (MXFP8 -> Q8_0 block scale/overflow bug)" is dead.

## 3. Steps 3+4 — the nan does not follow the device, and Q8_0 experts per se are not broken

`scripts/inkling_native/probe_deq_nan.sh` (new); 8 KiB corpus slice
(`logs/corpora/deq-probe.raw`), `-c 1024 -b 1024 -ub 512 -s 42` (`-b` = `-c` per the footgun),
`-ngl` never pinned nonzero except the explicit `-ngl 0` CPU legs. Full log:
`logs/v21-probe-deq-nan.log`.

| probe | file | placement | chunk 1 | chunk 2 |
|---|---|---|---|---|
| P0 | NATIVE | auto (CUDA) | 3.5669 | 5.3751 |
| P1 | DEQ-Q80 | auto (CUDA) | **nan** | **nan** |
| P2 | DEQ-Q80 | `-ngl 0` (**CPU**) | **nan** | **nan** |
| P3 | NATIVE | `-ngl 0` (**CPU**) | **nan** | **nan** |
| P4 | DEQ-Q80 | auto + `-ot 'exps\.weight=CPU'` | **nan** | **nan** |

- (c) file-size / CPU-spill: **refuted**. P2 puts the whole model on CPU and P4 keeps the
  experts on CPU; both still nan. The nan does not move with the device.
- (b) as written ("Q8_0-as-3-D-`MUL_MAT_ID`-weight path is broken"): **refuted in that form**.
  The healthy public IQ4_NL build stores blk.20-23's 8 ex-MXFP8 expert tensors as Q8_0, at
  identical shapes and bit-identical bytes to DEQ-Q80's (§2), and scores PPL 5.5636 (doc 20 §2).
  A type and shape that is fine in one file is not intrinsically broken.
- Chunk 1 is already nan, so this is not an accumulation/long-context effect.
- `test-backend-ops -o MUL_MAT_ID -b CUDA0`: **1150/1150 passed**, 115 of them
  `type_a=q8_0` (`logs/v21-bops-mmid-cuda0.log`). So the op is correct at the shapes the suite
  covers (largest `n_mats=37`, `k=2048`) — the trigger, if it is in the op, is shape- or
  size-dependent (`m=2304/4608`, `n_mats=128`, 1.44 GB src0), not generic.

## 4. NEW FINDING — NATIVE is nan on the CPU path

P3 is the surprise: the same NATIVE file that scores PPL 5.5757 and self-KLD ~0 under auto-fit
(CUDA) returns **nan on both chunks with `-ngl 0`**. Native MXFP8/NVFP4 experts therefore have a
second, independent defect in the CPU path — most likely the CPU `MUL_MAT_ID` handling of the
native types (they are new types; `ggml-cpu` support is only as complete as the CUDA side made
it). Two consequences:

1. Any conclusion of the form "NATIVE is numerically healthy" is **CUDA-scoped** until re-measured.
2. The brief's premise "the two files differ in exactly one way (expert storage type)" now has
   an extra live variable: **native expert types and Q8_0 expert types fail on CPU for
   different reasons, or for the same shared reason.** Doc 20's NATIVE-base KL table (§3.2) is
   still valid as a CUDA-vs-CUDA comparison, which is what it was measured as.

## 5. Where this leaves the KL baseline

The direct `NATIVE vs dequantized-reference` KL is **still unmeasured**, but the reason changed:
not "the reference is garbage", but "the runtime cannot evaluate this reference file as it
currently stands, on either backend". Because DEQ-Q80's numbers are provably correct (§2), the
baseline is recoverable — via the runtime path, not by re-converting.

Do not read this as doc 20 being wrong about the *measurement*: its nan reproductions are real
and independently reproduced here (P1, plus chunk-level `nan`). What is now falsified is its
diagnosis sketch (§2.1 "prime suspect"). Doc 20 §7's caveat should be rewritten as "the
reference file's data is finite and correct; the runtime path is at fault" — **out of my fence**
(`docs/mxfp8/20-*` is not in my editable list), so it is flagged here instead of edited.

## 6. Next step (cheap, no build, python-only) — layer bisect with a hybrid file

The discriminator the brief's step 3 wanted, without touching `ggml/**`: copy the healthy public
`Inkling-2-Mini-IQ4_NL.gguf` and, with `gguf-py`, swap **one layer's** `ffn_gate_up_exps` +
`ffn_down_exps` to DEQ-Q80's Q8_0 blocks (raw bytes + type field), then run
`scripts/inkling_native/probe_deq_nan.sh`-style 2-chunk probe.

- If a single ex-NVFP4 layer swapped to Q8_0 turns the healthy file nan -> pure runtime bug in
  the Q8_3D `MUL_MAT_ID` path at this shape, localized to one layer, written up as a reproducer
  with exact numbers for the C++ owner (who owns `ggml-cuda/mmvq.cu` today).
- If it stays finite -> the trigger is cumulative across the 40 layers, or magnitude-driven, and
  the fallback reference is a **BF16/F32-expert** dequantized build (doc 20 §2.1's other
  recommendation), which needs a converter change in `conversion/inkling2.py` (allowed here) and
  ~26 GB of disk.

Either way the recovered reference makes the `NATIVE vs dequantized` KL measurable at last.

## 7. Artifacts

New scripts: `scripts/inkling_native/scan_gguf_finite.py`, `check_deq_vs_iq4nl.py`,
`probe_deq_nan.sh`. New logs: `logs/v21-scan-deq-q80.log`, `logs/v21-scan-deq-experts.log`,
`logs/v21-check-deq-vs-iq4nl.log`, `logs/v21-probe-deq-nan.log`,
`logs/v21-bops-mmid-cuda0.log`. New corpus:
`logs/corpora/deq-probe.raw` (8 KiB). No model file, no source file outside
`scripts/inkling_native/`, no doc other than this one was modified.
