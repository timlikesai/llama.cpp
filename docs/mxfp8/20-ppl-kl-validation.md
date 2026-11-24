# 20 — Phase 5: PPL + KL validation of the lossless NATIVE file

Status: **FINAL** (2026-11-21, within 40-min budget). Headline: the lossless file validates;
the *dequantized reference file is numerically broken* (section 2.1).
Brief: `docs/mxfp8/briefs/U-ppl-kl-validation.md`
Constraint respected: **no builds** — existing binaries in `build-cuda/bin` + python only.

## 0. Claim under test

The NATIVE file's expert tensors are byte-identical to the HF checkpoint. Therefore any
measurable difference between `Inkling-2-Mini-NATIVE.gguf` and the dequantized
`Inkling-DEQ-Q80.gguf` must come from the **runtime path**, not from quantization loss.
A PPL/KL comparison is how that is demonstrated — and how a silent convention bug that
produces plausible-but-wrong text would be caught.

Hypothesis to falsify: NATIVE is statistically indistinguishable from DEQ-Q80.
If it is NOT indistinguishable, that invalidates the byte-exactness claim. Say so loudly.

## 1. Setup (corpus, flags, binaries)

**Binaries** (no builds performed): `build-cuda/bin/llama-perplexity` at tree `6c1424737`,
libs dated Nov 21 11:30. `llama-perplexity --help` shows `-ngl` default is **`auto`**, so
`-ngl` is simply omitted on every run (brief rule: never pin a nonzero `-ngl`).

**Corpus.** Canonical **WikiText-2 raw test** split, as `tools/perplexity/README.md`
prescribes.
- Source: `https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet` (732,610 B, 4358 rows).
- The two zip mirrors named in older notes are **dead**: `s3.amazonaws.com/repo.metamind.io/.../wikitext-2-v1.zip` returns an XML error body, `raw.githubusercontent.com/pytorch/text/data/...` returns 404. See `## Corrections to my brief`.
- Extracted with `uv run --with pyarrow python3` (system python has no pyarrow/pandas); joined the `text` column with blank lines -> `logs/corpora/wiki.test.raw`, **1,284,763 B / 5,781 lines**.
- Measured slice: first **65,536 B** -> `logs/corpora/wiki.test.slice64k.raw`, sha256 `319564b537e2799d881ff4c6d8d51c21c771231b3a7d26f1b920f88e80604af5`, **16,705 tokens**, 8 chunks at `n_ctx=2048`.
- Slice, not full split, because each KL base logit file costs ~0.4 MB/token and each run costs GPU time (see 40-min budget). 16.7k tokens is ample for the error bars reported.

**Flags, identical for every file:**
`-f <slice64k> -c 2048 -b 2048 -ub 512 -s 42`, `-ngl` omitted, KL runs add
`--kl-divergence --kl-divergence-base <file>`.

**Flag trap found (cost me the first full sequence).** `-c 512` does **not** give a 512-token
evaluation. `tools/perplexity/perplexity.cpp:2012` computes
`params.n_parallel = max(1, n_batch / n_ctx)`, then `:2014` sets `n_ctx = n_parallel * n_ctx`.
With defaults (`-b 2048`) and `-c 512` that silently builds a 4-slot 2048-token context, and
the run dies with `init_batch: failed to prepare attention base ubatches` /
`decode: failed to find a memory slot for batch of size 2048` after ~0 s. **`-b` must equal
`-c`.** `--ppl-stride` additionally destabilised the derived `n_ctx` (2048 -> 2304 -> 3072) and
was dropped in favour of the upstream default stride (`n_ctx/2`).

## 2. Perplexity results

WikiText-2-raw test slice, 16,705 tokens, 8 chunks @ `n_ctx=2048`, seed 42, `-ngl auto`,
CUDA on both 5060 Tis. Logs: `logs/u-ppl-kl.log`, `logs/u-ppl-kl2.log` (both complete - the
scripts redirect `2>&1` before teeing), `logs/u-probe-native.log` (stdout only - llama.cpp
logs to stderr and that one-off probe did not merge them, so it holds the per-chunk PPL line
but not the load log; the same per-chunk values appear in full in `logs/u-ppl-kl.log`).

| file | mean PPL | +/- | status |
|---|---|---|---|
| **Inkling-2-Mini-NATIVE.gguf** (ours, bit-exact) | **5.575741** | 0.144215 | ok |
| **Inkling-2-Mini-IQ4_NL.gguf** (public lossy) | **5.563596** | 0.143720 | ok |
| **Inkling-DEQ-Q80.gguf** (dequantized reference) | **nan** | nan | **BROKEN - see 2.1** |
| Inkling-2-Mini-NATIVE-scale1.gguf (sensitivity control) | **5,260,013.945** | 318,084.744 | broken **by design** |

Per-chunk PPL, NATIVE (identical between the v1 and v2/KL code paths, which cross-checks
the two implementations): `5.0931, 6.2767, 6.5986, 6.3100, ..., 5.5757`.

The two finite models are within **0.22%** of each other (5.5757 vs 5.5636), and the
uncertainty on each is +/-2.6%, so on this corpus the public lossy IQ4_NL build and our
bit-exact native build are **statistically indistinguishable in PPL**. That is a *weak*
result on its own (overlapping error bars), which is why the KL numbers in section 3 matter
more; but it is the first sign that nothing in our path is degraded.

### 2.1 FINDING: the dequantized reference `Inkling-DEQ-Q80.gguf` produces non-finite logits

Every one of the 8 chunks returns `nan` perplexity:

```
perplexity: calculating perplexity over 8 chunks, n_ctx=2048, batch_size=2048, n_seq=1
[1]nan,[2]nan,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan,
E Unexpected negative standard deviation of log(prob)
```

Reproduced **twice, on two independent code paths**: the plain `perplexity` path (run V0,
no KL flags) and the `perplexity_v2` + `--kl-divergence-base` path (run U1). So it is not a
KL-harness artifact - the model itself is emitting non-finite logits.

This is **not** a refutation of byte-exactness of the NATIVE file. It is a defect in the
converter's **dequantized** output (`--outtype q8_0`, no `--native-experts`) or in the
runtime path that file takes. Evidence about which:
- The file *generates* correct text (`logs/deq-q80-run.log` -> `68 B: 64 ...`), so it is not
  globally destroyed; the non-finite values appear under the long, dense prefill of a real
  corpus.
- Its load log carries `tensor overrides to CPU are used with mmap enabled` - at 27.8 GB it
  does not fit in 2x16 GB and **spills to CPU**, a path the other three files do not take.
- Prime suspect, **UNVERIFIED**: the 8 formerly-MXFP8 expert layers dequantized into Q8_0.
  e4m3 max magnitude is 448; combined with the F32 `.scale`, a q8_0 block scale/overflow bug
  would produce exactly this inf -> nan signature in the MoE output.

**Consequence for this brief:** task 3 as written ("KL of NATIVE vs the dequantized
reference") is **not computable** - the reference has no usable distribution. Section 3
uses the strongest available substitute and states what that costs.

**Recommended action (not taken here - I may not edit `conversion/`, and no rebuilds):**
dequantize MXFP8 experts to F32/BF16 rather than Q8_0, or fix the block scale, then
re-run `scripts/inkling_native/ppl_kl_u2.sh` unchanged.

## 3. KL divergence vs the dequantized reference

### 3.0 What this build exposes (better than the brief assumed)

No python harness and no top-k `--probabilities` fallback was needed. This
`llama-perplexity` build has **full-vocabulary KL built in** - confirmed in the binary
(`kl_divergence_result`, `compute_logprobs`, `saving all logits to`) and documented in
`tools/perplexity/README.md`:
1. base model + `--kl-divergence-base file.kld` -> records every logit vector to disk
   (magic `_logits_`, then `n_ctx`, `n_vocab`, `n_chunk`, the token array, then the logits);
2. candidate model + `--kl-divergence --kl-divergence-base file.kld` -> mean/max KLD, PPL
   ratio, mean/percentile delta-p, RMS delta-p, "Same top p".

Header of the file I generated: `n_ctx=2048`, `n_vocab=200058` (the *unpadded* vocab, vs
`inkling2_loop.vocab_size = 201024` in the GGUF), `n_chunk=8`, size 3,274,680,372 B.
Size scales as positions x n_vocab x 2 B, consistent with the README's "37 GiB for LLaMA 3
on Wikitext-2".

### 3.1 The base model is the broken one

Run with `--kl-divergence-base` on `Inkling-DEQ-Q80.gguf` produced a structurally valid but
numerically degenerate 3.27 GB file, and every comparison against it was self-contradictory:

| candidate vs DEQ-Q80 base | Mean KLD | Max KLD | Same top p | ln(PPL ratio) |
|---|---|---|---|---|
| NATIVE | 0.000000 +/- 0.000000 | 0.000000 | 0.012 +/- 0.012 % | -nan |
| SCALE1 | 0.000000 +/- 0.000000 | 0.000000 | 0.061 +/- 0.027 % | -nan |
| IQ4_NL | 0.000000 +/- 0.000000 | 0.000000 | 0.012 +/- 0.012 % | -nan |

**KLD of exactly 0 together with Same-top-p of ~0% is impossible** for a real pair of
distributions (identical distributions imply identical argmax, so Same-top-p would be
~100%). The zero is the arithmetic consequence of a base whose probabilities are all
zero/`nan`: `sum(p log p/q)` collapses to 0. So these KL numbers are **void**, and I am not
reporting them as a result. This is exactly the "harness can't see the difference" failure
mode brief item 4 warns about - caught here by the internal consistency check rather than by
the scale control, because the scale control also returned 0.000000 KL.

Section 3.2 re-runs the whole KL chain with a base that has finite logits.

### 3.2 KL with a finite base (NATIVE), and why that is still a real test

Because DEQ-Q80 has no usable distribution, the base became
`Inkling-2-Mini-NATIVE.gguf` itself (`logs/u-ppl-kl2.log`, runs V1/V2, same flags, seed 42,
base file `logs/corpora/u-kld-native.bin`).

What this **can** and **cannot** prove:
- It **cannot** re-derive "our file == the dequantized checkpoint values" - that comparison is
  blocked by 2.1.
- It **can** prove the KL instrument has resolution and no systematic bias, by giving a
  self-comparison (must be ~0) and a known-lossy model (must be clearly nonzero). Both hold,
  below. Together with section 2's PPL agreement and the offline byte-level checks
  (`scripts/inkling_native/verify_*`), that is what actually supports the lossless claim today.

| candidate vs **NATIVE** base | Mean KLD | Max KLD | PPL ratio | RMS Δp | Same top p |
|---|---|---|---|---|---|
| NATIVE (self, positive control) | **-0.000000 +/- 0.000000** | 0.000070 | 1.000413 +/- 0.000309 | 0.001% | **100.000%** |
| IQ4_NL (public, lossy) | **0.012061 +/- 0.000318** | 0.862515 | 0.998234 +/- 0.002064 | 2.948% | 95.687% |
| SCALE1 (broken `.scale`) | **13.787377 +/- 0.049874** | 34.917797 | 943764.579676 +/- 58505.485650 | 64.219% | 0.672% |
| DEQ-Q80 | **nan** (broken from both sides - see 3.3) | | | | |

### 3.3 DEQ-Q80 as the *candidate*: also nan

For completeness the dequantized file was also run against the healthy NATIVE base. Every
KL and delta-p statistic came back `nan` (e.g. `Median KLD: nan`, `Mean Δp: -nan ± 0.000 %`,
`Same top p: 0.012 ± 0.012 %`). Since the base in this run is known-good and finite, the
non-finite values must originate in **DEQ-Q80's own logits**, which corroborates 2.1 from the
opposite direction and rules out "the logit file was written badly" as the explanation.

The self-control is the important one: `Same top p = 100.000 ± 0.000 %` and `Mean KLD`
indistinguishable from zero, with only `Max KLD = 0.000070`. That residual is *not* model
error - the README states the logit file is stored as scaled **unsigned 16-bit**, not f32, so
a self-comparison measures exactly that downcast. The instrument's noise floor is therefore
~7e-5 KLD, and IQ4_NL sits **173x** above it. The KL path resolves real quantization damage
by two orders of magnitude, so a null result elsewhere means "no damage", not "no resolution".

**PPL of the reference under ours** (the third number task 3 asked for) is reported directly
by this tool as `Mean PPL(Q)` / `Mean PPL(base)` / their ratio: NATIVE 5.575741 vs its own
base 5.573439 (ratio 1.000413 +/- 0.000309, i.e. +0.04%, inside the downcast noise floor).

A useful validity check: `Mean PPL(base)` was **identical to 6 decimal places
(5.573439 ± 0.143983) in all four runs**, including the DEQ-Q80 candidate run which returned
`Mean PPL(Q) : nan`. The base is therefore read back deterministically, and the `nan` in that
run is unambiguously the candidate's own logits, not file corruption. The instrument's floor
for this setup is the self-comparison in the table above (KLD ~0, Max KLD 7e-5): differences
below ~1e-4 KLD are not resolvable, and IQ4_NL (0.0121) and SCALE1 (13.79) are far above it.

## 4. Sensitivity control (scale1)

**PASS on perplexity, decisively.** `Inkling-2-Mini-NATIVE-scale1.gguf` - identical file with
all 40 `.scale` sidecars forced to 1.0 - scores:

```
Mean PPL(Q) : 5260013.945012 +/- 318084.743819
```

That is **943,765x worse** than the correct NATIVE file (5.5757), on the same corpus, flags
and seed (`Mean PPL(Q)/PPL(base) = 943764.579676 ± 58505.485650` per the tool). A single wrong
scale convention moves PPL by nearly six orders of magnitude, so section 2's NATIVE/IQ4_NL
agreement (0.22% apart) is far outside the noise of this measurement.

**And it proves the more important point by contradiction:** in the *DEQ-based* table (3.1)
SCALE1 returned the same `Mean KLD 0.000000` as NATIVE did. A control that cannot be
distinguished from the model under test means the measurement was dead, exactly as brief
item 4 predicted. Because I caught that before reporting, the KL numbers I do report come
from the NATIVE base, where SCALE1 and IQ4_NL separate cleanly from the self-control.

### 4.1 KL of SCALE1 vs the NATIVE base - the control the harness *should* fail to hide

```
Mean    KLD:  13.787377 ±   0.049874
Maximum KLD:  34.917797
Median  KLD:  13.582924
Mean    Δp: -50.397 ± 0.440 %
RMS Δp    :  64.219 ± 0.355 %
Same top p:   0.672 ± 0.090 %
```

Mean KLD **13.79** against a noise floor of 7e-5 - a factor of ~200,000. The argmax token is
preserved in only **0.67%** of positions, i.e. the broken-scale file is a different model.
The KL instrument has more than enough dynamic range for the claim under test, so the
near-zero results in 3.2 are informative rather than merely undetectable.

## 5. Long-context / KV memory check

**Not run - budget.** Deferred honestly rather than guessed at. Child E's math (fp8 KV
296/552/1064/2088 MiB at 32k/64k/128k/262k; f16 uncapped 262k = 42,590 MiB) is **UNVERIFIED**
by me.

Two things I did observe that bear on Phase 6 and cost no extra GPU time:
- At `-c 2048` the auto-fit already splits these models across both cards and **disables
  Flash Attention**: `resolve_fused_ops: layer 24 is assigned to device CUDA1 but Flash
  Attention is assigned to device CUDA0 (usually due to missing support)` ->
  `Flash Attention not supported, set to disabled`. Without FA the KV cache is the plain
  f16 layout, so the fp8-KV savings in Child E's table are not being realised here, and any
  long-context measurement must decide whether to force FA onto one card.
- Layer 24 is the extra (MTP/output) layer landing on CUDA1; the split point is what breaks FA,
  so `-sm`/`-ts` choice, not just ctx size, gates fp8 KV at long context.

To finish this: `llama-perplexity -m <NATIVE> -f <slice> -c 8192 -b 8192` (keep `-b` = `-c`,
see section 1) and read the per-device buffer sizes from the load log; repeat at 32k+ with and
without `--flash-attn on --cache-type-k fp8 --cache-type-v fp8`.

## 6. Corrections to my brief

1. **`logs/g0-prompt-long.txt` is 14,168 B, not "only 2.8 KB"** (the 2.8 KB figure matches
   `g2b-cpu-run.log`, a different file). Irrelevant to the outcome - I used WikiText-2 anyway -
   but the size in the brief is wrong.
2. **Task 3's premise is unmeasurable as written.** It asks for KL of our file "against the
   dequantized reference", but that reference (`Inkling-DEQ-Q80.gguf`) emits non-finite logits
   and `nan` PPL on every chunk (2.1). It was described to me as known-good because it
   generates `68 B: 64 ...` correctly; generating short coherent text does **not** imply finite
   full-corpus logits. The brief's artifact table needs a warning column.
3. **No python harness / no `--probabilities` top-k fallback was needed.** The brief guessed
   the build might not expose full logits; it does, natively, via `--kl-divergence-base` +
   `--kl-divergence` (3.0). The top-k route would have been strictly weaker.
4. **The WikiText-2 mirrors that `scripts/get-wikitext-2.sh`-era notes point at are dead**
   (metamind S3 returns an XML error, pytorch/text `data` branch 404s). The working source is
   the `Salesforce/wikitext` HF **parquet**, which needs pyarrow - absent from system python;
   `uv run --with pyarrow` worked without installing anything globally.
5. **`-c 512` is a trap in this tool**, not a small-context run: `perplexity.cpp:2012` derives
   `n_parallel = n_batch / n_ctx` and multiplies ctx back up, so a small `-c` with default
   `-b 2048` silently requests 4 slots and fails to allocate. The brief's "never pin -ngl"
   warning is right, but there is a second, undocumented footgun on `-b`/`-c`.
6. Minor: the KL base file records `n_vocab = 200058` (unpadded) while the GGUF advertises
   `vocab_size = 201024`. Not a bug, but it changes the on-disk size estimate (~0.40 MB/token).

## 7. Verdict

**What is now demonstrated:**
- Our bit-exact NATIVE repack is **numerically healthy** on a real corpus: PPL 5.5757 +/- 0.1442
  over 16,705 WikiText-2 tokens, and its own logits reproduce at KLD ~0 / Same-top-p 100%.
- It is **not distinguishable from the public lossy IQ4_NL build on PPL** (5.5757 vs 5.5636,
  0.22% apart against +/-2.6% error bars); IQ4_NL sits at KLD 0.01206 +/- 0.00032 vs NATIVE,
  173x the instrument's noise floor - so the KL axis does resolve loss, and the PPL axis alone
  is the weaker of the two for this claim.
- A wrong scale convention is catastrophic and obvious (SCALE1: PPL 5.26e6). **No plausible
  silent-convention-bug survives this test**, which is the practical point of Phase 5.

**What is NOT demonstrated:** the direct "NATIVE == dequantized checkpoint" KL, because the
dequantized reference file is itself broken (nan). The byte-exactness claim in
`docs/mxfp8/06-lossless-repack-design.md` therefore still rests on the **offline tensor byte
comparison against the safetensors checkpoint**, not on this runtime comparison. I did not
contradict it; I could not corroborate it the way the brief intended.

**Do not smooth this over:** the headline is that `Inkling-DEQ-Q80.gguf`, the file the project
has been treating as its numeric reference, produces `nan` perplexity. Every past conclusion
that used it as a reference is suspect until that is fixed.

## 8. Follow-ups and caveats

1. **`Inkling-DEQ-Q80.gguf` emits non-finite logits** (2.1, 3.3) - the intended KL reference
   does not exist. Fix the dequantized build (likely the MXFP8 -> Q8_0 expert dequant), then
   re-run `scripts/inkling_native/ppl_kl_u.sh` unchanged; the NATIVE-base chain in
   `scripts/inkling_native/ppl_kl_u2.sh` is the working template.
2. **Section 5 (long-context KV memory, Phase 6 groundwork) was not run** - budget. Child E's
   fp8/f16 KV table remains unverified by measurement.
3. Only 64 KB of the 1.28 MB WikiText-2 test split was measured (16,705 tokens). Error bars are
   tight enough for the claims made, but the 0.22% NATIVE-vs-IQ4_NL PPL difference is real on
   the KL axis (KLD 0.0121 +/- 0.0003) and invisible on PPL - a full-split run would firm it up.
4. The `.kld` base files are **3.27 GB each** and still on disk
   (`logs/corpora/u-kld-deq-q80.bin`, `logs/corpora/u-kld-native.bin`). Left in place
   deliberately per the never-delete rule; the NATIVE one is reusable and valid, the DEQ one is
   not. Disk was 251 GB free before and is not materially affected.
5. **Scope honoured:** no source, model, or conversion file was modified and no build was run.
   Artifacts created: this doc, `scripts/inkling_native/ppl_kl_u.sh`,
   `scripts/inkling_native/ppl_kl_u2.sh`, corpora under `logs/corpora/`, and
   `logs/u-ppl-kl.log` / `logs/u-ppl-kl2.log` / `logs/u-probe-native.log`.
6. GPUs were idle before I started (`nvidia-smi`: 13 MiB used on both) and no build-lane
   contention was observed; timings (~36-80 s/run) were stable across repeats, so the numbers
   are not contaminated. Absolute timings are not a deliverable here - PPL and KL are.
