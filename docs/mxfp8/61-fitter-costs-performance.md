# The fitter is costing performance: `-ub 512 --fit off` beats the deployed config on every metric

Measured 2026-11-24 on the lossless `Inkling-2-Mini-FULL.gguf`, ctx **262,144** (the deployed
context), same prompt (`logs/prompt-code.txt`), same seed, `-n 384`, `temp 0`, two reps per cell,
**all cells deterministic to 5 digits on acceptance**.

## The 2x2

| config | acceptance | mean len | prefill t/s | decode t/s | compute buffers |
|---|---:|---:|---:|---:|---:|
| `-ub 256` `--fit on` + ngram chain  *(deployed)* | 0.44423 | 2.64 | 1,028 | 50.3 | 8,302 MiB |
| `-ub 256` `--fit on` + dspark only               | 0.63018 | 2.26 | 1,030 | 48.1 | 8,302 MiB |
| `-ub 512` `--fit off` + ngram chain              | 0.28677 | 2.44 | 1,139 | 44.6 | 4,894 MiB |
| **`-ub 512` `--fit off` + dspark only**          | **0.72204** | 2.44 | **1,141** | **51.25** | **4,894 MiB** |

**The candidate beats the deployed configuration on all four metrics**: +11.0% prefill, +1.9%
decode, +62.5% relative acceptance, and **−41% compute-buffer footprint** at the same context.

## The counter-intuitive part, which is the actual finding

**A larger micro-batch with the fitter OFF uses 3.4 GiB LESS memory than a smaller micro-batch with
the fitter ON.** Per-device compute buffers at ctx 262,144:

| | `-ub 512 --fit off` | `-ub 256 --fit on` |
|---|---:|---:|
| CUDA0 | 1,959 MiB | 2,793 MiB |
| CUDA1 | 2,154 MiB | 3,954 MiB |
| CUDA_Host | 782 MiB | 1,555 MiB |
| **total** | **4,894** | **8,302** |

Both offload 52/52 layers and 4/4 draft layers, so this is not a spill difference. It is the
mechanism established in doc 45 and TODO #35: **the per-device compute reserve tracks the address
span the allocator produces, not the tensors in it**, and the fitter's layer rebalancing drives
`splits` from 2 up to as many as 1,719, each split adding cross-device copies and address-span slack.
Disabling the fitter removes the rebalancing, so the span collapses — even while `-ub` doubles.

This is the first time #35 has a **price tag** rather than a description: ~3.4 GiB and ~11% of prefill
at the deployed context.

## The chain's value inverts with batch shape

| `-ub` | dspark only | + ngram chain | chain delta |
|---:|---:|---:|---:|
| 256 | 48.1 | **50.3** | **+4.6%** |
| 512 | **51.25** | 44.6 | **−13.0%** |

So the earlier "+7.9% from the chain" (measured at `-ub 256`, 262,144) was real — but it is **not a
property of the chain**, it is a property of the chain at that shape. At `-ub 512` the neural drafter
alone reaches 0.722 acceptance and the ngram proposals displace better drafts while paying their host
traffic (doc 45: the chain exactly doubles readback volume).

**Rule: re-measure the chain whenever `-ub` or the context changes. Never carry its benefit forward.**

## Why acceptance depends on shape at all

`-ub` changes the target's micro-batch, hence the batch shape the draft's injected context is
computed from, hence floating-point reduction order. Combined with the cache-state result (doc 60: a
prompt-cache hit moves acceptance 0.658 → 0.47 for the same reason), the picture is consistent:
**acceptance is a property of (model, prompt, batch shape, cache state), not of the drafter alone.**

That reframes the "acceptance gap vs the #32218/#32773 builds" (0.770 vs 0.647): those runs used
different fitter plans, therefore different shapes. It is plausibly not an implementation gap at all.

## Caveats on the recommendation

1. **`--fit off` removes the fitter's protection.** The deployment chose `--fit on` because pinning
   `-ngl 99` was the cause of a 4,682-restart loop *when combined with auto-fit*. Here `--fit off`
   + `-ngl 99` is a different, self-consistent configuration: no fitting, all layers offloaded, and
   it fits with 3.4 GiB to spare. But it means a future model or context change that no longer fits
   will fail at load rather than back off. That is an operational judgement for Tim, not a technical
   one.
2. **Two `failed to allocate` retries** appear in the `-ub 512` logs (the fitter's draft-model probe).
   My own rule says retries predict run-to-run spread — but here acceptance is bit-identical across
   reps and prefill varies 1,140.22 / 1,142.41 (0.2%), so the retries are not costing reproducibility
   in this configuration.
3. **Single-sequence only.** All numbers are `--parallel 1`. The deployed server's concurrency
   changes both the shape and the cache behaviour.
4. **One prompt domain.** Code. The chain-vs-no-chain and `-ub` results may differ on prose, where
   acceptance is lower to begin with.

## Follow-up running now

The freed 3.4 GiB may be enough for `f8_e4m3` target KV, which measured **+14% decode** but never fit
at 262,144. Sweeping `-ub` {256, 512, 1024} x cache {q4_1, f8_e4m3} at 262,144 with `--fit off`
(`measure_variant.sh freespace`).

## Update: the candidate is NOT a universal win — it is domain-dependent

Repeating the comparison across three prompt domains (code / prose / repetitive-technical), ctx
262,144, same seed, `temp 0`, `-n 384`, **vision off**:

| domain | config | acceptance | prefill | decode |
|---|---|---:|---:|---:|
| code | deployed (`-ub 256` q4_1 fit-on + chain) | 0.44423 | 1,026 | 50.36 |
| code | **candidate** (`-ub 512` f8 fit-off, dspark only) | **0.77333** | **1,129** | **55.65** |
| prose | deployed | 0.56265 | 960 | **53.91** |
| prose | candidate | **0.64565** | **1,012** | 50.70 |
| repetitive | deployed | **0.52000** | 1,063 | **43.80** |
| repetitive | candidate | 0.47074 | **1,126** | 43.14 |

**Prefill is better with the candidate in every domain** (+6% to +10%) — that part is unconditional,
and it is the fitter result.

**Decode is not.** The candidate wins decisively on code (+10.5%), loses on prose (−6.0%), and is a
wash on repetitive (−1.5%). The reason is the ngram chain, which the candidate drops: on prose and
repetitive text the ngram speculators find real matches, and although that *lowers* the acceptance
ratio (more proposals, same accepts), it *raises* throughput. On code the neural drafter alone reaches
0.773 and the chain only displaces it.

So the honest statement is: **`-ub 512 --fit off` + `f8_e4m3` is the better shape; whether to keep the
chain depends on the workload.** The full 2x2 (shape x spec) per domain is running, along with the
same matrix with vision enabled — the earlier runs could not have seen the projector's buffers because
my `NOVISION` flag was inverted (see below), and the deployed container runs vision.

### A flag bug in my own harness, recorded because it nearly produced a wrong claim

`mode_candidate` was written with `[ "${NOVISION:-1}" = 1 ] || MM=()` intending "clear the mmproj
args when NOVISION=1". Passing `NOVISION=0` therefore *cleared* them, so the run I labelled as
vision-on was vision-**off**. The label came from `${NOVISION:+-novision}`, which appends whenever the
variable is *set and non-empty* — so `NOVISION=0` produced a filename saying `-novision` while I
believed the opposite. Two independent confusions in one line: an inverted sense and a `:+` test on a
boolean-as-string.

Fixed to `VISION=1` meaning "add `-mm`", with the output filename keyed off the same variable. The
general rule: an opt flag should be named for what it turns **on**, and the filename suffix should be
derived from the same expression that controls the behaviour, not from a second reading of it.

## Final matrix: vision on, ctx 262,144, all three domains, full 2x2 (shape x spec)

| domain | deployed<br>`256 q4_1 fit-on +chain` | dep-dspark<br>`256 q4_1 fit-on` | cand-chain<br>`512 f8 fit-off +chain` | candidate<br>`512 f8 fit-off` |
|---|---|---|---|---|
| code        | 50.56  | 48.25 | 50.92 | **55.72** |
| prose       | **54.14** | 51.32 | 50.31 | 50.86 |
| repetitive  | 43.75  | **43.88** | 43.14 | 42.97 |
| prefill (t/s) | 970–1,069 | 964–1,034 | **1,138–1,142** | **1,134–1,136** |
| compute buffers | 8,302 MiB | 8,302 MiB | **4,883 MiB** | **4,883 MiB** |

Decode t/s per cell; acceptance in the same order as before (candidate highest on code 0.773,
dep-dspark highest on prose 0.699).

**Vision costs nothing measurable on throughput**: code 55.72 (on) vs 55.65 (off), prose 50.86 vs
50.70, repetitive 42.97 vs 43.14 — all inside run-to-run noise. The projector is on the CPU, so it
adds ~151 ms per image and no GPU pressure. The candidate configuration fits with vision enabled.

### The chain's value inverts with `-ub`, on both domains

| domain | chain at `-ub 256` | chain at `-ub 512` |
|---|---:|---:|
| code | **+4.8%** (48.25 -> 50.56) | **−8.6%** (55.72 -> 50.92) |
| prose | **+5.5%** (51.32 -> 54.14) | −1.1% (50.86 -> 50.31) |

That is the cleanest statement of what the ngram speculators are worth: they help when the neural
drafter is leaving tokens on the table (narrow micro-batch, acceptance ~0.63) and hurt when it is not
(wide micro-batch, acceptance ~0.77), because their proposals displace better ones and they double
host readback traffic (doc 45).

### Bottom line, honestly

Averaged over the three domains: deployed 49.48, candidate 49.85, cand-chain 48.12, dep-dspark 47.82
t/s decode. So on a mixed workload the candidate is **+0.7% decode, +10% prefill, −41% compute-buffer
footprint** — better, but not the +10% decode that the code-only cell suggested. It is decisively
better for code-heavy use (+10.2% decode), and slightly worse for prose (−6.0%).

**The 3.4 GiB the fitter was wasting is the more valuable asset than the throughput delta.** It is
what let `f8_e4m3` KV fit at 262,144 for the first time, and it is available for other things — most
plausibly `--parallel 2`, which for a serving box is probably worth more than 10% of prefill.
