# 45 — Profiling the SHIPPED configuration (2026-11-23)

Everything measured in `37-…`, `28-…`, `29-…`, `30-…` was a hand-written CLI config. This is the first
profile of what actually runs: the flag vector is **read out of `docker-compose.yml` at runtime** by
`scripts/inkling_native/profile_shipped.sh`, `llama-server` is launched under nsys with those args, and
the workload is real HTTP requests against it. If the deployment changes, the profile follows.

Command: `docker compose down llama-cpp && bash scripts/inkling_native/profile_shipped.sh both`
Artifacts: `logs/ps-both.{nsys-rep,sqlite}`, `logs/ps-both-{kernels,sync,sync2,peer,gaps}.txt`,
`logs/ps-req-*.json`, `logs/ps-both-server.log`. Nothing deleted, nothing filtered.

Workload: one cold 64 KB WikiText-2 slice prefill (`max_tokens 1`), then three 6 KB code prompts
generating 256 tokens each, `cache_prompt:false` so the drafter and not the prompt cache dominates.

## Headline: this model is neither compute- nor bandwidth-bound. It is host-bound.

| metric | value |
|---|---|
| wall time | 39.09 s |
| **device 0 busy** | **12.78 s = 32.7%** |
| **device 1 busy** | **13.90 s = 35.6%** |
| kernel time, both devices | 24.29 s (of 78 device-seconds available) |
| **`cudaEventSynchronize`** | **13,798 ms** over 15,622 calls |
| **`cudaStreamSynchronize`** | **5,942 ms** over 38,632 calls |
| `cudaLaunchKernelExC` | 5,818 ms over **821,508** calls |
| total kernel launches | **1,229,891** (graph launches: 9,042; graph executions: 973 / 195 distinct graphs) |
| **PCIe traffic** | **62.19 GB H2D + 42.10 GB D2H = 104 GB in 39 s** |
| memcpy device time | 1,755 ms H2D + 635 ms D2H + 176 ms peer |

Two-thirds of the time the GPUs are idle. Launch count is ~31,500 kernels/second across two cards —
that alone is a host-bound signature.

## Finding 1 (the big one): the tap channel moves ~68 GB over PCIe, ~56× more than it should

The dominant transfer size class, both directions, is **2,359,296 bytes**:

```
H2D:  2359296 B  x 14,381   = 33.929 GB
D2H:  2359296 B  x 14,260   = 33.644 GB
```

`2,359,296 = n_embd(2304) x n_ubatch(256) x 4 B` — exactly one ubatch of concatenated target taps. And
the second-most-common small class, `9,216 B = 2304 x 4`, is one *token's* taps (decode) — only ~1,050
and ~715 of those, i.e. **decode tap traffic is trivial; prefill tap traffic is the monster** ✓ which is
consistent with the ~-15% prefill we already measured from forcing host copies (`#3`).

**But the count does not add up, and that is the interesting part.** A 16k-token prefill at `-ub 256`
is 64 ubatches; with 4 taps that is 256 extractions, and even counting the draft-side re-injection at
`--spec-draft-n-ubatch 64` (256 chunks x 4 = 1,024 uploads) the expectation is on the order of 10³, not
1.4 x 10⁴. So either taps are being extracted/uploaded far more often than the graph needs, or whole
2.25 MB ubatch blocks are re-sent where a 64-token slice would do. **I have not proven which** — the
timeline windowing needed for that is `logs/ps-both-sync2.txt`, and it required fixing a bug in my own
`z_sync_attrib.py` (`start[0]` on an int) to produce it. Windows 46-58 show ~330 memcpys per 100 ms
alongside 80-118 ms of kernel busy, i.e. copies and compute interleaved, so this is steady-state
workload behavior, not load-time noise.

Why this matters more than any kernel tuning on this page: the copies themselves cost only ~510 ms of
device time, but the gaps tell the real story. The largest steady-state idle gaps on *both* devices are
pairs of these very transfers, back to back:

```
dev0  gap 49.1 ms  after memcpy[D2H] 2359296 B  | before memcpy[H2D] 2359296 B
dev0  gap 48.1 ms  after memcpy[D2H] 2359296 B  | before memcpy[H2D] 2359296 B   (x8 of the top 10)
dev1  gap 64.8 ms  after memcpy[D2H] 2359296 B  | before memcpy[H2D] 2359296 B
dev1  gap 64.4 ms  after memcpy[D2H] 2359296 B  | before memcpy[H2D] 2359296 B
```

A 2.25 MB copy is ~0.1 ms. The 45-65 ms around it is the **host** being asked and answered synchronously
— extract, then compute, then inject, with the devices waiting on each step. This is the same failure
class as the alias-straddling peer traffic that `1ab82fa7f` removed (+78% prefill): the fix is to stop
round-tripping, not to make the round trip faster.

**Action (`#3`, designed, not yet implemented):** re-enable the device feature buffer and read only the
anchor's row. `layer_inp_dev` is laid out `[n_embd * n_taps, n_batch]` (`llama-graph.cpp:4597-4622`
writes tap `col` into the column block at `col*n_embd`), so one row already *is* the 9,216-float vector
the markov head wants. The device branch of `build_inp_embd_state` (`:2914-2923`) gathers by row already
— but from `h_nextn_src`, asserting `ne[0] == n_embd_state`, and `h_nextn` is 2,304 wide and already
carries DSpark's confidence output. So: add a row read of the target's `layer_inp_dev` (either a second
source for that branch, or a ~20-line `llama_get_layer_inp_dev_row()` used by the host fill), and delete
the `n_embd_state == 0` precondition at `common/speculative.cpp:1150`. Host traffic per draft step then
drops from 2.25 MB x 4 per ubatch to 36 KB per token, or to zero if the gather goes in-graph.

### Finding 1, resolved by measurement (`#18`)

One env-gated print at the universal readback entry (`ggml_backend_tensor_get_async`,
`ggml/src/ggml-backend.cpp:475`) — patch kept at `logs/io-trace-instrumentation.patch`, reverted from
the tree after use — identified the tensors outright:

```
RD layer_inp-5   2359296 B  op=DUP  ne=2304,256,1,1     <- the taps, exactly as suspected
RD layer_inp-23  2359296 B
RD layer_inp-41  2359296 B
RD layer_inp-48  2359296 B                               <- ids 5/23/41/48 == mtp.target_layer_ids
RD result_output 2400696 B  n=14                         <- 3 tokens x 200058 vocab x 4B (spec verify)
```

Two proven facts, one still open:

1. **The ngram chain doubles everything.** Identical workload, only `--spec-type` differing:
   `draft-dspark` = 217 readbacks / 94.78 MB of taps; `draft-dspark,ngram-mod,ngram-map-k` = **434
   readbacks / 189.55 MB**, with every size class doubled in lockstep (9→18, 14→28). Throughput at
   `-c 4096` was a wash (1021.1/42.5 vs 1014.8/42.8 t/s), so the chain's price is host traffic, not
   device time — at 262144 ctx that doubling applies to a far larger prompt, which is most of why the
   shipped config moves 104 GB over PCIe.
2. **Extraction is NOT slot-amplified.** `llama-context.cpp:2786-2813` loops the `embeddings_layer_inp`
   bitmask and copies only enabled layers, sized by `ggml_nbytes(t)` which tracks `n_tokens` — 4 copies
   per ubatch, verified (23.62 MB per tap for a ~2,563-token prompt = 2,563 x 9,216 B ✓ exact). My
   earlier guess that this was the 48-vs-4 loop-slot bug class (`1ab82fa7f`-shaped) was **wrong**.
3. **Still open:** the residual gap between the profile's 14,260 full-ubatch copies and the
   arithmetic-after-the-chain-doubling (~1-2 k). The remaining suspect is **whole-prompt re-injection on
   every draft round** (`--spec-draft-n-ubatch 64` chunks the injection, but if the prompt is re-injected
   per drafting round rather than once, the count multiplies by the number of rounds). Unproven — needs a
   counter in the injection loop, so it is folded into `#3` rather than asserted here.

## Finding 1, FINAL accounting (supersedes the numbers above — read this one)

Instrumenting **all six** transfer entry points (`ggml_backend_tensor_set/get`, async and 2-D variants,
in `ggml/src/ggml-backend.cpp`, env-gated; patch in `logs/io-trace-instrumentation.patch`, reverted after
use) and running the identical workload inside `llama-server` gives the true picture:

| | traced bytes |
|---|---|
| total traced (one server session, 4 requests) | **26.71 GB** |
| one-time weight uploads (51 tensors ≥ 100 MB, e.g. `output.weight` 926 MB, `blk.20-23.ffn_gate_up_exps` 700 MB each) | **16.80 GB = 63%** |
| **steady state** | **~9.9 GB** |
| └ `inp_target_features` H2D, 2,359,296 B × 387 | 0.913 GB |
| └ `inp_target_features (reshaped)(permuted)` H2D × 299 | 0.705 GB |
| └ `result_output` logits D2H, 2,400,696 B × 209 (+52 MB × 9) | 0.970 GB |
| └ `layer_inp-5/23/41/48` D2H, 2.36 MB × 92 each | ~0.88 GB |

**So my headline above ("~68 GB of the 104 GB is the tap channel") was wrong by roughly 40×**, and the
corrected claim is: the tap/injection path is about **1.6 GB per four requests, ~16% of steady-state
traffic — the single largest steady-state item, but not the bulk of the 104 GB.** The reason nsys
reports 62 GB H2D is that it counts the **whole session**, including one-time weight upload *and* the
fitter's repeated context construction: the log line
`failed to measure the memory of the extra model, fitting without it` means a context was built, failed,
and the model was loaded/probed again — each pass pushing ~17 GB over PCIe. That is a startup-cost bug,
now filed separately, and it is also why run 1 and run 3 had byte-identical memcpy totals (deterministic,
not stale).

Two further corrections of mine, both wrong-then-checked:

- **`EXT` counts are not copies.** My counter sits at the top of `extract_layer_inputs`, before the
  bitmask loop, so the draft context's 387 `EXT` calls extract nothing (nothing ever enables bits on
  `ctx_dft` — `llama_set_embeddings_layer_inp` is called on `ctx_tgt` at `speculative.cpp:604` and `:1136`
  and nowhere else). I had written that up as "0.92 GB wasted on the draft" — retracted. It is also why
  the correct tool was the tensor trace, not the call counter.
- My aggregator first multiplied count×count instead of size×count, which sorted "top by bytes" by
  frequency and initially hid the weight uploads entirely.

**What survives, and it is still the actionable part:** the target-feature injection is uploaded
**twice** — as `inp_target_features` and again as `CUDA0#inp_target_features (reshaped)(permuted)`
(387 + 299 calls, 1.62 GB). And `build_inp_target_features` (`src/models/dflash.cpp:224-237`) already
contains the branch that would eliminate both host writes:

```c
if (g.ubatch.embd_row) {
    GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp);
    inp.embd_rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, g.n_tokens);
    ggml_set_input(inp.embd_rows);
    return ggml_get_rows(g.ctx0, g.layer_inp_src, inp.embd_rows);   // zero host traffic
}
```

Inkling takes the `else` branch solely because `common/speculative.cpp:1150` requires `n_embd_state == 0`
for `use_dev_features`, and Inkling's markov head needs 9,216-float state rows. So the fix for `#3` is
exactly as designed, and its measured prize is now honest: **remove ~1.6 GB of H2D per four requests
(~16% of steady-state PCIe traffic), plus the 45-65 ms idle gaps bracketing those copies, plus whatever
the per-device permuted duplicate costs on a split graph.** Not the 40× I first claimed.

### The trap that nearly produced two false findings

My first `llama-cli` trace run printed its result line (`[ Prompt: 1016.3 t/s | Generation: 42.3 t/s ]`)
and then **never exited** — `--jinja` puts llama-cli in conversation mode, waiting on stdin. It sat there
holding 9,610 + 11,476 MiB. Consequences, both of which I nearly reported as findings:

- the ngram-chain test appeared to fail with `failed to allocate CUDA0 buffer of size 7964370176`, which
  I wrote up as "a fixed-size ~8 GB allocation, batch-independent" — **retracted**, it was contention;
- `docker compose up -d llama-cpp` then failed with a 21.4 GB CUDA0 request (the fitter trying to put
  the whole model on one card) — **also contention**, not a config regression.

Rules added: every non-interactive harness invocation gets `< /dev/null`; and before believing any
allocation failure, `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`.
A process that printed a summary is not necessarily a process that finished.

## Finding 2: index/copy kernels, not math, are eating ~4 s

| kernel | calls | ms | note |
|---|---|---|---|
| `concat_cont<unsigned int>` | 33,254 | 746 | **unsigned int** => index concats, not my f32 bias table |
| `cpy_scalar_transpose<__half>` | 14,900 | 624 | scalar, transposing, half — a permute-copy the vector path isn't taking |
| `scale_f32` | 104,614 | 592 | 5.7 us/call: launch-latency-bound |
| `rms_norm_f32` | 39,714 | 388 | |
| `k_bin_bcast<op_mul>` / `<op_add>` | 72,859 / 57,763 | 336 / 243 | 4.2-4.6 us/call, same story |
| `cpy_scalar_contiguous<float,__half>` | 30,018 | 279 | conversions that should ride a vec kernel |
| `concat_non_cont<unsigned int>` | 24,192 | 242 | |

~1.8 s of device time in copy/concat/scale at 5-20 µs per call — the *count* is the cost. Note this
**refutes my own first reading**: I expected the sinks `concat` from `cb7f67131` here, but every concat
in the profile is `unsigned int`, so that f32 table concat is not what these 57k calls are. Attributing
each of these needs the graph-node dump (`GGML_ALLOC_DUMP` / `scripts/inkling_native/parse_graph.py`),
which is `#4`; I am not going to name owners from kernel names alone.

## Finding 3: MXFP8 MMQ costs ~4x per byte-pair versus NVFP4 — ~2x of which may be mine

| kernel | calls | ms | us/call |
|---|---|---|---|
| `mul_mat_q<40 = NVFP4>` | 8,360 | 5,741 | 687 |
| `mul_mat_q<44 = MXFP8>` | **760** | **2,161** | **2,844** |
| `flash_attn_tile` | 5,184 | 3,632 | 701 |

Type 44 is my MXFP8. Per unit of K it processes 8-bit values versus NVFP4's 4-bit on a `1X`/K=32 tile
against the FP4 path's `4X`/K=64 — so some multiple is inherent (PTX gives 8-bit no wider form, see
`44-…` §2). But 4.1x per call for 2x the bytes suggests the tile loader is leaving something on the
table: mine (`mmq-load-tiles.cuh:1755`) does one `memcpy` of the 32-byte payload per thread per block.
At 8.9% of all device time, halving it is ~4% overall. Worth an attempt after Findings 1-2, because it
is arithmetic-bound and therefore harder.

## Finding 4: q4_1 KV dequant is *not* a problem, and that settles a question

`dequantize_block_q4_1` : 13,542 calls, 171 ms = **0.7%** of device time. So the q4_1 cache choice costs
essentially nothing in dequant work — the ~7 t/s q4_1-vs-q4_0 decode gap in `41-…` must come from the
extra per-block `min` in the inner loop, not from cache pressure. Fine either way; recorded so nobody
re-litigates the cache type on performance grounds.

## What is *not* here, deliberately

- The mmproj is excluded (`PROFILE_VISION=1` includes it) so vision's CPU work doesn't blur GPU
  attribution. The projector runs on CPU anyway (`42-…`).
- `--metrics` is dropped (timing overhead in the trace); the `--server` token is dropped because it is
  `/app/tools.sh`'s entrypoint selector, **not** a llama.cpp flag — anyone copying compose args to a
  native run must strip it, or they get `error: invalid argument: --server`.
- Load-time transfers appear in "biggest single transfers" (926 MB, 700 MB x4, 382 MB x8 at 15-25 GB/s)
  and the 1.2-1.3 s top gaps are model load, not steady state. Excluded from the findings above.
- `ab_nsys_report.py` reports `{'bare': 1238964}` for graph vs bare launches; that classifier cannot see
  graph correlation, so "graphs are not being used" would be an over-claim. The honest evidence is
  821,508 `cudaLaunchKernelExC` against 9,042 `cudaGraphLaunch`.

## Ranked actions coming out of this profile

1. **`#3` device-side tap delivery** — removes ~68 GB of PCIe and the 45-65 ms sync gaps around it. Big
   prefill lever, moderate change, and it likely also closes `#8` (the missing anchor taps) because the
   prompt's final row is still resident in the device buffer.
2. **`#3b` (new, from Finding 1) attribute the 56x copy multiplication** — instrument extract/inject call
   sites with counts per phase before designing further; if taps are being re-extracted per chunk this
   may be a plain bug with a much cheaper fix than the device path.
3. **`#4` kill the 5-20 us copy/concat/scale kernels** — ~1.8 s of device time and most of the launch
   count; requires the graph-node attribution `parse_graph.py` already does.
4. **`#5-done` / Finding 3: MXFP8 tile loader vectorization** — bounded (~4% ceiling) but self-contained.
5. **`#6` MoE decode launch-bound work** — still true, but this profile shows the *prefill-side* host
   traffic is bigger, so it now ranks below #3.
