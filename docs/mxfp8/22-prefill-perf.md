# 22 — Inkling prefill at ~5-10% of peak: profile, find the dominant stall

Status: **partial — the lane captured nsys evidence and timed out before analyzing it; the parent read it and confirmed the dominant stall is host-side synchronization (64.8% of CUDA API time, ~7k calls). Cause not yet attributed per-op.**
Single question: prefill runs at
~507-509 t/s on the NATIVE file, which is ~3.2 TFLOP/s against tens of TFLOP/s of
MMA on two 5060 Tis, and ~4.4 GB/s against ~900 GB/s combined. Neither axis is
saturated. What stalls, and can we fix the dominant one inside `ggml/src/ggml-cuda/**`?

Fence: may edit `ggml/src/ggml-cuda/**`, `scripts/inkling_native/`, this doc, `logs/`.
`src/**` and `conversion/**` are read-only here - patches for those go in the
"Proposed patches outside my fence" section.

## Hypothesis ranking (from the brief, my starting bet)

| # | hypothesis | verdict | number |
|---|---|---|---|
| 1 | `MUL_MAT_ID` forces device syncs, serializing 48 MoE applications | **CONFIRMED (symptom) / cause not yet attributed** | `cudaStreamSynchronize` = **64.8% of all CUDA API time, 4.457 s over 6,991 calls** (avg 638 µs, max 97 ms) — `logs/w-nsys-api-sum.txt` |
| 1b | MoE cache host slot maps cause the syncs | **REFUTED** | `--moe-cache-mib` **defaults to 0 = disabled** (`common/arg.cpp:2751-2759`), and prefill is unchanged with it explicit: 506.26 vs 509.17 t/s (`logs/x-sync-default.log`, `logs/x-sync-nocache.log`) |
| 2 | per-expert `.scale` applied as a full-size elementwise `MUL` (48x/pass) | open | note `cudaMemcpyAsync` = 1,493 calls / 1.201 s (median 3 µs, **max 61 ms**) — copies are happening at a rate worth attributing |
| 3 | Flash Attention silently OFF on the 2-card split | **confirmed earlier by another lane** | `resolve_fused_ops: layer 24 assigned to CUDA1 but FA assigned to CUDA0` -> `Flash Attention not supported, set to disabled` |
| 4 | MXFP8 activation quantizer scalar | open | |
| 5 | MMQ tile config for MXFP8 mirrored from MXFP4, untuned | open | |
| 6 | NVFP4 running in a slower prec mode than its best one | open | |
| 7 | graph capture / scheduling | **partially interesting** | `cudaGraphInstantiate` **429 calls**, `cudaMemGetInfo` 250 calls (max 118 ms), while the run reports `graphs reused = 94 / reallocs = 0` — the instantiate count deserves an explanation |

## Measurements

**The lane captured nsys data and timed out before analyzing it.** Artifacts (all preserved): `logs/w-nsys-prefill.nsys-rep` (4.3 MB), `logs/w-nsys-prefill.sqlite` (10 MB), `logs/w-nsys-kern-sum.txt` (37 KB, per-kernel), `logs/w-nsys-api-sum.txt` (7 KB, per-API), captured over the fixed 2,817-token prefill + 64-token decode run.

Baseline repeat, same harness: prefill 506.26 t/s, decode 20.58 t/s (`logs/x-sync-default.log`).

**Headline: this is not a compute or bandwidth problem, it is a serialization problem.** Two-thirds of CUDA API wall time is host-side `cudaStreamSynchronize`, ~7,000 invocations in a single short run. That fully explains the arithmetic in the intro: ~3.2 TFLOP/s against tens available, ~4.4 GB/s against ~900 GB/s — the device is idle waiting between small pieces of work.

## Where to go next (for whoever picks this up)

Attribute the syncs per op before optimizing anything — the kernel summary is already on disk (`logs/w-nsys-kern-sum.txt`), and `nsys stats --report cuda_api_trace --report gpu_trace logs/w-nsys-prefill.nsys-rep` will interleave API calls with the kernels that precede them, so the pattern per MoE layer should be obvious. Cheaper alternative: run the same profile with `-ot 'exps.weight=CPU'` and with a **non-loop** Inkling build if one can be synthesized — if the sync count scales with layer *applications* (48) rather than layers (24), the loop replay is implicated; if it scales with `n_expert_used`, the MMoE `ids`/`get_rows` path is.

Also worth ruling in or out quickly: whether `ggml_cuda_mul_mat_id_needs_sync` is returning true for our types (`ggml/src/ggml-cuda/ggml-cuda.cu`, the `[TAG_MUL_MAT_ID_CUDA_GRAPHS]` path) — that function is literally named for this symptom, and a `false` there is what keeps MMoE inside the graph capture.

## Fix

not yet — see above. Any fix must keep `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on both backends and 237/237 / 306/306 green.

## Proposed patches outside my fence

(to fill)

## ANSWERED by docs/mxfp8/26-sync-attribution.md (brief Z, later lane)

- H1's cause is attributed: **~77% of the 6,991 syncs are cross-GPU tensor-copy boundaries**
  (5,460 `cudaMemcpyPeerAsync` between CUDA0 and CUDA1, each bracketed by a main-thread
  `cudaStreamSynchronize`, 3.12 s of the 4.46 s). `MUL_MAT_ID`'s `needs_sync` fallback is real but
  only ~21% (746 nodes x 2). `cudaMemGetInfo` x250 and `cudaGraphInstantiate` x429 are load/teardown
  artifacts, not per-layer polling or graph rebuilds.
- Why the copies exist: `-sm layer` puts layer 24+ on CUDA1, but `src/models/inkling2-loop.cpp:68`
  aliases those to the *physical* tensors of layers 0..23 on CUDA0, so ~33 GB of weights/control
  tensors ping-pong over PCIe per prefill (5.4-10.7 GB/s). Evidence + per-size attribution in
  `logs/z-{sync-thread-window,compute-copies,timeline,bigcopy-context}.txt`.
- Direction of the fix: device assignment for looped layers (outside 22's fence), not the MMID path.
  `-sm none` isolates it: prefill 506 -> 642 t/s (`logs/z-sm-none.log`), but decode halves, so it is
  a measurement, not a config to ship.
