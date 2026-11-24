# 40. Compute-buffer placement, and the indexed-bias result that supersedes the fold work

Status: findings + decision. Written 2026-11-22/23 overnight, before merging upstream PR #32808.
All numbers are from this box (2x RTX 5060 Ti 16 GB, `-sm layer`, FA on, draft KV q4_0).

## 1. Why the per-device compute buffers were 3.2 GiB on one card

Two earlier conclusions of mine were wrong, and the instruments that produced them were the
problem, not the reasoning:

- **Per-tensor peaks are not the live set.** `docs/mxfp8/34` attributed the flat
  (~context-independent) compute term to the SWA bias chain because the allocator dump showed
  40 distinct `GET_ROWS` / `CONT` / `ADD` tensors of 144 MiB each. Summing peaks suggested
  ~4.6 GiB. Acting on that, the SWA fold landed and moved the reserve by **36 MiB**. The
  allocator reuses one span for sequential same-shape nodes, so 40 nodes cost about one buffer.
- **Peak-summing across reserve passes is meaningless too.** There are 13 reserve passes per run
  with different graph shapes and unrelated address spaces. Once the dump was scoped to a single
  pass, the picture changed completely.

Per-pass, per-chunk interval union at ctx 8192 (`-ub 512`):

| device | chunk | tensors | span | covered | gaps |
|---|---|---|---|---|---|
| CUDA1 | 0 | 3,300 | 3,205 MiB | **613 MiB** | **2,592 MiB** |
| CUDA0 | 0 | 4,183 | 872 MiB | 584 MiB | 288 MiB |

So CUDA1's compute buffer was ~80% address holes, and the hole total was *constant* at
2,592 MiB across passes and contexts. The gaps sit around the cross-device copies of per-slot
layer outputs (`CUDA1#l_out-12#3`, `-30`, `-42` — the replayed slots of the loop model, which
carry activation buffers but no weights) and below the dense bias chain parked at the top of the
buffer. Two consequences:

- The reserve follows the **address span the allocator produces**, not the number or width of
  tensors placed in it. Deleting nodes cannot recover it. This is why three attempts
  (f16 gather of the table side, sink-node deletion, SWA fold) all measured ~neutral on memory.
- Fixing placement inside `ggml-alloc` is an upstream-sized problem, and it was the wrong one to
  take on: the reason those spans exist is that the graph genuinely holds `n_kv`-wide tensors.

## 2. What was actually wrong: materializing the relative bias at all

The dense layers gather `kq_b` as `{n_head, n_kv, n_tokens}`. At ctx 8192 / `-ub 512` that is
**192 MiB per node**, times gather + permute-copy, per dense layer. Flash Attention exists to
avoid materializing anything proportional to context, and this model's relative-bias path was
defeating it. My fold work (#30, `8def03281`, the sink contraction `cb7f67131`) reduced the
constant and the node count, but it could not remove the ctx-proportional term because the
gathered tensor still had to exist: `ggml_get_rows` hard-codes an F32 return, `ggml_mul_mat`
hard-codes an F32 return (`ggml.c:3402`), and `ggml_flash_attn_ext` has **no bias operand** —
`mask` is the only channel in, and CUDA wants it dense per-head. So the only way to shrink it was
to feed the kernel something narrower.

## 3. Upstream already has that: PR #32808 (and #32740 for the API)

- **#32740** "ggml: compact bias on the existing FLASH_ATTN_EXT" — API + CPU reference + the
  per-backend `supports_op` guards. Note: its own commit `9cd1953` moved the CUDA and Inkling
  graph changes *out* of that PR.
- **#32808** "metal : support indexed bias in FLASH_ATTN_EXT (Inkling 2)" — despite the title it
  carries the **whole** path: `fattn-common.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh`,
  `fattn-mma-f16.cuh`, `fattn.cu`, plus `llama-graph.{h,cpp}`, `llama-kv-cache.{h,cpp}`,
  `src/models/inkling2.cpp` and a `test-backend-ops` case. Self-contained (includes the API).

Contract: `ggml_flash_attn_ext_add_indexed_bias(a, table, indices)` adds
`table[indices[key, query], head, query]` to the mask in F32, with
`ggml_flash_attn_ext_has_indexed_bias(a)` to query it, and a `dense_bias_mask()` fallback for
backends that cannot. Their `set_input_relative_pos` gains a `rows` argument (so the kv_rows path
works) and **keeps** the additive mask with a single clamped bucket — no `-INF` row needed, which
is simpler than my fold and more accurate (f32 bias vs my f16 gather).

## 4. Measured head-to-head (both trees include my alias-aware placement `e046294a7`)

Same public `Inkling-2-Mini-IQ4_NL.gguf`, same flags, two repeats:

| build | prefill | decode | compute @8192 | ctx 131072 `-ub 512` | ctx 262144 `-ub 512` | PPL (16,705 tok) |
|---|---|---|---|---|---|---|
| ours (fold chain) | 844.6 / 846.2 | 46.3 / 46.2 | 872 + 3,205 MiB | **OOM** | **OOM** | 5.5681 |
| #32808 | **1025.3 / 1022.3** | **47.5 / 47.4** | 619 + 3,092 + 137 MiB | works (2,203 + 3,089 + 1,577) | **works** (935 + 1,115 + 782) | 5.5720 |

**+21% prefill, +2.4% decode, and 256K becomes reachable at the default micro-batch** — the
context that could not reserve on our branch at `-ub 512` at all. Total compute at 262144 goes
from >7 GiB (OOM) to ~2.8 GiB.

The PPL pair (0.07% apart) and the greedy divergence on this file are arithmetic-path differences,
not a defect: their bias is summed in F32 inside the kernel, ours was gathered from an F16 table.
Greedy text near a tie can flip between two valid implementations, which is the same effect already
documented for speculative verification.

## 5. Decision

Adopt #32808 into `mxfp8-weights`. What that retires and what it keeps:

| ours | fate |
|---|---|
| `77bd382db` dense mask fold, `8def03281` window fold, the `-INF` bucket + `LLAMA_REL_POS_FOLD_*` enum | **superseded** — their path never materializes the tensor, so there is nothing to fold. Their design is simpler and more accurate. |
| `b18b212c6` same-type gather call site | superseded for Inkling; `ggml_get_rows_same_type` stays as a ggml utility (still covered by the new op tests) |
| `cb7f67131` sink logit inside the contraction | **re-apply on top** — their `inkling2.cpp` still does `mul_mat` then `sub`; the +4.7% prefill is independent of where the bias lands |
| `e046294a7` alias-aware placement, `4a5c1fa3c` fitter zero-span guard, MXFP8 type/CPU/CUDA/MMQ, converter, drafter commits | keep, untouched |

Process notes worth keeping:

- The pass that found this was `gh api search/issues?q=indexed+bias` plus reading the PR's
  *commit list*, not just its title: #32808 is titled "metal" and contains the CUDA kernel.
  Titles are not specs.
- Checking upstream before optimizing again would have saved several hours tonight. The last time
  I said this (`690b87d88`) I meant it, and I still went and did three more optimizations on the
  subsystem upstream was actively rewriting.
- Three null/negative results recorded rather than hidden: `--cuda-graphs` style knob
  (`GGML_CUDA_GRAPH_CHUNK_SIZE` 0/256/512/2048 — all within noise), host expert spill for speed
  (13.5 t/s all-host, 35.9 partial — PCIe-bound as the arithmetic predicted; it is a capacity
  escape hatch only), and the SWA fold's memory effect.
