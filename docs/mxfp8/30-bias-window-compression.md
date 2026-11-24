# 30 — The relative-bias chain owns 91.6% of the context-proportional compute buffer (ablation-proven)

## 0. STATUS: **implemented and committed** as `77bd382db graph : fold the causal mask into the Inkling relative bias`

Sections 1-7 below are the investigation that led here; §7b is what shipped, with one amendment from the follow-up below. Measured on the lossless file, 2× 5060 Ti, `-sm layer`, FA on:

| | baseline | folded | Δ |
|---|---|---|---|
| compute @8k | 5,479.13 MiB | 5,104.67 | −374 |
| compute @32k | 9,652.11 MiB | **6,894.69** | **−2,757 (−28.6%)** |
| **slope** | 0.1698 | **0.0728 MiB/token** | **−57%** |
| prefill / decode | 901.28 / 40.36 t/s | **906.09 / 40.40** | no regression |
| **ctx 131072** | **could not reserve at any KV type** (~25 GiB compute) | **runs: 910.80 / 41.11 t/s** at `-ub 256` with `f8_e4m3` KV (compute 9,893 MiB + KV 2,148 MiB) | **unlocked** |

Gates: correctness string byte-identical on CUDA **and** CPU (`A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:`), `test-backend-ops -p mxfp8` 3/3 backends on CPU, CUDA0 and CUDA1, `test-llama-archs` 587/587, dense `Qwen3.5-0.8B-Q8_0` control unaffected (629 MiB compute, answers `68`).

### The amendment that mattered: `ggml_get_rows` cannot return f16

§7b assumed the gather could emit F16 if the table were F16. **It cannot** — `ggml/src/ggml.c`, `ggml_get_rows()` hard-codes the result type:

```c
// TODO: implement non F32 return
enum ggml_type type = GGML_TYPE_F32;
if (a->type == GGML_TYPE_I32) { type = a->type; }
```

That is **also the complete explanation for the §1 null result**: casting the table to f16 before the gather changed nothing, because the gather returned f32 regardless. A confusing measurement became deterministic the moment I read the op instead of assuming its behaviour. It is why the shipped form narrows *after* the gather:

```
unfolded:  get_rows f32 (1.0) + cont f32 (1.0) + add f32 (1.0) + cast f16 (0.5)  = 3.5 units
folded:    get_rows f32 (1.0) + cast f16 (0.5) + cont f16 (0.5)                  = 2.0 units
```

**Remaining lever:** implementing that TODO (return `a->type`) makes the gather itself half-width — 2.0 → 1.5 units, with the CUDA kernel already supporting `GGML_TYPE_F16` (`getrows.cu:303`), so it is mostly the graph-builder line plus backend support checks. Blast radius is every `get_rows` caller (embeddings especially), so it needs its own careful change rather than a drive-by. That is what stands between us and 256K fitting.

### Also found: SIGFPE at ctx 262144 with `-ub 128`

`llama-completion -c 262144 -ub 128 --flash-attn on` dies with a floating point exception (core dump) *after* printing the memory breakdown. `-c 262144 -ub 256` fails cleanly on compute-buffer size (27 GiB) instead, and `-ub 128` is fine at ctx 8192 with FA on **and** off — so it is a long-context × small-ubatch interaction, not a general `-ub 128` bug. **Attribution untested**: it may well be pre-existing, though the fitter's decisions depend on the buffer sizes this change altered.

---

## 1. Method: ablate, don't fit

Doc 29 fitted `3.5 units` of `n_kv × n_ubatch × n_head × 4 B` to a measured slope and got 0.164 vs 0.170 MiB/token. Convincing — but a fit with two free parameters on three points cannot separate competing causes, and it made a prediction I could test cheaply: **if the chain's f32 copies are the term, halving their dtype must halve the slope.**

I implemented exactly that (cast the `{n_head, extent+1, n_tokens}` table to f16 *before* `get_rows` when `flash_attn`, and keep the broadcast add in f16 — the mask input is already f16 in that mode per `llama-graph.cpp:1149`, so today's code widens f16→f32, adds in f32, and narrows back). Built clean, correctness string intact.

| | ctx 8k | ctx 32k | slope |
|---|---|---|---|
| baseline | 5,479.13 | 9,652.11 | **0.1698** MiB/token |
| my f16 chain | 5,334.99 | 9,507.97 | **0.1698** MiB/token |

**Slope identical to four digits.** Only a fixed 144 MiB came off, and prefill *regressed* 901 → 822 t/s. So I reverted it. Two conclusions: (i) the dtype of those buffers does not set the reserve slope, and (ii) doc 29's fit — right about *which* tensor family, wrong about *why* it costs what it costs — must not be used to size a fix. Whether the f16 path executed at all is an open question I did not chase (see §5); it doesn't matter, because the ablation below is unambiguous.

## 2. The ablation

Delete the chain's contribution at the single point where it enters the mask — `src/llama-graph.cpp`, in `build_attn(llm_graph_input_mem_hybrid_iswa_rel *, ...)`, replacing `kq_mask = ggml_add(ctx0, kq_b, kq_mask)` with a pass-through of `attn_mask.raw`. Logits are wrong by design; only `sched_reserve` numbers are read. (`logs/abl-8192.log`, `logs/abl-32768.log`; extracted from the **final** reserve pass, summing the three backend lines.)

| ctx | with chain | **chain removed** | Δ |
|---|---|---|---|
| 8,192 | 5,479.13 | **4,274.71** | −1,204 MiB |
| 32,768 | 9,652.11 | **4,627.21** | −5,025 MiB |
| **slope** | 0.1698 | **0.0143** | **−91.6%** |

**Compute is now nearly flat in context** — 4,275 → 4,627 MiB across a 4× context increase. The residual 0.0143 MiB/token is the honest floor (per-layer f16 mask input + i32 `rel_pos` + whatever else is genuinely `n_kv`-proportional).

So: doc 29's identification of the *tensor* was correct; my dtype-based remedy was not; and the chain is worth **~92% of the entire long-context compute problem.**

## 3. Why the fix must change the representation, not the dtype

The chain materializes `kq_b` at `n_kv` width, but the gather index space has only **`extent + 1 = 1025`** distinct values, and distances clamp: `src/llama-kv-cache.cpp:2018` is `distance >= 0 && distance < extent ? distance : extent`, with `relative_extent = **1024**` for this checkpoint (`relative_length = 16` is `n_rel`, the table's rank; `sliding_window = 1024`; `relative_clip = 8.0`).

I initially briefed this as 8192 — wrong, and worth being explicit because it changes the size of the prize: **~7/8 of the gathered columns are already redundant at ctx 8192, and ~99.4% at 256K.**

The tail is not merely similar, it is **identical**: `ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0)` (`src/models/inkling2.cpp:357`) adds a zero row that *becomes* bucket `extent`, and the sink subtraction happens **after** the pad —

```cpp
table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
```

— so every far key carries exactly `-sinks[head, token]`: a **per-(head, token) constant**, i.e. rank-1 in the key dimension. A windowed representation is therefore *numerically exact*, not approximate. That is the property that makes this a clean optimization.

The obstacle is documented in the code itself — `llama-graph.cpp:4499-4501`:

> `// TODO: support kv_rows - fill the distances over the kv_rows cell lists and pass the rows to build_attn_mha,`
> `//       needs a kv_rows flash attention with a per-head mask (CUDA has none)`

CUDA `FLASH_ATTN_EXT` wants a **dense per-head f16 mask**, which is precisely why the wide tensor exists. So the real fix is kernel-side: hand FA the `{n_head, 1025, n_tokens}` table plus the i32 `rel_pos` and index them in-kernel (the kernel already reads a mask row per token). That is option 1 in doc 29 §5 and, per this ablation, the only option that matters — worth ~42 GiB at 256K.

## 4. What it unlocks

Budget: 31.6 GiB total, 17.66 GiB weights, **~13.4 GiB free**. Using the ablated flat compute (~4.3 GiB at `-ub 512`, +0.0143/token) and corrected KV (both iswa lines; dense = 8 slots × 4 KiB/token f16, halved by f8_e4m3):

| ctx | KV f16 | KV fp8 | compute | **fp8 total** | fits 13.4 GiB? |
|---|---|---|---|---|---|
| 128K | 4.0 GiB | 2.0 | ~4.6 GiB | **6.6 GiB** | ✅ comfortably |
| 256K | 8.0 | **4.0** | ~4.7 | **8.7 GiB** | ✅ yes |
| 512K | 16.1 | 8.1 | ~4.8 | **12.9 GiB** | ⚠️ marginal — and ✅ at ~9.4 GiB with q4 KV |

Today, without the fix, 128K alone needs 25 GiB of compute and **cannot be configured at any KV type.** This single change is the difference between "256K works" and "256K is impossible."

Caveats I won't smooth over: (a) the flat **~4.0-4.3 GiB batch-proportional term is still unattributed** (MoE activations across 48 slots) and becomes the wall after this fix — halving `-ub` halves it; (b) 512K also needs `n_ctx` ≤ trained 1048576 ✓ and may hit the CUDA0/CUDA1 per-device granularity, so the per-device split must be checked, not just the total; (c) my 901 t/s baseline must be held.

## 5. Settled: the f16 path DID execute, and dtype is genuinely irrelevant to the reserve

I chased the obvious alternative explanation — that we run the **loop** model and it might carry its own copy of the attention block. It does not: `src/models/inkling2-loop.cpp:84,103,107,112` all call the same `llama_model_inkling2::graph::build_layer` (declared at `src/models/inkling2.cpp:273`), which is the function I edited, and `inkling2-loop.cpp` contains no `attn_rel_bias` / `get_rows` / `kq_b` of its own. Zero hits.

So the f16 chain ran and the slope still did not move. **Conclusion: the reserve peak for this chain is insensitive to element width — therefore the ctx-proportional term is driven by tensor *count* (the number of distinct live buffers in the chain), not by bytes per element.** Two independent experiments now support this: the ablation (§2, removing nodes → −91.6%) and this dtype intervention (§1, halving width → 0.0%). Any fix must therefore **reduce the number of live `n_kv`-wide tensors**, i.e. fuse or never materialize — shrinking the dtype of the existing ones cannot help. That also predicts the f16-chain idea would have delivered nothing even if I had kept it, which is why it was reverted rather than tuned.

The one thing still unverified is *which* buffers the allocator holds simultaneously (it is not `GGML_SCHED_DEBUG`-visible: `sched_reserve` uses `ggml_backend_sched_reserve_size`, which skips the split printer). A temporary node-level dump inside `ggml_gallocr` would name them; I judged the count-vs-width conclusion already established by the two experiments above and did not spend a build on it.

## 6. What the fix has to be (sharpened by §5)

Because width doesn't matter and count does, and because `ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)` (`ggml/include/ggml.h:2247-2255`) has **no bias/sinks/alibi operand** — `mask` is the only bias channel, and it must be a dense per-head f16 tensor — the options are:

1. **Add a bias operand to `FLASH_ATTN_EXT`** (or a `gather_add` fused op) so `table` + `rel_pos` + `mask` are combined in-kernel. Removes all `n_kv`-wide intermediates at once — the only approach that attacks *count*, which §5 says is the actual driver. Upstream-sized, CUDA-first (the `llama-graph.cpp:4499` TODO shows upstream already wants a per-head-mask-capable FA).
2. **Cut one node without new kernels**: skip `ggml_cont(ggml_permute(...))` by teaching the broadcast `ggml_add` to consume the strided view. Per §5 this should be the *cheap* win measured in whole-tensor units (1.0 of ~3.5), and unlike the dtype idea it is predicted to work. Needs a check that CUDA/CPU `ADD` accept that stride order — and note the redundant `ggml_cast` I left in the FA branch is itself a wasted full-width buffer, because `ggml_cast` never no-ops on a same-type input (`ggml/src/ggml.c:3636-3644` always builds a CPY node). Verify the final mask type before adding any narrowing cast.
3. Not an option: any further dtype-only idea, including the `n_head`-broadcast trick, unless it also reduces live tensor count.

## 7. Node-level truth (temporary `GGML_ALLOC_DUMP` instrumentation, since reverted)

Added an env-gated dump of `node_alloc.dst.size_max` per node inside `ggml_gallocr_reserve_n_impl`, ran ctx 8192 and 32768 (`logs/dump-8192.tsv`, `logs/dump-32768.tsv`, 9,456 / 9,312 lines), then `git checkout` — **the patch is not in the tree.** What the per-node sizes say at ctx 32768, where one `n_kv`-wide f32 unit (`{24, 32768, 512}`) is **exactly 1,536 MiB**:

| node | count | each |
|---|---|---|
| `GET_ROWS` f32 (`kq_b`) | **8** (= 4 dense layers × 2 loop slots) | 1,536 MiB |
| `CONT` f32 (`kq_b-N`) | one per dense slot | 1,536 MiB |
| `ADD` f32 (mask combine) | one per dense slot | 1,536 MiB |
| `CAST` f16 (FA handoff) | one per dense slot | 768 MiB |

Sum of all node maxima is 95,557 MiB but the reserve is 9,652 MiB — **the allocator reuses buffers across layers, so peak-live is roughly ONE dense layer's 3.5-unit chain.** That reconciles everything: the chain's *widths* are irrelevant to the peak (my §1 result), while the *number of simultaneously-live units per layer* is exactly what the slope measures (my §2 ablation). I could not confirm whether CUDA rejects an f16 `ADD` and lets the scheduler duplicate it through f32 — my greps for a support table found nothing in the layout I expected — and I stopped rather than spend another build on it, because the count-not-width conclusion holds either way.

## 7b. The fix that fits "no major rewrite": fold the causal/window mask into the bias table as extra buckets

The composite FA mask is `bias(rel_pos) + causal_mask`, and **both terms are functions of the same distance**:

- `bias` — already a table lookup over `0..extent` (8192→ no, `extent = 1024`) plus the clamped far bucket.
- `causal_mask` for a dense layer — 0 for `k <= q`, `-INF` for `k > q`: a two-valued function of the sign of the distance.
- for an SWA layer — additionally `-INF` beyond the 1024 window: one more distance range.

So extend the table by **two or three columns** and give `set_input_relative_pos` a couple more branches (`llama-kv-cache.cpp:2018`, host-side, no new ops, no kernels):

| bucket | meaning | value |
|---|---|---|
| `0 .. extent-1` | exact distance | `table[:, d] - sinks` |
| `extent` | far past (dense) | `-sinks` — already the case |
| `extent+1` | future / empty cell / padding | `-INF` |
| `extent+2` | beyond SWA window | `-INF` |

Then the gathered tensor **is** the final f16 mask: the `ADD`, the f32 widening of the input mask and the trailing `CAST` all disappear, and with no `ADD` in the chain there is nothing to force f32 — the gather and its permute can be f16 natively. Estimate: **3.5 units → ~1.5 (−57%)**, i.e. ctx term 0.170 → ~0.073 MiB/token, and compute at 32k from 9.65 → ~6.2 GiB. Needs `-INF` in f16 to behave in FA (check; f16 `-INF` is representable, and today's code already casts the sum to f16 with `-INF` in it — so this is already exercised).

**Gating (must fall back otherwise):** `n_seq == 1` (a multi-sequence mask is not a function of distance), `attn_mask.rows == nullptr` (already asserted on this path), and only layers using `llm_graph_input_mem_hybrid_iswa_rel`. Everything else keeps the current path, so the change is contained.

**And the honest limit:** this does *not* flatten the curve — a residual ~0.073 MiB/token remains, which at 256K is still ~18 GiB. Truly flat requires the gather to stop producing `n_kv`-wide output, i.e. FA indexing the table itself (option 1 in §6, a kernel change). Sequencing is therefore: ship 7b for the −57% with no rewrite, measure, then decide whether the kernel operand is worth it for 256K.

## 7c. What CUDA Flash Attention actually does with the mask (measured from source, `ggml/src/ggml-cuda/fattn*.cuh`) — pins the design

1. **FA really adds the per-head mask values**: `fattn-mma-f16.cuh:859` `KQ_C[...].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);` — so Inkling's per-head relative bias is applied inside FA as an additive term. (Line `:793` has a `slope *` variant — an alibi-style path — worth a look as a possible existing hook.)
2. **Non-finite mask entries are treated as exclusion, not as a value**: `fattn.cu:51` `selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));` in `flash_attn_mask_to_sparse_indices`. Two consequences for §7b: folding `-INF` into a table bucket is **exactly** the semantics FA already relies on (safe), and conversely any bias value that could go non-finite would silently drop keys — so the folded `-INF` must be the *only* non-finite value the gather can produce. Also relevant to #21, since Inkling's SituGLU composition has `atanh`/`sigmoid` in it.
3. **Strides are honored on the sparse-index path**: `fattn.cu:114-115` `s31 = mask->nb[1] / sizeof(half)`, `s33 = mask->nb[3] / sizeof(half)`. So a permuted **view** can be consumed there — which is what makes dropping the `ggml_cont` in §6 option 2 plausible rather than hopeless. Unverified: whether the *main* MMA kernel's `tile_mask` load path (the `nbatch_fa + 8` packed layout at `:859`) likewise accepts arbitrary strides, or requires the repacked dense form. That single question decides whether the `CONT` node (1 of ~3.5 units, 1,536 MiB per dense layer at 32k) can simply be deleted. **Check this first next session — it is the cheapest available win.**

## 8. Rules followed

Nothing deleted (all logs kept, including the reverted attempts). No `-ngl` pinned; both iswa KV lines read; final reserve pass only; no `llama-bench`. `conversion/**`, `tools/mtmd/**`, `ggml/src/ggml-cuda/mmq*` untouched. `git checkout` used only on the two files I edited this session, after confirming no other tracked changes existed.
