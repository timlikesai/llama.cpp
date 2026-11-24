# 29-compute-buffer-scaling - the ~0.17 MiB/token compute-buffer growth is the per-layer gathered relative-position bias

Brief: `docs/mxfp8/briefs/AD-compute-buffer-scaling.md`. Note the brief asked for the
filename `29-compute-buffer-scaling.md`; a sibling lane owns `29-fa-vs-f16-kv-memory.md`
(the KV matrix), so this file keeps the `-compute-buffer-scaling` suffix to avoid
colliding with it. No file was renamed or deleted.

**Verdict in one line:** the growth is real and it is **one dense (non-SWA) layer's
materialized relative-bias tensor `kq_b`, 3.5 live copies of `n_kv x n_ubatch x n_head`** -
built in `src/models/inkling2.cpp:356-365` and `src/llama-graph.cpp:4504-4511`.
It is **not** per-applied-layer allocation: holding all 8 dense slots live at once (the
`e046294a7` failure class, 48 slots for 24 physical layers) would put the ctx-slope at
1.31 MiB/token, i.e. 7.8x the measured 0.1698 MiB/token.

---

## 1. Measurement (this session, new)

Driver: `scripts/inkling_native/ad_cbuf_sweep.py` (new). 16 runs, one `llama-completion`
per config, `Inkling-2-Mini-NATIVE.gguf` and a Qwen3.5-0.8B-Q8_0 control, FA on,
1 seq, `n_seq_max=1`, `-lv 4`. Parsed from the **final** `sched_reserve` pass, and both
`llama_kv_cache: size =` lines (the two traps from doc 24 are avoided: last pass only,
both KV lines). Logs kept: `logs/ad-cb-<tag>.log`, table: `logs/ad-cb-sweep.tsv`,
driver transcript `logs/ad-cb-sweep-driver2.log`.

| model | ctx | n_ubatch | FA | CUDA0 | CUDA1 | Host | **compute total** | dense KV | SWA KV |
|---|---|---|---|---|---|---|---|---|---|
| inkling | 2048 | 512 | on | 690.77 | 3,691.94 | 64.59 | **4,447.30** | 64.0 | 240.0 |
| inkling | 8192 | 512 | on | 1,434.77 | 3,907.77 | 136.59 | **5,479.13** | 256.0 | 240.0 |
| inkling | 32768 | 512 | on | 4,170.77 | 5,056.75 | 424.59 | **9,652.11** | 1,024.0 | 240.0 |
| inkling | 8192 | 256 | on | 715.94 | 1,952.43 | 66.79 | **2,735.16** | 256.0 | 200.0 |
| inkling | 32768 | 256 | on | 2,011.94 | 3,248.43 | 210.79 | **5,471.16** | 1,024.0 | 200.0 |
| inkling | 2048 | 2048 | on | 1,382.77 | 5,881.77 | 282.34 | **7,546.88** | 64.0 | 320.0 |
| inkling | 8192 | 2048 | on | 4,010.56 | 4,668.53 | 168.09 | **8,847.18** | 256.0 | 480.0 |
| inkling | 32768 | 2048 | on | - | - | - | **FAILED** | 1,024.0 | 480.0 |
| inkling | 8192 | 512 | off | 1,897.74 | 4,361.74 | 206.55 | **6,466.03** | 256.0 | 240.0 |
| inkling | 32768 | 512 | off | 4,947.21 | 5,152.18 | 151.01 | **10,250.40** | 1,024.0 | 240.0 |
| qwen3.5-0.8B | 2048 | 512 | on | 74.22 | 503.07 | 28.10 | 605.39 | 24.0 | - |
| qwen3.5-0.8B | 8192 | 512 | on | 98.22 | 527.07 | 52.10 | 677.39 | 96.0 | - |
| qwen3.5-0.8B | 32768 | 512 | on | 194.22 | 623.07 | 148.10 | 965.39 | 384.0 | - |
| qwen3.5-0.8B | 8192 | 2048 | on | 392.45 | 2,108.28 | 208.41 | 2,709.14 | 96.0 | - |
| qwen3.5-0.8B | 32768 | 2048 | on | 776.45 | 2,492.28 | 592.41 | 3,861.14 | 384.0 | - |

Reproduced exactly: the 8k/512 and 32k/512 FA-on rows match `logs/fav-fa-on-{8k,32k}.log`
(1,434.77 / 3,907.77 / 136.59 and 4,170.77 / 5,056.75 / 424.59), so the brief's
0.174 MiB/token is confirmed, not a one-off.

### The two axes, separated (the experiment the brief called highest-information)

ctx-slope at fixed n_ubatch (8k -> 32k, total MiB/token of context):

| n_ubatch | inkling slope | qwen slope |
|---|---|---|
| 256 | 0.1113 | (not run) |
| 512 | **0.1698** | 0.01172 |
| 2048 | allocation fails at 32k | 0.04688 |

- **It follows context, and it is not a per-slot allocation**: the qwen control's slope
  is exactly proportional to n_ubatch (0.01172 -> 0.04688 for 512 -> 2048, ratio 4.00),
  i.e. pure scratch. Inkling's slope also rises with n_ubatch (0.1113 -> 0.1698, ratio 1.53)
  but sub-linearly, which is what a *peak-live count* of reused buffers looks like, not what
  a fresh allocation per slot looks like.
- Normalizing to bytes per **(context cell x batch token)** - the natural unit for a
  `{n_kv, n_tokens, ...}` tensor: inkling **347.8 B** at ub 512, **455.9 B** at ub 256;
  qwen **24.0 B** at both ub 512 and 2048 (perfectly linear). The inkling/qwen ratio is
  14.5x for a model with 24 heads vs 8, and the extra structure is the rel-bias path.
- Total compute is close to linear in n_ubatch at fixed ctx (2,735.16 x 2 = 5,470 vs
  5,479 measured at ub 256 -> 512, ctx 8k). So the whole compute buffer is a
  per-batch-token scratch pool; the context term rides on top of it.

## 2. Naming the allocation

One dense layer's additive bias, in graph order (`src/models/inkling2.cpp`):

```
356: // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
357: table  = ggml_mul_mat(pad(attn_rel_bias), Rcur)            # small: {n_head, n_extent+1, n_tokens}, n_extent=1024
363: kq_b   = ggml_get_rows(table, rel_pos)                     # F32 {n_head, n_kv, n_tokens}   <- 1.0 unit
364: kq_b   = ggml_reshape_4d(kq_b, n_head, n_kv, n_tok/n_stream, n_stream)
365: kq_b   = ggml_cont(ggml_permute(kq_b, 2, 0, 1, 3))         # F32 {n_tokens, n_head, n_kv}   <- 1.0 unit
```
then in `src/llama-graph.cpp`, `build_attn(llm_graph_input_mem_hybrid_iswa_rel *, ...)`
(function head `:4482`):
```
4504: kq_mask = attn_mask.raw;                                   # F16 {n_kv, n_tokens} input, 1/24 unit
4506: kq_mask = ggml_cast(kq_mask, GGML_TYPE_F32)               # F32 {n_kv, n_tokens}           <- 0.04 unit
4508: kq_mask = ggml_add(kq_b, kq_mask)                         # broadcast -> F32               <- 1.0 unit
4510: if (flash_attn) kq_mask = ggml_cast(kq_mask, F16)         # FA wants an f16 mask            <- 0.5 unit
```

Unit = `n_kv x n_ubatch x n_head x 4 B`. At n_ubatch 512, n_head 24 that is 49,152 B per
context cell, so the chain is `3.5 x 49,152 = 172,032 B = 0.164 MiB` per context cell.
**Measured 0.1698 MiB (3.5% high)** - the residual is per-buffer alignment plus the f16
`attn_inp_kq_mask` input and `attn_inp_rel_pos` (i32) copies that the 17-split layout
replicates per device. The code comment at `inkling2.cpp:353` already states the shape
out loud ("only the gathered bias is n_kv x n_tokens x n_head"); this is the confirmation
that it is also the dominant memory term.

Shapes are independently visible in the reserve log (`logs/fav-fa-on-32k.log:44190+`):
`attn_inp_kq_mask [f16, ne = {32768, 512, 1, 1}]` x 8 uses (one per dense slot) and
`attn_inp_rel_pos [i32, ne = {32768, 512, 1, 1}]` x 8 GET_ROWS uses, plus the same pair at
`{1536, 512, 1, 1}` x 40 for the SWA slots.

## 3. Answer to the brief's question: is it per-applied-layer (48) allocation? No.

| hypothesis | predicted ctx-slope at ub 512 | measured | verdict |
|---|---|---|---|
| 1 dense layer's chain live (3.5 units) | 0.164 MiB/tok | 0.1698 | **confirmed (+3.5%)** |
| all 8 dense slots live (the 48-vs-24 slot-count class of error) | 1.31 MiB/tok | 0.1698 | excluded (7.8x) |
| the loop doubling the dense term (as it legitimately does for KV) | 0.328 MiB/tok | 0.1698 | excluded (2.0x) |

`ggml-alloc` *is* reusing the per-layer scratch across the replayed slots; the loop
expansion costs compute-buffer memory only through the *device-split replication*
(splits = 17, and the dense slots straddle CUDA0/CUDA1/CPU, which is why CUDA1's flat
term is 2.7x CUDA0's and why the Host term tripled 8k -> 32k), not through a 2x
slot-count multiplier. The 2x KV effect documented in `docs/mxfp8/24-fa-kv-memory.md`
(8 dense caches for 4 unique dense layers) is genuinely per-slot because KV is state;
the compute buffer is not, and the loop does not double it.

Why FA does not help here: FA removes the `n_kv x n_tokens` *attention output*
materialization, but this arch feeds attention an **additive per-head bias**, and the graph
has to hand `ggml_flash_attn_ext` a materialized mask of shape
`{n_tokens, n_head, n_kv}`. The bias is a gather from a `{16, 1025}`-per-head table, so
essentially 100% of those bytes are repetition of at most 1,025 distinct values per head -
it is the cost of having no bias-table path in FA, not of the loop.

## 4. Cost model for 128K / 256K / 512K (weights aside)

`compute_bytes ~= n_ubatch * (A + 3.5 * n_head * 4B * n_kv_dense)`, with
`A ~= 8.0 MiB/batch-token` fitted from the ctx=2048 rows (MoE/FFN/conv state: 128 experts,
n_expert_used 8, ffn 1152, 48 slots - not further attributed, see §6).
`n_kv_dense = n_ctx`, `n_kv_swa = PAD(min(n_ctx, 1024 + n_ubatch), 256)`.

| n_ctx | n_ubatch | ctx term | flat term | total compute (fit) | measured |
|---|---|---|---|---|---|
| 8,192 | 512 | 1.31 GiB | 4.01 GiB | 5.32 GiB | 5.35 GiB |
| 32,768 | 512 | 5.25 GiB | 4.01 GiB | 9.26 GiB | 9.43 GiB |
| 131,072 | 512 | 21.0 GiB | 4.01 GiB | **25.0 GiB** | extrapolated |
| 262,144 | 512 | 42.0 GiB | 4.01 GiB | **46.0 GiB** | extrapolated |

Consequences for the 2x16 GB box (31.6 GiB total, ~18.4 GiB of weights already placed):
- **128K is out of reach with the current graph even at `-ub 512`** (26 GiB of compute
  buffer on 2 cards) - and the same config at `-ub 2048` would need ~93 GiB. This is the
  real blocker, and it is *not* KV (fp8 KV only halves KV, and KV at 256K is ~4-8 GiB).
- Confirmed by direct failure: `-c 32768 -ub 2048` could not reserve at all - the fitter
  asked for a **20,860 MiB CUDA0 compute buffer** and died in `cudaMalloc`:
  `ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 21873757952` /
  `graph_reserve: failed to allocate compute buffers`
  (`logs/ad-cb-ink-ctx32768-ub2048.log:675-678`), while `-c 32768 -ub 512` fits. The
  `-ub`-vs-`-c` trade-off in doc 28's config note is this term, not a fitter quirk.
- Before 128K, the compute buffer must be made O(n_extent) instead of O(n_ctx) per dense
  layer (§5), or n_head-broadcast and gathered inside the kernel.

## 5. Fix options, ranked (proposed, not implemented - the brief asked me to name the allocation first)

1. **`n_head`-broadcast the additive mask and gather inside `flash_attn_ext`.** The mask
   `kq_b` is `n_head x n_kv` per token, built from a `{n_rel=16, n_extent+1=1025}` table
   per head: pass `table` + `rel_pos` to FA and index them in-kernel (CUDA FA already
   reads a per-token mask row). Removes the whole ctx term (0.164 -> ~0 MiB/token) and the
   3 GiB of per-dense-layer traffic it implies. Largest win, biggest change, touches a
   per-backend kernel, so it is an upstream-sized change, not a lane patch.
2. **Compute the bias chain in f16.** The add result is discarded as f32 (it is cast to
   f16 for FA one line later), so `get_rows`/`cont`/`add` can all be f16: 3.5 units ->
   1.5 units, i.e. **-57% of the ctx term** (5.59 -> 2.4 GiB at 32K, ub 512) with a small
   double-rounding change in the mask values. Cheapest real win; gate it behind an
   op-check and validate on the `17 times 4` probe + PPL KL.
3. **Drop the `ggml_cont(ggml_permute(...))` at `inkling2.cpp:363`** by letting the
   following broadcast `ggml_add` work on the permuted (strided) layout, if ggml's add
   supports the 4D broadcast with that stride order (it does for views it can stride, so
   needs a check). -29% (3.5 -> 2.5 units), no numeric change.
4. **Reserve-side only mitigation:** cap the reserve graph's `n_kv` for the mask to
   `min(n_ctx, n_ubatch + extent)`... **rejected**: it would break the reserve-vs-eval
   size match (`~llama_context` "matches expectation" check, `llama-context.cpp:3051`)
   and the mask genuinely covers the whole cache. Not a fix, recorded so nobody tries it.

## 6. Not attributed (honest gaps)

- The **flat ~4.0 GiB at ub 512** (A, above) is batch-proportional and ctx-independent; it
  is dominated by the 128-expert MoE FFN activations across 48 slots, but I did not
  attribute it tensor by tensor. It is *not* the brief's question and does not grow with
  context, so it does not change the 128K conclusion (it is 4 GiB at 8K and 4 GiB at 512K).
- The **sub-linear ubatch response** (slope 455.9 -> 347.8 B/cell-token going ub 256 -> 512)
  is a peak-live-allocation artifact of the fitter's layer split; I fitted a line in
  `(1, n_ubatch)` to the two slopes, which puts the pure-ctx intercept at 54,920 B/cell.
  A node-level dump would settle it, but `GGML_SCHED_DEBUG=2` prints nothing during
  `sched_reserve` (`logs/ad-cb-dbg-ctx32768-ub512.log`, 33 KB, no `node #` lines) because
  the fitting path uses `ggml_backend_sched_reserve_size`, which does not run the split
  printer. Naming the tensors therefore rests on the code read plus the 3.5-unit fit; I did
  not add the temporary instrumented print the brief allowed, because the coefficient
  already discriminates the candidates (3.5 vs 28 vs 168 units).
- Fitter variance: with `-fit auto` the CUDA0/CUDA1 weight split changes between configs
  (7,855/9,344 MiB at 8k vs 10,289/6,911 at 32k in the `fav-*` logs), so the *per-device*
  compute numbers are less stable than the sum. The sum and the slope are stable to the
  last digit across repeats.

## 7. Rules followed

Nothing deleted. No `-ngl` pinned, no `llama-bench`, cache type string not needed here,
build dir untouched (no compile ran), `conversion/**` and `docs/mxfp8/23-mtp-draft.md`
untouched. New files: `scripts/inkling_native/ad_cbuf_sweep.py`, this doc,
`logs/ad-cb-*.{log,tsv}`.
