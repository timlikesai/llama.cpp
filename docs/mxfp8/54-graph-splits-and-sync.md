# 54. Decode is bound by device-boundary crossings, not kernel launches

Two claims I had to retract to get here. First I read the shipped profile's **2,029,000 direct
launches vs 5,340 `cudaGraphLaunch`** as "CUDA graphs are bypassed". Wrong: `GGML_CUDA_GRAPH_DEBUG=1`
shows `CUDA graph warmup complete` and `graphs reused = 11`, and my instrumented
`ggml_cuda_graph_check_compability` produced **zero** `GRAPHOFF` records — no `MUL_MAT_ID` in this
model needs a stream sync, so nothing disqualifies capture. The launch count is mostly *capture-time*
traffic (665 instantiates x ~2k nodes). Second, I guessed the crossings were the dense-bias mask; the
names say otherwise.

## What the data shows

`nsys stats -r cuda_api_sum` on the shipped config (4 requests, 262,144 ctx, DSpark drafting):

| API | calls | share |
|---|---:|---:|
| `cudaEventSynchronize` | 13,845 | 37.7% |
| `cudaLaunchKernelExC` | 1,356,553 | 20.3% |
| `cudaLaunchKernel` | 587,054 | 14.7% |
| `cudaStreamSynchronize` | 29,415 | 12.0% |
| `cudaMemcpyAsync` | 66,144 | 4.8% |
| `cudaMemcpyPeerAsync` | 47,837 | 2.2% |
| `cudaGraphLaunch` | 5,340 | 0.7% |

Device busy was measured at **32.7% / 35.6%**. ~43k syncs and ~48k peer copies over ~4k generated
tokens is **~11 syncs and ~12 peer copies per token** — and the scheduler dump explains why:

```
graphs: 35
splits per graph:            min 2   median 17   max 17
cross-split inputs per graph: median 179
```

The tensors crossing a device boundary, by frequency:

```
 58 x attn_inp_k_idxs      0K -> CUDA0        16 x attn_inp_kq_mask   2K -> CUDA0
 58 x attn_inp_v_idxs      0K -> CUDA0        14 x attn_inp_kq_mask  16K -> CUDA0
 29 x inp_pos              0K -> CUDA0        10 x embd              9K -> CUDA0
 28 x attn_inp_k_idxs      0K -> CUDA1        10 x out_ids           0K -> CUDA1
 20 x rs_s_copy            0K -> both         10 x l_out-12          9K -> CUDA1
```

**Every crossing is a 0-16 KB index/mask/position tensor.** They are not bandwidth; each one is a
`cudaMemcpyPeerAsync` plus the event/sync pair that orders it, and the graph ping-pongs CUDA0<->CUDA1
**17 times per pass**. That is where the 65% idle time goes.

## Why 17 boundaries when `-sm layer` should give one

Weights are placed contiguously (`load_tensors: layer 0..12 -> CUDA0, layer 13.. -> CUDA1`), so the
*model* has one boundary. The extra 16 come from the shared attention inputs: `attn_inp_k_idxs`,
`attn_inp_v_idxs`, `attn_inp_kq_mask`, `inp_pos` are produced once (on the device that owns the batch)
but consumed by layers on **both** devices, and the loop model's dense/SWA cache split plus the draft
context's shared embeddings and output head interleave device segments. Each alternation is a split.

## The lever

Not faster kernels — **fewer crossings**. Ranked:

1. **Replicate the small shared inputs per device instead of copying them across splits.** They are
   0-16 KB; producing them once per device (or memcpy-ing within a device) removes the peer copy and
   its sync pair entirely. ggml already materializes per-device input copies (`CUDA0#embd#0` naming),
   so the mechanism exists — the question is why these index tensors don't use it.
2. **Co-locate the consumers.** If the loop's dense slots and the drafter's shared head/embeddings sit
   on the same device as the batch inputs, the alternation count drops. `-ot` is the knob, but note
   `llama-cli` rejects `-ot`/`-ts` outright (it spawns a curated server subprocess) — use the server or
   `llama-perplexity` for placement experiments.
3. **Reduce the number of graphs.** 35 graphs for one short run; each new batch shape re-captures, and
   capture is where the 2M launches actually come from. A constant verification batch shape (the
   full-block idea from PR #32773) would cut captures — but I measured full-block as a ~20% decode
   *loss* pre-merge, so it needs re-testing on the merged build before anyone believes it.

## Instrumentation notes

- The check that decides this is `ggml-cuda.cu:3313 ggml_cuda_graph_check_compability`, and it is
  **all-or-nothing per graph**: one `MUL_MAT_ID` needing a sync costs the whole graph its capture
  (`break` at :3338). Worth knowing for any future MoE type.
- `GGML_CUDA_GRAPH_DEBUG=1` gates the debug prints, and llama-cli needs `-v` for them to reach the log
  (without `-v` this build writes only warnings — a 155-byte file that looks like "no problem found").
- The instrumentation is preserved at `logs/graphoff-instrumentation.patch` and reverted from the tree.

## Where the 16-17 splits come from (measured, not inferred)

The split-backend sequence is a perfect alternation, 20 of 35 graphs identical:

```
CUDA0 CUDA1 CUDA0 CUDA1 CUDA0 CUDA1 CUDA0 CUDA1 CUDA0 CUDA1 CUDA0 CUDA1 CUDA0 CUDA1 CUDA0 CUDA1
```

and the first node of each segment names the cause:

```
# 1 CUDA0 inputs=11  FILL cache_r_l0        # 9 CUDA0 inputs= 0  SSM_CONV node_6419
# 2 CUDA1 inputs= 1  FILL cache_r_l13       #10 CUDA1 inputs=35  REPEAT node_6431
# 3 CUDA0 inputs= 0  SSM_CONV node_3047     #11 CUDA0 inputs= 0  FILL cache_r_l36
# 4 CUDA1 inputs=39  REPEAT node_3059       #12 CUDA1 inputs= 1  SSM_CONV node_8010
# 5 CUDA0 inputs= 0  FILL cache_r_l24       #13 CUDA0 inputs= 8  REPEAT node_8020
# 6 CUDA1 inputs= 1  SSM_CONV node_4276     #14 CUDA1 inputs= 0  FILL cache_r_l43
# 7 CUDA0 inputs= 8  REPEAT node_4286       #15 CUDA0 inputs= 0  SSM_CONV node_10153
# 8 CUDA1 inputs= 0  FILL cache_r_l31       #16 CUDA1 inputs=75  REPEAT node_10165
```

`cache_r_lN` is the loop model's per-slot residual/conv state, and `SSM_CONV`/`REPEAT` are its
companion ops. Each is assigned to the device that owns slot `N`'s layer, and because the loop replays
the 24 physical layers into 48 slots, **the slot order interleaves the two cards' layer ranges** — so
every replayed pass crosses the boundary again instead of running one contiguous segment per device.

The copy is inserted by `ggml-backend.cpp:1795`:

```c
if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
    // create a copy of the input in the split's backend
```

so a peer copy happens precisely when the destination device cannot hold the source buffer. For a
**leaf input filled from host** (`inp_pos`, `attn_inp_k_idxs`, `attn_inp_v_idxs`, `out_ids`,
`rs_s_copy`) that is the worst available choice: it creates a device->device dependency and the event
pair that orders it, to move 0 KB.

## Proposal (not implemented — and I am not claiming a win)

Assign small leaf inputs to the host buffer so each consuming device does its own H2D instead of one
device copying to the other. Cost: 2 pinned H2Ds of a few KB instead of 1 H2D + 1 peer copy. Benefit:
no cross-device ordering edge per split, which is what the 13,845 `cudaEventSynchronize` calls are
waiting on. Scope: a threshold-gated branch in the assignment pass, plus a way for `set_input` to fill
every device copy (the scheduler already names them `CUDA0#embd#0`, so they are reachable).

Cheaper first experiment if anyone picks this up: move the `-sm layer` boundary so a whole loop pass
lands on one device. Not possible here — 24 physical layers must span 2 x 16 GB — but it is the reason
the problem is loop-model-specific, and a single-device run of a model that fits one card should show
`splits = 1` and no peer copies. I could not run that control: the smallest public Inkling export is
IQ4_NL at 17.6 GB, which does not fit a 16 GB card.

## Config can't fix it — measured (mode `splitsprobe`)

The cheapest hypothesis was that a smarter placement reduces the crossings. It goes the other way. Each
row is the largest `splits =` seen in `sched_reserve` for that placement (ctx 8,192, `-v` required —
without it the line is DEBUG and simply isn't there):

| placement | splits |
|---|---:|
| `-sm layer` (**deployed**) | **17** |
| `-ot token_embd.weight=CUDA0,output.weight=CUDA1` | 70 |
| `-ot .ffn_{gate,down,up}_exps.weight=CUDA1` (experts to one card) | 71 |
| `-ot blk.0.attn_norm.weight=CUDA0,token_embd.weight=CUDA0,output.weight=CUDA0` | 59 |
| `-sm row` | refused to run (arch/fit unsupported) |

Every override multiplies the crossing count by 3-4x. This retroactively explains a result I could not
account for earlier: sharding the two vocab matrices across cards cost **−4.4% decode** — it wasn't the
placement being slow on its own, it was 70 splits instead of 17, i.e. four times the peer copies and
event-sync pairs.

**Conclusion:** the deployed `-sm layer` is the best *configuration*, so #36 is strictly a code problem —
either replicate the small host-origin inputs per device (removing the crossing without moving weights)
or stop the loop model's slot order from alternating devices. Both live in the scheduler/graph layer,
and neither is a knob.
