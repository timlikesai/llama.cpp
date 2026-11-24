# 26 — Attribute the ~7k device syncs (Brief Z job 1) + FA/KV memory (job 2)

Status: **job 1 root cause found and measured; job 2 not measured (out of time) — scoped below.**
Fence: `ggml/src/ggml-cuda/**`, `scripts/inkling_native/`, this doc, `docs/mxfp8/22-*`, `logs/`.
Every claim is either `file:line` (this tree) or a number in `logs/z-*.txt`.

## BOTTOM LINE (job 1)

**~77% of the 6,991 `cudaStreamSynchronize` calls are the cross-GPU tensor-copy boundaries of the
split graph, not `MUL_MAT_ID`.** The Inkling loop graph is split over both 5060 Tis by layer, but
looped layers 24..47 reuse the *physical* tensors of layers 0..23, which live on the other GPU.
Every node boundary that crosses devices becomes one `cudaMemcpyPeerAsync` (+ event handshake)
**plus one blocking `cudaStreamSynchronize` on the main thread** — 5,460 copies / 5,411 syncs per
run, ~33 GB of peer traffic in the 5.5 s prefill, and the host sits inside
`cudaStreamSynchronize` for **90-100 ms of every 100 ms window**.

| share | source | per-run count | sync time |
|---|---|---|---|
| **77%** | CUDA0<->CUDA1 tensor copies of the split graph: each cross-device boundary is `cudaMemcpyPeerAsync` + `cudaEventRecord` + `cudaStreamWaitEvent` (`ggml-cuda.cu:3273-3286`) immediately followed by a full stream sync on the main thread | 5,460 copies / 5,411 syncs | 3.12 s |
| **21%** | `MUL_MAT_ID` fallback path, 2 syncs + 2 `cudaMemcpyAsync` per node (`ggml-cuda.cu:2719-2747`) | ~746 MMID nodes | 0.01 s (the wait shows up as 1.20 s of `cudaMemcpyAsync` API time instead - pageable host memory makes the "async" copy block) |
| ~2% | init/load: `set_tensor` per weight at load + `init_tensor` padding memset (`ggml-cuda.cu:1064`, `1076-1243`) | ~80 | 1.24 s (this is where the max-single 97 ms lives) |

All 6,991 syncs are issued by **one thread** (the main `llama-completion` thread, tid
289967436183886) — so they are host-side blocking in the decode loop, not worker threads.
Per-thread API windows around each sync (`logs/z-sync-thread-window.txt`) name the call sites:

| n | sync ms | 3 preceding api calls on the same thread | then |
|---|---|---|---|
| **2,700** | **1,997** | `cudaMemcpyPeerAsync, cudaEventRecord, cudaStreamWaitEvent` | `cudaMemcpyPeerAsync` |
| 582 | 403 | `cudaEventRecord, cudaStreamWaitEvent, cudaGraphLaunch` | `cudaMemcpyPeerAsync` |
| 576 | 60 | 3x `cudaLaunchKernelExC` | `cudaMemcpyPeerAsync` |
| 384 | 287 | `cudaEventRecord, cudaStreamWaitEvent, cudaLaunchKernelExC` | `cudaMemcpyPeerAsync` |
| 379 | 192 | `cudaGraphKernelNodeGetAttribute, cudaGraphDestroy, cudaGraphLaunch` | `cudaMemcpyPeerAsync` |
| 268/144/124/72 | 182 | other launch triples | `cudaMemcpyPeerAsync` |
| **741 + 77** | **12** | `cudaMemcpyAsync, cudaStreamSynchronize, cudaMemcpyAsync` | `cudaMemcpyAsync` |

Sum of the peer-copy-adjacent rows: **5,411 syncs / 3.12 s of the 4.46 s (70%)**. The
`peerAsync -> eventRecord -> waitEvent` triple is exactly
`ggml_backend_cuda_device_cpy_tensor` (`ggml-cuda.cu:3273-3286`), which itself does **not** sync;
the sync belongs to the caller between consecutive cross-device tensor copies, and the last row is
the `MUL_MAT_ID` fallback (`ggml-cuda.cu:2719-2747`).

Arithmetic that pins the MMID share (from `logs/z-sync-attrib-nsys.txt`, script
`scripts/inkling_native/z_sync_attrib.py`):

```
sync count by PRECEDING api          (prev api -> next api) around syncs
cudaStreamWaitEvent   2751  2016 ms   cudaStreamWaitEvent -> cudaMemcpyPeerAsync  2702
cudaLaunchKernelExC   1423  1159 ms   cudaLaunchKernelExC -> cudaMemcpyPeerAsync  1190
cudaGraphLaunch       1176  1184 ms   cudaGraphLaunch     -> cudaMemcpyPeerAsync  1055
cudaMemcpyAsync        888    20 ms   cudaMemcpyAsync     -> cudaMemcpyAsync       743   <- MMID pair
cudaLaunchKernel       340     0 ms   cudaLaunchKernel    -> cudaMemcpyPeerAsync   340
cuLaunchKernel         124    44 ms   cuLaunchKernel      -> cudaMemcpyPeerAsync   124
```
(The first version of this section derived the "followed by `cudaMemcpyPeerAsync`" pairing from a
process-global next-record lookup; the table above is the corrected same-thread version and the
conclusion is unchanged.)

`cudaMemcpyPeerAsync` is called exactly 5,460 times, matching the 5,460 `cudaEventRecord` /
`cudaStreamWaitEvent` triples one-for-one. The 741+77 `cudaMemcpyAsync <-> sync` pairs are the
`MUL_MAT_ID` fallback (`2719`/`2746`), whose own memcpies bound it at 1,493/2 = 746 nodes.

### The copies are loop-layer cross-device traffic, not activations-per-token

`logs/z-compute-copies.txt` — every compute-phase transfer is `cudaMemcpyPeerAsync`, sizes
mirrored device0->device1 and device1->device0 in equal counts:

| bytes | n (per direction) | matches |
|---|---|---|
| 4,718,592 | 1,785 | `attn_k/attn_v.weight` BF16 [2304,1024] = 4,718,592 B **and** 2048x576 f32 |
| 18,846,720 | 240 | 2048 x 2304 f32 hidden state (n_hidden 2304) |
| 9,437,184 | 260 | 2048 x 1152 f32 (ffn dim 1152) |
| 27,648 / 12,288 / 9,216 | 336 / 336 / ~413 | `attn_res_query` [2304,3] f32, `attn_q_clip` [128,24] f32, `attn_res_query` [2304] f32 — per-layer control tensors |

The small per-layer control tensors (`attn_res_query`, `attn_q_clip`, 24 unique layers each)
crossing the link is the fingerprint: **weight/control tensors of physical layers are being
fetched across PCIe because the graph node belongs to the other device.** The load-time warning
says the same thing in one line (`logs/z-base-verbose.log`):

```
W resolve_fused_ops: layer 24 is assigned to device CUDA1 but Flash Attention is assigned to device CUDA0
```
Layer 24 is the first *loop* application; `src/models/inkling2-loop.cpp` `post_arch_tensors` sets
`layers[il] = layers[layer_phys(il)]` (`src/models/inkling2-loop.cpp:68`, `layer_phys` at `:3`),
so il=24..47 point at tensors the fitter placed on CUDA0 while their graph nodes belong to CUDA1.

Peer traffic in the compute phase: 66.4 GB of memcpy activity over 6.19 s = **10.7 GB/s**
(`logs/z-peer-doublecount.txt`; CUPTI may record a peer copy once per participating device, so
read this as 5.4-10.7 GB/s — either way it is the whole budget of a PCIe Gen4 x16 link between
two consumer cards with no NVLink).

### Load phase is a separate, smaller sync source (do not confuse it)

`t=3.3-4.6 s` (windows 33-45 of `logs/z-timeline.txt`): 700 syncs, 800 copies, 17 GB H2D, **zero
kernel time** — that is model load: 20x `ffn_gate_up_exps.weight` @ 382,205,952 B (NVFP4, blk 0-19),
20x `ffn_down_exps.weight` @ 191,102,976 B, 4x 700,710,912 B + 4x 350,355,456 B (MXFP8 blk 20-23),
1x 926,318,592 B (`token_embd`/`output.weight`) — sizes cross-checked against the GGUF tensor
table in `logs/z-gguf-sizes.txt`. Those go through `set_tensor`/`set_tensor_n` and are one-sync-per
tensor, i.e. load is serialized at ~4 GB/s. Not the prefill stall, but it is why "max 97 ms" and
"max 61 ms" appear in the W lane's API summary.

### CUDA graphs

429 graphs captured, 1,177 `cudaGraphLaunch`, 527 device-side graph executions, 364 ms device time.
All 43,007 kernel activity records report `graphId = 0` (this nsys build traces graph kernels at
node level without stamping the id), so "graphs on/off" cannot be read from `graphId`; the
`cudaGraphLaunch` count can. Because `ggml_cuda_graph_check_compability` (`ggml-cuda.cu:3313-3349`)
does `use_cuda_graph = false; break;` on the **first** MMID node that `ggml_cuda_mul_mat_id_needs_sync`
flags, ~746 unsupported MMID nodes knock CUDA graphs out for the whole cgraph on that device —
real, but a *second-order* cost next to the peer ping-pong.

### Fix

**Rejected as insufficient (in-fence):** making the cross-device copies non-blocking (dropping the
`cudaStreamSynchronize(cudaStreamPerThread)` in the buffer-interface copies
`ggml_backend_cuda_buffer_cpy_tensor{,_n}`, `ggml-cuda.cu:1183` / `:1234`, or hoisting the sync out
of the caller that sits between consecutive `ggml_backend_cuda_device_cpy_tensor` triples) would
remove ~5,400 host blocks but leave the same ~33 GB of peer traffic on the critical path, and would
break the liveness guarantee the scheduler relies on (a copy source is a graph buffer the allocator
may hand to the next node). `device_cpy_tensor` already does the event-based wait
(`ggml-cuda.cu:3281-3286`), so there is no simple "use events instead" win left here. Would need an
upstream design conversation about batching the per-node cross-device copies.

**Actual fix (out of fence, for the parent to land or report):** the device for a looped layer's
nodes must follow the device of its **physical** tensors — i.e. after `inkling2-loop.cpp`
`post_arch_tensors` aliases `layers[il] = layers[layer_phys(il)]`, the fitter /
`resolve_fused_ops` / `llama-context` must not place il>=24 nodes on CUDA1. Until then the
workaround for users is a single-device config, which measurement supports:

```
-sm layer (baseline, -n 8)  : prefill 506.26 t/s   decode 20.58 t/s   (logs/x-sync-default.log)
-sm none   (same -n 8)      : prefill 642.40 t/s   decode 11.46 t/s   (logs/z-sm-none.log)
-sm layer (baseline, -n 64) : prefill 507.87 t/s   decode 22.26 t/s   (logs/z-base-verbose.log)
```
`-sm none` (no per-layer split) removes the ping-pong and buys **+27% prefill** with the same
2,817-token prompt; decode regresses (20.6 -> 11.5 t/s, both at `-n 8`) because `-sm none` also loses the
CUDA1 weight capacity and the fitter starts overriding tensors to CPU
(`W llama_model_loader: tensor overrides to CPU are used with mmap enabled` in that same log),
so it is *not* a shippable configuration — it is the measurement that isolates the split as the cause.
`-sm row` cannot even load here: `E llama_model_load: device CUDA0 does not support split buffers`.

**Upstream report skeleton** (facts above, all reproducible from `logs/w-nsys-prefill.nsys-rep`):
1. looped/re-used layer tensors + `-sm layer` cross-device placement -> 5,460 peer copies and
   ~5,400 host-blocking syncs per run;
2. each cross-device tensor boundary is serialised by a full `cudaStreamSynchronize` on the main
   thread (2,700 of them sit directly between two `cudaMemcpyPeerAsync` triples), so no copy can
   overlap compute;
3. `resolve_fused_ops` turns FA completely off on such a split (job 2 blocker, below).

## Hypothesis scoreboard (job 1)

| # | hypothesis (brief Z) | verdict | number |
|---|---|---|---|
| 1 | MMID fallback / `needs_sync` | **PARTIAL 21%** | 746 nodes x 2 syncs = 1,492 (`ggml-cuda.cu:2719-2747`); also kills CUDA graphs for the whole cgraph (`ggml-cuda.cu:3328-3338`) |
| 1b | MoE cache host slots | REFUTED (prior lane, confirmed) | 506.26 vs 509.17 t/s |
| 2 | scales with applications (48) vs layers (24) | **applications** | peer copies of per-layer control tensors = 336 = 14 x 24 unique, and the trigger is il>=24 (the second pass) |
| 3 | per-expert `.scale` `get_rows`/`repeat_4d` | **CONTRIBUTES, not via host sync** | `k_get_rows_float_vec` 313 calls / 377 ms = 14.7% of all device kernel time; 4,718,592-B `cpy_scalar`/`concat_cont` traffic is the visible part |
| 4 | `cudaMemGetInfo` polled per layer | REFUTED | 250 calls, 0.127 s, clustered at load (`logs/z-timeline.txt` shows them outside compute windows) |
| 5 | graph instantiate x429 | EXPLAINED | 429 capture / 1,177 replay / 429 teardown at exit; not a per-ubatch rebuild |
| NEW | cross-device peer copies (loop layers) | **PRIMARY 77%** | 5,460 copies / 5,411 syncs, 3.12 s of the 4.46 s sync time, 5.4-10.7 GB/s peer traffic |

## Job 2 — FA / KV memory matrix

**Not measured.** Time spent on job 1 root cause; the matrix needs a single-card config and the
time to run 6 configs. What I did establish, which changes the shape of the question:

1. **FA cannot be measured at all on the shipped 2-GPU config.** `resolve_fused_ops` disables FA
   outright (`logs/z-base-verbose.log`, first line after load). So Tim's report "enabling FA uses
   more memory" is about a single-device run, and the loop-layer device conflict above is the
   first thing to fix.
2. The failed **20,156,364,928 B (18.8 GiB) CUDA0 compute-buffer** request
   (`ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 20156364928`, then
   `sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism`) is
   the same size class as Tim's "FA uses more memory" symptom and is a *compute buffer*, not KV.
   Any memory matrix must log **per-device KV bytes and compute-buffer bytes separately**, or it
   will measure the wrong line.
3. Model geometry for the theoretical column, read off the GGUF (`logs/z-gguf-sizes.txt`):
   `n_embd 2304`, `ffn 1152`, 128 experts, 24 unique layers / 48 applications, vocab 201,024,
   `attn_k/attn_v` are `[2304, 1024]` BF16 -> `n_kv_head 8`, `head_dim 128` (matches brief's
   `head_dim_ctl 128`, `head_count_ctl [16,8]`). So full-attention KV per token per layer =
   `2 (K,V) x 8 heads x 128 dim x sizeof(type)`; SWA layers cap at `sliding_window 1024`.
4. Cheapest way to run the matrix when someone picks it up: single device + `-ot` to force the
   expert tensors off the critical link is NOT needed for memory — just `--device CUDA0` with
   `-c 2048/8192/32768`, `{fa on, off}` x `{q8_0, f16}` KV, and read the two numbers the loader
   prints per device (`KV buffer size`, `compute buffer size`). Two of those 8 configs will OOM at
   32k on one 16 GB card; record them as OOM rather than skipping.

## Corrections to my brief

- "syncs come from `MUL_MAT_ID`" -> only 21%. The other 77% are the cross-device tensor copies
  (peer copy + event handshake, then a full stream sync before the next one). The in-fence sites
  are `ggml-cuda.cu:3273-3286` (`device_cpy_tensor`) and `:1179/1229` (`buffer_cpy_tensor{,_n}`);
  the blocking `cudaStreamSynchronize` itself is issued by the caller at the backend boundary, which
  is outside this fence.
- "1,493 `cudaMemcpyAsync` calls are unexplained; 1,493 ~ 31 x 48" -> explained: `2 x 746` MMID
  fallback nodes. No other source issues `cudaMemcpyAsync` during compute.
- "`graphs reused = 94` vs `cudaGraphInstantiate` 429 deserves an explanation" -> 429 is one graph
  per distinct cgraph size x device, instantiated at capture, destroyed at teardown; it is not a
  rebuild loop. (My rerun of the same harness reports 62 reuse, the W lane 94 — the counter is
  per-run-phase, don't use it as a proxy for CUDA-graph health.)
- Brief says a python KLD lane may be using the GPU: GPUs were idle when I measured
  (13 MiB / 13 MiB) and my baseline reproduces the W lane within 0.3% (507.87 vs 506.26 t/s).

## Artifacts

- `scripts/inkling_native/z_sync_attrib.py` — nsys-sqlite sync/api/kernel/memcpy attribution tool.
- `logs/z-sync-attrib-nsys.txt` `logs/z-timeline.txt` `logs/z-compute-copies.txt`
  `logs/z-bigcopy-context.txt` `logs/z-peer-doublecount.txt` `logs/z-gguf-sizes.txt`
  `logs/z-base-verbose.log` `logs/z-sm-none.log` `logs/z-sm-row.log` `logs/z-sqlite-tables.txt`
  `logs/z-sqlite-schema.txt` `logs/z-sync-callsites.txt`.
- No source changes in this lane: nothing I found justified editing `ggml/src/ggml-cuda/**`
  (see Fix), and `src/**` is outside the fence.
