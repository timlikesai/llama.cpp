# 27 — Make looped-layer device placement alias-aware (Brief AB)

Status: **done, measured.** Prefill 507.6 -> 847.4 t/s with FA still off (like the baseline),
902.8 t/s once FA resolves; peer copies 5,460 -> 1,106 and `cudaStreamSynchronize` 6,991 -> 972;
decode 22.2 -> 39.6/40.4 t/s. Every claim below is a number in `logs/ab-*.log` / `logs/ab-*.txt`
or a `file:line` in this tree.

Fence: `src/**`, `ggml/src/ggml-cuda/ggml-cuda.cu` (device-assignment / `resolve_fused_ops` only),
this doc, `logs/`, plus one new measurement script `scripts/inkling_native/ab_peer_traffic.py`.
No CUDA, converter, or test-source changes were needed; `ggml-cuda.cu` was not touched.

## The finding being fixed (from `docs/mxfp8/26-sync-attribution.md`, not re-derived)

With `-sm layer` on 2x RTX 5060 Ti an Inkling-2 looped model spent ~77% of its 6,991
`cudaStreamSynchronize` calls (3.12 s) blocked on 5,460 `cudaMemcpyPeerAsync` (~33-57 GB)
because a *replayed* cache slot (`il >= n_phys`) was placed from its own slot index while the
tensors it reads were placed from the *physical* index:

- `src/models/inkling2-loop.cpp:3` `layer_phys(il)` is the alias, `:68` `post_arch_tensors`
  copies `layers[layer_phys(il)]` into `layers[il]`, so the weights belong to the physical layer;
- `src/llama-model.cpp` `load_tensors` decided `pimpl->dev_layer[il]` from `il` alone, so the
  alias and its physical twin sat on different cards;
- `dev_layer()` is what `resolve_fused_ops` (`src/llama-context.cpp:693`), the KV cache
  (`src/llama-kv-cache.cpp:217`), the recurrent state (`src/llama-memory-recurrent.cpp:89`) and
  the MoE cache (`src/llama-moe-cache.cpp:428`) ask, so the slot's *nodes and KV* followed the
  slot while its *weights* stayed on the other device -> one peer copy + one blocking sync per
  tensor per boundary.

Pre-change evidence in this tree (same model, `-sm layer`): `logs/ab-before-run.log` and
every other 2-GPU Inkling log from earlier lanes ends with
`W resolve_fused_ops: layer 24/25 is assigned to device CUDA1 but Flash Attention is assigned to
device CUDA0` followed by `Flash Attention not supported, set to disabled` - the split also
turned FA off completely.

## Change (brief design 1, alias-aware placement)

Three edits, +18/-8 lines, no new subsystem:

1. `src/llama-model.h` `llama_model`: new virtual
   `virtual int layer_phys(int il) const { return il; }` next to `dev_layer()`, documented
   contract: identity unless the model replays layers, and `il` below the physical count returns
   `il` (so the result doubles as the slot's rank among unique layers).
   `src/models/models.h:2553` marks the existing `llama_model_inkling2_loop::layer_phys` as
   `override` - it already satisfied the contract, nothing there changed.
2. `src/llama-model.cpp` `load_tensors`: the split is computed in **unique-layer space**
   (`n_layer_unit = max(layer_phys(il)) + 1`, then `i_gpu_start` / `act_gpu_layers` / the
   `upper_bound` over `splits` all use that unit index), so slots that add no weights no longer
   consume device budget, and
   `pimpl->dev_layer[il] = get_layer_buft_list(layer_phys(il))` makes an alias inherit the
   physical layer's device. `dev_output` uses `n_layer_unit` (was `n_layer_all`).
3. Nothing else. The fitter in `common/fit.cpp` and the CUDA backend are untouched: making the
   slot follow its physical layer is enough, because `ggml_backend_sched` places a node on the
   device of the weight tensors it reads.

Design alternatives considered and rejected: splitting in slot space and then re-pinning the
aliases (leaves one card holding every weight - that is `-sm none`, whose decode collapses to
11.5 t/s because CUDA1's capacity is lost); teaching the allocator about shared tensors
generally (a real upstream conversation, but not needed to get the traffic off the link).

For a model with no aliasing `layer_phys()` is the identity, `n_layer_unit == n_layer_all`, and
every expression in `load_tensors` evaluates exactly as before - gate 4 confirms it on a dense
control.

## Results (gates 1-2)

Model `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` (24 physical layers, 48 cache slots,
loop core `[2,14)`), `-sm layer`, harness
`llama-completion -m ... -c 8192 -f logs/g0-prompt-long.txt -n 64 --temp 0 -no-cnv -s 42`.
No other process was on the GPUs during any of these runs.

| config | prefill t/s | decode t/s | `cudaMemcpyPeerAsync` | `cudaStreamSynchronize` | Flash Attention |
|---|---|---|---|---|---|
| before, `-sm layer` (`logs/ab-before-run.log`) | 507.61 | 22.18 | 5,460 | 6,991 / 4.457 s | force-disabled |
| before, nsys (`logs/w-nsys-run.log`, lane Z) | 503.79 | - | 5,460 | 6,991 | force-disabled |
| reference `-sm none`, not shippable | 642.40 | 11.46 | - | - | - |
| **after, `-fa off`** (`logs/ab-after-run-faoff.log`) | **847.40** | **39.59** | - | - | off (as baseline) |
| **after, default/auto** (`logs/ab-after-run.log`) | **902.76** | **40.40** | 1,106 | 972 / 0.083 s | **enabled** |
| after, nsys (`logs/ab-after-nsys-run.log`) | 888.84 | - | 1,106 | 972 | enabled |

Traffic evidence (`scripts/inkling_native/ab_peer_traffic.py`, output in
`logs/ab-peer-traffic.txt` and `logs/ab-peer-traffic-b256.txt`; the nsys capture for "before" is
lane Z's `logs/w-nsys-prefill.sqlite`, same model/prompt/`-sm layer`, whose 503.79 t/s matches my
fresh 507.61 t/s baseline to within 1%):

| run (2,817-token prefill) | `cudaMemcpyPeerAsync` | `cudaEventRecord`/`cudaStreamWaitEvent` | `cudaStreamSynchronize` | peer MiB (device-side records) | dominant copy size |
|---|---|---|---|---|---|
| before, `-b 2048` | 5,460 | 5,460 / 5,460 | 6,991 (4.457 s) | 57,353 | 4,718,592 B x 3,570 (=1,785 each way, the `attn_k`/`attn_v` fingerprint) |
| after, `-b 2048` | 1,106 (-80%) | 1,218 / 2,219 | **972 (-86%, 0.083 s)** | 8,867 (-85%) | 4,718,592 B x 1,470 |
| after, `-b 256` | 2,054 | 2,262 / 4,121 | 1,102 | 8,867 (identical) | 2,359,296 B x 3,234 |

The residual traffic is per-token activation handoff at the CUDA0/CUDA1 boundary, not per-tensor
weight fetch: halving the ubatch 8x (2048 -> 256) leaves the peer total at exactly the same
8,867 MiB for the same token count while the dominant record size drops by the same factor.
The weight fingerprint of the before run (`4,718,592 B` x 1,785 per direction) is gone. The
remaining syncs are the `MUL_MAT_ID` fallback pairs (742, unchanged) that lane Z measured at 21%.

Memory now splits across both cards instead of piling on one (`logs/ab-after-run.log` vs
`logs/ab-before-run.log`):

```
before: CUDA0 self 14731 MiB (model 13152) | CUDA1 self 5742 MiB (model 4048)   + 19.2 GiB compute-buffer OOM retry
after : CUDA0 self  9558 MiB (model 7855)  | CUDA1 self 13484 MiB (model 9344)  + no OOM, no CPU tensor overrides
```

Gate 1 (prefill materially above 510 **with** decode >= 22): **passed** - 847/39.6 with FA off
and 903/40.4 with FA, vs 507.6/22.2 before. The +67% prefill at matched FA state is the
placement fix; FA coming back is worth a further +6.5% prefill / +2% decode.

## Correctness (gates 3-4)

| check | result |
|---|---|
| `llama-completion -p "Q: What is 17 times 4? A:" -n 24 --temp 0 -no-cnv -s 42 --flash-attn off` (2-GPU CUDA) | `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` - byte-identical to the pre-change reference `logs/native-fixed-gpu.log` (`logs/ab-gate3-cuda-faoff.log`) |
| same probe with FA auto (now enabled) | `A: 68 B: 64 C: 72 D: 70 Answer: A Explanation: 17 x 4` - first 4 answers identical, then greedy diverges because FA changes the attention numerics (`logs/ab-gate3-cuda.log`) |
| same probe `-ngl 0` (CPU) `-c 2048` | `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` (`logs/ab-gate3-cpu-ngl0-c2048.log`) |
| `test-backend-ops -p mxfp8 -b CUDA0` | 237/237 (`logs/ab-gate3-ops-cuda0.log`) |
| `test-backend-ops -p mxfp8 -b CUDA1` | 237/237 (`logs/ab-gate3-ops-cuda1.log`) |
| `test-backend-ops -p mxfp8 -b CPU` | 306/306 (`logs/ab-gate3-ops-cpu.log`) |
| dense control `Qwen3.5-0.8B-Q8_0` on both GPUs | loads, `offloaded 26/26 layers to GPU`, plain contiguous split (units 0-12 CUDA0, 13-25 CUDA1 - identical to the pre-change formula), FA enabled, 21,454 / 345 t/s (`logs/ab-gate4-dense-2gpu.log`, `logs/ab-gate4-dense-2gpu-run.log`) |

## Open items / risks

1. **FA now auto-enables on a looped multi-GPU split** (the device mismatch that disabled it is
   gone). That is the intended consequence, but it changes greedy tokens versus the FA-off
   reference, so gate 3's exact string holds with `--flash-attn off` on CUDA and with `-ngl 0` on
   CPU, not with FA on. Whether FA-on is bit-comparable for this model is a separate question
   (KLD lane territory), not something this change should decide.
2. **`-ngl` semantics for looped models moved to unique-layer space** (`-ngl 12` now means 12
   *physical* layers, previously 12 *slots*, which for slots >= n_phys selected aliased slots
   whose weights sat elsewhere). Nothing in this repo pins `-ngl` for loop models, and non-loop
   models are unaffected by construction.
3. **Pre-existing, out of fence:** `common/fit.cpp:496`
   `sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl)` divides by zero when
   the user pins `-ngl 0` and the `nd > 1` memory-deficit branch is reached - `-ngl 0` at the
   default context SIGFPEs in the fitter (`logs/ab-gate3-cpu-ngl0-gdb.log`), independent of this
   change (my change is provably inert at `-ngl 0`: every unit resolves to CPU either way).
   Use `-c 2048` (or a CPU device list) for CPU comparisons. `common/**` is not in my fence.
4. **Other aliasing archs are still alias-unaware**: `src/models/metis16.cpp:164` and
   `src/models/nanbeige.cpp:23-31` build aliased slots too but do not override `layer_phys`, so
   they keep the old (wrong under `-sm layer`) placement. The hook now exists; they need their
   own alias function and a machine that can host them to test - deliberately not done blind.
   `llama_meta_device_get_split_state` (`src/llama-model.cpp:429`) is still slot-index based; it
   only feeds `-sm tensor`, which this arch refuses.
5. The "before" nsys capture is lane Z's sqlite, not a re-run: I rebuilt before capturing, so the
   pre-change binary was gone. Its prompt, model, `-sm layer` config and t/s all match my fresh
   baseline, and the residual-traffic claim does not depend on it (the batch-size sweep in the
   after build carries that argument on its own).

## Artifacts

- `logs/ab-before-run.log` `logs/ab-after-run.log` `logs/ab-after-run-faoff.log`
- `logs/ab-after-nsys-prefill.nsys-rep` + `.sqlite` (`-b 2048`), `logs/ab-after-nsys-b256.nsys-rep` + `.sqlite`
- `logs/ab-peer-traffic.txt` `logs/ab-peer-traffic-b256.txt` `logs/ab-before-nsys-attrib.txt` `logs/ab-after-nsys-attrib.txt`
- `logs/ab-gate3-*.log` `logs/ab-gate4-dense-2gpu*.log` `logs/ab-build-1.log`
- `scripts/inkling_native/ab_peer_traffic.py` (new): peer-copy/sync counters + per-size histogram from an nsys sqlite export.
