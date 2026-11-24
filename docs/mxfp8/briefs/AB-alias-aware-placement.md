# Brief AB — Make looped-layer device placement alias-aware (measured +27% prefill available)

## The finding you are fixing (all measured; don't re-derive)

`docs/mxfp8/26-sync-attribution.md` has the full evidence. Summary: with `-sm layer`, Inkling-2-Mini spends **~77% of its 6,991 `cudaStreamSynchronize` calls (3.12 s)** sitting around **CUDA0↔CUDA1 peer copies** — 5,460 `cudaMemcpyPeerAsync` per run, **~33 GB of peer traffic** in a 5.5 s prefill (5.4–10.7 GB/s), host inside a sync 90–100 ms of every 100 ms.

**Cause:** `src/models/inkling2-loop.cpp:68` (`post_arch_tensors`) aliases the replayed layers onto the physical tensors: `layers[il] = layers[layer_phys(il)]` for `il >= n_phys`. The fitter/allocator is unaware of the aliasing, so with a split at layer 24 the replayed applications on **CUDA1** read tensors that live on **CUDA0**, once per tensor per boundary. Per-size attribution confirms it's *weights*, not activations: 4,718,592 B (`attn_k`/`attn_v` BF16 `[2304,1024]`) ×1,785 each way.

**Already rejected, don't retry:** (a) making the cross-device copies non-blocking — removes the waiting but leaves 33 GB on the critical path and breaks copy liveness (`ggml-cuda.cu:1183`/`:1234`, and `device_cpy_tensor` already uses events at `:3281-3286`); (b) `-sm none` as a *config* — it does give prefill 506→642 t/s (+27%) but decode collapses 20.6→11.5 because CUDA1's capacity is lost and tensors spill to CPU (`logs/z-sm-none.log`). It is the isolation experiment, not the answer. (c) MoE cache — off by default, no effect.

Model facts: 24 unique layers, 48 applications; `loop_start 2`, `loop_end 13`, `loop_count 2` → layers 2..13 are applied 3× total; unique-once layers are 0,1,14..23. Weight bytes: NVFP4 experts 10.68 GiB (blk 0–19), MXFP8 experts 3.92 GiB (blk 20–23), BF16/F32 rest. Per card: 15.79 GiB free.

## Job

Make device assignment for a replayed layer **follow the device of its physical tensors**, so no aliased weight is ever read cross-device. Design space to evaluate, in the order you should test them:

1. **Alias-aware fit/split.** Where the fitter or `llama_model::load_tensors` decides per-layer devices, treat `il` and `layer_phys(il)` as one unit — they must land on the same buffer/device. Look at `src/llama-model.cpp` (device assignment + `n_gpu_layers` handling), `src/llama-model-loader.cpp` (the `W tensor overrides to CPU are used with mmap enabled` message lives nearby), `src/llama-context.cpp` (`resolve_fused_ops`, FA assignment), and how `n_layer_phys()` / `layer_phys()` are already exposed as virtuals (`inkling2-loop.cpp` overrides them — the plumbing to know "this layer is an alias" **already exists**; the placement code just doesn't consult it).
   If the right place is "compute the split point in *unique-layer* space, then map back", the arithmetic is: hot set = layers 2..13 (12 unique, applied 3×) plus their expert weights; cold set = 0,1,14..23 (12 unique, applied once). Putting the whole aliased group on one card and the once-used layers on the other eliminates cross-device reads *and* keeps both cards loaded — that is the configuration the +27% number is pointing at.
2. **Report-only outcome is acceptable.** If this needs an upstream design conversation (e.g. the allocator must know about shared tensors generally, which would also help other weight-reuse models), write the RFC into the doc with the measurements, the code anchors, and the two candidate designs. That is a real deliverable — do not force a hack.

## Gates

1. **Prefill improves toward the isolated ceiling without wrecking decode.** Fixed harness: `llama-completion -m ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf -c 8192 -f logs/g0-prompt-long.txt -n 64 --temp 0 -no-cnv -s 42`. Baselines: prefill **506.26–509.17**, decode **20.58–22.26** (`logs/x-sync-default.log`, `logs/z-base-verbose.log`); `-sm none` reference point prefill 642.40 / decode 11.46. Target: prefill materially above 510 **with decode ≥ 22**.
2. **The peer-copy count actually drops.** Re-capture with nsys on the same command and show `cudaMemcpyPeerAsync` (was 5,460) and `cudaStreamSynchronize` (was 6,991, 4.457 s) in a before/after table. Reuse `scripts/inkling_native/z_sync_attrib.py`. **A t/s number without the traffic evidence is not accepted** — the whole finding was that wall clock hid the mechanism.
3. **Correctness invariant:** `llama-completion -p "Q: What is 17 times 4? A:" -n 24 --temp 0 -no-cnv -s 42` prints `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on CUDA **and** `-ngl 0` CPU; `test-backend-ops -p mxfp8` stays **237/237** on CUDA0 and CUDA1 and **306/306** on CPU.
4. **No regression for non-loop models**: run one dense control (e.g. `~/models/timlikesai/Qwen3.5-0.8B-Q8_0.gguf`) before/after and confirm nothing changes for a model with no aliasing.

## Rules

- **You are the only builder.** `cmake --build build-cuda -j 22`, never reconfigure, tee to `logs/ab-build-<n>.log`, read the complete output. Another lane may be measuring python-side KLD — it does not compile, but it may use the GPU; if timings wobble, say so and re-run.
- Fence: you own `src/**`, `ggml/src/ggml-cuda/ggml-cuda.cu` (only `resolve_fused_ops`/device-assignment areas), `docs/mxfp8/27-alias-aware-placement.md`, `logs/`. **Do not touch `conversion/**`, `scripts/inkling_native/**` beyond adding your own script, `tests/test-backend-ops.cpp`, or `ggml/src/ggml-cuda/mmq*`** (that path is committed and green).
- **Never pin a nonzero `-ngl`** (disables fitting, OOMs this 2-card setup). `llama-bench` is unusable here (no fitter; `--tensor-split` defaults to 0).
- No cherry-picks from stale branches, no rebase, never touch `master`. Check `git status` before editing and never revert work that isn't yours. **Never delete a file or log**; nothing in `~/models/timlikesai/` gets removed. No docker/systemctl.
- **Write `docs/mxfp8/27-alias-aware-placement.md` within your first 4 tool calls** (the finding, your chosen approach, a results table) and update it continuously. Fifteen lanes have timed out this session with nothing on disk — partial-on-disk beats perfect-and-lost, every time.
- 45-minute budget; finalize at 38; leave the tree building.
